"""Sequential workloads within one allocation."""
from pathlib import Path
import json,os,sys,time,threading,subprocess,signal,codecs
from .contracts import fingerprint,validate
from .io import put,Redactor,sha256


def terminate(process):
    if process.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    else:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name=='posix':
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:process.kill()
        process.wait()


class Runner:
    def __init__(self,root,secrets=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.secrets=secrets or {};self.pause=threading.Event();self.cancel=threading.Event()
        self.stop=threading.Event();self.reload=threading.Event();self.current=None
        self.lock=threading.RLock();self.process=None;self.listeners=[]
        self.services=None

    def status(self):
        with self.lock:value={'current':self.current,'paused':self.pause.is_set(),'stop_requested':self.stop.is_set()}
        if self.services:value['inference']=self.services.status()
        return value

    def event(self,kind,**fields):
        record={'event':kind,'unix_time':time.time(),**fields}
        line=Redactor(self.secrets.values()).clean(json.dumps(record))
        with self.lock:
            with (self.root/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
        print(line,flush=True)
        for listener in list(self.listeners):
            try:listener(record)
            except Exception:pass  # Optional delivery cannot prevent workload progress.

    def receipt(self,identity):
        p=self.root/'jobs'/identity/'receipt.json'
        return json.loads(p.read_text()) if p.exists() else None

    def run_one(self,job,snapshot,commit,remaining=None):
        # Validate the individual job without requiring dependency definitions here.
        validate({'version':1,'jobs':[{**job,'depends_on':[]}]})
        digest=fingerprint(job);identity=job['id'];folder=self.root/'jobs'/identity
        old=self.receipt(identity)
        if old:
            if old['spec_sha256']!=digest:raise ValueError('Changed specification requires a new job ID')
            if old['status']=='completed':return {**old,'status':'already_completed'}
            return {**old,'status':'interrupted' if old['status']=='running' else old['status']}
        snapshot=Path(snapshot).resolve();script=(snapshot/job['script']).resolve()
        if not script.is_relative_to(snapshot) or not script.is_file():raise ValueError('Script outside snapshot or missing')
        missing=[name for name in job.get('secrets',[]) if not self.secrets.get(name)]
        if missing:return {'job_id':identity,'status':'waiting_for_secrets','missing':missing}
        folder.mkdir(parents=True,exist_ok=True);output=folder/'output';output.mkdir(exist_ok=True)
        timeout=float(job.get('timeout_seconds',3600))
        if remaining is not None:timeout=min(timeout,remaining)
        if timeout<=0:return {'job_id':identity,'status':'deferred'}
        from .resources import job_plan,discover
        plan=job_plan(job,self.services.topology if self.services else discover())
        receipt={'job_id':identity,'source_commit':commit,'spec_sha256':digest,'status':'running',
            'started_unix':time.time(),'script':job['script'],'args':job.get('args',[]),
            'script_sha256':sha256(script)}
        put(folder/'receipt.json',receipt);self.cancel.clear()
        with self.lock:self.current=dict(receipt)
        self.event('job_started',job_id=identity,source_commit=commit)
        environment={k:v for k,v in os.environ.items() if k not in self.secrets}
        environment.update({name:self.secrets[name] for name in job.get('secrets',[])})
        environment.update(plan['environment'])
        environment.update({'QUEUE_OUTPUT_DIR':str(output),'QUEUE_JOB_ID':identity,'QUEUE_SOURCE_COMMIT':commit,
            'QUEUE_SHARED_DIR':str(self.root/'shared'),'QUEUE_CHECKPOINT_REQUEST':str(folder/'checkpoint-request.json'),
            'PYTHONUNBUFFERED':'1','PYTHONPATH':str(snapshot)+os.pathsep+environment.get('PYTHONPATH','')})
        (self.root/'shared').mkdir(exist_ok=True)
        outcome='failed';reader=None;process=None
        try:
            interpreter=os.environ.get('QUEUE_WORKLOAD_PYTHON',sys.executable)
            receipt['python']=interpreter
            receipt['resources']={'chips':plan['chips'],'cpus':plan['cpus']}
            command=[interpreter,'-u',str(script),*job.get('args',[])]
            if plan['cpus']:command=[sys.executable,'-m','queue_runner.affinity_exec',*command]
            process=subprocess.Popen(command,cwd=snapshot,
                env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=os.name=='posix')
            self.process=process
            def pump():
                decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');redactor=Redactor(self.secrets.values())
                with (folder/'log.txt').open('w',encoding='utf-8') as log:
                    while True:
                        block=os.read(process.stdout.fileno(),4096)
                        if not block:break
                        safe=redactor.feed(decoder.decode(block));log.write(safe);log.flush()
                    safe=redactor.feed(decoder.decode(b'',final=True),final=True);log.write(safe);log.flush()
            reader=threading.Thread(target=pump,daemon=True);reader.start();started=time.monotonic()
            while process.poll() is None:
                if self.cancel.is_set():outcome='cancelled';terminate(process);break
                if time.monotonic()-started>=timeout:outcome='timed_out';terminate(process);break
                time.sleep(.05)
            else:outcome='completed' if process.returncode==0 else 'failed'
            receipt['exit_code']=process.poll()
        except BaseException as error:
            if process is not None:terminate(process)
            receipt['error']=Redactor(self.secrets.values()).clean(str(error))
            if isinstance(error,(KeyboardInterrupt,SystemExit)):
                outcome='interrupted';raise
        finally:
            # Jobs own their Linux process group, including children whose parent
            # exited early. Do not leave GPU work or an inherited log pipe alive.
            if process is not None and os.name=='posix':
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if reader is not None:reader.join(timeout=10)
            receipt.update(status=outcome,finished_unix=time.time())
            put(folder/'receipt.json',receipt)
            with self.lock:self.current=None;self.process=None
            self.event('job_finished',job_id=identity,status=outcome)
        return receipt
