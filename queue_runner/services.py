"""Resident services between compatible queue workloads."""
from pathlib import Path
import json,os,sys,subprocess,threading,time,hashlib,codecs,signal,tempfile
from . import resources
from .sidecar import validate_profile
from .runner import terminate
from .io import put,Redactor


class Manager:
    def __init__(self,root,*,topology=None,secrets=None,network=None,event=None,command_factory=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.topology=topology if topology is not None else resources.discover()
        put(self.root/'hardware.json',self.topology)
        self.secrets=secrets or {};self.network=network;self.event=event or (lambda *a,**k:None)
        self.command_factory=command_factory;self.process=None;self.record={'phase':'disabled'}
        self.lock=threading.RLock();self.shutdown=threading.Event();self.finished={};self.current_key=None
        self.state=None;self.log_thread=None;self.forwarded=None;self.last_phase=None
        self.monitor=threading.Thread(target=self._watch,daemon=True);self.monitor.start()

    def status(self):
        with self.lock:
            value=dict(self.record)
            if self.process is not None and self.state and self.state.exists():
                try:value.update(json.loads(self.state.read_text()))
                except (OSError,ValueError):pass
            return value

    def _stop(self,reason):
        if self.process is not None:
            process=self.process;terminate(process)
            if os.name=='posix':
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            if self.log_thread:self.log_thread.join(timeout=2)
            if self.state:put(self.state.parent/'termination.json',{'reason':reason,'exit_code':process.poll(),'unix_time':time.time()})
            self.process=None
            self.event('service_stopped',service_id=self.record.get('id'),reason=reason)
        if self.forwarded is not None:
            try:self.network.call('serve','--tcp='+str(self.forwarded),'off')
            except Exception:pass
            self.forwarded=None
        self.current_key=None;self.record={'phase':'stopped','reason':reason}

    def _watch(self):
        while not self.shutdown.wait(.2):
            with self.lock:
                if self.process is None:continue
                status=self.status();phase=status.get('phase')
                if self.process.poll() is not None:
                    exit_code=self.process.returncode
                    self.finished[self.current_key]=exit_code
                    previous=dict(self.record)
                    self._stop('service_process_exited');self.record={**previous,'phase':'failed' if exit_code else 'stopped','exit_code':exit_code,
                        'error':status.get('error','Service process exited')}
                    if exit_code:self.event('service_failed',service_id=previous.get('id'),exit_code=exit_code)
                elif phase=='ready' and self.last_phase!='ready':
                    self.event('service_ready',service_id=self.record.get('id'),private_api_url=self.record.get('private_api_url'))
                self.last_phase=phase

    def before_job(self,manifest,job,snapshot,commit,remaining):
        primary=resources.job_plan(job,self.topology)
        enabled=[s for s in manifest.get('services',[]) if s.get('enabled',False)]
        with self.lock:
            if not enabled:
                self._stop('disabled');return primary
            spec=enabled[0];snapshot=Path(snapshot).resolve();path=(snapshot/spec['profile']).resolve()
            try:
                if not path.is_relative_to(snapshot):raise ValueError('Service profile outside snapshot')
                profile=validate_profile(json.loads(path.read_text()))
                key=spec['id']+'-'+hashlib.sha256(path.read_bytes()).hexdigest()
                partition=resources.service_plan(profile,self.topology,primary)
                if os.name!='posix':raise ValueError('Resident TPU services require Linux')
                if not self.secrets.get('QUEUE_CONTROL_TOKEN'):raise ValueError('Queue control token required for inference API')
                if remaining<profile.get('min_remaining_seconds',1800):raise ValueError('Insufficient remaining time for model service')
            except (OSError,ValueError,TypeError,KeyError) as error:
                self._stop('resource_conflict');self.record={'phase':'deferred','reason':str(error),'id':spec['id']};return primary
            if self.process is not None and self.current_key==key:return primary
            self._stop('profile_changed')
            if key in self.finished:
                code=self.finished[key]
                self.record={'phase':'failed' if code else 'stopped','exit_code':code,
                    'reason':'Change service ID or profile for an explicit restart','id':spec['id']};return primary
            folder=self.root/'services'/key/('attempt-'+str(time.time_ns()));folder.mkdir(parents=True,exist_ok=True);self.state=folder/'state.json'
            profile={**profile,'max_seconds':min(profile.get('max_seconds',7200),max(1,int(remaining)))}
            path=folder/'profile.json';put(path,profile);put(self.state,{'phase':'starting'})
            cache=Path(tempfile.gettempdir())/'queue-model-cache'
            interpreter=os.environ.get('QUEUE_WORKLOAD_PYTHON',sys.executable)
            command=self.command_factory(profile,self.state) if self.command_factory else [interpreter,'-u','-m','queue_runner.sidecar',
                '--profile',str(path),'--state',str(self.state),'--cache',str(cache)]
            env={k:v for k,v in os.environ.items() if k not in self.secrets}
            env.update(partition['environment']);env.update(VLLM_API_KEY=self.secrets['QUEUE_CONTROL_TOKEN'],
                QUEUE_SERVICE_STATE=str(self.state),PYTHONPATH=str(snapshot)+os.pathsep+env.get('PYTHONPATH',''),
                HF_HOME=str(cache/'huggingface'),XDG_CACHE_HOME=str(cache/'xdg'),VLLM_XLA_CACHE_PATH=str(cache/'xla'),
                JAX_COMPILATION_CACHE_DIR=str(cache/'jax'),PIP_CACHE_DIR=str(cache/'pip'),
                HF_HUB_DISABLE_TELEMETRY='1',VLLM_NO_USAGE_STATS='1')
            if self.secrets.get('HF_TOKEN'):env['HF_TOKEN']=self.secrets['HF_TOKEN']
            wrapped=[interpreter,'-m','queue_runner.affinity_exec',*command]
            self.process=subprocess.Popen(wrapped,cwd=snapshot,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
            self.current_key=key;self.last_phase=None
            self.record={'id':spec['id'],'phase':'starting','pid':self.process.pid,'source_commit':commit,
                'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cpus':partition['cpus'],'chips':partition['chips']}
            if self.network:
                try:
                    port=profile.get('port',8000)
                    self.network.call('serve','--bg','--tcp='+str(port),'tcp://127.0.0.1:'+str(port))
                    self.forwarded=port
                    base=self.network.control_url.rsplit(':',1)[0]
                    self.record['private_api_url']=base+':'+str(port)+'/v1'
                except Exception as error:self.record['network_error_type']=type(error).__name__
            put(folder/'launch.json',self.record)
            process=self.process
            def pump():
                decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');redactor=Redactor(self.secrets.values())
                with (folder/'log.txt').open('w',encoding='utf-8') as out:
                    while True:
                        block=os.read(process.stdout.fileno(),4096)
                        if not block:break
                        out.write(redactor.feed(decoder.decode(block)));out.flush()
                    out.write(redactor.feed(decoder.decode(b'',final=True),final=True))
            self.log_thread=threading.Thread(target=pump,daemon=True);self.log_thread.start()
            self.event('service_started',service_id=spec['id'],source_commit=commit)
            return primary

    def close(self):
        self.shutdown.set();self.monitor.join(timeout=1)
        with self.lock:self._stop('runner_shutdown')
