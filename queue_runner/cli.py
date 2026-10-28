"""Queue worker, local notification receiver and private control client."""
from pathlib import Path
import argparse,json,os,time,uuid,secrets,threading,tempfile,urllib.request,shutil
from .io import put,Redactor


def get_secret(label):
    if os.environ.get(label):return os.environ[label]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(label)
    except Exception:return None


def worker(args):
    from .runner import Runner
    from .source import GitSource
    from .scheduler import run_queue
    from .control import start
    from .network import Tailnet
    from .notifications import send
    run_id=args.run_id or 'kaggle-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    root=Path(args.root).resolve()/run_id
    if root.exists() and not args.resume:raise ValueError('Run directory exists; choose a new run ID or explicit --resume')
    names=['TS_AUTHKEY','QUEUE_CONTROL_TOKEN','QUEUE_NOTIFY_TOKEN','QUEUE_NOTIFY_URL','HF_TOKEN']
    private={name:value for name in names if (value:=get_secret(name))}
    r=Runner(root,secrets=private);source=GitSource(args.repo,args.branch,root/'.source')
    commit,manifest=source.refresh()
    put(root/'source.json',{'repository':args.repo,'branch':args.branch,'initial_commit':commit,'worker_id':run_id})
    network=None;server=None;publisher=None;publisher_thread=None;stop_sync=threading.Event();notification_threads=[]
    def notify(record):
        if record['event'] not in ['allocation_acquired','ready','job_started','job_finished','runner_finished']:return
        packet={'worker_id':run_id,'event':record['event'],
            'control_url':network.control_url if network else None,
            **{k:record[k] for k in ['job_id','status'] if k in record}}
        def deliver():
            error=None
            for delay in [0,2,5]:
                if delay:time.sleep(delay)
                try:
                    if send(private.get('QUEUE_NOTIFY_URL'),private.get('QUEUE_NOTIFY_TOKEN'),packet,
                            proxy=network.proxy if network else None):
                        put(root/'notification-delivery.json',{'event':record['event'],'delivered':True,'unix_time':time.time()});return
                    return
                except Exception as e:error=type(e).__name__
            put(root/'notification-delivery.json',{'event':record['event'],'delivered':False,'error_type':error,'unix_time':time.time()})
        t=threading.Thread(target=deliver,daemon=True);notification_threads.append(t);t.start()
    r.listeners.append(notify)
    try:
        r.event('allocation_acquired',worker_id=run_id)
        if private.get('QUEUE_CONTROL_TOKEN'):
            server=start(r,private['QUEUE_CONTROL_TOKEN'],port=args.port)
        if args.network!='off' and private.get('TS_AUTHKEY') and server:
            network=Tailnet(Path(tempfile.gettempdir())/('queue-private-'+uuid.uuid4().hex))
            try:
                network.start(private['TS_AUTHKEY'],run_id,port=args.port)
                r.event('tailnet_connected',worker_id=run_id)
            except Exception as error:
                network.close();network=None
                r.event('tailnet_unavailable',error_type=type(error).__name__)
                if args.network=='required':raise
        elif args.network=='required':raise ValueError('TS_AUTHKEY and QUEUE_CONTROL_TOKEN Kaggle Secrets are required for private control')
        sink=manifest.get('artifact_sink')
        if sink:
            if not private.get('HF_TOKEN'):r.event('artifact_sink_unavailable',reason='HF_TOKEN is not configured')
            else:
                from huggingface_hub import HfApi,download_bucket_files
                from .artifacts import Sink
                token=private['HF_TOKEN']
                publisher=Sink(root,sink['bucket'],sink['prefix'],client=HfApi(token=token),
                    download=lambda bucket,pairs,**kw:download_bucket_files(bucket,pairs,token=token,**kw),secrets=private.values())
                def sync_loop():
                    while True:
                        try:publisher.sync();put(root/'artifact-sync.json',{'ok':True,'unix_time':time.time()})
                        except Exception as error:
                            put(root/'artifact-sync.json',{'ok':False,'error':Redactor(private.values()).clean(str(error)),'unix_time':time.time()})
                        if stop_sync.wait(60):
                            try:publisher.sync()
                            except Exception as error:put(root/'artifact-sync.json',{'ok':False,'error_type':type(error).__name__,'unix_time':time.time()})
                            return
                publisher_thread=threading.Thread(target=sync_loop,daemon=True);publisher_thread.start()
        r.event('ready',worker_id=run_id,private_control=bool(network))
        return run_queue(source,r,max_seconds=args.hours*3600,idle_seconds=args.idle_minutes*60,
                         poll_seconds=args.poll_seconds,secret_provider=get_secret)
    finally:
        stop_sync.set()
        if publisher_thread:publisher_thread.join(timeout=120)
        # Final callback gets a short opportunity to finish before disconnecting.
        for t in notification_threads:
            if t.is_alive():t.join(timeout=2)
        if server:server.shutdown();server.server_close()
        if network:network.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('worker');p.add_argument('--repo',required=True);p.add_argument('--branch',default='main')
    p.add_argument('--root',default='/kaggle/working/queue-runs');p.add_argument('--run-id');p.add_argument('--resume',action='store_true')
    p.add_argument('--hours',type=float,default=8);p.add_argument('--idle-minutes',type=float,default=5)
    p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--network',choices=['auto','off','required'],default='auto')
    p=commands.add_parser('listen');p.add_argument('--root',required=True);p.add_argument('--token-file',required=True)
    p.add_argument('--port',type=int,default=8766);p.add_argument('--hours',type=float,default=48);p.add_argument('--agent-id')
    p=commands.add_parser('tokens');p.add_argument('--directory',required=True)
    p=commands.add_parser('control');p.add_argument('--url',required=True);p.add_argument('--token-file',required=True)
    p.add_argument('action',choices=['status','events','logs','reload','pause','resume','checkpoint','cancel','stop']);p.add_argument('--job-id')
    args=parser.parse_args()
    if args.command=='worker':print(json.dumps(worker(args)),flush=True)
    elif args.command=='tokens':
        folder=Path(args.directory).resolve();folder.mkdir(parents=True,exist_ok=True)
        for name in ['control.token','notify.token']:
            p=folder/name
            if not p.exists():p.write_text(secrets.token_hex(32));p.chmod(0o600)
        print(json.dumps({'control_token_file':str(folder/'control.token'),'notify_token_file':str(folder/'notify.token'),'secret_values_printed':False}))
    elif args.command=='listen':
        from .notifications import listen
        executable=shutil.which('ai-devkit') or shutil.which('npx')
        command=[executable] if executable and 'ai-devkit' in Path(executable).name else [executable,'--yes','ai-devkit@latest'] if executable else None
        server=listen(args.root,Path(args.token_file).read_text().strip(),port=args.port,agent_id=args.agent_id,agent_command=command)
        print(json.dumps({'listening':'127.0.0.1:'+str(args.port),'agent_delivery_enabled':bool(args.agent_id and command)}),flush=True)
        try:
            deadline=time.monotonic()+args.hours*3600
            while time.monotonic()<deadline:time.sleep(min(30,max(0,deadline-time.monotonic())))
        finally:server.shutdown();server.server_close()
    else:
        token=Path(args.token_file).read_text().strip();path='/'+args.action
        if args.action=='logs':
            from urllib.parse import urlencode
            path+='?'+urlencode({'job_id':args.job_id or ''})
        request=urllib.request.Request(args.url.rstrip('/')+path,headers={'Authorization':'Bearer '+token},
            data=None if args.action in ['status','events','logs'] else b'{}')
        with urllib.request.urlopen(request,timeout=15) as response:print(response.read().decode())


if __name__=='__main__':main()
