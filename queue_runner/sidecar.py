"""Isolated vLLM TPU environment and server lifecycle."""
from pathlib import Path
import argparse,json,os,re,signal,subprocess,sys,time,urllib.request,shutil,hashlib
from .io import put,Redactor
from .resources import chip_bounds

def validate_profile(p):
    if not isinstance(p,dict) or not isinstance(p.get('model'),str) or not p['model']:raise ValueError('Model ID required')
    if not re.fullmatch('[a-f0-9]{40}',p.get('revision','')):raise ValueError('Pin an immutable Hugging Face model revision')
    chip_bounds(p.get('chips'))
    if p.get('deployment_blocked_reason'):raise ValueError(p['deployment_blocked_reason'])
    if p.get('backend','vllm_jax')!='vllm_jax':raise ValueError('No adapter installed for selected inference backend')
    if p.get('runner','generate') not in ['generate','pooling']:raise ValueError('Unknown model runner')
    if p.get('dtype','bfloat16') not in ['auto','bfloat16','float16','float32']:raise ValueError('Unknown model dtype')
    for name,default,low,high in [('numa_node',None,0,1024),('cpu_threads',8,1,224),('port',8000,1024,65535),
        ('max_model_len',4096,128,131072),('max_num_seqs',4,1,128),('startup_seconds',1800,1,7200),
        ('max_seconds',7200,1,86400),('min_remaining_seconds',1800,0,86400),('cpu_offset',0,0,223)]:
        value=p.get(name,default)
        if type(value)!=int or not low<=value<=high:raise ValueError('Invalid '+name)
    utilization=p.get('memory_utilization',.75)
    if type(utilization) not in [float,int] or not .1<=utilization<=.85:raise ValueError('Keep memory utilization between .1 and .85')
    if p.get('port',8000)==8765:raise ValueError('Inference port collides with queue control')
    return p


def server_command(python,profile,cache):
    command=[str(python),'-m','vllm.entrypoints.cli.main','serve',profile['model'],
        '--revision',profile['revision'],'--host','127.0.0.1','--port',str(profile.get('port',8000)),
        '--dtype',profile.get('dtype','bfloat16'),'--tensor-parallel-size',str(len(profile['chips'])),
        '--max-model-len',str(profile.get('max_model_len',4096)),
        '--max-num-seqs',str(profile.get('max_num_seqs',4)),
        '--gpu-memory-utilization',str(profile.get('memory_utilization',.75)),
        '--download-dir',str(cache/'models')]
    if profile.get('runner')=='pooling':command+=['--runner','pooling']
    if profile.get('quantization'):command+=['--quantization',profile['quantization']]
    if profile.get('tool_call_parser'):command+=['--tool-call-parser',profile['tool_call_parser'],'--enable-auto-tool-choice']
    return command


def ensure_environment(cache,state):
    if sys.version_info[:2]!=(3,12) or sys.platform!='linux':raise RuntimeError('Pinned vLLM wheel requires Linux CPython 3.12')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    Path(state).parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    put(state,{'phase':'waiting_environment'})
    with (cache/'environment-install.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        return _ensure_environment(cache,state)


def _ensure_environment(cache,state):
    if shutil.disk_usage(cache).free<32*1024**3:raise RuntimeError('At least 32 GiB free staging space required for this profile')
    lock=Path(__file__).resolve().parents[1]/'environments/vllm-tpu-py312.lock'
    digest=hashlib.sha256(lock.read_bytes()).hexdigest()
    env=cache/('vllm-tpu-0.32.0-'+digest[:16]);python=env/'bin/python';marker=env/'installation.json'
    if not marker.exists():
        put(state,{'phase':'installing','package':'vllm-tpu==0.32.0'})
        subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',
            '--extra-index-url','https://download.pytorch.org/whl/cpu',
            '--require-hashes','--only-binary=:all:','-r',str(lock)],check=True)
        subprocess.run([str(python),'-m','pip','check'],check=True)
        frozen=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        put(marker,{'package':'vllm-tpu==0.32.0','lock_sha256':digest,'freeze':frozen})
    shutil.copyfile(marker,Path(state).parent/'environment.json')
    return python


def run(profile,state,cache):
    validate_profile(profile);state=Path(state);started=time.monotonic();process=None
    token=os.environ.get('VLLM_API_KEY')
    if not token:raise ValueError('Authenticated inference requires the queue control token')
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        Path(cache).mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(cache).free<profile.get('min_free_disk_gib',32)*1024**3:raise RuntimeError('Insufficient staging space for selected model profile')
        python=ensure_environment(Path(cache),state)
        if stopping:return
        # This short-lived child closes its TPU client before the serving child starts.
        probe="import jax,json; d=jax.devices(); assert len(d)=="+str(len(profile['chips']))+" and all(v.platform=='tpu' for v in d), str(d); print(json.dumps({'jax':jax.__version__,'devices':[str(v) for v in d]}))"
        put(state,{'phase':'device_check'})
        output=subprocess.check_output([str(python),'-c',probe],text=True,timeout=180)
        (state.parent/'devices.txt').write_text(output)
        if stopping:return
        command=server_command(python,profile,Path(cache));put(state,{'phase':'loading','model':profile['model']})
        process=subprocess.Popen(command)
        deadline=time.monotonic()+profile.get('startup_seconds',1800);ready=False
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not stopping and time.monotonic()-started<profile.get('max_seconds',7200):
            if process.poll() is not None:raise RuntimeError('vLLM exited with code '+str(process.returncode))
            if not ready:
                try:
                    request=urllib.request.Request('http://127.0.0.1:'+str(profile.get('port',8000))+'/v1/models',headers={'Authorization':'Bearer '+token})
                    with opener.open(request,timeout=2) as response:models=json.load(response)
                    if not models.get('data'):raise ValueError('No served model returned')
                    put(state,{'phase':'ready','model':profile['model'],'revision':profile['revision'],
                        'local_api_url':'http://127.0.0.1:'+str(profile.get('port',8000))+'/v1'})
                    ready=True
                except (OSError,ValueError):
                    if time.monotonic()>=deadline:raise TimeoutError('vLLM readiness deadline exceeded')
            time.sleep(1)
        put(state,{'phase':'stopped','reason':'requested' if stopping else 'service_time_limit'})
    except BaseException as error:
        put(state,{'phase':'failed','error':Redactor([token,os.environ.get('HF_TOKEN','')]).clean(str(error))})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--state',required=True);p.add_argument('--cache',required=True)
    args=p.parse_args();run(json.loads(Path(args.profile).read_text()),args.state,args.cache)


if __name__=='__main__':main()
