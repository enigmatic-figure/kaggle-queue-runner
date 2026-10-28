import json,os,sys,time,subprocess
import pytest
from queue_runner import services


@pytest.mark.skipif(os.name!='posix',reason='Linux resource isolation')
def test_resident_service_survives_cpu_jobs_and_yields_before_full_tpu_job(tmp_path):
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<2:pytest.skip('Two allowed CPUs required')
    topology={'nodes':{0:[cpus[0]],1:[cpus[1]]},'chips':{i:i//4 for i in range(8)}}
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    (snapshot/'model.json').write_text(json.dumps({'model':'Qwen/Qwen3-8B','revision':'a'*40,
        'numa_node':0,'chips':[0,1,2,3],'cpu_threads':1,'port':8000}))
    script=tmp_path/'fake_service.py'
    script.write_text('import os,time,json,pathlib\np=pathlib.Path(os.environ["QUEUE_SERVICE_STATE"])\n'
        'p.write_text(json.dumps({"phase":"ready","cpus":sorted(os.sched_getaffinity(0)),"chips":os.environ["TPU_VISIBLE_CHIPS"]}))\ntime.sleep(30)\n')
    events=[]
    m=services.Manager(tmp_path/'run',topology=topology,secrets={'QUEUE_CONTROL_TOKEN':'example-key'},
        command_factory=lambda profile,state:[sys.executable,str(script)],event=lambda kind,**kw:events.append(kind))
    manifest={'services':[{'id':'llm-v1','profile':'model.json','enabled':True}]}
    job={'resources':{'tpu_chips':[],'numa_node':1}}
    try:
        m.before_job(manifest,job,snapshot,'a'*40,3600)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and m.status().get('phase')!='ready':time.sleep(.05)
        assert m.status()['phase']=='ready'
        assert m.status()['cpus']==[cpus[0]] and m.status()['chips']=='0,1,2,3'
        pid=m.process.pid
        m.before_job(manifest,job,snapshot,'b'*40,3500)
        assert m.process.pid==pid
        m.before_job(manifest,{},snapshot,'b'*40,3400)
        assert m.process is None and m.status()['phase']=='deferred'
        with pytest.raises(ProcessLookupError):os.kill(pid,0)
    finally:m.close()


def test_vllm_command_keeps_api_on_loopback_and_uses_budgeted_context(tmp_path):
    from queue_runner.sidecar import server_command,validate_profile
    profile={'model':'Qwen/Qwen3-8B','revision':'a'*40,'numa_node':0,'chips':[0,1,2,3],
        'port':8000,'max_model_len':4096,'max_num_seqs':4,'memory_utilization':.75}
    validate_profile(profile)
    command=server_command('/env/bin/python',profile,tmp_path)
    assert command[command.index('--host')+1]=='127.0.0.1'
    assert command[command.index('--tensor-parallel-size')+1]=='4'
    assert '--trust-remote-code' not in command
    assert command[command.index('--max-model-len')+1]=='4096'
    assert '--api-key' not in command  # Key belongs in the child environment.
    with pytest.raises(ValueError):validate_profile({**profile,'revision':'main'})


@pytest.mark.skipif(os.name!='posix',reason='Linux resource isolation')
def test_queue_reconciles_resident_service_and_pins_cpu_job_before_child_imports(tmp_path):
    from queue_runner.runner import Runner
    from queue_runner.source import GitSource
    from queue_runner.scheduler import run_queue
    from queue_runner import resources
    cpus=sorted(os.sched_getaffinity(0));chosen=cpus[0]
    origin=tmp_path/'origin';origin.mkdir()
    def git(*args):subprocess.run(['git','-C',str(origin),*args],check=True,stdout=subprocess.DEVNULL)
    git('init','-b','main');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    (origin/'job.py').write_text('import os,json,pathlib\npathlib.Path(os.environ["QUEUE_OUTPUT_DIR"],"observed.json").write_text(json.dumps({"cpus":sorted(os.sched_getaffinity(0)),"jax":os.environ.get("JAX_PLATFORMS")}))')
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[
        {'id':'cpu','script':'job.py','resources':{'tpu_chips':[],'numa_node':0,'cpu_threads':1}}]}))
    git('add','.');git('commit','-m','cpu')
    r=Runner(tmp_path/'run');calls=[]
    class Service:
        topology={'nodes':{0:[chosen]},'chips':{}}
        def before_job(self,manifest,job,snapshot,commit,remaining):calls.append(job['id'])
        def status(self):return {'phase':'deferred'}
    r.services=Service()
    run_queue(GitSource(str(origin),'main',tmp_path/'cache'),r,max_seconds=5,idle_seconds=.1,poll_seconds=.02)
    value=json.loads((tmp_path/'run/jobs/cpu/output/observed.json').read_text())
    assert calls==['cpu'] and value=={'cpus':[chosen],'jax':'cpu'}
    assert r.status()['inference']['phase']=='deferred'


@pytest.mark.skipif(os.name!='posix',reason='Linux resource isolation')
@pytest.mark.parametrize('exit_code',[0,7])
def test_finished_service_records_exit_code_without_restart_loop(tmp_path,exit_code):
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<2:pytest.skip('Two allowed CPUs required')
    topology={'nodes':{0:[cpus[0]],1:[cpus[1]]},'chips':{i:i//4 for i in range(8)}}
    (tmp_path/'model.json').write_text(json.dumps({'model':'Qwen/Qwen3-8B','revision':'a'*40,'numa_node':0,'chips':[0,1,2,3]}))
    m=services.Manager(tmp_path/'run',topology=topology,secrets={'QUEUE_CONTROL_TOKEN':'example'},
        command_factory=lambda p,s:[sys.executable,'-c','raise SystemExit('+str(exit_code)+')'])
    manifest={'services':[{'id':'bad-v1','profile':'model.json','enabled':True}]}
    job={'resources':{'tpu_chips':[],'numa_node':1}}
    try:
        m.before_job(manifest,job,tmp_path,'a'*40,3600)
        deadline=time.monotonic()+5
        phase='failed' if exit_code else 'stopped'
        while time.monotonic()<deadline and m.status().get('phase')!=phase:time.sleep(.05)
        assert m.status()['exit_code']==exit_code
        m.before_job(manifest,job,tmp_path,'a'*40,3500)
        assert m.process is None and m.status()['phase']==phase
    finally:m.close()


def test_invalid_optional_service_profile_defers_without_blocking_primary(tmp_path):
    topology={'nodes':{0:[0],1:[1]},'chips':{i:i//4 for i in range(8)}}
    m=services.Manager(tmp_path/'run',topology=topology)
    try:
        result=m.before_job({'services':[{'id':'llm','enabled':True,'profile':'missing.json'}]},
            {'resources':{'tpu_chips':[],'numa_node':1}},tmp_path,'a'*40,3600)
        assert result['chips']==[] and m.status()['phase']=='deferred'
    finally:m.close()
