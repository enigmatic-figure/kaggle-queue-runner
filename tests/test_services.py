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
        m.before_job(manifest,job,snapshot,'b'*40,100)
        assert m.process.pid==pid  # Existing residency does not need a new startup budget.
        m.before_job(manifest,{},snapshot,'b'*40,3400)
        assert m.process is None and m.status()['phase']=='deferred'
        with pytest.raises(ProcessLookupError):os.kill(pid,0)
        m.before_job(manifest,job,snapshot,'b'*40,3300)
        assert m.process.pid!=pid
        launches=[json.loads(p.read_text()) for p in (tmp_path/'run/services').rglob('launch.json')]
        assert sorted(p['source_commit'] for p in launches)==['a'*40,'b'*40]
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


def test_model_profiles_select_pooling_precision_and_actual_chip_count(tmp_path):
    from queue_runner.sidecar import server_command,validate_profile
    profile={'model':'Qwen/Qwen3-Embedding-0.6B','revision':'a'*40,'numa_node':0,'chips':[0],
        'runner':'pooling','dtype':'auto','backend':'vllm_jax'}
    validate_profile(profile)
    command=server_command('/env/python',profile,tmp_path)
    assert command[command.index('--tensor-parallel-size')+1]=='1'
    assert command[command.index('--dtype')+1]=='auto'
    assert command[command.index('--runner')+1]=='pooling'


def test_pool_rejects_conflicting_multi_model_layout_before_launching(tmp_path):
    assert hasattr(services,'Pool'), 'Missing multi-model service pool'
    profile={'model':'embedding','revision':'a'*40,'numa_node':0,'chips':[0],'port':8001,'cpu_threads':1}
    (tmp_path/'a.json').write_text(json.dumps(profile));(tmp_path/'b.json').write_text(json.dumps({**profile,'port':8002,'cpu_offset':1}))
    pool=services.Pool(tmp_path/'run',topology={'nodes':{0:[0,1],1:[2]},'chips':{i:i//4 for i in range(8)}})
    try:
        pool.before_job({'services':[{'id':'a','profile':'a.json','enabled':True},{'id':'b','profile':'b.json','enabled':True}]},
            {'resources':{'tpu_chips':[],'numa_node':1}},tmp_path,'a'*40,3600)
        assert pool.status()['phase']=='deferred' and 'overlap' in pool.status()['reason']
        assert not pool.children
    finally:pool.close()


@pytest.mark.skipif(os.name!='posix',reason='Linux process affinity')
def test_two_service_processes_remain_disjoint_and_handoff_together(tmp_path):
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<3:pytest.skip('Three allowed CPUs required')
    topology={'nodes':{0:cpus[:2],1:cpus[2:3]},'chips':{i:i//4 for i in range(8)}}
    script=tmp_path/'server.py';script.write_text('import pathlib,os,json,time\np=pathlib.Path(os.environ["QUEUE_SERVICE_STATE"]);p.write_text(json.dumps({"phase":"ready","observed_chips":os.environ["TPU_VISIBLE_CHIPS"],"observed_cpus":sorted(os.sched_getaffinity(0))}));time.sleep(30)')
    specs=[]
    for i in range(2):
        path=tmp_path/f'{i}.json';path.write_text(json.dumps({'model':'embed','revision':'a'*40,'chips':[i],
            'numa_node':0,'cpu_threads':1,'cpu_offset':i,'port':8001+i}))
        specs.append({'id':'e'+str(i),'profile':path.name,'enabled':True})
    def factory(root,**kw):return services.Manager(root,**kw,command_factory=lambda p,s:[sys.executable,str(script)])
    pool=services.Pool(tmp_path/'run',topology=topology,secrets={'QUEUE_CONTROL_TOKEN':'example'},manager_factory=factory)
    try:
        pool.before_job({'services':specs},{'resources':{'tpu_chips':[],'numa_node':1}},tmp_path,'a'*40,3600)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and pool.status()['phase']!='ready':time.sleep(.05)
        states=pool.status()['services'];assert pool.status()['phase']=='ready'
        assert [states['e'+str(i)]['observed_chips'] for i in range(2)]==['0','1']
        assert states['e0']['observed_cpus']!=states['e1']['observed_cpus']
        processes=[m.process for m in pool.children.values()]
        pool.before_job({'services':specs},{},tmp_path,'a'*40,3500)
        assert pool.status()['phase']=='deferred' and all(p.poll() is not None for p in processes)
    finally:pool.close()


@pytest.mark.skipif(os.name!='posix' or sys.version_info[:2]!=(3,12),reason='Pinned Linux Python 3.12 installer')
def test_parallel_models_install_shared_environment_only_once(tmp_path,monkeypatch):
    import concurrent.futures
    from types import SimpleNamespace
    from queue_runner import sidecar
    calls=[]
    def run(command,**kwargs):
        if 'venv' in command:
            calls.append('venv');time.sleep(.15);(tmp_path/'unused').touch()
            __import__('pathlib').Path(command[-1]).mkdir(parents=True,exist_ok=True)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(sidecar.subprocess,'run',run)
    monkeypatch.setattr(sidecar.subprocess,'check_output',lambda *a,**k:'example==1\n')
    monkeypatch.setattr(sidecar.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(sidecar.ensure_environment,tmp_path/'cache',tmp_path/str(i)/'state.json') for i in range(2)]
        values=[f.result() for f in futures]
    assert calls==['venv'] and values[0]==values[1]
