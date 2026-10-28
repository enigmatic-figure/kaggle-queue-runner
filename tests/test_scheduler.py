import json,threading,time,subprocess
from queue_runner import scheduler
from queue_runner.source import GitSource
from queue_runner.runner import Runner


def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def test_same_allocation_accepts_new_work_after_start_without_repeating_completed_job(tmp_path):
    assert hasattr(scheduler,'run_queue'), 'Missing live queue dispatch loop'
    origin=tmp_path/'origin';origin.mkdir();git(origin,'init','-b','main')
    git(origin,'config','user.name','Test');git(origin,'config','user.email','test@example.invalid')
    script="from pathlib import Path\nimport os,time\ntime.sleep(.3)\nPath(os.environ['QUEUE_OUTPUT_DIR'],'result.txt').write_text('done')"
    (origin/'job.py').write_text(script)
    first={'id':'one','script':'job.py','timeout_seconds':10}
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[first]}))
    git(origin,'add','.');git(origin,'commit','-m','first')
    source=GitSource(str(origin),'main',tmp_path/'cache');runner=Runner(tmp_path/'state')
    thread=threading.Thread(target=scheduler.run_queue,args=(source,runner),
        kwargs={'max_seconds':10,'idle_seconds':1,'poll_seconds':.1});thread.start()
    deadline=time.time()+5
    while runner.current is None and time.time()<deadline:time.sleep(.02)
    assert runner.current is not None
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[first,
        {'id':'two','script':'job.py','timeout_seconds':10,'depends_on':['one']}]}))
    git(origin,'add','.');git(origin,'commit','-m','second')
    thread.join(timeout=10);assert not thread.is_alive()
    a=runner.receipt('one');b=runner.receipt('two')
    assert a['status']==b['status']=='completed' and a['source_commit']!=b['source_commit']
    assert (tmp_path/'state/jobs/one/output/result.txt').read_text()=='done'


def test_await_work_policy_uses_session_budget_and_retains_checkpoint_reserve(tmp_path):
    class Source:
        def refresh(self):return 'a'*40,{'jobs':[],'policy':{'idle_mode':'await_work','checkpoint_reserve_seconds':.1}}
    runner=Runner(tmp_path)
    result=scheduler.run_queue(Source(),runner,max_seconds=.4,idle_seconds=.03,poll_seconds=.01)
    assert result['reason']=='checkpoint_reserve'
    assert .25<=result['elapsed_seconds']<.45


def test_profile_job_selects_its_service_set_before_execution(tmp_path):
    origin=tmp_path/'origin';origin.mkdir();git(origin,'init','-b','main')
    git(origin,'config','user.name','Test');git(origin,'config','user.email','test@example.invalid')
    (origin/'job.py').write_text('pass')
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[
        {'id':'test','script':'job.py','service_set':'embedders'}],
        'service_sets':{'embedders':[{'id':'e','profile':'e.json','enabled':True}]}}))
    git(origin,'add','.');git(origin,'commit','-m','set')
    runner=Runner(tmp_path/'run');seen=[]
    class Services:
        topology={'nodes':{},'chips':{}}
        def before_job(self,manifest,*args):seen.append(manifest['services'][0]['id'])
        def status(self):return {}
    runner.services=Services()
    scheduler.run_queue(GitSource(str(origin),'main',tmp_path/'cache'),runner,max_seconds=5,idle_seconds=.03,poll_seconds=.01)
    assert seen==['e']


def test_selected_model_set_can_be_primary_during_idle_api_session(tmp_path):
    origin=tmp_path/'origin';origin.mkdir();git(origin,'init','-b','main')
    git(origin,'config','user.name','Test');git(origin,'config','user.email','test@example.invalid')
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[],
        'services':[{'id':'api','profile':'model.json','enabled':True}],
        'policy':{'idle_mode':'await_work','serve_when_idle':True,'checkpoint_reserve_seconds':.5}}))
    git(origin,'add','.');git(origin,'commit','-m','serve')
    runner=Runner(tmp_path/'run');seen=[]
    class Services:
        topology={'nodes':{1:[0]},'chips':{}}
        def before_job(self,manifest,job,snapshot,commit,remaining):seen.append((job['resources']['tpu_chips'],commit))
        def status(self):return {'phase':'ready'}
    runner.services=Services()
    result=scheduler.run_queue(GitSource(str(origin),'main',tmp_path/'cache'),runner,max_seconds=1.5,idle_seconds=.01,poll_seconds=.02)
    assert seen and seen[0][0]==[] and result['reason']=='checkpoint_reserve'
