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
