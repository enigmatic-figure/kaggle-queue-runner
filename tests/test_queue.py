from pathlib import Path
import json,subprocess,sys,time,os
import pytest
from queue_runner import contracts,source,runner


def test_manifest_requires_unique_versioned_ids_and_relative_python_entrypoints():
    assert hasattr(contracts,'validate'), 'Missing workload contract validation'
    good={'version':1,'jobs':[{'id':'experiment-v1','script':'workloads/train.py','timeout_seconds':60}]}
    contracts.validate(good)
    for bad in [{'version':2,'jobs':[]},{'version':1,'jobs':good['jobs']*2},
                {'version':1,'jobs':[{'id':'x','script':'../outside.py'}]},
                {'version':1,'jobs':[{'id':'x','script':'/tmp/outside.py'}]},
                {'version':1,'jobs':[{'id':'x','script':'workloads/x.py','timeout_seconds':-1}]},
                {'version':1,'jobs':[{'id':'x','script':'x.py','enabled':'false'}]},
                {'version':1,'jobs':[{'id':'x','script':'x.py','secrets':'TOKEN'}]},
                {'version':1,'jobs':[{'id':'x','script':'x.py','depends_on':['y']},
                                     {'id':'y','script':'y.py','depends_on':['x']}]}]:
        with pytest.raises(ValueError):contracts.validate(bad)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def test_fetch_sees_work_added_while_waiting_but_active_snapshot_stays_immutable(tmp_path):
    assert hasattr(source,'GitSource'), 'Missing latest-at-allocation Git source'
    origin=tmp_path/'origin';origin.mkdir();git(origin,'init','-b','main')
    git(origin,'config','user.name','Test');git(origin,'config','user.email','test@example.invalid')
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[{'id':'one','script':'job.py'}]}))
    (origin/'job.py').write_text("print('first')")
    git(origin,'add','.');git(origin,'commit','-m','first')
    remote=source.GitSource(str(origin),'main',tmp_path/'cache')
    first,manifest=remote.refresh();snapshot=remote.snapshot(first,tmp_path/'active')
    (origin/'job.py').write_text("print('second')")
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[{'id':'two','script':'job.py'}]}))
    git(origin,'add','.');git(origin,'commit','-m','second')
    second,new=remote.refresh()
    assert first!=second and new['jobs'][0]['id']=='two'
    assert (snapshot/'job.py').read_text()=="print('first')"


def test_job_receipts_skip_completed_jobs_redact_logs_and_reject_changed_ids(tmp_path):
    assert hasattr(runner,'Runner'), 'Missing sequential job runner'
    code=tmp_path/'code';code.mkdir();(code/'job.py').write_text("import os; print(os.environ['TEST_SECRET']); print('done')")
    r=runner.Runner(tmp_path/'state',secrets={'TEST_SECRET':'not-a-real-secret'})
    spec={'id':'one','script':'job.py','secrets':['TEST_SECRET'],'timeout_seconds':5}
    result=r.run_one(spec,code,'a'*40)
    assert result['status']=='completed'
    assert 'not-a-real-secret' not in (tmp_path/'state/jobs/one/log.txt').read_text()
    assert '[REDACTED]' in (tmp_path/'state/jobs/one/log.txt').read_text()
    assert r.run_one(spec,code,'b'*40)['status']=='already_completed'
    with pytest.raises(ValueError,match='new job ID'):
        r.run_one({**spec,'args':['changed']},code,'b'*40)


def test_timeout_leaves_terminal_receipt_and_does_not_stall_queue(tmp_path):
    assert hasattr(runner,'Runner'), 'Missing bounded workload execution'
    code=tmp_path/'code';code.mkdir();(code/'job.py').write_text('import time; time.sleep(30)')
    r=runner.Runner(tmp_path/'state');started=time.monotonic()
    result=r.run_one({'id':'timeout','script':'job.py','timeout_seconds':.2},code,'c'*40)
    assert result['status']=='timed_out' and time.monotonic()-started<8
    assert json.loads((tmp_path/'state/jobs/timeout/receipt.json').read_text())['status']=='timed_out'


def test_workload_can_use_base_python_without_runner_environment_dependencies(tmp_path,monkeypatch):
    base=sys._base_executable
    if os.path.normcase(base)==os.path.normcase(sys.executable):pytest.skip('Needs a venv to distinguish interpreters')
    code=tmp_path/'code';code.mkdir()
    (code/'job.py').write_text("import os,sys,json; from pathlib import Path; Path(os.environ['QUEUE_OUTPUT_DIR'],'python.json').write_text(json.dumps({'prefix':sys.prefix}))")
    monkeypatch.setenv('QUEUE_WORKLOAD_PYTHON',base)
    r=runner.Runner(tmp_path/'state');r.run_one({'id':'base','script':'job.py'},code,'d'*40)
    got=json.loads((tmp_path/'state/jobs/base/output/python.json').read_text())
    assert os.path.normcase(got['prefix'])==os.path.normcase(sys.base_prefix)


def test_controller_interrupt_propagates_after_recording_interrupted_job(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();(code/'job.py').write_text('pass')
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(runner.subprocess,'Popen',interrupted)
    r=runner.Runner(tmp_path/'state')
    with pytest.raises(KeyboardInterrupt):r.run_one({'id':'interrupted','script':'job.py'},code,'e'*40)
    assert r.receipt('interrupted')['status']=='interrupted'


@pytest.mark.skipif(os.name!='posix',reason='Kaggle/Linux process-group cleanup')
def test_exited_parent_does_not_leave_background_child_holding_log_pipe(tmp_path):
    code=tmp_path/'code';code.mkdir()
    (code/'job.py').write_text("import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])")
    r=runner.Runner(tmp_path/'state');started=time.monotonic()
    assert r.run_one({'id':'child','script':'job.py'},code,'f'*40)['status']=='completed'
    assert time.monotonic()-started<3
