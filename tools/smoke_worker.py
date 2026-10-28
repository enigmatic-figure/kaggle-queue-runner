"""End-to-end CLI smoke test without Kaggle, accelerator allocation or tailnet changes."""
from pathlib import Path
import tempfile,shutil,subprocess,sys,json,os

project=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='queue-cli-check-') as temporary:
    root=Path(temporary);origin=root/'origin';origin.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(origin),*args],text=True).strip()
    git('init','-b','main');git('config','user.name','Queue test');git('config','user.email','test@example.invalid')
    shutil.copytree(project/'queue_runner',origin/'queue_runner',ignore=shutil.ignore_patterns('__pycache__'))
    (origin/'job.py').write_text("from queue_runner.sdk import output_dir,checkpoint,progress\n"
        "p=output_dir()/'complete';p.mkdir();(p/'result.txt').write_text('done');checkpoint('complete');progress(stage='complete')")
    (origin/'workloads.json').write_text(json.dumps({'version':1,'jobs':[
        {'id':'one','script':'job.py','timeout_seconds':10},
        {'id':'two','script':'job.py','timeout_seconds':10,'depends_on':['one']}]}))
    git('add','.');git('commit','-m','smoke')
    env={k:v for k,v in os.environ.items() if k not in ['TS_AUTHKEY','QUEUE_CONTROL_TOKEN','QUEUE_NOTIFY_URL','QUEUE_NOTIFY_TOKEN','HF_TOKEN']}
    subprocess.run([sys.executable,'-m','queue_runner.cli','worker','--repo',str(origin),
        '--root',str(root/'runs'),'--run-id','smoke','--network','off','--hours','.01',
        '--idle-minutes','.003','--poll-seconds','.05'],cwd=project,env=env,check=True,timeout=30)
    receipts=[json.loads((root/'runs/smoke/jobs'/identity/'receipt.json').read_text()) for identity in ['one','two']]
    assert all(r['status']=='completed' for r in receipts)
    assert all((root/'runs/smoke/jobs'/identity/'output/complete/checkpoint.ready.json').exists() for identity in ['one','two'])
    print(json.dumps({'cli_worker':'passed','sequential_jobs':2,'closed_checkpoints':2,'external_credentials_used':False}))
