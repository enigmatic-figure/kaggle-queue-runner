"""Generate the stable notebook. Workload changes do not modify this file."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1];out=root/'kaggle';out.mkdir(exist_ok=True)
code='''from pathlib import Path
import subprocess,sys,uuid,os

REPOSITORY = "https://github.com/enigmatic-figure/kaggle-queue-runner.git"
BRANCH = "main"

# This clone happens AFTER Kaggle allocates the worker, so it sees queue-time edits.
runtime = Path('/tmp') / ('queue-bootstrap-' + uuid.uuid4().hex)
subprocess.run(['git','clone','--depth','1','--branch',BRANCH,REPOSITORY,str(runtime)],check=True)
commit = subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
print({'bootstrap_commit':commit},flush=True)
# Isolate runner/storage dependencies from the Kaggle model/JAX environment.
engine = runtime / '.engine'
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(engine)],check=True)
python = engine / 'bin' / 'python'
subprocess.run([str(python),'-m','pip','install','--quiet','-e',str(runtime)+'[hub]'],check=True)
environment = {**os.environ, 'QUEUE_WORKLOAD_PYTHON': sys.executable}
subprocess.run([str(python),'-u','-m','queue_runner.cli','worker',
    '--repo',REPOSITORY,'--branch',BRANCH,'--hours','8','--idle-minutes','5',
    '--network','auto'],cwd=runtime,env=environment,check=True)
'''
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
    'cells':[{'id':'overview','cell_type':'markdown','metadata':{},'source':['# Kaggle Queue Runner\n',
        'A stable bootstrap. Update workloads.json and scripts in the controlled public repository while this notebook waits. Checks and useful work share this allocation. Kaggle Secrets supply optional private networking and artifact credentials.']},
        {'id':'bootstrap','cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'execution_count':None,'outputs':[]}]}
(out/'queue_runner.ipynb').write_text(json.dumps(notebook,indent=2))
metadata={'id':'finnbogiabrahamson/kaggle-queue-runner','title':'Kaggle Queue Runner','code_file':'queue_runner.ipynb',
    'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':True,
    'enable_internet':True,'machine_shape':'TpuV5E8','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(out/'queue_runner.ipynb')
