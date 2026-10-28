"""Install the isolated CPU environment and verify Parquet statistics and CMA-ES."""
from pathlib import Path
import json,os,shutil,subprocess,sys,tempfile

project=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='queue-cpu-smoke-') as scratch:
    root=Path(scratch)
    for folder in ['queue_runner','workloads','environments']:
        shutil.copytree(project/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (root/'workloads/smoke_payload.py').write_text('''
import pyarrow as pa, pyarrow.parquet as pq, pathlib, runpy, sys, cma
p=pathlib.Path('sample.parquet')
pq.write_table(pa.table({'value':[1.,2.,None,5.,5.],'label':['a','b',None,'a','b']}),p)
es=cma.CMAEvolutionStrategy([2.,-2.],.5,{'verbose':-9,'seed':1,'maxfevals':24})
while not es.stop():
    points=es.ask();es.tell(points,[sum(v*v for v in x) for x in points])
assert es.result.fbest<8
sys.argv=['parquet_audit.py','--input',str(p),'--memory-gb','1']
runpy.run_path('workloads/parquet_audit.py',run_name='__main__')
''')
    env={**os.environ,'JAX_PLATFORMS':'cpu','PJRT_DEVICE':'CPU','TPU_VISIBLE_CHIPS':'',
        'QUEUE_OUTPUT_DIR':str(root/'output'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'PYTHONPATH':str(root)}
    (root/'output').mkdir()
    subprocess.run([sys.executable,'workloads/cpu_task.py','--script','workloads/smoke_payload.py'],cwd=root,env=env,check=True)
    value=json.loads((root/'output/parquet-audit/summary.json').read_text())
    assert value['rows']==5
    assert value['columns'][0]['nulls']==1 and value['columns'][0]['mean']==3.25
    assert (root/'output/parquet-audit/checkpoint.ready.json').is_file()
    print(json.dumps({'cpu_environment':'passed','parquet_statistics':'passed','cma_es':'passed','accelerator_used':False}))
