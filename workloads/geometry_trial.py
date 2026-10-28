# Workload-specific dependencies; the runner itself uses the standard library.
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','--quiet','huggingface_hub==2.3.0','faiss-cpu'],check=True)
"""End-to-end bounded real-coordinate trial. Not a full-corpus optimization."""
from pathlib import Path
import os,time,json,traceback,shutil,platform
import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from queue_runner.sdk import checkpoint,progress
root=Path(os.environ['QUEUE_OUTPUT_DIR']);root.mkdir(parents=True,exist_ok=True)
result={'stage':'starting','full_graph_optimization':False,'model_training':False}
def save():
    p=root/'report.partial';p.write_text(json.dumps(result,indent=2));p.replace(root/'report.json')
    print(json.dumps(result),flush=True)
    progress(stage=result['stage'],processed_members=result.get('processed_members',0))

try:
    import jax,psutil,faiss
    from huggingface_hub import download_bucket_files
    from corpus_atlas.core import sha256,seal
    from corpus_atlas.constraints import knn_candidates,neighbor_centroids
    from corpus_atlas.constraint_blocks import symmetric_csr,block_step
    from corpus_atlas.constraint_tpu import ForceBackend
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision','highest')
    devices=jax.devices()
    result.update(stage='hardware',jax=jax.__version__,python=platform.python_version(),
        backend=jax.default_backend(),device_count=len(devices),
        device_kinds=[d.device_kind for d in devices],
        host_ram_bytes=psutil.virtual_memory().total,logical_cpus=os.cpu_count(),
        working_disk=shutil.disk_usage('/kaggle/working')._asdict())
    save()
    if len(devices)!=8 or any(d.platform!='tpu' for d in devices):
        raise RuntimeError('Requires eight actual TPU devices; refusing CPU fallback')
    faiss.omp_set_num_threads(min(16,os.cpu_count() or 1))
    bucket='IntellAgents/wiki727-suffix-atlas-20261025'
    prefix='constraints/four-pass-reference8192-20261026-v1'
    files={'result/input_normalized.npy':'73c001c2b901f336cc4dbaa697ed3f38e85d450713f0c0ddcca32679e4680c58',
           'result/embedding_ids.npy':'f68de31df5f223070d96ca5329f8d1e157c6a30239de33545ecc79f539daadc5'}
    data=root/'input';data.mkdir(exist_ok=True)
    download_bucket_files(bucket,[(prefix+'/'+name,data/Path(name).name) for name in files],
                          token=False,raise_on_missing_files=True)
    for name,digest in files.items():
        if sha256(data/Path(name).name)!=digest:raise ValueError('Real input provenance mismatch')
    ref=np.load(data/'input_normalized.npy');ids=np.load(data/'embedding_ids.npy')
    if ref.shape!=(8192,768) or ids.shape!=(8192,) or len(np.unique(ids))!=len(ids):
        raise ValueError('Input shape/identity mismatch')
    result.update(stage='building_reference_graph',input_bucket=bucket,input_prefix=prefix,
        input_sha256=files,members=len(ref),dimensions=ref.shape[1],neighbors=16,
        edge_capacity=32768,owner_capacity=512,checkpoint_interval_blocks=4)
    save();started=time.perf_counter();nn=knn_candidates(ref,16,device='cpu')
    result['graph_build_seconds']=time.perf_counter()-started
    centers=neighbor_centroids(ref,nn)
    nearest=np.empty(len(ref))
    for start in range(0,len(ref),128):
        rows=np.arange(start,min(start+128,len(ref)))
        nearest[rows]=np.linalg.norm(ref[rows,None].astype(np.float64)-ref[nn[rows]],axis=2).min(1)
    scales=1.05*np.maximum(nearest,1e-4)
    radii=np.linalg.norm(ref.astype(np.float64)-centers,axis=1)+.10*np.maximum(nearest,1e-4)
    pointer,indices=symmetric_csr(nn)
    edges=np.unique(np.sort(np.column_stack((np.repeat(np.arange(len(nn)),nn.shape[1]),nn.ravel())),axis=1),axis=0)
    margins=np.minimum(scales[edges[:,0]],scales[edges[:,1]])
    def score(x):
        energy=0.;worst=0.
        for start in range(0,len(edges),2048):
            ij=edges[start:start+2048]
            distance=np.linalg.norm(x[ij[:,0]].astype(np.float64)-x[ij[:,1]],axis=1)
            deficit=np.maximum(0,margins[start:start+len(ij)]-distance)
            energy+=float(np.sum(deficit**2));worst=max(worst,float(deficit.max()))
        return {'fixed_graph_hinge_energy':energy,'maximum_deficit':worst}
    x=ref.copy();result['before']=score(x);result['blocks']=[]
    backend=ForceBackend(dimensions=768,edge_capacity=32768,owner_capacity=512)
    result['stage']='optimizing_trial';save();started=time.perf_counter()
    for start in range(0,len(x),512):
        owners=np.arange(start,min(start+512,len(x)));tick=time.perf_counter()
        summary=block_step(x,pointer,indices,scales,owners,centers[owners],radii[owners],force_backend=backend)
        if summary['energy_after']>summary['energy_before']:raise ValueError('Accepted update increased incident energy')
        result['blocks'].append({'start':start,'seconds':time.perf_counter()-tick,**summary})
        result['processed_members']=int(start+len(owners));save()
        if len(result['blocks'])%4==0 or start+len(owners)==len(x):
            with (root/'coordinates.partial').open('wb') as stream:np.save(stream,x)
            (root/'coordinates.partial').replace(root/'coordinates.npy')
            closed=root/'checkpoints'/f"block-{len(result['blocks']):04d}";closed.mkdir(parents=True)
            shutil.copy2(root/'coordinates.npy',closed/'coordinates.npy')
            shutil.copy2(root/'report.json',closed/'report.json')
            shutil.copy2(data/'embedding_ids.npy',closed/'embedding_ids.npy')
            checkpoint(closed.relative_to(root))
    result.update(stage='auditing',loop_seconds=time.perf_counter()-started,after=score(x),
        norm_max_error=float(abs(np.linalg.norm(x.astype(np.float64),axis=1)-1).max()),
        confinement_max_excess=float(np.maximum(0,np.linalg.norm(x.astype(np.float64)-centers,axis=1)-radii).max()))
    save()
    if (result['after']['fixed_graph_hinge_energy']>=result['before']['fixed_graph_hinge_energy']
            or result['norm_max_error']>1e-6 or result['confinement_max_excess']>1e-6):
        raise ValueError('Trial failed independent energy/norm/confinement audits')
    np.save(root/'embedding_ids.npy',ids);np.save(root/'neighbors.npy',nn)
    result.update(stage='complete',acceptance_scope='One fixed-graph sweep on 8192 real rows; not global feasibility, full-scale throughput, or retrieval improvement',
        source_sha256={p.relative_to(Path(__file__).parent).as_posix():sha256(p) for p in Path(__file__).parent.rglob('*.py')})
    save();seal(root,{'kind':'tpu_real8192_block_trial','full_graph_optimization':False})
    final=root/'final';final.mkdir()
    for name in ['coordinates.npy','report.json','embedding_ids.npy','neighbors.npy','manifest.json']:
        shutil.copy2(root/name,final/name)
    checkpoint('final')
except BaseException:
    result.update(stage='failed',error=traceback.format_exc());save();raise
