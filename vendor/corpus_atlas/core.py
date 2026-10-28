"""Numerical atlas operations (no model inference)."""

import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False,ensure_ascii=True)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def seal(folder,metadata):
    folder=Path(folder)
    files={p.relative_to(folder).as_posix():sha256(p) for p in sorted(folder.rglob('*'))
           if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.partial')}
    result={**metadata,'complete':True,'files':files}
    put(folder/'manifest.json',result)
    return result


def checked(folder):
    folder=Path(folder);path=folder/'manifest.json'
    if not path.exists():return None
    manifest=json.loads(path.read_text())
    if not manifest.get('complete'):return None
    for name,checksum in manifest['files'].items():
        target=(folder/name).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_file() or sha256(target)!=checksum:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return manifest

def suffix_view(x, width, geometry):
    if x.ndim!=2 or not 1<=width<=x.shape[1] or geometry not in ('raw','spherical'):
        raise ValueError('Invalid suffix width or geometry')
    view=np.array(x[:,-width:],dtype=np.float32,order='C',copy=True)
    if not np.isfinite(view).all():raise ValueError('Nonfinite source coordinates')
    norm=np.linalg.norm(view,axis=1)
    valid=norm>0
    if geometry=='spherical':view[valid]/=norm[valid,None]
    return view,norm,valid


def duplicate_inventory(source,width,folder,checkpoint=None):
    folder=Path(folder);previous=checked(folder)
    if previous:
        if checkpoint:checkpoint(folder)
        return json.loads((folder/'summary.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    if not 1<=width<=source.shape[1]:raise ValueError('Invalid duplicate suffix width')
    x=np.ascontiguousarray(source[:,-width:],dtype='<f2')
    records=x.view(np.dtype((np.void,width*2))).reshape(-1)
    _,first,inverse,counts=np.unique(records,return_index=True,return_inverse=True,return_counts=True)
    np.save(folder/'group_labels.npy',inverse.astype(np.uint32))
    np.save(folder/'representative_corpus_rows.npy',first.astype(np.int64))
    np.save(folder/'group_member_counts.npy',counts.astype(np.uint32))
    top=np.argpartition(counts,-min(50,len(counts)))[-min(50,len(counts)):]
    top=top[np.argsort(-counts[top])]
    report={'rows':len(source),'width':width,'unique_patterns':len(counts),'duplicate_groups':int((counts>1).sum()),
        'rows_in_duplicate_groups':int(counts[counts>1].sum()),'largest_group':int(counts.max()),
        'definition':'Bitwise identical FP16 saved suffixes before renormalization; signed zeros remain distinct',
        'top_groups':[{'group':int(i),'members':int(counts[i]),'representative_corpus_row':int(first[i])} for i in top]}
    put(folder/'summary.json',report);seal(folder,{'kind':'exact_suffix_duplicate_inventory'})
    if checkpoint:checkpoint(folder)
    return report


class MomentAccumulator:
    def __init__(self,clusters,width,backend='numpy'):
        self.backend=backend;self.means_ref=None;self.device_means=None
        if backend=='numpy':
            self.sums=np.zeros((clusters,width),dtype=np.float64);self.squares=np.zeros_like(self.sums)
        elif backend=='torch_cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA statistics requested but unavailable')
            self.torch=torch
            self.sums=torch.zeros((clusters,width),device='cuda',dtype=torch.float64)
            self.squares=torch.zeros_like(self.sums)
        else:raise ValueError('Unknown statistics backend')

    def update(self,labels,x):
        if self.backend=='numpy':
            np.add.at(self.sums,labels,x);np.add.at(self.squares,labels,x.astype(np.float64)**2)
        else:
            values=self.torch.as_tensor(x,device='cuda',dtype=self.torch.float64)
            indices=self.torch.as_tensor(labels,device='cuda',dtype=self.torch.int64)
            self.sums.index_add_(0,indices,values);self.squares.index_add_(0,indices,values.square())

    def totals(self):
        if self.backend=='numpy':return self.sums,self.squares
        return self.sums.cpu().numpy(),self.squares.cpu().numpy()

    def radii(self,x,labels,means):
        if self.backend=='numpy':return np.linalg.norm(x.astype(np.float64)-means[labels],axis=1)
        if self.means_ref is not means:
            self.device_means=self.torch.as_tensor(means,device='cuda',dtype=self.torch.float64);self.means_ref=means
        values=self.torch.as_tensor(x,device='cuda',dtype=self.torch.float64)
        indices=self.torch.as_tensor(labels,device='cuda',dtype=self.torch.int64)
        return self.torch.linalg.vector_norm(values-self.device_means[indices],dim=1).cpu().numpy()

def _index(centers,geometry,backend):
    import faiss
    metric=faiss.METRIC_INNER_PRODUCT if geometry=='spherical' else faiss.METRIC_L2
    cpu=faiss.IndexFlat(centers.shape[1],metric)
    if backend=='gpu':
        if not hasattr(faiss,'StandardGpuResources') or faiss.get_num_gpus()<1:
            raise RuntimeError('GPU Faiss unavailable; refusing silent CPU fallback')
        resources=faiss.StandardGpuResources();resources.setTempMemory(512*1024**2)
        options=faiss.GpuClonerOptions();options.useFloat16=False
        index=faiss.index_cpu_to_gpu(resources,0,cpu,options)
    elif backend=='cpu':resources=None;index=cpu
    else:raise ValueError('backend must be cpu or gpu')
    index.add(np.ascontiguousarray(centers,dtype=np.float32))
    return index,resources


def _blocks(n,size):
    for start in range(0,n,size):yield start,min(start+size,n)


def _profile(source,sample,ids,width,geometry,folder,config,checkpoint):
    existing=checked(folder)
    if existing:
        if checkpoint:checkpoint(folder)
        return
    folder.mkdir(parents=True,exist_ok=True)
    x,norm,valid=suffix_view(sample,width,geometry);x=x[valid];row_ids=ids[valid]
    if len(x)<2:raise ValueError('Insufficient nonzero sample rows')
    mean=x.mean(axis=0,dtype=np.float64)
    covariance=(x.astype(np.float64).T@x)/len(x)-np.outer(mean,mean)
    eigen=np.maximum(np.linalg.eigvalsh(covariance),0)[::-1]
    p=eigen/eigen.sum() if eigen.sum()>0 else eigen
    entropy=float(-np.sum(p[p>0]*np.log(p[p>0])))
    np.save(folder/'sample_mean.npy',mean)
    np.save(folder/'sample_covariance.npy',covariance)
    np.save(folder/'sample_eigenvalues.npy',eigen)
    put(folder/'sample_geometry.json',{'sample_rows':len(x),'zero_sample_rows':int((~valid).sum()),
        'geometry':geometry,'width':width,'source_coordinate_range':[source.shape[1]-width,source.shape[1]],
        'sample_mean_norm':float(np.linalg.norm(mean)),'effective_rank':math.exp(entropy),
        'variance_top_1_fraction':float(p[0]),'variance_top_8_fraction':float(p[:8].sum()),
        'coordinate_min':float(x.min()),'coordinate_max':float(x.max()),'coordinate_mean':float(x.mean(dtype=np.float64)),
        'raw_suffix_norm_quantiles':dict(zip(['0','0.5','0.9','0.99','1'],map(float,np.quantile(norm,[0,.5,.9,.99,1])))),
        'scope':'PCA and nearest-neighbor diagnostics on fixed uniform sample, not a full-corpus neighbor graph'})
    index,resources=_index(x,'raw',config['backend'])
    q=min(config.get('knn_sample_rows',4096),len(x));k=min(config.get('knn_k',32),len(x)-1)
    distances,neighbors=index.search(np.ascontiguousarray(x[:q]),k+1)
    out_ids=np.empty((q,k),dtype=np.int64);out_dist=np.empty((q,k),dtype=np.float32)
    for i in range(q):
        keep=neighbors[i]!=i
        chosen=neighbors[i][keep][:k]
        out_ids[i]=row_ids[chosen];out_dist[i]=np.sqrt(np.maximum(distances[i][keep][:k],0))
    np.save(folder/'knn_query_corpus_rows.npy',row_ids[:q])
    np.save(folder/'knn_corpus_rows.npy',out_ids);np.save(folder/'knn_distances.npy',out_dist)
    seal(folder,{'kind':'sample_geometry','width':width,'geometry':geometry})
    if checkpoint:checkpoint(folder)


def _job(source,sample,width,geometry,k,seed,folder,config,checkpoint,deadline=float('inf')):
    import faiss
    previous=checked(folder)
    if previous:
        if checkpoint:checkpoint(folder)
        return previous['summary']
    folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();n=len(source);batch=config['batch_rows']
    training,_,valid=suffix_view(sample,width,geometry);training=training[valid]
    if len(training)<k:raise ValueError(f'{k} clusters need at least {k} nonzero sample rows')
    fit=faiss.Kmeans(width,k,niter=config['iterations'],nredo=config['redos'],seed=seed,
        spherical=geometry=='spherical',gpu=config['backend']=='gpu',verbose=False,
        min_points_per_centroid=1,max_points_per_centroid=math.ceil(len(training)/k)+1)
    fit.train(training)
    centers=np.array(fit.centroids,copy=True)
    np.save(folder/'assignment_centroids.npy',centers)
    faiss.write_index(fit.index if config['backend']=='cpu' else faiss.index_gpu_to_cpu(fit.index),str(folder/'assignment.faiss'))
    put(folder/'fit.json',{'seed':seed,'training_rows':len(training),'iterations':config['iterations'],
        'redos':config['redos'],'objective_history':list(map(float,fit.obj)),
        'backend':config['backend'],'faiss_version':faiss.__version__,'seconds':time.perf_counter()-started,
        'full_corpus_refit':False})
    labels=np.lib.format.open_memmap(folder/'labels.npy',mode='w+',dtype=np.int32,shape=(n,))
    second=np.lib.format.open_memmap(folder/'second_labels.npy',mode='w+',dtype=np.int32,shape=(n,))
    margins=np.lib.format.open_memmap(folder/'assignment_margins.npy',mode='w+',dtype=np.float32,shape=(n,))
    counts=np.zeros(k,dtype=np.int64)
    moments=MomentAccumulator(k,width,config.get('statistics_backend','numpy'))
    for lo,hi in _blocks(n,batch):
        if time.time()>deadline:raise TimeoutError('Runtime budget reached; completed bundles are preserved')
        x,_,valid=suffix_view(source[lo:hi],width,geometry)
        labels[lo:hi]=-1;second[lo:hi]=-1;margins[lo:hi]=np.nan
        if valid.any():
            scores,assigned=fit.index.search(x[valid],min(2,k));where=lo+np.flatnonzero(valid)
            lab=assigned[:,0].astype(np.int32)
            labels[where]=lab;second[where]=assigned[:,1] if k>1 else -1
            if k>1:margins[where]=(scores[:,0]-scores[:,1]) if geometry=='spherical' else (scores[:,1]-scores[:,0])
            counts+=np.bincount(lab,minlength=k)
            moments.update(lab,x[valid])
        if lo==0 or hi==n or hi%(batch*50)==0:
            put(folder/'progress.json',{'phase':'assign','rows':hi,'total':n})
    sums,squares=moments.totals()
    means=np.divide(sums,counts[:,None],out=np.zeros_like(sums),where=counts[:,None]>0)
    variance=np.maximum(np.divide(squares,counts[:,None],out=np.zeros_like(squares),where=counts[:,None]>0)-means**2,0)
    np.save(folder/'member_centroids.npy',means);np.save(folder/'coordinate_variance.npy',variance)
    radii=np.lib.format.open_memmap(folder/'radii.npy',mode='w+',dtype=np.float32,shape=(n,))
    for lo,hi in _blocks(n,batch):
        if time.time()>deadline:raise TimeoutError('Runtime budget reached; completed bundles are preserved')
        x,_,valid=suffix_view(source[lo:hi],width,geometry)
        radii[lo:hi]=np.nan
        if valid.any():radii[lo+np.flatnonzero(valid)]=moments.radii(x[valid],labels[lo:hi][valid],means)
        if lo==0 or hi==n or hi%(batch*50)==0:put(folder/'progress.json',{'phase':'radii','rows':hi,'total':n})
    for a in (labels,second,margins,radii):a.flush()
    nonempty=np.flatnonzero(counts)
    near_index,resources=_index(means[nonempty].astype(np.float32),'raw',config['backend'])
    nk=min(config.get('neighbor_centroids',8),max(len(nonempty)-1,0))
    near_dist,near_id=near_index.search(means[nonempty].astype(np.float32),min(nk+1,len(nonempty)))
    neighbor_ids=np.full((k,nk),-1,dtype=np.int32);neighbor_dist=np.full((k,nk),np.nan,dtype=np.float32)
    for pos,c in enumerate(nonempty):
        keep=near_id[pos]!=pos
        neighbor_ids[c]=nonempty[near_id[pos][keep][:nk]]
        neighbor_dist[c]=np.sqrt(np.maximum(near_dist[pos][keep][:nk],0))
    np.save(folder/'nearest_centroid_ids.npy',neighbor_ids);np.save(folder/'nearest_centroid_distances.npy',neighbor_dist)
    valid_ids=np.flatnonzero(labels>=0)
    order=valid_ids[np.argsort(labels[valid_ids],kind='stable')]
    ordered_radii=np.asarray(radii[order]);starts=np.r_[0,np.cumsum(counts)]
    stats=[];quantiles=config['quantiles'];thresholds=config['radii'];examples=config.get('examples_per_cluster',5)
    for c in range(k):
        begin,end=starts[c:c+2];values=ordered_radii[begin:end];ids=order[begin:end]
        if len(values):
            ranks=np.argsort(values,kind='stable')
            close=ids[ranks[:examples]];far=ids[ranks[-examples:][::-1]]
            stats.append({'cluster':c,'members':int(counts[c]),'member_centroid_norm':float(np.linalg.norm(means[c])),
                'radius_mean':float(values.mean(dtype=np.float64)),'radius_std':float(values.std(dtype=np.float64)),
                'radius_quantiles':{str(q):float(v) for q,v in zip(quantiles,np.quantile(values,quantiles))},
                'members_within_radius':[int(np.count_nonzero(values<=r)) for r in thresholds],
                'variance_trace':float(variance[c].sum()),'closest_corpus_rows':list(map(int,close)),
                'farthest_corpus_rows':list(map(int,far)),
                'mean_assignment_margin':float(np.nanmean(margins[ids])) if k>1 else None})
        else:stats.append({'cluster':c,'members':0})
    put(folder/'clusters.json',stats)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(stats),folder/'clusters.parquet',compression='zstd')
    # The audit compares the actual assignment index with an independent CPU exact index.
    audit_ids=valid_ids[np.linspace(0,len(valid_ids)-1,min(config.get('audit_rows',1024),len(valid_ids)),dtype=int)]
    audit_x,_,_=suffix_view(source[audit_ids],width,geometry)
    cpu,_=_index(centers,geometry,'cpu');reference,ref_ids=cpu.search(audit_x,1)
    actual,_=fit.index.search(audit_x,1)
    audit_error=float(np.max(np.abs(actual-reference))) if len(audit_ids) else 0
    put(folder/'assignment_audit.json',{'rows':len(audit_ids),'maximum_score_error':audit_error,
        'label_agreement':float(np.mean(labels[audit_ids]==ref_ids[:,0])),'ties_may_differ':True})
    if audit_error>config.get('assignment_audit_tolerance',.001):raise RuntimeError('GPU assignment accuracy audit failed')
    summary={'job':folder.name,'width':width,'geometry':geometry,'requested_clusters':k,
        'nonempty_clusters':int((counts>0).sum()),'empty_clusters':int((counts==0).sum()),
        'assigned_rows':int(counts.sum()),'unassigned_zero_rows':int(n-counts.sum()),
        'cluster_size_min':int(counts[counts>0].min()),'cluster_size_max':int(counts.max()),
        'cluster_size_mean':float(counts[counts>0].mean()),
        'radius_thresholds':thresholds,'radius_center':'arithmetic mean of all assigned full-corpus member vectors',
        'margin_definition':'best-minus-second inner product' if geometry=='spherical' else 'second-minus-best squared L2',
        'seconds':time.perf_counter()-started,'seed':seed,'natural_cluster_count':False,
        'statistics_backend':config.get('statistics_backend','numpy'),'accumulation_dtype':'float64'}
    put(folder/'summary.json',summary)
    del labels,second,margins,radii,fit
    seal(folder,{'kind':'cluster_job','summary':summary})
    if checkpoint:checkpoint(folder)
    print(json.dumps(summary),flush=True)
    return summary


def analyze(source_path, output, config, checkpoint=None):
    import faiss
    faiss.omp_set_num_threads(config.get('cpu_threads',min(os.cpu_count() or 1,16)))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['identity']!=identity:
        raise ValueError('Output identity differs; use a new output directory')
    expected=config['source'];source_path=Path(source_path)
    if source_path.stat().st_size!=expected['bytes'] or sha256(source_path)!=expected['sha256']:
        raise ValueError('source checksum or byte count mismatch')
    if expected['rows']*expected['dimensions']*2!=expected['bytes']:raise ValueError('Source shape/size mismatch')
    if any(not 1<=w<=expected['dimensions'] for w in config['widths']):raise ValueError('Invalid ladder width')
    execution=config.get('execution_widths',config['widths'])
    if sorted(execution)!=sorted(config['widths']):raise ValueError('Execution widths must match the declared ladder')
    if any(g not in ('raw','spherical') for g in config['geometries']):raise ValueError('Invalid geometry')
    if config['batch_rows']<1 or any(k<1 for k in config['cluster_counts']):raise ValueError('Invalid batch or cluster count')
    if any(r<0 for r in config['radii']) or config['radii']!=sorted(config['radii']):raise ValueError('Radii must be sorted and nonnegative')
    if not identity_path.exists():put(identity_path,{'identity':identity,'config':config})
    source=np.memmap(source_path,dtype='<f2',mode='r',shape=(expected['rows'],expected['dimensions']))
    existing=checked(output)
    if existing:
        # Root checksum list covers every completed child; return without rewriting immutable completion.
        if checkpoint:checkpoint(output)
        return json.loads((output/'summary.json').read_text())
    n=len(source);rng=np.random.default_rng(config['seeds'][0]);size=min(config['sample_rows'],n)
    ids=np.sort(rng.choice(n,size=size,replace=False));sample=np.asarray(source[ids])
    np.save(output/'training_corpus_rows.npy',ids)
    jobs=[];comparisons=[]
    deadline=time.time()+config.get('runtime_budget_hours',float('inf'))*3600
    for width in execution:
        profile=output/'profiles'/f'w{width:04d}';profile.mkdir(parents=True,exist_ok=True)
        if not checked(profile):
            norms=np.lib.format.open_memmap(profile/'suffix_norms.npy',mode='w+',dtype=np.float32,shape=(n,))
            for lo,hi in _blocks(n,config['batch_rows']):norms[lo:hi]=suffix_view(source[lo:hi],width,'raw')[1]
            norms.flush();del norms
        for geometry in config['geometries']:
            _profile(source,sample,ids,width,geometry,profile/geometry,config,checkpoint)
            for k,seed in itertools.product(config['cluster_counts'],config['seeds']):
                name=f'w{width:04d}_{geometry}_k{k:06d}_s{seed}'
                summary=_job(source,sample,width,geometry,k,seed,output/'jobs'/name,config,checkpoint,deadline)
                jobs.append(summary)
                put(output/'progress.json',{'complete':False,'finished_jobs':len(jobs),'total_jobs':len(config['widths'])*len(config['geometries'])*len(config['cluster_counts'])*len(config['seeds']),'last_job':name})
        seal(profile,{'kind':'suffix_profile','width':width})
        if checkpoint:checkpoint(profile)
    for geometry,k,seed in itertools.product(config['geometries'],config['cluster_counts'],config['seeds']):
        for left,right in zip(config['widths'],config['widths'][1:]):
            name=f'{geometry}_k{k:06d}_s{seed}_{left}_to_{right}'
            folder=output/'comparisons'/name;done=checked(folder)
            if done:comparison=json.loads((folder/'summary.json').read_text())
            else:
                folder.mkdir(parents=True,exist_ok=True)
                a=np.load(output/'jobs'/f'w{left:04d}_{geometry}_k{k:06d}_s{seed}'/'labels.npy',mmap_mode='r')
                b=np.load(output/'jobs'/f'w{right:04d}_{geometry}_k{k:06d}_s{seed}'/'labels.npy',mmap_mode='r')
                comparison,table=compare_assignments(a,b)
                comparison.update(left_width=left,right_width=right,geometry=geometry,clusters=k,seed=seed)
                np.savez_compressed(folder/'contingency.npz',**table);put(folder/'summary.json',comparison)
                seal(folder,{'kind':'cross_width_correspondence'})
            if checkpoint:checkpoint(folder)
            comparisons.append(comparison)
    duplicates=[]
    for width in config.get('exact_duplicate_widths',[]):
        duplicates.append(duplicate_inventory(source,width,output/'duplicates'/f'w{width:04d}',checkpoint))
    report={'complete':True,'rows':n,'identity':identity,'jobs':jobs,'comparisons':comparisons,'duplicates':duplicates,
        'source_coordinate_basis':'saved cache; original model output retained only its first saved dimensions',
        'full_corpus_membership':True,'centroid_fit':'fixed uniform sample; all rows assigned and summarized',
        'interpretation':'Partition counts are chosen resolutions, not discovered semantic categories'}
    put(output/'summary.json',report)
    put(output/'progress.json',{'complete':True,'finished_jobs':len(jobs)})
    seal(output,{'kind':'corpus_atlas','identity':identity})
    if checkpoint:checkpoint(output)
    return report

def compare_assignments(left, right):
    if left.shape!=right.shape:raise ValueError('Assignments must share corpus row order and length')
    valid=(left>=0)&(right>=0);a=left[valid].astype(np.uint64);b=right[valid].astype(np.uint64)
    keys,counts=np.unique((a<<32)|b,return_counts=True)
    la=(keys>>32).astype(np.int32);lb=(keys&np.uint64(0xffffffff)).astype(np.int32)
    n=int(valid.sum())
    if n:
        row=np.bincount(la,weights=counts);col=np.bincount(lb,weights=counts)
        choose=lambda x:np.sum(x*(x-1)/2)
        total=n*(n-1)/2;sumrow=choose(row);sumcol=choose(col);sumcell=choose(counts.astype(float))
        expected=sumrow*sumcol/total if total else 0;denominator=(sumrow+sumcol)/2-expected
        ari=(sumcell-expected)/denominator if denominator else 1.
        mutual=np.sum(counts/n*np.log(counts.astype(float)*n/(row[la]*col[lb])))
        entropy=lambda v:-np.sum(v[v>0]/n*np.log(v[v>0]/n))
        h1,h2=entropy(row),entropy(col);nmi=2*mutual/(h1+h2) if h1+h2 else 1.
    else:ari=None;nmi=None
    return {'compared_rows':n,'excluded_rows':int(len(left)-n),'adjusted_rand_index':None if ari is None else float(ari),
            'normalized_mutual_information':None if nmi is None else float(nmi),
            'occupied_correspondence_cells':len(keys)}, {'left_cluster':la,'right_cluster':lb,'count':counts}
