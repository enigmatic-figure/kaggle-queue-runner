"""Constraint geometry and balanced spatial split primitives."""
import numpy as np


def prepare_parent_shards(source, members, output, shard_rows=1048576, batch_rows=32768,
                          device='cpu', source_identity=None, publish=None, progress=None):
    """Stream the entire cleaned corpus into identity-bound FP32 master shards.

    This is initialization, not geometric optimization. A verified immutable
    source hash can be supplied to avoid rereading the whole input for hashing.
    Closed shards are resumable; no source coordinates or membership are changed.
    """
    import hashlib,json
    from pathlib import Path
    from .core import checked,put,seal
    ids=np.asarray(members)
    if (source.ndim!=2 or ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
            or len(ids)==0 or np.any(ids[1:]<=ids[:-1]) or ids[0]<0 or ids[-1]>=len(source)
            or shard_rows<1 or batch_rows<1):
        raise ValueError('Invalid parent identity/shape/batch configuration')
    if source_identity is None:
        source_identity=hashlib.sha256(memoryview(np.ascontiguousarray(source)).cast('B')).hexdigest()
    identity={'source':source_identity,'source_shape':list(source.shape),'source_dtype':str(source.dtype),
              'members':len(ids),'member_order_sha256':hashlib.sha256(memoryview(np.ascontiguousarray(ids)).cast('B')).hexdigest(),
              'dimensions':source.shape[1],'storage_dtype':'float32','norm':'L2 normalized in float64 before FP32 save',
              'shard_rows':shard_rows,'preparation_only':True}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scope=output/'identity.json'
    if scope.exists() and json.loads(scope.read_text())!=identity:
        raise ValueError('Parent preparation identity differs from existing run')
    put(scope,identity)
    if checked(output):
        if publish:publish(output)
        return json.loads((output/'summary.json').read_text())
    if device!='cpu':
        import torch
    total_norm_error=0.;parts=0
    for start in range(0,len(ids),shard_rows):
        stop=min(len(ids),start+shard_rows);folder=output/'shards'/f'part-{start:08d}'
        existing=checked(folder)
        if existing:
            stats=json.loads((folder/'summary.json').read_text())
            total_norm_error=max(total_norm_error,stats['norm_max_error'])
        else:
            folder.mkdir(parents=True,exist_ok=True)
            points=np.lib.format.open_memmap(folder/'coordinates.npy',mode='w+',dtype=np.float32,
                                             shape=(stop-start,source.shape[1]))
            max_error=0.
            try:
                for cursor in range(start,stop,batch_rows):
                    end=min(stop,cursor+batch_rows)
                    block=np.array(source[ids[cursor:end]],dtype=np.float64,copy=True)
                    if not np.isfinite(block).all():raise ValueError('Nonfinite source coordinate')
                    if device=='cpu':
                        norm=np.linalg.norm(block,axis=1)
                        if np.any(norm==0):raise ValueError('Cannot normalize zero vector')
                        result=(block/norm[:,None]).astype(np.float32)
                    else:
                        values=torch.from_numpy(block).to(device)
                        norm=torch.linalg.vector_norm(values,dim=1)
                        if bool((norm==0).any()):raise ValueError('Cannot normalize zero vector')
                        result=(values/norm[:,None]).float().cpu().numpy()
                        del values,norm
                    max_error=max(max_error,float(np.abs(np.linalg.norm(result.astype(np.float64),axis=1)-1).max()))
                    points[cursor-start:end-start]=result
                    if progress:progress({'phase':'normalizing_parent','rows':end,'total':len(ids)})
                points.flush()
            finally:
                points._mmap.close()
            np.save(folder/'embedding_ids.npy',ids[start:stop].astype(np.uint32))
            stats={'start_row':start,'stop_row':stop,'members':stop-start,'norm_max_error':max_error,
                   'first_embedding_id':int(ids[start]),'last_embedding_id':int(ids[stop-1])}
            put(folder/'summary.json',stats)
            seal(folder,{'kind':'normalized_parent_initialization_shard','source_identity':source_identity,
                         'member_order_sha256':identity['member_order_sha256']})
            total_norm_error=max(total_norm_error,max_error)
        if publish:publish(folder)
        parts+=1
        if progress:progress({'phase':'parent_shard_sealed','rows':stop,'total':len(ids),'shards':parts})
    summary={**identity,'shards':parts,'norm_max_error':total_norm_error,'complete':True,
             'geometrically_optimized':False}
    put(output/'summary.json',summary);seal(output,{'kind':'complete_parent_initialization','members':len(ids)})
    if publish:publish(output)
    return summary


def neighbor_centroids(x, neighbors, batch_size=256):
    if batch_size<1 or neighbors.ndim!=2 or len(neighbors)!=len(x):
        raise ValueError('Invalid centroid batch or neighbor table')
    result=np.empty(x.shape,dtype=np.float64)
    for start in range(0,len(x),batch_size):
        result[start:start+batch_size]=x[neighbors[start:start+batch_size]].mean(1)
    return result


def knn_candidates(x, k, device='cpu'):
    """Exhaustive Float32 candidates with oversampling and direct Float64 rerank.

    Candidate coverage is numerical, not a Float64 rank certificate. Ties are
    resolved by row ID among the retrieved pool. GPU scores are chunked.
    """
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not 1<=k<len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid neighbor search input')
    pool=min(len(x)-1,k+32); candidates=np.empty((len(x),pool),dtype=np.int64)
    if device=='cpu':
        import faiss
        index=faiss.IndexFlatL2(x.shape[1]); index.add(x.astype(np.float32))
        _,raw=index.search(x.astype(np.float32),min(len(x),pool+1))
        for i,row in enumerate(raw): candidates[i]=row[row!=i][:pool]
    else:
        import torch
        points=torch.as_tensor(x.astype(np.float32),device=device)
        norm=torch.sum(points*points,dim=1)
        precision=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('highest')
            for start in range(0,len(x),256):
                block=points[start:start+256]
                scores=norm[start:start+len(block),None]+norm[None,:]-2*(block@points.T)
                scores[torch.arange(len(block),device=device),torch.arange(start,start+len(block),device=device)]=float('inf')
                candidates[start:start+len(block)]=torch.topk(scores,pool,largest=False,sorted=False).indices.cpu().numpy()
        finally:
            torch.set_float32_matmul_precision(precision)
    output=np.empty((len(x),k),dtype=np.int64)
    for start in range(0,len(x),128):
        candidate=candidates[start:start+128]
        distance=np.sum((x[start:start+len(candidate),None,:]-x[candidate])**2,axis=2)
        order=np.lexsort((candidate,distance),axis=1)[:,:k]
        output[start:start+len(candidate)]=np.take_along_axis(candidate,order,axis=1)
    return output


def _arrays(y, centers, radii):
    y=np.asarray(y,dtype=np.float64); c=np.asarray(centers,dtype=np.float64)
    r=np.asarray(radii,dtype=np.float64)
    if (y.ndim!=2 or y.shape[1]<2 or c.shape!=y.shape or r.shape!=(len(y),)
        or not all(np.isfinite(v).all() for v in (y,c,r)) or (r<0).any()):
        raise ValueError('Invalid points, ball centers, or radii')
    return y,c,r

def project_sphere_ball(y, centers, radii):
    """Closest unit vector in each closed ball, with deterministic tie handling."""
    y,c,r=_arrays(y,centers,radii)
    cn=np.linalg.norm(c,axis=1)
    if np.any(r < np.abs(cn-1)-1e-12): raise ValueError('Empty sphere/ball intersection')
    yn=np.linalg.norm(y,axis=1)
    result=y/np.maximum(yn[:,None],1e-300)
    result[yn==0]=np.eye(1,y.shape[1])[0]
    active=(cn>0)&(np.linalg.norm(result-c,axis=1)>r+1e-13)
    for i in np.flatnonzero(active):
        axis=c[i]/cn[i]
        cosine=np.clip((1+cn[i]**2-r[i]**2)/(2*cn[i]),-1,1)
        tangent=result[i]-np.dot(result[i],axis)*axis
        if np.linalg.norm(tangent)<1e-12:
            tangent=np.eye(1,y.shape[1],int(np.argmin(np.abs(axis))))[0]
            tangent-=np.dot(tangent,axis)*axis
        tangent/=np.linalg.norm(tangent)
        result[i]=cosine*axis+np.sqrt(max(0,1-cosine*cosine))*tangent
    return result

def separation_bounds(centers, radii, edges, batch_size=4096):
    c=np.asarray(centers,dtype=np.float64); r=np.asarray(radii,dtype=np.float64)
    e=np.asarray(edges,dtype=np.int64)
    if (c.ndim!=2 or r.shape!=(len(c),) or e.ndim!=2 or e.shape[1]!=2
        or not np.isfinite(c).all() or not np.isfinite(r).all() or (r<0).any()
        or (e<0).any() or (e>=len(c)).any() or batch_size<1): raise ValueError('Invalid bounds inputs')
    lower=np.empty(len(e));upper=np.empty(len(e))
    for start in range(0,len(e),batch_size):
        ij=e[start:start+batch_size]
        d=np.linalg.norm(c[ij[:,0]]-c[ij[:,1]],axis=1)
        total=r[ij[:,0]]+r[ij[:,1]]
        lower[start:start+len(ij)]=np.maximum(0,d-total)
        upper[start:start+len(ij)]=np.minimum(2,d+total)
    return lower,upper

def spatial_five_way(x, ids, leaf_size=128, seed=1729):
    """KD spatial leaves, then locally and globally balanced five-way assignment.

    Balances membership in each spatial leaf. Coordinate/pair-distance marginal
    equality is an audited approximation, not a mathematical guarantee.
    """
    x=np.asarray(x); ids=np.asarray(ids)
    if (x.ndim!=2 or len(x)!=len(ids) or len(np.unique(ids))!=len(ids)
        or not np.isfinite(x).all() or leaf_size<5): raise ValueError('Invalid split inputs')
    leaves=np.empty(len(x),dtype=np.uint32); pending=[np.arange(len(x))]; leaf=0
    while pending:
        rows=pending.pop()
        if len(rows)<=leaf_size:
            leaves[rows]=leaf; leaf+=1; continue
        axis=int(np.argmax(np.var(x[rows].astype(np.float64),axis=0)))
        order=np.lexsort((ids[rows],x[rows,axis])); rows=rows[order]
        middle=len(rows)//2
        pending.extend([rows[middle:],rows[:middle]])
    rng=np.random.default_rng(seed); groups=np.empty(len(x),dtype=np.uint8)
    loads=np.zeros(5,dtype=np.int64)
    for k in range(leaf):
        rows=np.flatnonzero(leaves==k); rng.shuffle(rows)
        for start in range(0,len(rows),5):
            part=rows[start:start+5]
            tie=rng.random(5)
            chosen=np.lexsort((tie,loads))[:len(part)]
            groups[part]=chosen; loads[chosen]+=1
    return groups,leaves

def relax(x, edges, margins, centers, radii, epochs=8, device='cpu', edge_batch=4096):
    """Projected symmetric hinge relaxation of a declared, fixed sparse edge set.

    Rebuild constraints externally between calls. Confinement is enforced by
    joint projection; unsatisfied separation remains explicitly in the audit.
    Torch evaluates batched forces; Float64 CPU projection is the reference.
    This is not the full distributed/sharded solver.
    """
    import torch
    x,c,r=_arrays(x,centers,radii); x=x.copy()
    e=np.asarray(edges,dtype=np.int64); m=np.asarray(margins,dtype=np.float64)
    if (e.ndim!=2 or e.shape[1]!=2 or len(e)==0 or m.shape!=(len(e),)
        or not np.isfinite(m).all() or (m<0).any() or (m>2).any()
        or (e<0).any() or (e>=len(x)).any() or np.any(e[:,0]>=e[:,1])
        or len(np.unique(e,axis=0))!=len(e) or edge_batch<1):
        raise ValueError('Need unique canonical nonself edges with finite margins in [0,2]')
    if np.max(np.abs(np.linalg.norm(x,axis=1)-1))>1e-6 or np.any(np.linalg.norm(x-c,axis=1)>r+1e-8):
        raise ValueError('Initial configuration violates sphere or confinement')
    dtype=torch.float64 if device=='cpu' else torch.float32
    edge=torch.as_tensor(e,device=device); margin=torch.as_tensor(m,device=device,dtype=dtype)

    def assess(v):
        deficits=np.empty(len(e),dtype=np.float64)
        for start in range(0,len(e),edge_batch):
            ij=e[start:start+edge_batch]
            deficits[start:start+len(ij)]=np.maximum(0,m[start:start+len(ij)]-
                np.linalg.norm(v[ij[:,0]]-v[ij[:,1]],axis=1))
        return {'energy':float(np.mean(deficits**2)), 'mean_deficit':float(deficits.mean()),
            'maximum_deficit':float(deficits.max()), 'violated_edges':int((deficits>1e-6).sum()),
            'norm_max_error':float(np.max(np.abs(np.linalg.norm(v,axis=1)-1))),
            'confinement_max_excess':float(np.maximum(0,np.linalg.norm(v-c,axis=1)-r).max())}
    history=[assess(x)]
    for epoch in range(epochs):
        if history[-1]['maximum_deficit']<=1e-6: break
        points=torch.as_tensor(x,device=device,dtype=dtype)
        force=torch.zeros_like(points)
        active_degree=torch.zeros(len(x),dtype=torch.int64,device=device)
        for start in range(0,len(e),edge_batch):
            ij=edge[start:start+edge_batch]; left,right=ij[:,0],ij[:,1]
            delta=points[left]-points[right]; distance=torch.linalg.vector_norm(delta,dim=1)
            collision=distance<1e-12
            if bool(collision.any()):
                a=left[collision]; z=points[a]
                axis=(a*1664525+right[collision]*1013904223)%x.shape[1]
                direction=torch.zeros_like(z); direction[torch.arange(len(z),device=device),axis]=1
                direction-=torch.sum(direction*z,dim=1)[:,None]*z
                bad=torch.linalg.vector_norm(direction,dim=1)<1e-8
                if bool(bad.any()):
                    v=z[bad]; j=torch.argmin(torch.abs(v),dim=1)
                    fallback=torch.zeros_like(v); fallback[torch.arange(len(v),device=device),j]=1
                    fallback-=torch.sum(fallback*v,dim=1)[:,None]*v
                    direction[bad]=fallback
                direction/=torch.linalg.vector_norm(direction,dim=1)[:,None]
                delta[collision]=direction
            unit=delta/torch.clamp(distance,min=1e-12)[:,None]
            unit[collision]=delta[collision]
            push=torch.clamp(margin[start:start+len(ij)]-distance,min=0)[:,None]*unit
            force.index_add_(0,left,push); force.index_add_(0,right,-push)
            active=margin[start:start+len(ij)]>distance
            active_degree.index_add_(0,left[active],torch.ones_like(left[active]))
            active_degree.index_add_(0,right[active],torch.ones_like(right[active]))
        f=force.detach().cpu().numpy().astype(np.float64)
        step=.5/max(1,int(active_degree.max().item()))
        accepted=False
        for backtrack in range(16):
            candidate=project_sphere_ball(x+step*(.5**backtrack)*f,c,r)
            audit=assess(candidate)
            if audit['energy']<=history[-1]['energy']+1e-15:
                x=candidate; accepted=True; break
        if not accepted: audit=assess(x)
        history.append({**audit,'epoch':epoch+1,'accepted':accepted})
    return x,history


def run_four_pass_pilot(x, ids, output, epochs=8, device='cpu', k=16):
    """Bounded real-data experiment; not a 50M out-of-core execution engine."""
    from pathlib import Path
    import time, json, shutil, platform
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.stats import ks_2samp
    from corpus_atlas.core import put, seal, sha256
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a new experiment directory')
    x=np.asarray(x,dtype=np.float64).copy(); ids=np.asarray(ids)
    if (x.ndim!=2 or len(x)<50 or len(x)>100000 or len(ids)!=len(x)
        or len(np.unique(ids))!=len(ids) or not np.isfinite(x).all() or epochs<1
        or np.any(np.linalg.norm(x,axis=1)==0)):
        raise ValueError('Pilot requires 50..100000 unique finite nonzero members')
    x/=np.linalg.norm(x,axis=1)[:,None]
    output.mkdir(parents=True,exist_ok=True); started=time.time()
    settings={'kind':'four_pass_feasibility_pilot','rows':len(x),'dimensions':x.shape[1],
        'epochs_per_pass':epochs,'device':device,'neighbors':k,'margin_multiplier':1.05,
        'radius_slack_fraction_of_initial_nn':.10,'graph_refresh_epochs':2,
        'optimization_state':'Float64 CPU reference projection; GPU force calculation Float32',
        'stored_coordinates':'Float32, audited after readback',
        'search':'Exhaustive Float32 GPU matrix or CPU Faiss candidates, k+32 oversample, direct Float64 rerank; no global Float64 rank certificate',
        'semantic_retrieval_validated':False,'full_corpus_run':False,
        'source_vectors_modified':False,'runtime':{'python':platform.python_version(),
        'numpy':np.__version__,'faiss':faiss.__version__}}
    put(output/'settings.json',settings)
    np.save(output/'input_normalized.npy',x.astype(np.float32)); np.save(output/'embedding_ids.npy',ids)

    def neighbors(v):
        return knn_candidates(v,min(k,len(v)-1),device)

    def graph(v,nn,scales):
        e=np.column_stack((np.repeat(np.arange(len(v)),nn.shape[1]),nn.ravel()))
        e=np.unique(np.sort(e,axis=1),axis=0)
        return e,np.minimum(scales[e[:,0]],scales[e[:,1]])

    def diagnostics(v,nn):
        mu=v.mean(0); radius=np.linalg.norm(v-mu,axis=1)
        delta=np.empty(len(v))
        for i in range(0,len(v),256):
            block=np.arange(i,min(i+256,len(v)))
            delta[block]=np.linalg.norm(v[block,None,:]-v[nn[block]],axis=2).min(1)
        rng=np.random.default_rng(314159); left=rng.integers(len(v),size=10000)
        right=(left+rng.integers(1,len(v),size=len(left)))%len(v)
        pairs=np.empty(len(left))
        for i in range(0,len(left),256):
            pairs[i:i+256]=np.linalg.norm(v[left[i:i+256]]-v[right[i:i+256]],axis=1)
        medr=float(np.median(radius)); meddelta=float(np.median(delta))
        return {'nearest_distance_quantiles':np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
            'centroid_radius_quantiles':np.quantile(radius,[0,.01,.5,.99,1]).tolist(),
            'median_relative_separation':meddelta/medr if medr else None,
            'random_pair_distance_quantiles':np.quantile(pairs,[0,.01,.5,.99,1]).tolist(),
            'mean_distinct_pair_cosine':float((np.sum(v.sum(0)**2)-np.sum(v*v))/(len(v)*(len(v)-1))),
            'centroid':mu.tolist()},delta,radius

    def one_pass(initial, member_ids, name):
        folder=output/name; folder.mkdir(); v=initial.copy()
        nn0=neighbors(v); before,delta,_=diagnostics(v,nn0)
        scales=1.05*np.maximum(delta,1e-4)
        centers=neighbor_centroids(v,nn0)
        radii=np.linalg.norm(v-centers,axis=1)+.10*np.maximum(delta,1e-4)
        np.save(folder/'initial.npy',v.astype(np.float32)); np.save(folder/'embedding_ids.npy',member_ids)
        np.save(folder/'fixed_centers.npy',centers); np.save(folder/'fixed_radii.npy',radii)
        np.save(folder/'fixed_node_margin_scales.npy',scales)
        nn=nn0; history=[]; generation=0
        for start in range(0,epochs,2):
            e,m=graph(v,nn,scales); lower,upper=separation_bounds(centers,radii,e)
            keep=lower<m
            v,trace=relax(v,e[keep],m[keep],centers,radii,
                          epochs=min(2,epochs-start),device=device)
            history.append({'generation':generation,'epoch_start':start,'edges':len(e),
                'redundant_by_bounds':int((~keep).sum()),'impossible_by_bounds':int((upper<m).sum()),
                'trace':trace})
            generation+=1; nn=neighbors(v)
            put(output/'progress.json',{'phase':'optimizing','pass':name,'epochs_done':min(start+2,epochs)})
            # Audit fresh candidates in the same Float32 storage used by the checkpoint.
            probe=v.astype(np.float32).astype(np.float64)
            check_edges,check_margins=graph(probe,nn,scales)
            worst=0.
            for i in range(0,len(check_edges),4096):
                ij=check_edges[i:i+4096]
                actual=np.linalg.norm(probe[ij[:,0]]-probe[ij[:,1]],axis=1)
                worst=max(worst,float(np.max(check_margins[i:i+len(ij)]-actual)))
            if worst<=5e-6: break
        # The final refreshed graph, rather than just the graph used for the last step.
        np.save(folder/'optimized.npy',v.astype(np.float32))
        saved=np.load(folder/'optimized.npy').astype(np.float64)
        nn=neighbors(saved); after,_,_=diagnostics(saved,nn)
        e,m=graph(saved,nn,scales); distance=np.empty(len(e))
        for i in range(0,len(e),4096):
            part=e[i:i+4096]; distance[i:i+len(part)]=np.linalg.norm(saved[part[:,0]]-saved[part[:,1]],axis=1)
        deficit=np.maximum(0,m-distance); norm_error=np.abs(np.linalg.norm(saved,axis=1)-1)
        confinement=np.maximum(0,np.linalg.norm(saved-centers,axis=1)-radii)
        retained=np.array([len(set(a)&set(b))/len(a) for a,b in zip(nn0,nn)])
        displacement=np.linalg.norm(saved-initial,axis=1)
        # Export FP16 only as an independently audited optional encoding, not the optimizer state.
        half=saved.astype(np.float16).astype(np.float64)
        half_audit={'norm_max_error':float(np.max(np.abs(np.linalg.norm(half,axis=1)-1))),
            'confinement_max_excess':float(np.maximum(0,np.linalg.norm(half-centers,axis=1)-radii).max())}
        result={'members':len(v),'before':before,'after':after,'history':history,
            'epochs_used':min(start+2,epochs),
            'final_refreshed_graph_edges':len(e),'final_mean_deficit':float(deficit.mean()),
            'final_maximum_deficit':float(deficit.max()),'final_violated_edges':int((deficit>1e-5).sum()),
            'norm_max_error_after_fp32_save':float(norm_error.max()),
            'confinement_max_excess_after_fp32_save':float(confinement.max()),
            'initial_neighbor_retention_mean':float(retained.mean()),
            'displacement_quantiles':np.quantile(displacement,[0,.5,.99,1]).tolist(),
            'optional_fp16_rounding_audit':half_audit,'semantic_retrieval_validated':False,
            'declared_sample_graph_feasible':bool((deficit<=1e-5).all() and norm_error.max()<1e-6 and confinement.max()<1e-6)}
        pq.write_table(pa.table({'source_id':member_ids[e[:,0]],'target_id':member_ids[e[:,1]],
            'minimum_distance':m,'actual_distance':distance,'deficit':deficit}),folder/'constraints.parquet',compression='zstd')
        pq.write_table(pa.table({'embedding_id':member_ids,'displacement':displacement,
            'initial_neighbor_retention':retained,'norm_error':norm_error,'confinement_excess':confinement}),
            folder/'members.parquet',compression='zstd')
        put(folder/'result.json',result); seal(folder,{'kind':'constrained_relaxation_pass','rows':len(v)})
        return saved,result

    parent,parent_result=one_pass(x,ids,'parent')
    groups,leaves=spatial_five_way(parent,ids,leaf_size=128,seed=1729)
    piles=np.array([0,0,1,1,2],dtype=np.uint8)[groups]
    pq.write_table(pa.table({'embedding_id':ids,'five_way_group':groups,'spatial_leaf':leaves,'pile':piles}),
                   output/'split.parquet',compression='zstd')
    split_audit=[]
    for group in range(5):
        part=parent[groups==group]
        coordinate_ks=np.array([ks_2samp(parent[:,j],part[:,j]).statistic for j in range(parent.shape[1])])
        split_audit.append({'group':group,'members':len(part),
            'coordinate_ks_median':float(np.median(coordinate_ks)),
            'coordinate_ks_maximum':float(coordinate_ks.max()),
            'coordinate_mean_l2_error':float(np.linalg.norm(parent.mean(0)-part.mean(0)))})
    put(output/'split_audit.json',split_audit)
    results={'parent':parent_result}; sizes=[]
    for group,name in enumerate(['pile_a','pile_b','pile_c']):
        select=piles==group; sizes.append(int(select.sum()))
        _,results[name]=one_pass(parent[select],ids[select],name)
    summary={'passes_completed':4,'parent_members':len(x),'pile_sizes':sizes,
        'five_way_counts':np.bincount(groups,minlength=5).tolist(),
        'elapsed_seconds':time.time()-started,'full_corpus_run':False,
        'semantic_retrieval_validated':False,'all_passes_feasible':all(r['declared_sample_graph_feasible'] for r in results.values()),
        'pass_results':{name:{k:r[k] for k in ['members','final_mean_deficit','final_maximum_deficit',
            'final_violated_edges','norm_max_error_after_fp32_save','confinement_max_excess_after_fp32_save',
            'initial_neighbor_retention_mean','declared_sample_graph_feasible']} for name,r in results.items()}}
    put(output/'summary.json',summary); put(output/'progress.json',{'phase':'complete'})
    shutil.copy2(__file__,output/'constraints.py')
    seal(output,{'kind':'four_pass_constraint_pilot','complete_execution_not_feasibility':True})
    return summary
