"""Memory-bounded graph optimization with serialized block updates."""
import numpy as np


def neighbor_index(x, nlist=1024, nprobe=32, device='cuda', progress=None):
    """FP16 scalar-quantized IVF candidates, reranked later on FP32 masters.

    The compressed index is a search aid, never the optimized coordinate state.
    Candidate recall must be measured independently at each refreshed graph.
    """
    import faiss
    faiss.omp_set_num_threads(4)
    if x.shape[1]<2 or len(x)<nlist*40 or not 1<=nprobe<=nlist:
        raise ValueError('Insufficient training rows or invalid IVF settings')
    cpu=faiss.index_factory(x.shape[1],f'IVF{nlist},SQfp16')
    resource=None;index=cpu
    if device!='cpu':
        resource=faiss.StandardGpuResources();resource.setTempMemory(512*1024**2)
        resource.setDefaultNullStreamAllDevices()
        options=faiss.GpuClonerOptions();options.indicesOptions=faiss.INDICES_32_BIT
        index=faiss.index_cpu_to_gpu(resource,0,cpu,options)
        if hasattr(index,'reserveMemory'):index.reserveMemory(len(x))
    training=np.sort(np.random.default_rng(1729).choice(len(x),min(len(x),nlist*64),replace=False))
    index.train(np.asarray(x[training],dtype=np.float32,order='C'));index.nprobe=nprobe
    for start in range(0,len(x),32768):
        index.add(np.asarray(x[start:start+32768],dtype=np.float32,order='C'))
        if progress:progress({'phase':'building_neighbor_index','rows':min(len(x),start+32768),'total':len(x)})
    return index,resource


def search_neighbors(x, index, queries, k=16, pool=96, device='cuda', batch_rows=128):
    """Direct Float64 distances and stable row-ID ties on retrieved candidates."""
    queries=np.asarray(queries,dtype=np.int64)
    if queries.ndim!=1 or np.any(queries<0) or np.any(queries>=len(x)) or not 1<=k<len(x):
        raise ValueError('Invalid neighbor query rows/count')
    count=min(len(x),max(k+1,pool));result=np.empty((len(queries),k),dtype=np.int32)
    for start in range(0,len(queries),batch_rows):
        rows=queries[start:start+batch_rows]
        q=np.asarray(x[rows],dtype=np.float32,order='C')
        _,found=index.search(q,count)
        safe=np.maximum(found,0)
        if device=='cpu':
            vectors=np.asarray(x[safe],dtype=np.float64)
            distance=np.linalg.norm(vectors-q.astype(np.float64)[:,None],axis=2)
        else:
            import torch
            vectors=torch.as_tensor(np.asarray(x[safe],dtype=np.float32),device=device).double()
            qt=torch.as_tensor(q,device=device,dtype=torch.float64)
            distance=torch.linalg.vector_norm(vectors-qt[:,None],dim=2).cpu().numpy()
            del vectors,qt
        distance[(found<0)|(found==rows[:,None])]=np.inf
        order=np.lexsort((found,distance),axis=1)[:,:k]
        selected=np.take_along_axis(found,order,axis=1)
        if not np.isfinite(np.take_along_axis(distance,order,axis=1)).all():
            raise ValueError('Insufficient finite IVF candidates; increase nprobe/pool')
        result[start:start+len(rows)]=selected
    return result


def symmetric_csr(neighbors):
    """Unique bidirectional adjacency; incoming constraints survive ownership cuts."""
    from scipy.sparse import csr_matrix
    nn=np.asarray(neighbors)
    if nn.ndim!=2 or not np.issubdtype(nn.dtype,np.integer) or np.any(nn<0) or np.any(nn>=len(nn)):
        raise ValueError('Neighbor bounds/shape invalid')
    if np.any(nn==np.arange(len(nn))[:,None]):raise ValueError('Neighbor graph contains self edges')
    rows=np.repeat(np.arange(len(nn),dtype=np.int32),nn.shape[1])
    graph=csr_matrix((np.ones(nn.size,dtype=np.uint8),(rows,nn.ravel())),shape=(len(nn),len(nn)))
    graph.sum_duplicates();graph=graph.maximum(graph.T);graph.sort_indices()
    return graph.indptr.astype(np.int64),graph.indices.astype(np.int32)


def block_step(x, indptr, indices, scales, owners, centers, radii, *,device='cpu',retries=16,edge_batch=8192,force_backend=None):
    """One serial coordinate-block update of the global fixed-edge energy.

    Only owners can move. Every incident edge (incoming and outgoing) is scored
    against the current shared state, so accepted moves cannot increase the
    global fixed-graph hinge energy. This is not a convergence certificate.
    Projection and acceptance audit the actual FP32 candidate coordinates.
    """
    from .constraints import project_sphere_ball
    own=np.asarray(owners,dtype=np.int64)
    if own.ndim!=1 or len(own)==0 or np.any(own[1:]<=own[:-1]) or own[0]<0 or own[-1]>=len(x):
        raise ValueError('Owners must be sorted unique valid row indices')
    if np.asarray(centers).shape!=(len(own),x.shape[1]) or np.asarray(radii).shape!=(len(own),):
        raise ValueError('Owner region shapes differ')
    counts=indptr[own+1]-indptr[own]
    left=np.repeat(own,counts)
    right=np.concatenate([indices[indptr[i]:indptr[i+1]] for i in own])
    if len(left)==0:return {'accepted':False,'edges':0,'energy_before':0.,'energy_after':0.}
    edges=np.unique(np.sort(np.column_stack((left,right)),axis=1),axis=0)
    nodes=np.unique(edges);local=np.searchsorted(nodes,edges)
    owner_local=np.searchsorted(nodes,own)
    # Empty-degree owners are handled explicitly rather than indexing a wrong row.
    present=(owner_local<len(nodes))
    present[present]&=nodes[owner_local[present]]==own[present]
    original=np.array(x[own],dtype=np.float64,copy=True)
    points=np.array(x[nodes],dtype=np.float64,copy=True)
    margins=np.minimum(scales[edges[:,0]],scales[edges[:,1]]).astype(np.float64)
    if force_backend is not None:
        positions=np.where(present,owner_local,-1)
        owner_force,max_degree=force_backend(points,local,margins,positions)
        owner_force=np.asarray(owner_force,dtype=np.float64)
        if owner_force.shape!=original.shape or not np.isfinite(owner_force).all() or max_degree<0:
            raise ValueError('Invalid force backend output')
    else:
        import torch
        dtype=torch.float64 if device=='cpu' else torch.float32
        p=torch.as_tensor(points,device=device,dtype=dtype)
        e=torch.as_tensor(local,device=device,dtype=torch.int64)
        m=torch.as_tensor(margins,device=device,dtype=dtype)
        force=torch.zeros_like(p);degree=torch.zeros(len(nodes),device=device,dtype=torch.int64)
        for start in range(0,len(edges),edge_batch):
            ij=e[start:start+edge_batch];a,b=ij.T
            delta=p[a]-p[b];distance=torch.linalg.vector_norm(delta,dim=1)
            active=m[start:start+len(ij)]>distance
            if not bool(active.any()):continue
            a,b,delta,distance,margin=a[active],b[active],delta[active],distance[active],m[start:start+len(ij)][active]
            collision=distance<1e-12
            direction=delta/distance.clamp_min(1e-12)[:,None]
            if bool(collision.any()):
                # Canonical pair orientation supplies equal/opposite collision escape.
                z=p[a[collision]]
                axis=torch.argmin(torch.abs(z),dim=1)
                tangent=torch.zeros_like(z);tangent[torch.arange(len(z),device=device),axis]=1
                tangent-=torch.sum(tangent*z,dim=1)[:,None]*z
                tangent/=torch.linalg.vector_norm(tangent,dim=1)[:,None]
                direction[collision]=tangent
            push=(margin-distance)[:,None]*direction
            force.index_add_(0,a,push);force.index_add_(0,b,-push)
            degree.index_add_(0,a,torch.ones_like(a));degree.index_add_(0,b,torch.ones_like(b))
        owner_force=np.zeros_like(original)
        owner_force[present]=force[torch.as_tensor(owner_local[present],device=device)].cpu().numpy()
        max_degree=int(degree.max())
    if max_degree==0:
        return {'accepted':False,'edges':len(edges),'energy_before':0.,'energy_after':0.}
    def assess(owner_coordinates):
        trial=points.copy();trial[owner_local[present]]=owner_coordinates[present]
        deficit=np.empty(len(edges),dtype=np.float64)
        for start in range(0,len(edges),edge_batch):
            ij=local[start:start+edge_batch]
            deficit[start:start+len(ij)]=np.maximum(0,margins[start:start+len(ij)]-
                np.linalg.norm(trial[ij[:,0]]-trial[ij[:,1]],axis=1))
        return float(np.sum(deficit**2)),float(deficit.max())
    before,initial_deficit=assess(original)
    for backtrack in range(retries):
        candidate=project_sphere_ball(original+(.5/max_degree)*(.5**backtrack)*owner_force,centers,radii).astype(np.float32)
        value,worst=assess(candidate.astype(np.float64))
        if value<=before:
            x[own]=candidate
            return {'accepted':True,'edges':len(edges),'energy_before':before,'energy_after':value,
                    'maximum_deficit_before':initial_deficit,'maximum_deficit_after':worst,'backtracks':backtrack}
    return {'accepted':False,'edges':len(edges),'energy_before':before,'energy_after':before,
            'maximum_deficit_before':initial_deficit,'maximum_deficit_after':initial_deficit,'backtracks':retries}
