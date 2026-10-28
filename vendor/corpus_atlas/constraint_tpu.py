"""TPU-friendly edge-force kernel; CPU Float64 remains the audit reference."""


def pair_forces(a,b,margin,valid):
    """Canonical left/right hinge pushes, including deterministic collision escape.

    Pads use valid=False. Shapes stay fixed under JIT; no dynamic-sized indexing.
    This kernel does not perform coordinate commits or assert global feasibility.
    """
    import jax.numpy as jnp
    a,b,margin,valid=map(jnp.asarray,(a,b,margin,valid))
    delta=a-b
    distance=jnp.linalg.norm(delta,axis=-1)
    unit=delta/jnp.maximum(distance[...,None],1e-12)
    axis=jnp.argmin(jnp.abs(a),axis=-1)
    tangent=(jnp.arange(a.shape[-1])==axis[...,None]).astype(a.dtype)
    tangent=tangent-jnp.sum(tangent*a,axis=-1,keepdims=True)*a
    tangent=tangent/jnp.maximum(jnp.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
    unit=jnp.where((distance<1e-12)[...,None],tangent,unit)
    amplitude=jnp.where(valid,jnp.maximum(0,margin-distance),0)
    return amplitude[...,None]*unit,distance


class ForceBackend:
    """Fixed-shape edge shards, reduced to forces for the mutable owners.

    Halo endpoints participate in degree accounting but never receive writes.
    Coordinate projection and Float64 acceptance remain in block_step.
    """

    def __init__(self, *, dimensions, edge_capacity=32768, owner_capacity=512, devices=None):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        devices=list(jax.devices() if devices is None else devices)
        if dimensions<2 or owner_capacity<1 or edge_capacity<1 or not devices or edge_capacity%len(devices):
            raise ValueError('Invalid dimensions/capacities or uneven device shards')
        self.dimensions=dimensions;self.edge_capacity=edge_capacity;self.owner_capacity=owner_capacity
        mesh=Mesh(np.array(devices),('workers',))
        self.vector_sharding=NamedSharding(mesh,P('workers',None))
        self.edge_sharding=NamedSharding(mesh,P('workers'))
        replicated=NamedSharding(mesh,P(None,None))

        def accumulate(a,b,margin,valid,left_owner,right_owner):
            push,distance=pair_forces(a,b,margin,valid)
            force=jnp.zeros((owner_capacity,dimensions),dtype=a.dtype)
            # owner_capacity is an out-of-bounds sentinel; negative indices wrap.
            force=force.at[left_owner].add(push,mode='drop')
            force=force.at[right_owner].add(-push,mode='drop')
            return force,(margin>distance)&valid
        self.kernel=jax.jit(accumulate,in_shardings=(self.vector_sharding,self.vector_sharding,
            self.edge_sharding,self.edge_sharding,self.edge_sharding,self.edge_sharding),
            out_shardings=(replicated,self.edge_sharding))

    def __call__(self,points,edges,margins,owner_positions):
        import jax
        import numpy as np
        p=np.asarray(points,dtype=np.float32)
        e=np.asarray(edges,dtype=np.int64);m=np.asarray(margins,dtype=np.float32)
        owners=np.asarray(owner_positions,dtype=np.int64)
        if (p.ndim!=2 or p.shape[1]!=self.dimensions or e.ndim!=2 or e.shape[1]!=2
                or m.shape!=(len(e),) or owners.ndim!=1 or len(owners)>self.owner_capacity
                or np.any(e<0) or np.any(e>=len(p)) or np.any(owners < -1)
                or np.any(owners>=len(p)) or not np.isfinite(p).all()
                or not np.isfinite(m).all() or np.any(m<0)):
            raise ValueError('Invalid edge-force input shapes, bounds, or values')
        real=owners>=0
        if len(np.unique(owners[real]))!=sum(real):raise ValueError('Duplicate owner positions')
        lookup=np.full(len(p),self.owner_capacity,dtype=np.int32)
        lookup[owners[real]]=np.flatnonzero(real)
        force=np.zeros((len(owners),self.dimensions),dtype=np.float64)
        degree=np.zeros(len(p),dtype=np.int64)
        for start in range(0,len(e),self.edge_capacity):
            ij=e[start:start+self.edge_capacity];n=len(ij)
            a=np.zeros((self.edge_capacity,self.dimensions),dtype=np.float32);b=np.zeros_like(a)
            margin=np.zeros(self.edge_capacity,dtype=np.float32);valid=np.zeros(self.edge_capacity,dtype=bool)
            left=np.full(self.edge_capacity,self.owner_capacity,dtype=np.int32);right=left.copy()
            a[:n]=p[ij[:,0]];b[:n]=p[ij[:,1]];margin[:n]=m[start:start+n];valid[:n]=True
            left[:n]=lookup[ij[:,0]];right[:n]=lookup[ij[:,1]]
            inputs=(a,b,margin,valid,left,right)
            shards=(self.vector_sharding,self.vector_sharding)+(self.edge_sharding,)*4
            result,active=jax.block_until_ready(self.kernel(*[
                jax.device_put(v,s) for v,s in zip(inputs,shards)]))
            force+=np.asarray(result)[:len(owners)].astype(np.float64)
            degree+=np.bincount(ij[np.asarray(active)[:n]].ravel(),minlength=len(p))
        return force,int(degree.max(initial=0))
