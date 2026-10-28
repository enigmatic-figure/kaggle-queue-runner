"""Observed CPU and TPU resource partitions."""
from pathlib import Path
import os,re


def cpu_list(value):
    result=set()
    for part in value.strip().split(','):
        if not part:continue
        bounds=[int(v) for v in part.split('-')]
        result.update(range(bounds[0],bounds[-1]+1))
    return sorted(result)


def discover(sysfs='/sys',allowed_cpus=None):
    root=Path(sysfs)
    allowed=set(allowed_cpus if allowed_cpus is not None else
        os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else range(os.cpu_count() or 1))
    nodes={};chips={}
    for p in sorted((root/'devices/system/node').glob('node[0-9]*')):
        try:nodes[int(p.name[4:])]=sorted(set(cpu_list((p/'cpulist').read_text()))&allowed)
        except (OSError,ValueError):continue
    for family,pattern in [('accel','accel[0-9]*'),('apex','apex_[0-9]*')]:
        for p in sorted((root/'class'/family).glob(pattern)):
            try:
                chip=int(re.search(r'(\d+)$',p.name).group(1));node=int((p/'device/numa_node').read_text())
                if node>=0:
                    if chip in chips and chips[chip]!=node:raise ValueError('Conflicting TPU NUMA mapping')
                    chips[chip]=node
            except OSError:continue
    return {'nodes':nodes,'chips':chips,'allowed_cpus':sorted(allowed)}


def validate_resources(value):
    if not isinstance(value,dict):raise ValueError('resources must be an object')
    chips=value.get('tpu_chips','all')
    if chips!='all' and (not isinstance(chips,list) or any(type(i)!=int or not 0<=i<8 for i in chips) or len(set(chips))!=len(chips)):
        raise ValueError('tpu_chips must be all or unique physical chip IDs 0..7')
    if 'numa_node' in value and (type(value['numa_node'])!=int or value['numa_node']<0):raise ValueError('Invalid NUMA node')
    if 'cpu_threads' in value and (type(value['cpu_threads'])!=int or value['cpu_threads']<1):raise ValueError('Positive CPU thread count required')


def _cpu_plan(spec,topology):
    node=spec.get('numa_node');cpus=None
    if node is not None:
        cpus=topology['nodes'].get(node)
        if not cpus:raise ValueError('Selected NUMA node has no allowed CPUs')
        cpus=list(cpus)
    threads=spec.get('cpu_threads',len(cpus) if cpus else 1)
    if cpus:cpus=cpus[:threads]
    env={name:str(min(threads,len(cpus)) if cpus else threads)
         for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
    if cpus:env['QUEUE_CPUSET']=','.join(map(str,cpus))
    return cpus,env


def job_plan(job,topology):
    spec=job.get('resources')
    if spec is None:return {'chips':list(range(8)),'cpus':None,'environment':{}}
    validate_resources(spec)
    chips=spec.get('tpu_chips','all');chips=list(range(8)) if chips=='all' else list(chips)
    cpus,env=_cpu_plan(spec,topology)
    if not chips:env.update(JAX_PLATFORMS='cpu',PJRT_DEVICE='CPU',TPU_VISIBLE_CHIPS='',CUDA_VISIBLE_DEVICES='')
    elif len(chips)!=8:
        if chips not in [[0,1,2,3],[4,5,6,7]]:raise ValueError('Partial TPU jobs currently require one contiguous 2x2 partition')
        env.update(TPU_VISIBLE_CHIPS=','.join(map(str,chips)),TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1',TPU_PROCESS_BOUNDS='1,1,1')
    return {'chips':chips,'cpus':cpus,'environment':env}


def service_plan(profile,topology,primary):
    chips=profile['chips'];node=profile['numa_node']
    if not topology['chips']:raise ValueError('TPU chip-to-NUMA mapping is unavailable; service deferred')
    if chips not in [[0,1,2,3],[4,5,6,7]]:raise ValueError('Service requires a contiguous four-chip v5e partition')
    if any(topology['chips'].get(i)!=node for i in chips):raise ValueError('Requested chips do not match observed NUMA mapping')
    if set(chips)&set(primary['chips']):raise ValueError('TPU chip overlap with primary workload')
    cpus,env=_cpu_plan(profile,topology)
    if primary['cpus'] is None or set(cpus)&set(primary['cpus']):raise ValueError('Primary must reserve disjoint CPU affinity before service can start')
    env.update(TPU_VISIBLE_CHIPS=','.join(map(str,chips)),TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1',
        TPU_PROCESS_BOUNDS='1,1,1',JAX_PLATFORMS='tpu',PJRT_DEVICE='TPU',TPU_PROCESS_PORT='8477',
        TPU_PROCESS_ADDRESSES='localhost:8477',CLOUD_TPU_TASK_ID='0',JAX_PROCESS_ID='0',JAX_NUM_PROCESSES='1')
    return {'chips':chips,'cpus':cpus,'environment':env}
