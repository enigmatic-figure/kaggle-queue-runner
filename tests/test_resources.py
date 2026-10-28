import os
import pytest
from queue_runner import resources


def test_topology_uses_sysfs_and_intersects_container_cpu_affinity(tmp_path):
    node=tmp_path/'devices/system/node'
    for index, cpus in [(0,'0-3'),(1,'4-7')]:
        p=node/f'node{index}';p.mkdir(parents=True);(p/'cpulist').write_text(cpus)
    for chip in range(8):
        p=tmp_path/f'class/accel/accel{chip}/device';p.mkdir(parents=True)
        (p/'numa_node').write_text(str(chip//4))
    result=resources.discover(tmp_path,allowed_cpus={1,2,5,6})
    assert result['nodes']=={0:[1,2],1:[5,6]}
    assert result['chips']=={i:i//4 for i in range(8)}


def test_cpu_only_job_cannot_initialize_tpu_and_leaves_chips_for_service():
    topology={'nodes':{0:[0,1],1:[2,3]},'chips':{i:i//4 for i in range(8)}}
    primary=resources.job_plan({'resources':{'tpu_chips':[],'numa_node':1,'cpu_threads':2}},topology)
    assert primary['environment']['JAX_PLATFORMS']=='cpu'
    assert primary['cpus']==[2,3] and primary['chips']==[]
    service=resources.service_plan({'numa_node':0,'chips':[0,1,2,3],'cpu_threads':2},topology,primary)
    assert service['environment']['TPU_VISIBLE_CHIPS']=='0,1,2,3'
    assert service['environment']['TPU_CHIPS_PER_PROCESS_BOUNDS']=='2,2,1'


def test_undeclared_job_reserves_all_chips_and_blocks_service():
    topology={'nodes':{0:[0],1:[1]},'chips':{i:i//4 for i in range(8)}}
    primary=resources.job_plan({},topology)
    assert set(primary['chips'])==set(range(8))
    with pytest.raises(ValueError,match='overlap'):
        resources.service_plan({'numa_node':0,'chips':[0,1,2,3]},topology,primary)


def test_numa_selection_requires_observed_chip_mapping_and_no_cpu_overlap():
    topology={'nodes':{0:[0],1:[1]},'chips':{i:i//4 for i in range(8)}}
    profile={'numa_node':0,'chips':[0,1,2,3]}
    with pytest.raises(ValueError,match='NUMA'):
        resources.service_plan({**profile,'chips':[4,5,6,7]},topology,resources.job_plan({'resources':{'tpu_chips':[]}},topology))
    with pytest.raises(ValueError,match='CPU'):
        resources.service_plan(profile,topology,resources.job_plan({'resources':{'tpu_chips':[],'numa_node':0}},topology))
    with pytest.raises(ValueError,match='mapping'):
        resources.service_plan(profile,{'nodes':topology['nodes'],'chips':{}},resources.job_plan({'resources':{'tpu_chips':[]}},topology))


def test_contract_rejects_invalid_resources_and_service_paths():
    from queue_runner.contracts import validate
    job={'id':'cpu-v1','script':'job.py'}
    for declaration in [{'tpu_chips':[1,1]},{'tpu_chips':[True]},{'tpu_chips':[],'cpu_threads':0}]:
        with pytest.raises(ValueError):validate({'version':1,'jobs':[{**job,'resources':declaration}]})
    with pytest.raises(ValueError):validate({'version':1,'jobs':[job],'services':[{'id':'llm','profile':'../bad.json'}]})
