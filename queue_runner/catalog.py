"""Portable deployment sets and finite useful-work backlogs."""
from pathlib import PurePosixPath
import copy,math


def safe_json_path(path):
    if not isinstance(path,str):raise ValueError('Configuration path must be a string')
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or '\\' in path or ':' in path or p.suffix!='.json':raise ValueError('Expected repository-relative JSON path')
    return path


def validate_policy(policy):
    if not isinstance(policy,dict) or policy.get('idle_mode','drain') not in ['drain','await_work']:raise ValueError('Unknown queue idle policy')
    reserve=policy.get('checkpoint_reserve_seconds',0)
    if type(reserve) not in [int,float] or not math.isfinite(reserve) or not 0<=reserve<=3600:raise ValueError('Invalid checkpoint reserve')
    if not isinstance(policy.get('serve_when_idle',False),bool):raise ValueError('serve_when_idle must be Boolean')
    if 'idle_resources' in policy:
        from .resources import validate_resources
        validate_resources(policy['idle_resources'])
        if policy['idle_resources'].get('tpu_chips')!=[]:raise ValueError('Idle controller must leave TPU chips available to serving')


def resolve(manifest,read_json):
    value=copy.deepcopy(manifest)
    if not value.get('configuration'):return value
    config=value['configuration']
    registry=read_json(safe_json_path(config['registry']));selection=read_json(safe_json_path(config['selection']))
    if registry.get('version')!=1:raise ValueError('Expected catalog version 1')
    sets=registry.get('sets',{});active=selection.get('active_set')
    if active is not None:
        if active not in sets:raise ValueError('Unknown selected service set: '+str(active))
        value['services']=copy.deepcopy(sets[active]['services'])
    value['service_sets']={name:copy.deepcopy(spec['services']) for name,spec in sets.items()}
    for name in selection.get('backlogs',[]):
        if name not in registry.get('backlogs',{}):raise ValueError('Unknown backlog: '+str(name))
        backlog=read_json(safe_json_path(registry['backlogs'][name]))
        value['jobs'].extend(copy.deepcopy(backlog['jobs']))
    value['policy']={k:selection[k] for k in ['idle_mode','checkpoint_reserve_seconds','serve_when_idle','idle_resources'] if k in selection}
    validate_policy(value['policy'])
    for job in value['jobs']:
        if 'service_set' in job and job['service_set'] not in sets:raise ValueError('Unknown job service set')
    return value
