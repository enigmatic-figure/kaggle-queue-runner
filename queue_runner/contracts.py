"""Workload contracts."""
import hashlib,json,re,math
from pathlib import PurePosixPath


def validate(value):
    if not isinstance(value,dict) or value.get('version')!=1 or not isinstance(value.get('jobs'),list):raise ValueError('Expected version 1 workload list')
    seen=set()
    for job in value['jobs']:
        if not isinstance(job,dict):raise ValueError('Each job must be an object')
        identity=job.get('id','');script=job.get('script','')
        if not isinstance(identity,str) or not isinstance(script,str):raise ValueError('Job IDs/scripts must be strings')
        p=PurePosixPath(script)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',identity) or identity in seen:raise ValueError('Use unique simple versioned job IDs')
        seen.add(identity)
        if not script or p.is_absolute() or '..' in p.parts or '\\' in script or ':' in script or p.suffix!='.py':raise ValueError('Expected relative Python script')
        timeout=job.get('timeout_seconds',3600)
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:raise ValueError('Positive finite job timeout required')
        if not isinstance(job.get('enabled',True),bool):raise ValueError('enabled must be Boolean')
        minimum=job.get('min_remaining_seconds',1)
        if isinstance(minimum,bool) or not isinstance(minimum,(int,float)) or not math.isfinite(minimum) or minimum<0:raise ValueError('Invalid minimum remaining time')
        if not isinstance(job.get('args',[]),list) or any(not isinstance(a,str) for a in job.get('args',[])):raise ValueError('Arguments must be strings')
        if not isinstance(job.get('secrets',[]),list) or any(not isinstance(s,str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',s) for s in job.get('secrets',[])):raise ValueError('Invalid secret label')
        if job.get('on_failure','continue') not in ['continue','stop']:raise ValueError('Unknown failure policy')
        if 'resources' in job:
            from .resources import validate_resources
            validate_resources(job['resources'])
        if not isinstance(job.get('depends_on',[]),list) or any(not isinstance(v,str) for v in job.get('depends_on',[])):raise ValueError('Dependency IDs must be strings')
    for job in value['jobs']:
        if any(d not in seen or d==job['id'] for d in job.get('depends_on',[])):raise ValueError('Unknown/self dependency')
    dependencies={job['id']:job.get('depends_on',[]) for job in value['jobs']};visited=set();active=set()
    def visit(identity):
        if identity in active:raise ValueError('Cyclic job dependencies')
        if identity in visited:return
        active.add(identity)
        for d in dependencies[identity]:visit(d)
        active.remove(identity);visited.add(identity)
    for identity in dependencies:visit(identity)
    services=value.get('services',[])
    if not isinstance(services,list) or len(services)>1:raise ValueError('At most one resident inference service is currently supported')
    for service in services:
        if not isinstance(service,dict):raise ValueError('Service must be an object')
        identity=service.get('id','');profile=service.get('profile','')
        if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',identity):raise ValueError('Invalid service ID')
        if not isinstance(service.get('enabled',False),bool):raise ValueError('Service enabled must be Boolean')
        if not isinstance(profile,str):raise ValueError('Service profile must be a path')
        p=PurePosixPath(profile)
        if p.is_absolute() or '..' in p.parts or '\\' in profile or ':' in profile or p.suffix!='.json':raise ValueError('Expected relative JSON service profile')
    return value


def fingerprint(job):
    return hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
