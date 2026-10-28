"""Refresh the workload list between pinned job executions."""
import time,uuid,math
from .contracts import fingerprint
from .io import put


def run_queue(source,runner,*,max_seconds=28800,idle_seconds=300,poll_seconds=30,secret_provider=None):
    if not all(math.isfinite(v) and v>0 for v in [max_seconds,idle_seconds,poll_seconds]):raise ValueError('Positive finite scheduler times required')
    started=time.monotonic();idle_started=None;failures=0;reason='session_limit'
    while time.monotonic()-started<max_seconds:
        if runner.stop.is_set():reason='stop_requested';break
        if runner.pause.is_set():runner.reload.wait(poll_seconds);runner.reload.clear();continue
        try:
            commit,manifest=source.refresh();failures=0
        except Exception as error:
            failures+=1;runner.event('source_refresh_failed',attempt=failures,error=str(error))
            if failures>=5:reason='source_unavailable';break
            runner.reload.wait(poll_seconds);runner.reload.clear();continue
        ran=False;pending=[]
        for job in manifest['jobs']:
            if not job.get('enabled',True):continue
            old=runner.receipt(job['id'])
            if old:
                if old['spec_sha256']!=fingerprint(job):raise ValueError('Changed specification requires a new job ID: '+job['id'])
                continue
            dependencies=[runner.receipt(d) for d in job.get('depends_on',[])]
            if any(not d or d['status']!='completed' for d in dependencies):
                pending.append({'id':job['id'],'reason':'dependencies'});continue
            if secret_provider:
                for name in job.get('secrets',[]):
                    if not runner.secrets.get(name):
                        value=secret_provider(name)
                        if value:runner.secrets[name]=value
            missing=[name for name in job.get('secrets',[]) if not runner.secrets.get(name)]
            if missing:pending.append({'id':job['id'],'reason':'missing_secrets','labels':missing});continue
            remaining=max_seconds-(time.monotonic()-started)
            if remaining<job.get('min_remaining_seconds',1):pending.append({'id':job['id'],'reason':'remaining_time'});continue
            destination=runner.root/'snapshots'/(job['id']+'-'+commit[:12]+'-'+uuid.uuid4().hex[:8])
            snapshot=source.snapshot(commit,destination)
            if runner.services:
                runner.services.before_job(manifest,job,snapshot,commit,remaining)
            result=runner.run_one(job,snapshot,commit,remaining=remaining)
            ran=True;idle_started=None
            put(runner.root/'status.json',{'phase':'job_finished','commit':commit,'last_result':result,**runner.status()})
            if result['status']!='completed' and job.get('on_failure','continue')=='stop':
                runner.stop.set();reason='job_failure'
            break  # Refresh the controlled repository before selecting the next job.
        if ran:continue
        if idle_started is None:idle_started=time.monotonic()
        put(runner.root/'status.json',{'phase':'waiting_for_work','commit':commit,'pending':pending,
            'idle_seconds':time.monotonic()-idle_started,**runner.status()})
        if time.monotonic()-idle_started>=idle_seconds:reason='idle_timeout';break
        runner.reload.wait(min(poll_seconds,max_seconds-(time.monotonic()-started)));runner.reload.clear()
    result={'reason':reason,'elapsed_seconds':time.monotonic()-started,**runner.status()}
    put(runner.root/'final.json',result);runner.event('runner_finished',reason=reason);return result
