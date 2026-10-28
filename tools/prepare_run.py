"""Prepare a reviewable future queue payload and selection; never submit or reset a queue."""
from pathlib import Path
import argparse,json,math


def main():
    root=Path(__file__).resolve().parents[1];registry=json.loads((root/'catalog/sets.json').read_text())
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hours',type=float,required=True)
    p.add_argument('--set',dest='deployment',default=registry['default_agent_set'],choices=registry['sets'])
    p.add_argument('--backlog',action='append',choices=['none',*registry['backlogs']]);p.add_argument('--output',required=True)
    args=p.parse_args()
    if not math.isfinite(args.hours) or not 0<args.hours<=24:raise ValueError('Choose the budget allowed for the intended allocation')
    output=Path(args.output).resolve()
    if output.exists():raise ValueError('Use a fresh output directory')
    notebook=json.loads((root/'kaggle/queue_runner.ipynb').read_text())
    for cell in notebook['cells']:
        if cell.get('cell_type')!='code':continue
        source=''.join(cell['source']);needle="'--hours','8'"
        if source.count(needle)!=1:raise ValueError('Notebook budget template changed; inspect it before preparing a payload')
        cell['source']=source.replace(needle,"'--hours',"+repr(str(args.hours))).splitlines(keepends=True)
    backlogs=args.backlog if args.backlog is not None else ['model-profiling']
    if 'none' in backlogs:
        if len(backlogs)!=1:raise ValueError('none cannot be combined with another backlog')
        backlogs=[]
    selection={'active_set':args.deployment,'backlogs':backlogs,'idle_mode':'await_work','serve_when_idle':True,
        'idle_resources':{'tpu_chips':[],'numa_node':1,'cpu_threads':1},'checkpoint_reserve_seconds':300}
    output.mkdir(parents=True)
    (output/'queue_runner.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    (output/'kernel-metadata.json').write_text((root/'kaggle/kernel-metadata.json').read_text())
    (output/'selection.to-publish.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({'prepared':str(output),'hours_requested':args.hours,'active_set':args.deployment,
        'backlogs':backlogs,'submitted':False,'quota_and_platform_limit_verified':False,
        'next_step':'Review selection, publish it to catalog/selection.json, then submit this payload at the chosen time.'}))


if __name__=='__main__':main()
