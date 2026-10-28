"""Profile selected resident endpoints and retain each request as a sealed artifact."""
from pathlib import Path
import argparse,json,math,os,time,urllib.request
from queue_runner.inference_client import Session
from queue_runner.sdk import output_dir,checkpoint,progress

PROMPTS=[
    'A kitten is sleeping beside a window.', 'The server completed its backup successfully.',
    'An experiment needs a reproducible random seed.', 'Two sentences express the same idea differently.',
    'The train arrived before sunrise.', 'A graph can contain overlapping communities.',
    'Rainfall increased during the summer.', 'A researcher checks the source of a surprising result.',
    'The nearest neighbor changed after optimization.', 'A document contains several related paragraphs.',
    'The smallest model can ask a teacher for help.', 'A useful checkpoint survives a stopped process.']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--requests',type=int,default=3);p.add_argument('--startup-seconds',type=float,default=2400)
    args=p.parse_args()
    if not 1<=args.requests<=len(PROMPTS) or args.startup_seconds<=0:raise ValueError('Invalid profiling budget')
    token=os.environ['QUEUE_CONTROL_TOKEN'];control=os.environ['QUEUE_CONTROL_URL'];root=output_dir();started=time.monotonic()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));states={};status={}
    while time.monotonic()-started<args.startup_seconds:
        request=urllib.request.Request(control+'/status',headers={'Authorization':'Bearer '+token})
        with opener.open(request,timeout=15) as response:status=json.load(response).get('inference',{})
        states=status.get('services',{})
        if status.get('phase')=='ready':break
        if status.get('phase') in ['deferred','failed','stopped']:break
        progress(stage='waiting_for_model',inference=status);time.sleep(2)
    summary={'model_loading_seconds':time.monotonic()-started,'source_commit':os.environ.get('QUEUE_SOURCE_COMMIT'),
        'initial_status':status,'services':{},'scope':'Startup and request profiling; not a retrieval or agent-quality benchmark'}
    for identity,info in states.items():
        record={'model':info.get('model'),'revision':info.get('revision'),'runner':info.get('runner'),'outcomes':[]}
        summary['services'][identity]=record
        if info.get('phase')!='ready':record['status']='not_ready';continue
        session=Session(info['local_api_url'],token,root/'private-traces'/(identity+'.jsonl'),
            metadata={'model_revision':info.get('revision'),'role':'profiling','source_commit':summary['source_commit']})
        for index,prompt in enumerate(PROMPTS[:args.requests]):
            tick=time.perf_counter();outcome={'request':index}
            try:
                if info.get('runner')=='pooling':
                    value=session.request('embeddings',{'model':info['model'],'input':[prompt]},timeout=120)
                    vector=value['data'][0]['embedding'];outcome.update(dimensions=len(vector),norm=math.sqrt(sum(v*v for v in vector)))
                else:
                    value=session.request('chat/completions',{'model':info['model'],
                        'messages':[{'role':'user','content':'Reply with one short sentence describing this text: '+prompt}],
                        'temperature':0},timeout=120)
                    outcome.update(response=value['choices'][0]['message'],usage=value.get('usage'))
                outcome['status']='complete'
            except Exception as error:outcome.update(status='failed',error_type=type(error).__name__)
            outcome['seconds']=time.perf_counter()-tick;record['outcomes'].append(outcome)
            part=root/'measurements'/(identity+'-'+str(index));part.mkdir(parents=True)
            (part/'request.json').write_text(json.dumps(outcome,indent=2));checkpoint(part.relative_to(root))
            progress(stage='profiling',service=identity,request=index,status=outcome['status'])
        record['status']='complete' if all(o['status']=='complete' for o in record['outcomes']) else 'request_failures'
    closed=root/'summary';closed.mkdir();(closed/'profile.json').write_text(json.dumps(summary,indent=2));checkpoint('summary')


if __name__=='__main__':main()
