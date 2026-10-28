import json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from queue_runner import inference_client


def test_api_trace_retains_messages_tool_results_and_provenance_without_credentials(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            assert self.headers.get('Authorization')=='Bearer example-private-token'
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            value={'choices':[{'message':{'role':'assistant','content':'example-private-token'}}],'usage':{'total_tokens':5}}
            data=json.dumps(value).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        assert hasattr(inference_client,'Session'), 'Missing portable session trace client'
        session=inference_client.Session('http://127.0.0.1:'+str(server.server_port)+'/v1','example-private-token',
            tmp_path/'session.jsonl',metadata={'model_revision':'a'*40,'role':'agent'})
        value=session.request('chat/completions',{'model':'example','messages':[{'role':'user','content':'inspect worktree'}]})
        session.record_tool('call-1','terminal',{'argv':['git','status']},{'exit_code':0,'stdout':'clean'})
        rows=[json.loads(line) for line in (tmp_path/'session.jsonl').read_text().splitlines()]
        assert rows[0]['request']['messages'][0]['content']=='inspect worktree'
        assert rows[0]['metadata']['model_revision']=='a'*40
        assert rows[1]['kind']=='tool' and rows[1]['result']['exit_code']==0
        assert 'example-private-token' not in (tmp_path/'session.jsonl').read_text()
        assert value['usage']['total_tokens']==5
    finally:server.shutdown();server.server_close()


def test_profiling_workload_records_real_http_request_and_sealed_metrics(tmp_path):
    import os,sys,subprocess
    from pathlib import Path
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def send(self,value):
            body=json.dumps(value).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
        def do_GET(self):
            base='http://127.0.0.1:'+str(self.server.server_port)
            self.send({'inference':{'phase':'ready','services':{'embed':{'model':'example','revision':'a'*40,
                'runner':'pooling','phase':'ready','local_api_url':base+'/v1'}}}})
        def do_POST(self):
            assert self.headers.get('Authorization')=='Bearer example'
            self.rfile.read(int(self.headers['Content-Length']))
            self.send({'data':[{'embedding':[0.,0.,1.]}]})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    root=Path(__file__).resolve().parents[1]
    try:
        env={**os.environ,'QUEUE_CONTROL_URL':'http://127.0.0.1:'+str(server.server_port),
            'QUEUE_CONTROL_TOKEN':'example','QUEUE_OUTPUT_DIR':str(tmp_path/'out'),'PYTHONPATH':str(root)}
        result=subprocess.run([sys.executable,str(root/'workloads/profile_services.py'),'--requests','1'],env=env,cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        value=json.loads((tmp_path/'out/summary/profile.json').read_text())
        assert value['services']['embed']['outcomes'][0]['dimensions']==3
        assert value['services']['embed']['outcomes'][0]['norm']==1
        assert (tmp_path/'out/summary/checkpoint.ready.json').is_file()
    finally:server.shutdown();server.server_close()
