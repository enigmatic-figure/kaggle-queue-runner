"""Authenticated readiness callbacks and optional active-agent delivery."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import urllib.request,json,hmac,threading,time,re,subprocess
from .io import put


def send(url,token,packet,proxy=None):
    if not url or not token:return False
    request=urllib.request.Request(url,data=json.dumps(packet).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}) if proxy else urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=10) as response:return response.status==200


def listen(root,token,host='127.0.0.1',port=8766,agent_id=None,agent_command=None):
    if not token:raise ValueError('Notification token required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):return self.reply(401,{'error':'unauthorized'})
            if self.path!='/notify':return self.reply(404,{'error':'unknown endpoint'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768:raise ValueError('Invalid body length')
                value=json.loads(self.rfile.read(length))
                if not isinstance(value,dict) or not re.fullmatch('[A-Za-z0-9_-]{1,120}',value.get('worker_id','')):raise ValueError('Invalid worker ID')
                if value.get('event') not in ['allocation_acquired','ready','job_started','job_finished','runner_finished']:raise ValueError('Unknown event')
                value['received_unix']=time.time()
                with lock:
                    put(root/'latest.json',value)
                    with (root/'inbox.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
                    put(root/(value['worker_id']+'.json'),value)
                if agent_id and agent_command:
                    # Only a fixed message plus constrained worker ID reaches the agent.
                    message='Kaggle worker '+value['worker_id']+' reported '+value['event']+'. Read the local queue inbox for structured details.'
                    subprocess.Popen([*agent_command,'agent','send','--id',agent_id,message],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                return self.reply(200,{'ok':True})
            except (ValueError,TypeError,json.JSONDecodeError):return self.reply(400,{'error':'invalid notification'})
    server=ThreadingHTTPServer((host,port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start();return server
