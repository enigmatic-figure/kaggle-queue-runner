"""Authenticated private worker controls."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
import hmac,json,re,threading,time
from .io import put


def start(runner,token,host='127.0.0.1',port=8765):
    if not token:raise ValueError('Control token required')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            body=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def authorized(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.reply(401,{'error':'unauthorized'});return False
            return True
        def do_GET(self):
            if not self.authorized():return
            url=urlsplit(self.path)
            if url.path=='/status':return self.reply(200,runner.status())
            if url.path=='/events':path=runner.root/'events.jsonl'
            elif url.path=='/logs':
                identity=parse_qs(url.query).get('job_id',[''])[0]
                if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,95}',identity):return self.reply(400,{'error':'invalid job ID'})
                path=runner.root/'jobs'/identity/'log.txt'
            else:return self.reply(404,{'error':'unknown endpoint'})
            if not path.exists():return self.reply(200,{'tail':''})
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-65536));tail=stream.read().decode(errors='replace')
            self.reply(200,{'tail':tail})
        def do_POST(self):
            if not self.authorized():return
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.reply(400,{'error':'invalid length'})
            if not 0<=length<=4096:return self.reply(413,{'error':'invalid body size'})
            if length:self.rfile.read(length)
            action=urlsplit(self.path).path.strip('/')
            if action=='pause':runner.pause.set()
            elif action=='resume':runner.pause.clear();runner.reload.set()
            elif action=='reload':runner.reload.set()
            elif action=='cancel':runner.cancel.set()
            elif action=='stop':runner.stop.set()
            elif action=='checkpoint':
                current=runner.status()['current']
                if not current:return self.reply(409,{'error':'no running workload'})
                put(runner.root/'jobs'/current['job_id']/'checkpoint-request.json',{'requested_unix':time.time()})
            else:return self.reply(404,{'error':'unknown endpoint'})
            runner.event('control',action=action);self.reply(200,{'ok':True,'action':action})
    server=ThreadingHTTPServer((host,port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start();return server
