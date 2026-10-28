"""Provider-neutral inference calls with private, redacted training traces."""
from pathlib import Path
from urllib.parse import urlsplit
import json,threading,time,urllib.request,uuid
from .io import Redactor


class Session:
    def __init__(self,base_url,token,trace,*,metadata=None):
        parsed=urlsplit(base_url)
        if parsed.scheme not in ['http','https'] or parsed.username or parsed.password:raise ValueError('Use an HTTP(S) endpoint with credentials in headers')
        self.base_url=base_url.rstrip('/');self.token=token;self.trace=Path(trace)
        self.trace.parent.mkdir(parents=True,exist_ok=True);self.metadata=metadata or {};self.lock=threading.Lock()
        self.redactor=Redactor([token]);self.session_id=uuid.uuid4().hex
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _record(self,kind,**value):
        record={'schema':'queue-session-jsonl-v1','session_id':self.session_id,'kind':kind,
            'unix_time':time.time(),'metadata':self.metadata,**value}
        with self.lock:
            with self.trace.open('a',encoding='utf-8') as stream:
                stream.write(self.redactor.clean(json.dumps(record,ensure_ascii=True))+'\n');stream.flush()
            self.trace.chmod(0o600)

    def request(self,endpoint,payload,*,timeout=300):
        if endpoint.startswith('/') or '..' in endpoint or '://' in endpoint:raise ValueError('Use a relative API endpoint')
        request=urllib.request.Request(self.base_url+'/'+endpoint,data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        started=time.perf_counter()
        try:
            with self.opener.open(request,timeout=timeout) as response:value=json.load(response)
        except Exception as error:
            self._record('request',endpoint=endpoint,request=payload,error=str(error),elapsed_seconds=time.perf_counter()-started)
            raise
        self._record('request',endpoint=endpoint,request=payload,response=value,elapsed_seconds=time.perf_counter()-started)
        return value

    def record_tool(self,call_id,name,arguments,result):
        self._record('tool',call_id=call_id,name=name,arguments=arguments,result=result)
