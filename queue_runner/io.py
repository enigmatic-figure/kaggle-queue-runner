from pathlib import Path
import json,os,re,hashlib


def put(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(q,p)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


class Redactor:
    def __init__(self,secrets=()):
        self.secrets=sorted(set(s for s in secrets if s),key=len,reverse=True)
        self.hold=max([256]+[len(s)+1 for s in self.secrets]);self.buffer=''

    def clean(self,text):
        for value in self.secrets:text=text.replace(value,'[REDACTED]')
        return re.sub(r'(?:hf_[A-Za-z0-9]{16,}|tskey-[A-Za-z0-9_-]{16,})','[REDACTED]',text)

    def feed(self,text,final=False):
        self.buffer+=text;self.buffer=self.clean(self.buffer)
        cut=len(self.buffer) if final else max(0,len(self.buffer)-self.hold)
        output=self.buffer[:cut];self.buffer=self.buffer[cut:];return output
