"""Optional manifest-last publication of closed workload checkpoints."""
from pathlib import Path
import json,time,tempfile,re
from .io import put,sha256,Redactor


class Sink:
    def __init__(self,root,bucket,prefix,*,client=None,download=None,secrets=()):
        self.root=Path(root).resolve();self.bucket=bucket
        if bucket=='IntellAgents/embeddibert-wiki727-cache-20260909':raise ValueError('Source bucket is read-only')
        if not prefix or '..' in prefix.split('/'):raise ValueError('Invalid artifact prefix')
        self.prefix=prefix.strip('/')+'/'+self.root.name
        if client is None:
            from huggingface_hub import HfApi
            client=HfApi()
        if download is None:
            from huggingface_hub import download_bucket_files
            download=download_bucket_files
        self.client=client;self.download=download;self.redactor=Redactor(secrets)
        self.receipt=self.root/'published.json'
        self.done=json.loads(self.receipt.read_text()) if self.receipt.exists() else {}

    def publish_checkpoint(self,marker):
        marker=Path(marker).resolve()
        if not marker.is_relative_to(self.root/'jobs'):raise ValueError('Checkpoint outside job tree')
        relative=marker.relative_to(self.root).as_posix();digest=sha256(marker)
        if self.done.get(relative,{}).get('sha256')==digest:return
        manifest=json.loads(marker.read_text());files=[]
        for name,expected in manifest['files'].items():
            p=(marker.parent/name).resolve()
            if not p.is_relative_to(marker.parent) or p.is_symlink() or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Invalid checkpoint file binding')
            if sha256(p)!=expected:raise ValueError('Checkpoint payload hash changed')
            files.append((p,self.prefix+'/'+p.relative_to(self.root).as_posix()))
        if files:self.client.batch_bucket_files(self.bucket,add=files)
        with tempfile.TemporaryDirectory(prefix='queue-readback-') as scratch:
            target=Path(scratch)/'file'
            for path,remote in files:
                self.download(self.bucket,[(remote,target)],raise_on_missing_files=True)
                if sha256(target)!=manifest['files'][path.relative_to(marker.parent).as_posix()]:raise ValueError('Checkpoint remote readback hash differs')
                target.unlink()
            remote=self.prefix+'/'+relative
            self.client.batch_bucket_files(self.bucket,add=[(marker,remote)])
            self.download(self.bucket,[(remote,target)],raise_on_missing_files=True)
            if sha256(target)!=digest:raise ValueError('Checkpoint marker readback hash differs')
        self.done[relative]={'sha256':digest,'verified_unix':time.time()};put(self.receipt,self.done)

    def sync(self):
        for marker in sorted((self.root/'jobs').rglob('checkpoint.ready.json')):self.publish_checkpoint(marker)
        with tempfile.TemporaryDirectory(prefix='queue-telemetry-') as scratch:
            files=[]
            candidates=[self.root/n for n in ['status.json','events.jsonl','final.json','published.json']]
            candidates+=list((self.root/'jobs').glob('*/receipt.json'))+list((self.root/'jobs').glob('*/log.txt'))
            for p in candidates:
                if not p.exists():continue
                with p.open('rb') as stream:
                    if p.suffix in ['.txt','.jsonl']:stream.seek(max(0,p.stat().st_size-65536))
                    content=stream.read().decode(errors='replace')
                relative=p.relative_to(self.root);target=Path(scratch)/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(self.redactor.clean(content),encoding='utf-8')
                files.append((target,self.prefix+'/telemetry/'+relative.as_posix()))
            if files:self.client.batch_bucket_files(self.bucket,add=files)
