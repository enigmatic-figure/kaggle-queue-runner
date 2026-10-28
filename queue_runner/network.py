"""Optional userspace Tailscale connection; authentication never enters outputs."""
from pathlib import Path
import hashlib,tarfile,shutil,os,subprocess,time,json,urllib.request,platform
from .io import sha256,Redactor

VERSION='1.104.4'
AMD64_SHA256='624ea093d47d62cf56f17b8332d74b8decda4a1faa610ebc60a1de334b2b3dc9'


def extract_binaries(archive,digest,destination):
    if sha256(archive)!=digest:raise ValueError('Tailscale archive hash mismatch')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as data:
        for name in ['tailscale','tailscaled']:
            items=[m for m in data.getmembers() if Path(m.name).name==name and m.isfile()]
            if len(items)!=1:raise ValueError('Expected one regular '+name+' binary')
            with data.extractfile(items[0]) as source,(destination/name).open('wb') as target:shutil.copyfileobj(source,target)
            (destination/name).chmod(0o700)


class Tailnet:
    def __init__(self,private_root):
        self.root=Path(private_root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.daemon=None;self.socket=self.root/'tailscaled.sock';self.control_url=None
        self.proxy='http://127.0.0.1:1055';self.binary=self.root/'bin/tailscale'

    def call(self,*args,timeout=45):
        return subprocess.check_output([str(self.binary),'--socket='+str(self.socket),*args],text=True,stderr=subprocess.PIPE,timeout=timeout)

    def start(self,auth_key,hostname,port=8765):
        if platform.system()!='Linux' or platform.machine() not in ['x86_64','AMD64']:
            raise RuntimeError('Bundled userspace worker targets Linux amd64; local controller uses installed Tailscale')
        archive=self.root/'tailscale.tgz'
        if not self.binary.exists():
            urllib.request.urlretrieve(f'https://pkgs.tailscale.com/stable/tailscale_{VERSION}_amd64.tgz',archive)
            extract_binaries(archive,AMD64_SHA256,self.binary.parent);archive.unlink()
        with (self.root/'daemon.log').open('ab') as log:
            self.daemon=subprocess.Popen([str(self.root/'bin/tailscaled'),'--tun=userspace-networking',
                '--state=mem:','--socket='+str(self.socket),'--socks5-server=127.0.0.1:1055',
                '--outbound-http-proxy-listen=127.0.0.1:1055'],stdout=log,stderr=log)
        keyfile=self.root/'auth.key';keyfile.write_text(auth_key);keyfile.chmod(0o600)
        try:
            for _ in range(100):
                if self.socket.exists():break
                if self.daemon.poll() is not None:raise RuntimeError('Private Tailscale daemon exited')
                time.sleep(.1)
            self.call('up','--auth-key=file:'+str(keyfile),'--hostname='+hostname,
                      '--accept-dns=false','--accept-routes=false','--timeout=30s')
        finally:keyfile.unlink(missing_ok=True)
        self.call('serve','--bg','--tcp='+str(port),'tcp://127.0.0.1:'+str(port))
        status=json.loads(self.call('status','--json'))
        addresses=status.get('TailscaleIPs',[])
        ipv4=next((ip for ip in addresses if ':' not in ip),None)
        if not ipv4:raise RuntimeError('No worker tailnet IPv4 address')
        self.control_url='http://'+ipv4+':'+str(port)
        return {'connected':True,'version':VERSION,'control_url':self.control_url,'transport':'private WireGuard TCP forwarding'}

    def close(self):
        if self.daemon is not None:
            self.daemon.terminate()
            try:self.daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:self.daemon.kill();self.daemon.wait()
            self.daemon=None
