"""Linux smoke check without enrolling a node or changing the host tailnet."""
from pathlib import Path
import tempfile,urllib.request,subprocess,time,json
from queue_runner.network import extract_binaries,VERSION,AMD64_SHA256

with tempfile.TemporaryDirectory(prefix='queue-userspace-check-') as temporary:
    root=Path(temporary);archive=root/'tailscale.tgz'
    urllib.request.urlretrieve(f'https://pkgs.tailscale.com/stable/tailscale_{VERSION}_amd64.tgz',archive)
    extract_binaries(archive,AMD64_SHA256,root/'bin')
    binary=root/'bin/tailscale';socket=root/'daemon.sock'
    with (root/'daemon.log').open('wb') as log:
        process=subprocess.Popen([str(root/'bin/tailscaled'),'--tun=userspace-networking',
            '--state=mem:','--socket='+str(socket)],stdout=log,stderr=log)
    try:
        for _ in range(100):
            if socket.exists():break
            if process.poll() is not None:raise RuntimeError('Userspace daemon exited')
            time.sleep(.1)
        status=json.loads(subprocess.check_output([str(binary),'--socket='+str(socket),'status','--json'],text=True))
        assert status['BackendState'] in ['NeedsLogin','NoState','Stopped']
        result={'binary_version':subprocess.check_output([str(binary),'version'],text=True).splitlines()[0],
            'archive_sha256_verified':True,'userspace_daemon_started':True,'backend_state':status['BackendState'],
            'authenticated_enrollment_tested':False,'host_tailnet_modified':False}
        print(json.dumps(result))
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
