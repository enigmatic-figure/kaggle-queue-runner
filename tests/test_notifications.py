import json,urllib.request,urllib.error
from queue_runner import notifications,network


def test_notification_is_authenticated_and_retained_in_private_inbox(tmp_path):
    assert hasattr(notifications,'listen'), 'Missing readiness notification receiver'
    server=notifications.listen(tmp_path,'notify-token',host='127.0.0.1',port=0)
    url='http://127.0.0.1:'+str(server.server_port)+'/notify'
    try:
        packet={'worker_id':'worker-1','event':'allocation_acquired','control_url':'http://100.64.0.1:8765'}
        req=urllib.request.Request(url,data=json.dumps(packet).encode(),method='POST')
        try:urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:assert e.code==401
        else:assert False,'Unauthenticated notification accepted'
        assert notifications.send(url,'notify-token',packet)
        saved=json.loads((tmp_path/'latest.json').read_text())
        assert saved['event']=='allocation_acquired' and saved['worker_id']=='worker-1'
    finally:server.shutdown();server.server_close()


def test_binary_archive_extraction_checks_hash_and_only_extracts_expected_regular_files(tmp_path):
    import tarfile,io,hashlib
    assert hasattr(network,'extract_binaries'), 'Missing verified binary installer'
    p=tmp_path/'package.tgz'
    with tarfile.open(p,'w:gz') as archive:
        for name in ['tailscale_1/tailscale','tailscale_1/tailscaled']:
            data=b'test';info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    network.extract_binaries(p,digest,tmp_path/'bin')
    assert (tmp_path/'bin/tailscale').read_bytes()==b'test'
    try:network.extract_binaries(p,'0'*64,tmp_path/'bad')
    except ValueError:pass
    else:assert False,'Incorrect archive digest accepted'
