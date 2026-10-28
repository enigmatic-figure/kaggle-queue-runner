import json,threading,urllib.request,urllib.error
from queue_runner.runner import Runner
from queue_runner import control,sdk


def test_private_api_requires_token_and_only_accepts_defined_controls(tmp_path):
    assert hasattr(control,'start'), 'Missing authenticated status/control endpoint'
    r=Runner(tmp_path);server=control.start(r,'secret-token',host='127.0.0.1',port=0)
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        try:urllib.request.urlopen(base+'/status')
        except urllib.error.HTTPError as e:assert e.code==401
        else:assert False,'Unauthenticated status request accepted'
        req=urllib.request.Request(base+'/reload',data=b'{}',headers={'Authorization':'Bearer secret-token'},method='POST')
        assert json.load(urllib.request.urlopen(req))['ok'] and r.reload.is_set()
        req=urllib.request.Request(base+'/shell',data=b'{}',headers={'Authorization':'Bearer secret-token'},method='POST')
        try:urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:assert e.code==404
        else:assert False,'Undefined execution endpoint accepted'
    finally:server.shutdown();server.server_close()


def test_checkpoint_marks_only_closed_files_with_hashes_and_rejects_changes(tmp_path,monkeypatch):
    assert hasattr(sdk,'checkpoint'), 'Missing workload checkpoint sealing'
    monkeypatch.setenv('QUEUE_OUTPUT_DIR',str(tmp_path))
    p=tmp_path/'epoch1';p.mkdir();(p/'weights.bin').write_bytes(b'weights')
    marker=sdk.checkpoint('epoch1');value=json.loads(marker.read_text())
    assert value['files']['weights.bin'] and marker.name=='checkpoint.ready.json'
    (p/'weights.bin').write_bytes(b'changed')
    try:sdk.checkpoint('epoch1')
    except ValueError:pass
    else:assert False,'Changed sealed checkpoint accepted'
    try:sdk.checkpoint('../elsewhere')
    except ValueError:pass
    else:assert False,'Checkpoint outside job output accepted'
