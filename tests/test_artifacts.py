from pathlib import Path
import json,shutil
import pytest
from queue_runner import artifacts,sdk


class FakeHub:
    def __init__(self,root):self.root=root;self.calls=[]
    def batch_bucket_files(self,bucket,add):
        self.calls.append([name for _,name in add])
        for path,name in add:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)


def test_checkpoint_publication_verifies_payload_before_publishing_marker_and_skips_repeats(tmp_path,monkeypatch):
    assert hasattr(artifacts,'Sink'), 'Missing durable checkpoint publication'
    root=tmp_path/'run';out=root/'jobs/one/output';p=out/'epoch1';p.mkdir(parents=True)
    monkeypatch.setenv('QUEUE_OUTPUT_DIR',str(out));(p/'weights.bin').write_bytes(b'payload')
    marker=sdk.checkpoint('epoch1');hub=FakeHub(tmp_path/'remote')
    def download(bucket,pairs,**kwargs):
        for name,destination in pairs:shutil.copyfile(hub.root/name,destination)
    sink=artifacts.Sink(root,'owner/results','runs',client=hub,download=download)
    sink.publish_checkpoint(marker)
    assert len(hub.calls)==2 and hub.calls[-1][0].endswith('/checkpoint.ready.json')
    sink.publish_checkpoint(marker);assert len(hub.calls)==2
    (p/'weights.bin').write_bytes(b'changed')
    other=sdk.output_dir()/'epoch2';other.mkdir();(other/'weights.bin').write_bytes(b'old');bad=sdk.checkpoint('epoch2')
    (other/'weights.bin').write_bytes(b'new')
    with pytest.raises(ValueError,match='hash'):sink.publish_checkpoint(bad)
