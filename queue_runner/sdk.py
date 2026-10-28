"""Small workload-side checkpoint contract."""
from pathlib import Path
import os,json,time
from .io import put,sha256


def output_dir():
    path=Path(os.environ['QUEUE_OUTPUT_DIR']);path.mkdir(parents=True,exist_ok=True);return path


def progress(**values):
    put(output_dir()/'progress.json',{'unix_time':time.time(),**values})


def checkpoint_requested():
    name=os.environ.get('QUEUE_CHECKPOINT_REQUEST')
    if not name:return False
    p=Path(name)
    if not p.exists():return False
    p.unlink();return True


def checkpoint(folder):
    """Seal a closed, immutable directory below the job output directory."""
    root=output_dir().resolve();path=(root/folder).resolve()
    if not path.is_relative_to(root) or not path.is_dir():raise ValueError('Checkpoint must be an existing job-output directory')
    files={}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():raise ValueError('Checkpoint symlinks are not supported')
        if p.is_file() and p.name!='checkpoint.ready.json' and not p.name.endswith('.partial'):
            files[p.relative_to(path).as_posix()]=sha256(p)
    marker=path/'checkpoint.ready.json'
    if marker.exists():
        if json.loads(marker.read_text())['files']!=files:raise ValueError('Sealed checkpoint changed; use a new directory')
        return marker
    put(marker,{'version':1,'job_id':os.environ.get('QUEUE_JOB_ID'),
                'source_commit':os.environ.get('QUEUE_SOURCE_COMMIT'),'files':files,'sealed_unix':time.time()})
    return marker
