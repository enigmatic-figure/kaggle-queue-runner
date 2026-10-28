"""Minimal workload template. Write closed checkpoints below QUEUE_OUTPUT_DIR."""
from queue_runner.sdk import output_dir,progress,checkpoint
import json,time,platform,os

out=output_dir();progress(stage='running',fraction=0.)
folder=out/'complete';folder.mkdir()
(folder/'result.json').write_text(json.dumps({'python':platform.python_version(),
    'cpus':os.cpu_count(),'source_commit':os.environ['QUEUE_SOURCE_COMMIT'],'unix_time':time.time()},indent=2))
checkpoint('complete');progress(stage='complete',fraction=1.)
