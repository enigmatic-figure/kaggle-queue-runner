"""Run a repository task in the isolated CPU analytics environment."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
from queue_runner.io import put


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--script',required=True);p.add_argument('args',nargs=argparse.REMAINDER)
    args=p.parse_args();snapshot=Path.cwd().resolve();script=(snapshot/args.script).resolve()
    if not script.is_relative_to(snapshot) or script.suffix!='.py' or not script.is_file():raise ValueError('Select a Python task in the pinned worktree')
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('Declare resources.tpu_chips=[] for CPU tasks')
    requirements=snapshot/'environments/cpu.txt';digest=hashlib.sha256(requirements.read_bytes()).hexdigest()
    env=Path(tempfile.gettempdir())/('queue-cpu-'+digest[:16]);python=env/'bin/python';marker=env/'ready.json'
    if not marker.exists():
        subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
        subprocess.run([str(python),'-m','pip','install','--only-binary=:all:','-r',str(requirements)],check=True)
        subprocess.run([str(python),'-m','pip','check'],check=True)
        freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        put(marker,{'requirements_sha256':digest,'freeze':freeze})
    output=Path(os.environ['QUEUE_OUTPUT_DIR']);put(output/'cpu-environment.json',json.loads(marker.read_text()))
    # The wrapper has already applied affinity; child processes inherit it.
    extra=args.args[1:] if args.args[:1]==['--'] else args.args
    subprocess.run([str(python),'-u',str(script),*extra],check=True)


if __name__=='__main__':main()
