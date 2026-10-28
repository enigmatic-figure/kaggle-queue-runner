"""Git source snapshots."""
from pathlib import Path
import subprocess,json,re
from .contracts import validate


class GitSource:
    def __init__(self,url,branch,cache):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]*',branch) or '..' in branch:raise ValueError('Invalid source branch')
        self.url=url;self.branch=branch;self.cache=Path(cache).resolve()
        if not self.cache.exists():
            self.cache.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--depth','1','--no-checkout','--single-branch','--branch',branch,url,str(self.cache)],check=True,capture_output=True)

    def _git(self,*args):
        return subprocess.check_output(['git','-C',str(self.cache),*args],text=True,stderr=subprocess.PIPE).strip()

    def refresh(self):
        self._git('fetch','--quiet','--depth','1','origin',self.branch)
        commit=self._git('rev-parse','FETCH_HEAD')
        manifest=validate(json.loads(self._git('show',commit+':workloads.json')))
        return commit,manifest

    def snapshot(self,commit,destination):
        if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full commit SHA required')
        destination=Path(destination).resolve()
        if destination.exists():raise ValueError('Snapshot destination already exists')
        destination.parent.mkdir(parents=True,exist_ok=True)
        self._git('worktree','add','--quiet','--detach',str(destination),commit)
        return destination
