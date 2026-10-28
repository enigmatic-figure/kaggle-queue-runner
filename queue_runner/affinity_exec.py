"""Set Linux affinity before importing accelerator libraries or execing a child."""
import os,sys


def main():
    cpus=os.environ.get('QUEUE_CPUSET')
    if cpus:
        if not hasattr(os,'sched_setaffinity'):raise RuntimeError('Explicit CPU isolation requires Linux')
        wanted={int(v) for v in cpus.split(',')}
        if not wanted<=os.sched_getaffinity(0):raise ValueError('Requested CPUs exceed container affinity')
        os.sched_setaffinity(0,wanted)
    command=sys.argv[1:]
    if not command:raise ValueError('Child command required')
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
