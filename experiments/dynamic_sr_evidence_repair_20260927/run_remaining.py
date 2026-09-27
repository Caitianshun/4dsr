"""Persistent exit-triggered fixed-time probes and one bounded replay batch."""
import argparse
import ctypes
import errno
import fcntl
import select
import subprocess
import time
import traceback
from context import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);a=p.parse_args()
    out=OUT/'remaining';out.mkdir(exist_ok=False);started=time.time();commands=[]
    def state(**kw):write_json(out/'status.json',dict(pid=os.getpid(),updated=time.time(),seconds=time.time()-started,**kw))
    def run(script,args,label):
        state(status='running',phase=label);cmd=[sys.executable,'-u',str(HERE/script),*args]
        row=dict(label=label,argv=cmd,started=time.time());commands.append(row);write_json(out/'commands.json',commands)
        with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=r.returncode,seconds=time.time()-row['started']);write_json(out/'commands.json',commands)
        if r.returncode:raise RuntimeError(label+' failed')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':read(OUT/'protocol.json')['physical_gpu'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    try:
        state(status='waiting_for_process_exit',phase='tails',after_pid=a.after_pid)
        try:
            libc=ctypes.CDLL(None,use_errno=True)
            fd=libc.pidfd_open(a.after_pid,0)
            if fd<0:
                error=ctypes.get_errno()
                if error==errno.ESRCH:raise ProcessLookupError()
                raise OSError(error,os.strerror(error))
            select.select([fd],[],[]);os.close(fd)
        except ProcessLookupError:pass
        assert read(OUT/'replay_tails/status.json')['status']=='completed'
        lock=(OUT/'pro6000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        for line in apps.splitlines()[1:]:
            if line.startswith(env['CUDA_VISIBLE_DEVICES']):assert '/opt/todesk/' in line,line
        write_json(out/'resources_start.json',dict(apps=apps,gpu=env['CUDA_VISIBLE_DEVICES']))
        for arm in ['Shared40','Baked40']:
            run('fit_fixed_time.py',['--arm',arm],arm+'_train')
            run('evaluate_probes.py',['--mode',arm],arm+'_evaluation')
        run('evaluate_probes.py',['--mode','references'],'references_evaluation')
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        run('run_replay.py',['--phase','long'],'bounded_long_replay')
        state(status='completed',phase='all_exits_evaluated')
    except BaseException:state(status='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
