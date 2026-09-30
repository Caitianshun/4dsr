"""Remote launch with immutable task ownership; never interrupt another service."""
from dv_common import *
import subprocess
import shlex
import argparse
from readiness import validate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);a=ap.parse_args()
    p=require_run_root(OUT);validate(p);task=next(t for t in p['task_plan'] if t['task_id']==a.task)
    gpu=task['physical_gpu'];base=OUT/'runs'/a.task/'attempt_01';unit='4dsr-prior-diagnosis-'+a.task.lower().replace('_','-')
    if (base/'dispatch.json').exists():
        old=read(base/'dispatch.json');assert old['protocol_sha256']==sha(OUT/'protocol.json')
        state=subprocess.check_output(['systemctl','--user','show',unit,'--property=LoadState,ActiveState'],text=True)
        if 'ActiveState=active\n' in state or (base/'train/complete.json').exists() or (base/'train/failed.json').exists():
            print(json.dumps(old));return
        if (base/'train').exists() or old.get('dispatch_state')=='started':
            raise RuntimeError('Prior service exited without a completion receipt; inspect the existing attempt, never restart blindly')
        assert 'LoadState=not-found\n' in state,('Prepared dispatch has an existing unit; inspect before relaunch',state)
    occ=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    blocked=[line for line in occ.splitlines() if gpu in line and 'gnome-remote-desktop-daemon' not in line]
    assert not blocked,('GPU has another compute job; no launch',blocked)
    base.mkdir(parents=True,exist_ok=True)
    argv=['python','-u',str(HERE/'train.py'),'--method',task['arm'],'--repeat',task['repeat'],'--out',str(base/'train')]
    shell='cd '+shlex.quote(str(ROOT))+' && source activate_a100.sh && export CUDA_VISIBLE_DEVICES='+shlex.quote(gpu)+' && exec '+shlex.join(argv)+' > '+shlex.quote(str(base/'train.log'))+' 2>&1'
    record=dict(unit=unit,physical_gpu=gpu,task=a.task,protocol_sha256=sha(OUT/'protocol.json'),prelaunch_occupancy=occ,started_unix=time.time(),argv=argv,dispatch_state='prepared')
    write(base/'dispatch.json',record)
    try:
        subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',shell],check=True)
    except BaseException as e:
        record.update(dispatch_state='launch_failed',error=repr(e));write(base/'dispatch.json',record);raise
    record.update(dispatch_state='started');write(base/'dispatch.json',record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
