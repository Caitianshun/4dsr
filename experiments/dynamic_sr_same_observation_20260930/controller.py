"""Persistent remote training; process-return-triggered retrieval and local evaluation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_same_observation_20260930'
REMOTE = '/home/ubuntu/3DGS/4dsr'
UNIT = '4dsr-p1-from-start-20260930'
PYTHON = '/home/cai_tianshun/Project/4dgs/.venv/bin/python'


def state(status, **kwargs):
    item = dict(status=status, updated_unix=time.time(), pid=os.getpid(), **kwargs)
    tmp = OUT / 'controller_state.tmp'
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2))
    tmp.replace(OUT / 'controller_state.json')
    print(json.dumps(item, ensure_ascii=False), flush=True)


def call(command, log, env=None):
    with (OUT/log).open('w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{command[0]} exited {result.returncode}; see {OUT/log}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Wait for the existing training process; never launch training again')
    parser.add_argument('--host', choices=['a100-train','a100-via-5090'], default='a100-train')
    args = parser.parse_args()
    host = args.host
    remote_out = REMOTE + '/output/dynamic_sr_same_observation_20260930'
    remote_train = REMOTE + '/experiments/dynamic_sr_same_observation_20260930/train.py'
    command = ('source ' + shlex.quote(REMOTE+'/activate_a100.sh') +
               ' && export CUDA_VISIBLE_DEVICES=1 && exec python ' + shlex.quote(remote_train) +
               ' --protocol ' + shlex.quote(remote_out+'/protocol.json') +
               ' --out ' + shlex.quote(remote_out+'/train') +
               ' > ' + shlex.quote(remote_out+'/train_stdout.log') + ' 2>&1')
    launch = ['systemd-run','--user','--unit='+UNIT,'--wait',
              '--property=WorkingDirectory='+REMOTE,'bash','-lc',command]
    # Reconnecting waits on the existing PID's exit event, not on progress files.
    # The completed receipt also allows recovery after the transient unit expires.
    wait_code = '''import json,os,select,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); unit=sys.argv[2]
if not (root/'train/complete.json').exists():
    pid=int(subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID','--value'],text=True).strip())
    if pid:
        try: fd=os.pidfd_open(pid)
        except ProcessLookupError: fd=None
        if fd is not None:
            select.select([fd],[],[]); os.close(fd)
result=json.loads((root/'train/complete.json').read_text())
assert result['status']=='completed_training' and result['updates']==20200
'''
    wait_command = shlex.join(['python3','-c',wait_code,remote_out,UNIT])
    state('waiting_remote_training_exit',host=host,physical_gpu='1',unit=UNIT,resuming=args.resume)
    if args.resume:
        call(['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',host,wait_command],'remote_resume.log')
    else:
        try:
            call(['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',host,shlex.join(launch)],'remote_service.log')
        except RuntimeError:
            # Verify the existing result/process through the known fallback.
            # This code cannot restart or duplicate the registered training.
            host = 'a100-via-5090'
            state('recovering_remote_exit_wait',host=host,unit=UNIT)
            call(['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',host,wait_command],'remote_resume.log')
    state('retrieving_training_outputs')
    call(['rsync','-a',host+':'+remote_out+'/',str(OUT)+'/'],'retrieval.log')
    complete = json.loads((OUT/'train/complete.json').read_text())
    assert complete['status']=='completed_training' and complete['updates']==20200
    checkpoint = OUT/'train'/complete['final_checkpoint']
    import hashlib
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==complete['final_sha256']
    # Check local evaluation resource again immediately before creating its context.
    uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert not any(uuid in line for line in jobs.splitlines()), ('Evaluation GPU busy',jobs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for split in ['test','dev','train_fixed']:
        state('unified_local_evaluation',split=split,physical_gpu='1')
        command = [PYTHON,str(ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'),
                   '--manifest',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'),
                   '--checkpoint',str(checkpoint),'--out',str(OUT/'eval'/split),'--split',split,
                   '--method','P1-from-start',
                   '--teacher-index',str(ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'),
                   '--roi-protocol',str(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json'),
                   '--cache-dir',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/evaluation_cache'),
                   '--lpips-device','cuda','--no-video']
        call(command,'evaluation_'+split+'.log',env)
    state('completed_evaluation',training=complete,
          evaluation_files={split:str(OUT/'eval'/split/'metrics.json') for split in ['test','dev','train_fixed']})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        state('failed',traceback=traceback.format_exc())
        raise
