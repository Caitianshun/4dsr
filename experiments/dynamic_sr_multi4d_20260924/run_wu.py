"""Persistent U40/W40 process chain: train, verify, evaluate on exit."""
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('leases', ROOT/'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
leases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True, type=Path)
    a = p.parse_args()
    s = leases.read(a.spec)
    out = ROOT/s['out']
    out.mkdir(parents=True, exist_ok=False)
    gpu = s['gpu']
    assert leases.GPU_PATTERN.fullmatch(gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4')
    status = dict(status='starting', host=socket.gethostname(), gpu=gpu, pid=os.getpid(), started_utc=leases.stamp(), spec=s)
    def run(label, cmd):
        snap = leases.gpu_snapshot(gpu)
        leases.require_available(snap)
        leases.write(out/(label+'_launch.json'), dict(command=cmd, snapshot=snap, utc=leases.stamp()))
        status.update(status=label)
        leases.write(out/'status.json', status)
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{gpu}.lock'):
            cmd = [sys.executable,str(HERE/'wu_train.py'),'--out',str(out/'train'),'--method',s['method']]
            for k in ['manifest','checkpoint','selection','teacher-index','schedule','prefix-schedule','resume']:
                cmd += ['--'+k, str(ROOT/s[k])]
            run('training',cmd)
            done=leases.read(out/'train/complete.json')
            assert done['total_sr_updates']==40000 and done['parameter_updates']==34000 and not done['smoke']
            checkpoints={str(i):leases.identity(out/f'train/checkpoint_{i}.pt') for i in [18000,40000]}
            leases.write(out/'checkpoint_index.json',checkpoints)
            for step in [18000,40000]:
                for split in ['train_fixed','dev','test']:
                    label=f'eval_{split}_{step}'
                    run(label,[sys.executable,str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'),
                        '--manifest',str(ROOT/s['manifest']),'--checkpoint',checkpoints[str(step)]['path'],
                        '--out',str(out/label),'--split',split,'--method',s['method']+'40',
                        '--original-prior-cameras','cam02,cam06,cam12,cam18','--roi-protocol',str(ROOT/s['roi'])])
                    e=leases.read(out/label/'complete.json')
                    assert e['status']=='completed_evaluation' and e['observations']==(16 if split=='train_fixed' else 60)
        status.update(status='completed_and_evaluated',finished_utc=leases.stamp())
        leases.write(out/'complete.json',status)
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc(),failed_utc=leases.stamp())
        leases.write(out/'failed.json',status)
        raise
    finally:
        leases.write(out/'status.json',status)


if __name__=='__main__':
    main()
