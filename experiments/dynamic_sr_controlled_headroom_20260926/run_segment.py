"""Persistent, GPU-locked segment: exact training exit -> fixed evaluations."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('leases',ROOT/'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
leases=importlib.util.module_from_spec(spec);spec.loader.exec_module(leases)


def evaluate(s,out,checkpoint,step,env):
    for split in ['train_fixed','dev','test']:
        dest=out/f'eval_{split}_{step}'
        cmd=[sys.executable,str(HERE/'evaluate.py'),'--manifest',str(ROOT/s['manifest']),'--checkpoint',str(checkpoint),
            '--teacher-index',str(ROOT/s['teacher_index']),'--out',str(dest),'--split',split,'--method',s['method']+('_HR_PRIVILEGED' if s['method']=='O' else ''),
            '--original-prior-cameras','cam02,cam06,cam12,cam18','--roi-protocol',str(ROOT/s['roi'])]
        leases.require_available(leases.gpu_snapshot(s['gpu']))
        with (out/f'eval_{split}_{step}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert leases.read(dest/'complete.json')['status']=='completed_evaluation'


def train(s,out,env):
    cmd=[sys.executable,str(HERE/'train.py'),'--method',s['method'],'--stop',str(s['stop']),'--out',str(out/'train')]
    for key in ['manifest','resume','schedule','start-identity','lr-curve']:
        cmd+=['--'+key,str(ROOT/s[key])]
    if s.get('target-index'):cmd+=['--target-index',str(ROOT/s['target-index'])]
    snapshot=leases.gpu_snapshot(s['gpu']);leases.require_available(snapshot);leases.write(out/'launch.json',dict(command=cmd,snapshot=snapshot))
    with (out/'train.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=leases.read(out/'train/complete.json');assert receipt['total_updates']==s['stop'] and not receipt['smoke']


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();s=leases.read(a.spec);out=ROOT/s['out'];out.mkdir(parents=True,exist_ok=False)
    status=dict(status='waiting_gpu_lock',pid=os.getpid(),method=s['method'],gpu=s['gpu'],started_utc=leases.stamp())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=s['gpu'],OMP_NUM_THREADS='4');leases.write(out/'status.json',status)
    try:
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{s["gpu"]}.lock'):
            status['status']='training';leases.write(out/'status.json',status);train(s,out,env)
            steps=[6000] if s['stop']==6000 else [18000,40000]
            leases.write(out/'checkpoint_index.json',{str(k):leases.identity(out/f'train/checkpoint_{k}.pt') for k in steps})
            status['status']='evaluating';leases.write(out/'status.json',status)
            for step in steps:evaluate(s,out,out/f'train/checkpoint_{step}.pt',step,env)
        status.update(status='completed_and_evaluated',finished_utc=leases.stamp());leases.write(out/'complete.json',status)
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());leases.write(out/'failed.json',status);raise
    finally:leases.write(out/'status.json',status)


if __name__=='__main__':main()
