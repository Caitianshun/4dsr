"""Persistent sequential local queue. Exit of each train triggers evaluation.

No polling discovers training completion: subprocess.run returns on process exit.
Only one GPU is used, with ownership rechecked before each child process.
"""
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260923/meeting_controls_v1'
SCRIPT=ROOT/'experiments/dynamic_sr_20260923/train_sharing_control.py'
PY='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
GPU='GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def check_gpu():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    for row in csv.reader(output.splitlines()):
        if row[0].strip()==GPU and row[2].strip()!='/opt/todesk/bin/ToDesk_Session':
            raise RuntimeError(f'GPU0 occupied by existing compute process: {row}')
    return output


def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    calibration=ROOT/'output/dynamic_sr_20260923/meeting_calibration_v2/calibration.json'
    smoke=ROOT/'output/dynamic_sr_20260923/meeting_smoke_v1/complete.json'
    if not smoke.exists() or json.loads(smoke.read_text())['parameter_updates']!=20:raise RuntimeError('smoke not completed')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),gpu_uuid=GPU,
        planned_branches=list('ABCD'),steps=6000,milestones=[1200,3000,6000],events=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(OUT/'status.json',state)
    def run(label,cmd):
        gpu_state=check_gpu();event=dict(label=label,command=cmd,start_utc=datetime.now(timezone.utc).isoformat(),gpu_before=gpu_state)
        state['current']=label;state['events'].append(event);write(OUT/'status.json',state)
        tick=time.monotonic()
        with (OUT/f'{label}.log').open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        event.update(returncode=rc,elapsed_seconds=time.monotonic()-tick);write(OUT/'status.json',state)
        if rc:raise RuntimeError(f'{label} failed: {rc}')
    try:
        for branch in 'ABCD':
            folder=OUT/branch
            cmd=[PY,str(SCRIPT),'train','--branch',branch,'--out',str(folder),
                '--manifest',str(ROOT/'data/dynamic_sr/meetroom_prepared/discussion/manifest.json'),
                '--checkpoint',str(ROOT/'output/dynamic_sr_20260919/meetroom_discussion_integrated_parent/checkpoint_final.pt'),
                '--cache-manifest',str(ROOT/'output/dynamic_sr_20260923/meeting_training_targets_v1/manifest.json'),
                '--calibration',str(calibration),'--steps','6000']
            run(f'{branch}_train',cmd)
            for step in [1200,3000,6000]:
                run(f'{branch}_eval_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),
                    '--checkpoint',str(folder/f'checkpoint_{step}.pt'),'--manifest',str(ROOT/'data/dynamic_sr/meetroom_prepared/discussion/manifest.json'),
                    '--out',str(folder/f'eval_{step}'),'--no-video'])
                run(f'{branch}_fit_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260923/evaluate_training_fit.py'),
                    '--run',str(folder),'--step',str(step)])
        # Integrity checks are chained to final process exit, not a future timer.
        draws=[];rows=[]
        for branch in 'ABCD':
            folder=OUT/branch;done=json.loads((folder/'complete.json').read_text());draws.append(done['draw_sha256'])
            for step in [1200,3000,6000]:
                m=json.loads((folder/f'eval_{step}/metrics.json').read_text())
                assert len(m['rows'])==60 and m['frame_indices']==list(range(0,120,2))
                assert len(list((folder/f'eval_{step}/predictions').glob('*.png')))==60
                assert (folder/f'own_target_fit_{step}.json').exists()
                rows.append(dict(branch=branch,step=step,aggregate=m['aggregate'],temporal=m['temporal_aggregate'],
                                 checkpoint_sha256=m['checkpoint_sha256'],evaluation_seconds=m['elapsed_seconds']))
        assert len(set(draws))==1
        write(OUT/'summary.json',dict(rows=rows,all_draws_equal=True,source='completed fixed checkpoints, no best-test selection'))
        state.update(status='completed_and_evaluated',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        write(OUT/'status.json',state);write(OUT/'complete.json',state)
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write(OUT/'status.json',state);write(OUT/'failed.json',state);raise


if __name__=='__main__':main()
