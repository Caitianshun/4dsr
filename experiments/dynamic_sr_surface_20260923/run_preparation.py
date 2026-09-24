"""Parallel local LR parent and frozen SR prior, with process-exit receipts."""
import csv, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_surface_20260923/preparation_v1'
DATA=ROOT/'data/dynamic_sr/zju_prepared/coreview387_pilot_v1'
GPUS=['GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5','GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec']


def write(path,obj):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(path)


def launch(label,gpu,cmd):
    snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    for row in csv.reader(snapshot.splitlines()):
        if row[0].strip()==gpu and row[2].strip()!='/opt/todesk/bin/ToDesk_Session':raise RuntimeError(f'Occupied GPU: {row}')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    state=dict(label=label,status='running',command=cmd,gpu=gpu,gpu_before=snapshot,started_utc=datetime.now(timezone.utc).isoformat())
    tick=time.monotonic()
    with (OUT/f'{label}.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=proc.pid;write(OUT/f'{label}.json',state);rc=proc.wait()
    state.update(status='completed' if rc==0 else 'failed',returncode=rc,elapsed_s=time.monotonic()-tick)
    write(OUT/f'{label}.json',state)
    if rc:raise RuntimeError(f'{label} exited {rc}')


def main():
    manifest=DATA/'manifest_hull.json';assert manifest.exists()
    OUT.mkdir(parents=True,exist_ok=False)
    script=ROOT/'experiments/dynamic_sr_surface_20260923'
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            p=pool.submit(launch,'parent',GPUS[0],[sys.executable,str(script/'train_parent.py'),'--manifest',str(manifest),
                 '--out',str(OUT/'parent'),'--coarse-steps','1000','--fine-steps','6000'])
            s=pool.submit(launch,'prior',GPUS[1],[sys.executable,str(script/'generate_prior.py'),'--manifest',str(manifest),
                 '--cameras','cam00,cam06,cam12,cam18','--tile','0'])
            p.result();s.result()
        write(OUT/'complete.json',dict(status='completed',parent=str(OUT/'parent/checkpoint_final.pt'),manifest=str(manifest),
            finished_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException:
        write(OUT/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
