"""Persistent, exit-driven reference evaluation chain; never trains a model."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_job_helpers',ROOT/'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--gpu',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    configuration=job.read(a.spec);state={'status':'running_reference_evaluations','cases':{},'gpu':a.gpu,'spec':job.identity(a.spec)}
    job.write(a.out/'status.json',state)
    try:
        with job.lock_file(f'/tmp/4dsr_soft_eval_{a.gpu}.lock'):
            with job.lock_file(f'/tmp/4dsr_soft_gpu_{a.gpu}.lock'):
                for name,case in configuration['cases'].items():
                    folder=a.out/name;folder.mkdir();evaluations={}
                    for split in case['splits']:
                        destination=folder/f'eval_{split}_6000'
                        command=[sys.executable,str(HERE/('postprocess.py' if case['role']=='B4_post' else 'evaluate.py')),
                                 '--manifest',case['manifest'],'--checkpoint',case['checkpoint'],'--out',str(destination),
                                 '--split',split,'--method',case['role'],'--original-prior-cameras',case['original_prior_cameras'],
                                 '--roi-protocol',case['roi_protocol']]
                        before=job.gpu_snapshot(a.gpu);job.require_available(before)
                        event={'status':'running','command':command,'gpu_before':before,'started_utc':job.stamp()}
                        job.write(folder/f'{split}_event.json',event)
                        import os
                        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
                        tick=time.monotonic()
                        with (folder/f'{split}.log').open('w') as log:
                            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        event.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,seconds=time.monotonic()-tick)
                        job.write(folder/f'{split}_event.json',event)
                        if result.returncode: raise RuntimeError(f'{name}/{split} failed; no training launched')
                        receipt=job.read(destination/'complete.json')
                        assert receipt['status']=='completed_evaluation' and receipt['parameter_updates']==0
                        assert receipt['checkpoint_sha256']==job.sha(case['checkpoint'])
                        assert receipt['metrics_sha256']==job.sha(destination/'metrics.json')
                        evaluations[f'{split}_6000']=str(destination.resolve())
                    entry=dict(role=case['role'],label=case['role'],checkpoint=case['checkpoint'],train_dir=case['train_dir'],evaluations=evaluations)
                    job.write(folder/'methods-entry.json',{'methods':{case['role']:entry}})
                    job.write(folder/'complete.json',{'status':'completed_reference_evaluations','parameter_updates':0})
                    state['cases'][name]={'out':str(folder.resolve()),'methods_entry':str((folder/'methods-entry.json').resolve())}
                    job.write(a.out/'status.json',state)
        state.update(status='completed_reference_evaluations',seconds=time.monotonic()-started,finished_utc=job.stamp(),parameter_updates=0)
        job.write(a.out/'complete.json',state);job.write(a.out/'status.json',state)
    except BaseException:
        state.update(status='failed_reference_evaluations',traceback=traceback.format_exc(),parameter_updates=0)
        job.write(a.out/'failed.json',state);job.write(a.out/'status.json',state)
        raise


if __name__=='__main__':main()
