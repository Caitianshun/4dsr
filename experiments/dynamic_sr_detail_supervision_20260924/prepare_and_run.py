"""Wait for an input producer's exit event, calibrate once, run fixed U/W/F."""
import argparse
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
from run_job import ROOT,HERE,soft,read,write,identity,stamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait-pid',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    plan=read(a.plan);common=plan['common'];state=dict(status='waiting_input_completion_event',plan=identity(a.plan),started_utc=stamp())
    write(a.out/'status.json',state)
    try:
        if a.wait_pid:
            # The bundled CUDA Python lacks pidfd_open; system Python has it.
            # One blocking kernel exit event, never a status/curve polling loop.
            waiter='import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\ntry:\n ready,_,_=select.select([fd],[],[],3600)\n if not ready: raise RuntimeError("Producer exceeded one-hour event wait")\nfinally: os.close(fd)\n'
            subprocess.run(['/usr/bin/python3','-c',waiter,str(a.wait_pid)],check=True)
        teacher=read(common['teacher_index'])
        assert teacher['status'].startswith('completed'), 'Full legal training teacher pool not ready'
        frozen=read(plan['core_freeze'])
        for row in frozen['files']:
            assert soft.sha(row['path'])==row['sha256'], f'Frozen core changed: {row["path"]}'
        calibration=a.out/'calibration'
        command=[sys.executable,str(HERE/'calibrate.py'),'--manifest',common['manifest'],'--checkpoint',common['parent'],
                 '--selection',common['selection'],'--teacher-index',common['teacher_index'],'--schedule',common['schedule'],'--out',str(calibration)]
        state.update(status='waiting_calibration_gpu',calibration_command=command);write(a.out/'status.json',state)
        with soft.lock_file(f'/tmp/4dsr_soft_gpu_{common["train_gpu"]}.lock'):
            snapshot=soft.gpu_snapshot(common['train_gpu']);soft.require_available(snapshot)
            state.update(status='calibrating',gpu_before=snapshot);write(a.out/'status.json',state)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=common['train_gpu'],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            with (a.out/'calibration.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt=read(calibration/'complete.json');assert receipt['status']=='completed_calibration'
        calfile=(calibration/'calibration.json').resolve()
        inputs=[common[k] for k in ['manifest','parent','selection','teacher_index','schedule','roi_protocol']]
        inputs+=[str(calfile)]+[r['path'] for r in frozen['files']]
        spec=dict(scene=plan['scene'],common=common,methods={'U':{},'W':{},'F':{'calibration':str(calfile)}},
                  file_identities=[identity(v) for v in inputs],required_receipts=[{'path':common['teacher_index'],'status':teacher['status']}])
        specfile=(a.out/'batch_spec.json').resolve();write(specfile,spec)
        state.update(status='running_fixed_batch');write(a.out/'status.json',state)
        with (a.out/'batch.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'remote_batch.py'),'run','--spec',str(specfile),'--out',str(a.out/'batch')],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(a.out/'batch/complete.json')['status']=='completed_sequential_batch'
        state.update(status='completed_calibration_and_fixed_batch',finished_utc=stamp())
        write(a.out/'complete.json',state);write(a.out/'status.json',state)
    except BaseException:
        state.update(status='failed_preparation_or_batch',traceback=traceback.format_exc(),failed_utc=stamp())
        write(a.out/'failed.json',state);write(a.out/'status.json',state)
        raise


if __name__=='__main__':main()
