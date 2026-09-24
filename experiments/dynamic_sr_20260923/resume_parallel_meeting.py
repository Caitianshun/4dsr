"""Take over the paused serial dispatcher without interrupting its C trainer.

D runs on the confirmed-idle local 3090. C completion is received via pidfd,
then evaluated on GPU0. D has immediate GPU1 provisional evaluation and final
GPU0 evaluation. User's parallel scheduling change is retained in the receipt.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv,json,os,select,signal,subprocess,time,traceback
from pathlib import Path
from run_meeting_controls import ROOT,OUT,PY,SCRIPT,GPU,write

GPU1='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec'
START=time.monotonic()
MANIFEST=str(ROOT/'data/dynamic_sr/meetroom_prepared/discussion/manifest.json')


def run(label,cmd,gpu):
    state=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    for row in csv.reader(state.splitlines()):
        if row[0].strip()==gpu and row[2].strip()!='/opt/todesk/bin/ToDesk_Session':raise RuntimeError(f'{gpu} occupied: {row}')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    receipt=dict(label=label,command=cmd,gpu=gpu,status='running',started_utc=datetime.now(timezone.utc).isoformat(),gpu_before=state)
    path=OUT/'parallel_events'/f'{label}.json';write(path,receipt);tick=time.monotonic()
    with (OUT/f'{label}.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt['pid']=proc.pid;write(path,receipt);rc=proc.wait()
    receipt.update(status='completed' if rc==0 else 'failed',returncode=rc,elapsed_seconds=time.monotonic()-tick)
    write(path,receipt)
    if rc:raise RuntimeError(f'{label} failed {rc}')


def evaluate(branch,gpu,provisional=False):
    folder=OUT/branch
    for step in [1200,3000,6000]:
        prefix='provisional_3090' if provisional else 'eval'
        run(f'parallel_{branch}_{prefix}_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),
            '--checkpoint',str(folder/f'checkpoint_{step}.pt'),'--manifest',MANIFEST,
            '--out',str(folder/f'{prefix}_{step}'),'--no-video'],gpu)
        if not provisional:
            run(f'parallel_{branch}_fit_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260923/evaluate_training_fit.py'),
                '--run',str(folder),'--step',str(step)],gpu)


def finish_c(handoff):
    assert handoff['current']=='C_train' and len(handoff['children'])==1
    child=int(next(iter(handoff['children'])))
    # PID is the previously recorded child; pidfd waits for its exit event.
    try:
        fd=os.pidfd_open(child);select.select([fd],[],[]);os.close(fd)
    except ProcessLookupError:
        pass
    c=json.loads((OUT/'C/complete.json').read_text());assert c['parameter_updates']==6000
    # Only the dispatcher was stopped. Its GPU child has exited successfully.
    serial=handoff['serial_pid']
    if Path(f'/proc/{serial}').exists():os.kill(serial,signal.SIGKILL)
    handoff.update(status='C_completed_without_interruption_dispatcher_superseded',handoff_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'parallel_handoff.json',handoff)
    evaluate('C',GPU)


def do_d():
    run('parallel_D_train',[PY,str(SCRIPT),'train','--branch','D','--out',str(OUT/'D'),
        '--manifest',MANIFEST,'--checkpoint',str(ROOT/'output/dynamic_sr_20260919/meetroom_discussion_integrated_parent/checkpoint_final.pt'),
        '--cache-manifest',str(ROOT/'output/dynamic_sr_20260923/meeting_training_targets_v1/manifest.json'),
        '--calibration',str(ROOT/'output/dynamic_sr_20260923/meeting_calibration_v2/calibration.json'),'--steps','6000'],GPU1)
    evaluate('D',GPU1,provisional=True)


def main():
    (OUT/'parallel_events').mkdir(exist_ok=False)
    handoff=json.loads((OUT/'parallel_handoff.json').read_text())
    state=dict(status='parallel_running',reason='user requested parallel GPUs to accelerate validation',
        pid=os.getpid(),gpu_assignment={'C':GPU,'D':GPU1},final_evaluation_gpu=GPU,
        serial_snapshot='status_serial_snapshot.json',started_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'status.json',state)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fc=pool.submit(finish_c,handoff);fd=pool.submit(do_d)
            fc.result();fd.result()
        evaluate('D',GPU)
        rows=[];draws=[]
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
        write(OUT/'summary.json',dict(rows=rows,all_draws_equal=True,source='fixed endpoints, unified GPU0 evaluation',
            training_hardware='A/B/C RTX PRO6000; D RTX3090; hardware effects on optimization not eliminated by unified evaluation'))
        state.update(status='completed_and_evaluated',elapsed_seconds_since_handoff=time.monotonic()-START,
            finished_utc=datetime.now(timezone.utc).isoformat(),all_draws_equal=True)
        write(OUT/'status.json',state);write(OUT/'complete.json',state)
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc());write(OUT/'status.json',state);write(OUT/'failed.json',state);raise


if __name__=='__main__':main()
