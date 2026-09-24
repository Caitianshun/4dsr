"""One fixed D repeat on A/B/C's training GPU to resolve hardware confounding."""
from datetime import datetime, timezone
import hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from run_meeting_controls import ROOT, PY, SCRIPT, GPU, write, check_gpu

OUT=ROOT/'output/dynamic_sr_20260923/meeting_D_gpu0_confirmation_v1'
BASE=ROOT/'output/dynamic_sr_20260923/meeting_controls_v1'


def main():
    assert json.loads((BASE/'complete.json').read_text())['status']=='completed_and_evaluated'
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((BASE/'D/config.json').read_text())
    state=dict(status='running',reason='Small mixed-hardware differences do not identify teacher effect; one same-GPU fixed D confirmation, no tuning',
        started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),gpu=GPU,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),events=[],
        comparison_endpoint=6000,reference=str(BASE),training_replicate='same explicit seed and camera streams as original A/B/C/D')
    start=time.monotonic()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    def run(label,cmd):
        event=dict(label=label,command=cmd,gpu_before=check_gpu(),started_utc=datetime.now(timezone.utc).isoformat())
        state['events'].append(event);write(OUT/'status.json',state);tick=time.monotonic()
        with (OUT/f'{label}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            event['pid']=proc.pid;write(OUT/'status.json',state);rc=proc.wait()
        event.update(returncode=rc,elapsed_seconds=time.monotonic()-tick);write(OUT/'status.json',state)
        if rc:raise RuntimeError(f'{label}: {rc}')
    try:
        folder=OUT/'D'
        run('train',[PY,str(SCRIPT),'train','--branch','D','--out',str(folder),
            '--manifest',cfg['manifest'],'--checkpoint',cfg['checkpoint'],'--cache-manifest',cfg['cache_manifest'],
            '--calibration',cfg['calibration'],'--steps','6000','--seed',str(cfg['seed'])])
        for step in [1200,3000,6000]:
            run(f'eval_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),
                '--checkpoint',str(folder/f'checkpoint_{step}.pt'),'--manifest',cfg['manifest'],
                '--out',str(folder/f'eval_{step}'),'--no-video'])
            run(f'fit_{step}',[PY,str(ROOT/'experiments/dynamic_sr_20260923/evaluate_training_fit.py'),
                '--run',str(folder),'--step',str(step)])
        done=json.loads((folder/'complete.json').read_text())
        assert done['draw_sha256']==json.loads((BASE/'A/complete.json').read_text())['draw_sha256']
        assert done['parameter_updates']==6000 and done['source_unchanged']
        for step in [1200,3000,6000]:
            m=json.loads((folder/f'eval_{step}/metrics.json').read_text())
            assert m['frame_indices']==list(range(0,120,2)) and len(m['rows'])==60
            assert (folder/f'own_target_fit_{step}.json').exists()
        state.update(status='completed_and_evaluated',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        write(OUT/'status.json',state);write(OUT/'complete.json',state)
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc());write(OUT/'status.json',state);write(OUT/'failed.json',state);raise


if __name__=='__main__':main()
