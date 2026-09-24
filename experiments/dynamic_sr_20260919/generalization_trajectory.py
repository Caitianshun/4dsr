"""Post-training held-out trajectories; no training or checkpoint selection.

Waits on the GPU1 queue exit event, then evaluates existing intermediate states.
The original test cameras have already become development evidence. The fixed
18k endpoint remains the reported main comparison; this is diagnostic only.
"""
import json,os,select,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260919'
DEST=OUT/'generalization_trajectory'
PY='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
SCENES=['cook_spinach','cut_roasted_beef']
CASES=['lr_long','sr_w01','sr_w10']
STEPS=[1200,6000,12000]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)

def main():
    DEST.mkdir(exist_ok=False)
    state=DEST/'state.json'
    s=dict(status='waiting_gpu',pid=os.getpid(),gpu='1',started=now(),completed=[],
           purpose='diagnostic held-out trajectories, no endpoint selection or gradient updates')
    write(state,s)
    try:
        dependency=OUT/'cut_roasted_beef_queue.json';q=read(dependency)
        if q['status'] not in ['complete','failed']:
            fd=os.pidfd_open(q['pid'])
            try:select.select([fd],[],[])
            finally:os.close(fd)
            q=read(dependency)
        if q['status']!='complete':raise RuntimeError('GPU predecessor failed')
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
        s.update(status='running',gpu_acquired=now());write(state,s)
        result={}
        for scene in SCENES:
            result[scene]={}
            manifest=ROOT/f'data/dynamic_sr/n3dv_prepared/{scene}/manifest.json'
            for case in CASES:
                name=scene+'_'+case;result[scene][case]={}
                for step in STEPS:
                    label=name+f'_step{step}';out=DEST/label
                    s['current']=label;write(state,s)
                    cmd=[PY,str(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),'--manifest',str(manifest),
                         '--checkpoint',str(OUT/name/f'checkpoint_{step}.pt'),'--out',str(out),'--no-video']
                    with (DEST/(label+'.log')).open('x') as f:
                        p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                    if p.returncode:raise RuntimeError(f'{label}: exit {p.returncode}')
                    ev=read(out/'metrics.json')
                    result[scene][case][str(step)]=dict(aggregate=ev['aggregate'],temporal=ev['temporal_aggregate'],
                        checkpoint_sha256=ev['checkpoint_sha256'],metrics=str(out/'metrics.json'))
                    s['completed'].append(label);write(state,s)
                ev=read(OUT/(name+'_evaluation')/'metrics.json')
                result[scene][case]['18000']=dict(aggregate=ev['aggregate'],temporal=ev['temporal_aggregate'],
                    checkpoint_sha256=ev['checkpoint_sha256'],metrics=str(OUT/(name+'_evaluation')/'metrics.json'))
        write(DEST/'metrics.json',dict(scope='development-only post-training trajectory diagnosis',
              no_checkpoint_selection=True,finished=now(),scenes=result))
        s.update(status='complete',finished=now(),current=None);write(state,s)
    except BaseException as e:
        s.update(status='failed',finished=now(),error=repr(e));write(state,s);raise

if __name__=='__main__':main()
