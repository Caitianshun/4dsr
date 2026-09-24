"""Serial local queue; the model does not poll per-step training logs."""
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/dynamic_sr_20260918'
NEW=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_20260919'
PY=Path('/home/cai_tianshun/Project/4dgs/.venv/bin/python')

def now():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def run(name,argv,state,status):
    status['current']=name;status['current_started']=now();write(state,status)
    with open(OUT/f'{name}.log','x') as log:
        result=subprocess.run([str(PY),*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}')
    status['completed'].append(name);write(state,status)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
    OUT.mkdir(exist_ok=True,parents=True)
    state=OUT/f'{a.scene}_queue.json'
    if state.exists():raise FileExistsError(state)
    status=dict(scene=a.scene,gpu=a.gpu,pid=os.getpid(),started=now(),status='running',completed=[])
    manifest=ROOT/f'data/dynamic_sr/n3dv_prepared/{a.scene}/manifest.json'
    parent=ROOT/f'output/dynamic_sr_20260918/{a.scene}_pilot_v1_lr_integrated/checkpoint_final.pt'
    cases=[('lr_long','none',0,False),('sr_w01','sr',.1,False),('sr_w10','sr',1.,False),
           ('hr_oracle_w10','hr',1.,False),('sr_w10_dense','sr',1.,True),('lr_dense','none',0,True)]
    try:
        for case,teacher,weight,dense in cases:
            name=f'{a.scene}_{case}';out=OUT/name
            args=[NEW/'controlled_fit.py','--manifest',manifest,'--checkpoint',parent,'--out',out,
                  '--teacher',teacher,'--weight',weight,'--steps',18000,'--milestones','1200,6000,12000,18000']
            if dense:args+=['--dense']
            run(name,args,state,status)
            run(name+'_eval',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',OUT/(name+'_evaluation'),'--no-video'],state,status)
            run(name+'_sampling',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',OUT/(name+'_sampling'),'--sampling-only','--no-video'],state,status)
        status.update(status='complete',finished=now(),current=None);write(state,status)
    except BaseException as e:
        status.update(status='failed',finished=now(),error=repr(e));write(state,status);raise

if __name__=='__main__':main()
