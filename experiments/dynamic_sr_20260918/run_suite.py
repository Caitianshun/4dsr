"""Sequential per-GPU pilot runner with immutable per-run output directories."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True,choices=['cook_spinach','cut_roasted_beef'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--modes',default='lr_native,lr_integrated,joint,appearance,frozen')
    p.add_argument('--prefix',default='pilot_v1')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    project=Path(__file__).resolve().parents[2]
    scripts=Path(__file__).parent
    root=project/'output/dynamic_sr_20260918'
    manifest=project/'data/dynamic_sr/n3dv_prepared'/a.scene/'manifest.json'
    (root/'logs').mkdir(parents=True,exist_ok=True)
    events=[]
    def call(label, command):
        path=root/'logs'/f'{a.scene}_{a.prefix}_{label}.log'
        if path.exists(): raise FileExistsError(path)
        start=time.monotonic()
        print(json.dumps(dict(event='start',label=label,command=command,log=str(path))),flush=True)
        with path.open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=project)
        item=dict(label=label,elapsed_s=time.monotonic()-start,log=str(path))
        events.append(item)
        print(json.dumps(dict(event='complete',**item)),flush=True)
    if a.evaluate:
        call('sampling_warmup',[sys.executable,str(scripts/'evaluate.py'),'--manifest',str(manifest),
             '--checkpoint',a.checkpoint,'--out',str(root/f'{a.scene}_{a.prefix}_sampling_warmup'),'--sampling-only'])
        call('eval_warmup',[sys.executable,str(scripts/'evaluate.py'),'--manifest',str(manifest),
             '--checkpoint',a.checkpoint,'--out',str(root/f'{a.scene}_{a.prefix}_warmup_evaluation'),'--bicubic-baseline'])
    for mode in a.modes.split(','):
        out=root/f'{a.scene}_{a.prefix}_{mode}'
        call(mode,[sys.executable,str(scripts/'run_experiment.py'),'--manifest',str(manifest),
                   '--out',str(out),'--task','branch','--checkpoint',a.checkpoint,
                   '--mode',mode,'--steps',str(a.steps)])
        if a.evaluate:
            call('eval_'+mode,[sys.executable,str(scripts/'evaluate.py'),'--manifest',str(manifest),
                              '--checkpoint',str(out/'checkpoint_final.pt'),'--out',str(out/'evaluation')])
    if a.probe:
        call('probe',[sys.executable,str(scripts/'probe_coupling.py'),'--manifest',str(manifest),
                      '--checkpoint',a.checkpoint,'--out',str(root/f'{a.scene}_{a.prefix}_coupling_probe.json')])
    path=root/f'{a.scene}_{a.prefix}_suite.json'
    path.write_text(json.dumps(dict(args=vars(a),events=events),indent=2))


if __name__=='__main__': main()
