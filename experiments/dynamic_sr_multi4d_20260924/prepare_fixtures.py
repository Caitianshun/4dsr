"""Boundary fixtures from a real short update: initialized Adam is essential."""
import argparse
from pathlib import Path
import torch
import full_state as fs


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state['stage']=='fine' and state['completed_updates']==20
    assert all(b['optimizer']['state'] for b in state['branches'])
    for step in [2999,3049,9999,15999]:
        fixture=dict(state,completed_updates=step,phase=3 if step>=10000 else 2)
        fixture['metadata']=dict(state['metadata'],engineering_fixture=True,warning='Real fine20 parameters and Adam; synthetic loop counter, NOT a trained long endpoint')
        fixture['sampler']=dict(state['sampler'],completed_batches=step)
        fs.save(a.out/f'fixture_{step}.pt',fixture)


if __name__=='__main__':main()
