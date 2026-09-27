"""CPU comparison of first post-boundary updates; no new optimization."""
import json
import math
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_evidence_repair_20260927'
def flatten(x,p=''):
 if torch.is_tensor(x) or x is None:yield p,x
 elif isinstance(x,dict):
  for k,v in x.items():yield from flatten(v,p+'/'+str(k))
 elif isinstance(x,(list,tuple)):
  for k,v in enumerate(x):yield from flatten(v,p+'/'+str(k))
def main():
 torch.set_num_threads(4);jobs=[('LiveB','LiveA',12001),('LiveB','LiveA',17551),('Restart12','LiveA',12001),('Restart12','LiveA',17551),('Restart17550','LiveA',17551)];result=[]
 for left,right,step in jobs:
  paths=[OUT/'replay_long'/r/'train'/f'boundary_{step}.pt' for r in [left,right]];a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];assert a['draw_sha256']==b['draw_sha256'];rows={}
  for group in ['gradients','parameters','base_optimizer','child_optimizer']:
   aa,bb=dict(flatten(a[group])),dict(flatten(b[group]));assert set(aa)==set(bb)
   for n,x in aa.items():
    y=bb[n]
    if x is None or y is None:rows[group+n]=dict(none_equal=(x is None)==(y is None));continue
    d=x.double()-y.double();norm=float(torch.linalg.vector_norm(y.double()));rows[group+n]=dict(maxabs=float(d.abs().max()) if d.numel() else 0.,relative_l2=float(torch.linalg.vector_norm(d))/max(norm,1e-30),reference_l2=norm,exact=torch.equal(x,y))
  losses=[]
  for run in [left,right]:
   records=[json.loads(x) for x in (OUT/'replay_long'/run/'train/training.jsonl').read_text().splitlines()];losses.append(next(r for r in records if r['step']==step))
  result.append(dict(left=left,right=right,step=step,paths=[str(p) for p in paths],rows=rows,pre_update_losses=losses,comparison='descriptive; ordinary LiveB/LiveA state diverged before boundary, while restart shares the exact LiveA pre-update state'))
 (OUT/'boundary_comparison.json').write_text(json.dumps(dict(status='completed',rows=result,parameter_updates=0),indent=2))
if __name__=='__main__':main()
