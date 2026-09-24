"""Finite teacher gradients and labelled synthetic stage-boundary fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from PIL import Image
import numpy as np
import full_state as fs
from multi_data import camera,sha


def main():
    p=argparse.ArgumentParser()
    for key in ['checkpoint','manifest','teacher-index','out']:p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from multi_model import load_model,render_model
    model=load_model(a.checkpoint);m=json.loads(a.manifest.read_text())
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state['completed_updates']==0 and state['stage']=='fine'
    identity=fs.digest(state['branches']);assert identity==fs.digest([fs.branch_state(g) for g in model.branches])
    idx=json.loads(a.teacher_index.read_text());entries={(e['camera'],e['frame']):e for e in idx['entries']}
    cams=[];targets=[]
    for cid in ['cam02','cam12']:
        r=next(r for r in m['observations'] if r['camera_id']==cid and r['frame_index']==40)
        cams.append(camera(m,r,torch.zeros(3,252,336)))
        entry=entries[cid,40];path=a.manifest.parent/entry['relative_path'];assert sha(path)==entry['sha256']
        with Image.open(path) as im:targets.append(torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float().cuda()/255.)
    rendered=[render_model(model,c) for c in cams]
    prediction=torch.stack([r['render'] for r in rendered]);target=torch.stack(targets)
    teacher=.4*(prediction-target).abs().mean();assert torch.isfinite(teacher)
    teacher.backward();norms={}
    for name,g in zip(['persistent_dynamic','static','transient'],model.branches):
        grads=[p.grad for group in g.optimizer.param_groups for p in group['params'] if p.grad is not None]
        assert all(bool(torch.isfinite(v).all()) for v in grads)
        norms[name]=dict(norm=sum(float(v.square().sum()) for v in grads)**.5,gradient_tensors=len(grads))
    visibility={name:[int(r[key].sum()) for r in rendered] for name,key in [('persistent_dynamic','visibility_filter_fg'),('static','visibility_filter_bg'),('transient','visibility_filter')]}
    (a.out/'teacher_gradient_observation.json').write_text(json.dumps(dict(teacher=float(teacher),gradients=norms,visibility=visibility),indent=2)+'\n')
    assert norms['persistent_dynamic']['norm']>0 and norms['static']['norm']>0
    # No optimizer/controller steps occur in this teacher-only diagnostic.
    assert identity==fs.digest([fs.branch_state(g) for g in model.branches])
    (a.out/'teacher_gradients.json').write_text(json.dumps(dict(status='passed',teacher=float(teacher),gradients=norms,parameter_updates=0,branch_identity_unchanged=True),indent=2)+'\n')
    differences=[]
    for name,g in [('persistent_dynamic',model.g),('static',model.bg)]:
        param=g._features_dc;flat=int(param.grad.abs().argmax());analytical=float(param.grad.flatten()[flat])
        original=float(param.flatten()[flat]);eps=.005
        values=[]
        with torch.no_grad():
            for sign in [1,-1]:
                param.flatten()[flat]=original+sign*eps
                pred=torch.stack([render_model(model,c)['render'] for c in cams])
                values.append(float(.4*(pred-target).double().abs().mean()))
            param.flatten()[flat]=original
        numerical=(values[0]-values[1])/(2*eps)
        relative=abs(numerical-analytical)/max(abs(analytical),1e-12)
        differences.append(dict(branch=name,parameter='SH_DC',index=flat,analytical=analytical,numerical=numerical,relative_error=relative))
    (a.out/'finite_difference.json').write_text(json.dumps(differences,indent=2)+'\n')
    assert all(d['relative_error']<.08 for d in differences),'Hybrid SH gradient finite difference failed'
    # Empty transient set can arise after pruning; exercise a full native render.
    model.tr.prune_points(torch.ones(len(model.tr._xyz),device='cuda',dtype=torch.bool))
    for g in model.branches:g.optimizer.zero_grad(set_to_none=True)
    empty=render_model(model,cams[0])['render'];assert torch.isfinite(empty).all()
    empty.mean().backward()
    assert model.bg._features_dc.grad is not None and torch.isfinite(model.bg._features_dc.grad).all() and model.bg._features_dc.grad.abs().sum()>0
    (a.out/'empty_transient.json').write_text(json.dumps(dict(status='passed',shape=list(empty.shape),transient_count=len(model.tr._xyz),background_backward_nonzero=True))+'\n')


if __name__=='__main__':main()
