"""One fixed, no-update legal-input calibration; never query development HR."""
from dv_common import *
shared=setup()
import importlib.util
import numpy as np
import torch
import random
from prior_modules import TexturePrior
from sr_attribute_router import render_model
from motion_model import load_model
from n3dv_data import load_manifest
from common import resized_camera

def main():
    torch.set_num_threads(4)
    p=read(OUT/'audit_manifest.json')['assets'];m=load_manifest(bound(p['manifest']))
    spec=importlib.util.spec_from_file_location('calibration_training_loader',ROOT/'experiments/dynamic_sr_20260918/run_experiment.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    records,cameras=loader.load_training(m);w,h=m['resolutions']['hr']
    model=load_model(bound(p['parent']),m);cache=TexturePrior(False,dict(teacher=p['teacher']),records,m)
    identity=shared.initial_identity(model)
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
    idx=read(OUT/'diagnostic_maps/prior_index.json');by={(e['camera'],e['frame']):e for e in idx['entries']}
    candidates=[i for i,r in enumerate(records) if (r['camera_id'],r['frame_index']) in by]
    chosen=[candidates[j] for j in np.linspace(0,len(candidates)-1,32,dtype=int)]
    named={n:v for n,v in shared.all_named(model).items() if v.requires_grad and not any(x in n for x in ['features_dc','features_rest','sh_dc','sh_rest','shs_deform'])}
    rows=[]
    for i in chosen:
        r=records[i];key=(r['camera_id'],r['frame_index']);camera=resized_camera(cameras[i],h,w)
        pred=render_model(model,camera)['render'];target=cache.teacher.get(i);error=(pred-target).abs()
        weights=torch.from_numpy(np.load(bound(by[key]))).cuda()[None,None]
        weights=torch.nn.functional.interpolate(weights,(h,w),mode='nearest')[0]
        base=error.mean();weighted=(error*weights).mean()
        a=torch.autograd.grad(base,list(named.values()),retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(weighted,list(named.values()),allow_unused=True)
        norm=lambda gs:float(torch.sqrt(sum(x.double().square().sum() for x in gs if x is not None)))
        n0,n1=norm(a),norm(b);assert n0>0 and n1>0 and np.isfinite(n0+n1)
        rows.append(dict(camera=key[0],frame=key[1],base_geometry_gradient=n0,weighted_geometry_gradient=n1,
            base_loss=float(base),weighted_loss=float(weighted),mean_weight=float(weights.mean())))
    scale=float(np.median([r['base_geometry_gradient'] for r in rows])/np.median([r['weighted_geometry_gradient'] for r in rows]))
    assert np.isfinite(scale) and scale>0
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    assert shared.initial_identity(model)==identity
    write(OUT/'texture_calibration.json',dict(status='completed',loss_scale=scale,batches=32,rows=rows,
        rule='median baseline SR geometry norm / median weighted SR geometry norm; once, no update',
        geometry_parameter_names=list(named),parent=p['parent'],maps=entry(OUT/'diagnostic_maps/prior_index.json'),
        script=entry(Path(__file__)),external_sr_coefficient=.1,parameter_updates=0,
        rng_restored=True,model_and_two_adam_unchanged=True,information_boundary='legal train LR, SwinIR, frozen U6000 demand maps only'))
    print(json.dumps(dict(loss_scale=scale,batches=32,parameter_updates=0)),flush=True)

if __name__=='__main__':main()
