"""One no-update check of base compatibility, selection gradient and projection."""
from dv_common import *
shared=setup()
import importlib.util
import torch
import numpy as np
from prior_modules import TexturePrior,render_sampling
from sr_attribute_router import render_model
from prepare_texture_maps import screen_radius,scalar_render
from motion_model import load_model
from n3dv_data import load_manifest
from common import resized_camera

def main():
    torch.set_num_threads(4)
    reg=read(OUT/'audit_manifest.json')['assets'];m=load_manifest(bound(reg['manifest']))
    spec=importlib.util.spec_from_file_location('diagnosis_base_loader',ROOT/'experiments/dynamic_sr_20260918/run_experiment.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    records,cameras=loader.load_training(m);w,h=m['resolutions']['hr']
    camera=resized_camera(cameras[0],h,w);model=load_model(bound(reg['parent']),m)
    identity=shared.initial_identity(model);cache=TexturePrior(False,dict(teacher=reg['teacher']),records,m)
    pred=render_model(model,camera)['render'];loss0=(pred-cache.teacher.get(0)).abs().mean()
    loss1=cache.loss(dict(render=pred),dict(sr_index=0));assert torch.equal(loss0,loss1)
    named=shared.all_named(model);params=[p for p in named.values() if p.requires_grad]
    grad0=torch.autograd.grad(loss0,params,retain_graph=True,allow_unused=True)
    grad1=torch.autograd.grad(loss1,params,retain_graph=True,allow_unused=True)
    repeat0=torch.autograd.grad(loss0,params,retain_graph=True,allow_unused=True)
    pixel0=torch.autograd.grad(loss0,pred,retain_graph=True)[0]
    pixel1=torch.autograd.grad(loss1,pred,retain_graph=True)[0]
    assert torch.equal(pixel0,pixel1)
    gradient_comparison=[]
    for a,b,c in zip(grad0,grad1,repeat0):
        assert (a is None)==(b is None)
        if a is not None:
            relative_norm=float((a-b).norm()/a.norm().clamp_min(1e-12))
            baseline_repeat=float((a-c).norm()/a.norm().clamp_min(1e-12))
            close=relative_norm <= max(5*baseline_repeat,1e-4)
            gradient_comparison.append(dict(max_abs=float((a-b).abs().max()),relative_l2=relative_norm,baseline_repeat_relative_l2=baseline_repeat,close=close,bitwise_equal=torch.equal(a,b)))
            assert close,gradient_comparison[-1]
    map_path=OUT/'diagnostic_maps'/records[0]['camera_id']/f"{records[0]['frame_index']:04d}.npy"
    weight=torch.from_numpy(np.load(map_path)).cuda()[None,None]
    weight=torch.nn.functional.interpolate(weight,(h,w),mode='nearest')[0]
    weighted=((pred-cache.teacher.get(0)).abs()*weight).mean()
    gw=torch.autograd.grad(weighted,params,allow_unused=True)
    grads={name:dict(connected=v is not None,norm=0. if v is None else float(v.norm()),finite=v is None or bool(torch.isfinite(v).all()))
        for name,v in zip([n for n,p in named.items() if p.requires_grad],gw)}
    assert all(g['finite'] for g in grads.values()) and grads['_xyz']['norm']>0 and grads['_scaling']['norm']>0
    assert any(g['norm']>0 for n,g in grads.items() if n.startswith('deformation.'))
    with torch.no_grad():
        off=render_sampling(model,camera,False)['render'];assert torch.equal(pred,off)
        state=shared.effective_state(model,camera.time);w0,h0=m['resolutions']['lr'];low=resized_camera(camera,h0,w0)
        radius,cov=screen_radius(low,state)
        _,actual,_=scalar_render(low,state,torch.ones((len(radius),3),device='cuda'))
        c=cov+torch.eye(2,device='cuda')[None]*.3;a,b,d=c[:,0,0],c[:,0,1],c[:,1,1]
        mid=(a+d)/2;expect=torch.ceil(3*(mid+(mid.square()-(a*d-b.square())).clamp_min(.1).sqrt()).sqrt())
        visible=actual>0;delta=(actual[visible]-expect[visible]).abs()
        assert int((delta>1).sum())==0
        unchanged=shared.initial_identity(model)==identity;assert unchanged
    write(OUT/'control_tests.json',dict(status='passed',script=entry(Path(__file__)),parameter_updates=0,
        module_off_loss_exact=True,module_off_pixel_gradients_exact=True,module_off_render_exact=True,
        parameter_gradient_comparison=gradient_comparison,
        parameter_gradient_tolerance='relative L2 no greater than max(5x same-baseline repeated-backward difference,1e-4); exact incoming pixel gradient is the compatibility gate',
        complete_model_and_two_adam_unchanged=unchanged,weighted_joint_gradients=grads,
        projection=dict(visible_points=int(visible.sum()),max_radius_difference=float(delta.max()),
                        exact_fraction=float((delta==0).float().mean()),formula='native radius including +0.3 for comparison'),
        maps_legal_only=read(OUT/'diagnostic_maps/prior_index.json')['information_boundary'],
        resume='same existing complete checkpoint/sampler/RNG restoration; per-run assertions before first update',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name()))
    print('control tests passed; zero parameter updates',flush=True)

if __name__=='__main__':main()
