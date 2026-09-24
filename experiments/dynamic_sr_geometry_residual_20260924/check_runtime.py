"""Training-only basis, center-routing, zero-parity, gradient and reload checks."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import torch
from time_basis import TimeBasis, raw_basis
from geometry_model import old, attach, render_model, load_model, synchronize_lr, geometry_state
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924'))
from training_support import read, initial_identity, load_teacher_index, TeacherCache, digest_state
from common import load_checkpoint, resized_camera, downsample, write_json
from n3dv_data import load_manifest
from run_experiment import load_training


def math_checks():
    result={}
    from scipy.interpolate import BSpline
    import numpy as np
    ts=torch.unique(torch.cat([torch.linspace(0,1,1001,dtype=torch.float64),torch.tensor([0,.2,.4,.6,.8,1],dtype=torch.float64)]))
    for kind in ['G','L']:
        basis=TimeBasis(kind); b=raw_basis(ts,kind)
        if kind=='G': expected=np.polynomial.legendre.legvander((2*ts-1).numpy(),7)
        else: expected=BSpline.design_matrix(ts.numpy(),basis.knots64.numpy(),3).toarray()
        error=float((b-torch.from_numpy(expected)).abs().max()); assert error<1e-12
        if kind=='L':
            assert float(b.min())>=0 and float((b.sum(-1)-1).abs().max())<1e-12
            assert int((b>0).sum(-1).max())<=4
            # Interior basis2 has support [0,.6]; coefficient direct gradient is zero outside.
            assert float(raw_basis(torch.tensor(.8,dtype=torch.float64),'L')[2])==0
        assert basis.protocol()['rank']==8
        assert abs(basis.protocol()['mean_row_square_norm']-1)<1e-6
        for bad in [-.1,1.1,float('nan')]:
            try: raw_basis(torch.tensor(bad),kind)
            except ValueError: pass
            else: raise AssertionError('Out-of-window silently accepted')
        result[kind]=dict(**basis.protocol(),independent_reference_max_abs=error)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','checkpoint','selection','teacher-index','reference-config','out']: p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=dict(bases=math_checks(),scope='Engineering checks on legal train cam02/frame40 only; no HR or held-out reads')
    m=load_manifest(a.manifest);g,h,o,ck=load_checkpoint(a.checkpoint)
    selection=torch.load(a.selection,map_location='cpu',weights_only=False)
    model=old.make_model(g,h,o,ck,m,'ordinary_split',selection)
    assert initial_identity(model)==read(a.reference_config)['initial_identity']
    records,cameras=load_training(m)
    idx=next(i for i,r in enumerate(records) if r['camera_id']=='cam02' and r['frame_index']==40)
    w,hh=m['resolutions']['hr'];camera=resized_camera(cameras[idx],hh,w)
    paths,_=load_teacher_index(a.teacher_index,m,records)
    target=TeacherCache(paths,(3,hh,w)).get(idx);lr=records[idx]['image'].cuda()
    params=[p for p in list(old.named_parameters(g).values())+list(model.children.parameters()) if p.requires_grad]
    def probe(enabled):
        out=render_model(model,camera,enabled)['render']
        loss=(downsample(out,lr.shape[-2:])-lr).abs().mean()+.1*(out-target).abs().mean()
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        return out.detach(),[x.detach().clone() if x is not None else None for x in grads]
    baseline,grads=probe(False)
    _, repeated = probe(False)
    repeat_error=sum(float((x-y).double().square().sum()) for x,y in zip(grads,repeated) if x is not None)**.5
    repeat_norm=sum(float(x.double().square().sum()) for x in grads if x is not None)**.5
    report['unchanged_U_gradient_repeat_relative_l2']=repeat_error/max(repeat_norm,1e-30)
    report['gradient_check']='Record CUDA old-parameter differences and one unchanged-U backward repeat; center-addition Jacobian is checked exactly. No optimizer or training repeated.'
    report['branches']={} 
    for kind in ['G','L']:
        attach(model,kind,m)
        assert not model.geometry_optimizer.state
        prediction,new=probe(True)
        err=float((prediction-baseline).abs().max());assert err<1e-6
        pairs=[(x,y) for x,y in zip(grads,new) if x is not None and y is not None]
        assert all((x is None)==(y is None) for x,y in zip(grads,new))
        grad_error=sum(float((x-y).double().square().sum()) for x,y in pairs)**.5
        grad_norm=sum(float(x.double().square().sum()) for x,y in pairs)**.5
        relative=grad_error/max(grad_norm,1e-30)
        assert all(bool(torch.isfinite(y).all()) for x,y in pairs)
        # The zero addition's derivative to all old centers is exactly identity;
        # CUDA deformation/rasterizer reductions are independently nondeterministic.
        leaf=torch.randn(5,3,device='cuda',requires_grad=True)
        co=torch.zeros(2,3,device='cuda',requires_grad=True)
        weight=torch.randn_like(leaf)
        vjp=torch.autograd.grad((leaf.index_add(0,torch.tensor([3,4],device='cuda'),co)*weight).sum(),leaf)[0]
        assert torch.equal(vjp,weight)
        # A gross mismatch still blocks launch. Small differences are explicitly
        # descriptive alongside the same U repeat, not a statistical noise bound.
        assert relative < .005, relative
        routes={}
        for route in ['LR','SR']:
            pred=render_model(model,camera)['render']
            loss=(downsample(pred,lr.shape[-2:])-lr).abs().mean() if route=='LR' else .1*(pred-target).abs().mean()
            grad=torch.autograd.grad(loss,model.geometry.coeff)[0]
            assert torch.isfinite(grad).all() and float(grad.norm())>0
            routes[route]=float(grad.norm())
        # Intercept the actual rasterizer input, proving only child world means move.
        original=old.rasterize
        try:
            old.rasterize=lambda g,c,xyz,cov,opacity,sh: {'means':xyz,'cov':cov,'opacity':opacity,'sh':sh}
            base=render_model(model,camera,False)
            with torch.no_grad(): model.geometry.coeff[0,2,0]=.001
            changed=render_model(model,camera)
            delta=changed['means']-base['means'];expected=torch.zeros_like(delta)
            expected[model.geometry.child_indices[0],0]=.001*model.geometry.basis(40)[2]
            assert torch.equal(changed['means'], base['means']+expected)
            for name in ['cov','opacity','sh']: assert torch.equal(changed[name],base[name])
            if kind=='L':
                shifted=copy.copy(camera);shifted.image_name='cam02/0100';shifted.time=100/300
                center=render_model(model,shifted)['means']
                cg=torch.autograd.grad(center.sum(),model.geometry.coeff)[0]
                assert torch.count_nonzero(cg[:,2])==0
        finally: old.rasterize=original
        # One diagnostic Adam update, distinct from formal zero-initialized training.
        g.update_learning_rate(7201);rate=synchronize_lr(model)
        model.geometry_optimizer.zero_grad(set_to_none=True)
        pred=render_model(model,camera)['render'];(.1*(pred-target).abs().mean()).backward()
        model.geometry_optimizer.step()
        payload=dict(model=g.capture(),hidden=vars(h),optim=vars(o),metadata=ck['metadata'],
                     motion_refinement=old.refinement_state(model),geometry_residual=geometry_state(model))
        path=a.out/f'{kind}_reload_probe.pt';torch.save(payload,path)
        restored=load_model(path,m)
        with torch.no_grad(): reload_error=float((render_model(model,camera)['render']-render_model(restored,camera)['render']).abs().max())
        assert reload_error<1e-6
        assert digest_state(model.geometry_optimizer.state_dict())==digest_state(restored.geometry_optimizer.state_dict())
        report['branches'][kind]=dict(zero_forward_max_abs=err,old_gradient_relative_l2=relative,
            coefficient_gradient_l2=routes,child_only_centers=True,unmodified_other_attributes=True,
            reload_max_abs=reload_error,optimizer_reload_exact=True,world_xyz_lr=rate)
        del restored
    report.update(status='passed',gpu=torch.cuda.get_device_name(),reference_identity=read(a.reference_config)['initial_identity'])
    write_json(a.out/'checks.json',report);print(json.dumps(report,allow_nan=False))

if __name__=='__main__':main()
