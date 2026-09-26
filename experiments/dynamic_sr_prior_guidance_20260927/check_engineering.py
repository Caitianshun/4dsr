"""Pre-registered routing, restart, and indirect-update engineering fixtures."""
import os
import subprocess
import sys
import time
import traceback
import numpy as np
from shared import *
from gradient_policy import render_model as routed_render, effective_state, appearance, backward_prior, group_name


def tensor_diff(a, b):
    diffs = []
    def rec(x, y):
        assert type(x) == type(y)
        if torch.is_tensor(x):
            assert x.shape == y.shape and x.dtype == y.dtype
            diffs.append(float((x.double()-y.double()).abs().max()) if x.numel() else 0.)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: rec(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for u, v in zip(x, y): rec(u, v)
        elif isinstance(x, np.ndarray): assert np.array_equal(x, y)
        else: assert x == y
    rec(a,b)
    return max(diffs)


def main():
    torch.set_num_threads(4)
    out=OUT/'engineering_v1';out.mkdir(exist_ok=False)
    p=paths();protocol=read(OUT/'protocol.json');m=load_manifest(p['manifest']);ev=evaluator()
    start=Path(protocol['start']);model=load_model(start,m)
    obs=next(o for o in m['observations'] if o['camera_id']=='cam02' and o['frame_index']==40)
    cam=ev.render_camera(m,obs,0)
    zero=[]
    with torch.no_grad():
        for c in ['cam00','cam01','cam02','cam18']:
            for f in [40,80]:
                o=next(o for o in m['observations'] if o['camera_id']==c and o['frame_index']==f)
                camera=ev.render_camera(m,o,0);reference=render_model(model,camera)['render']
                for detach in [False,True]:
                    delta=float((routed_render(model,camera,detach)['render']-reference).abs().max())
                    assert delta==0
                    zero.append(dict(camera=c,frame=f,detach=detach,maxabs=delta))
    target=torch.from_numpy(ev.legacy.read_rgb(Path(m['_root'])/obs['lr_path'])).permute(2,0,1).cuda()
    index=read(p['teacher']);entry=next(e for e in index['entries'] if e['camera']=='cam02' and e['frame']==40)
    teacher=torch.from_numpy(ev.legacy.read_rgb(Path(m['_root'])/entry['relative_path'])).permute(2,0,1).cuda()
    traces=[]
    for kind in ['SR_joint','SR_detached','LR']:
        model.g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
        r=routed_render(model,cam,kind=='SR_detached',trace=True)
        loss=(downsample(r['render'],target.shape[-2:])-target).abs().mean() if kind=='LR' else .1*(r['render']-teacher).abs().mean()
        loss.backward()
        states=r['effective_state'];grad={n:None if v.grad is None else float(v.grad.abs().max()) for n,v in states.items()}
        if kind=='SR_detached':
            assert grad['xyz'] is None or grad['xyz']==0
            assert all(grad[n] is not None and grad[n]>0 for n in ['cov','opacity','sh'])
        else: assert grad['xyz']>0
        traces.append(dict(kind=kind,effective_gradient_maxabs=grad,parameters={n:None if v.grad is None else float(v.grad.norm()) for n,v in all_named(model).items()}))
    # The two base/child states were concatenated before the only rasterizer call.
    assert len(states['xyz'])==132972
    # Separate synthetic leaf attributes prove the local interface does not
    # depend on shared-field compensation or a particular real residual.
    with torch.no_grad():
        synthetic={k:v[:0].clone() for k,v in states.items()}
        world=torch.tensor([[.05,.03,2.,1.],[-.1,.04,2.8,1.]],device='cuda')@torch.linalg.inv(cam.world_view_transform.cuda())
    synthetic=dict(xyz=world[:,:3].detach().requires_grad_(),cov=torch.diag_embed(torch.tensor([[.014,.009,.005],[.016,.006,.01]],device='cuda')).requires_grad_(),opacity=torch.tensor([[.5],[.4]],device='cuda',requires_grad=True),sh=torch.zeros((2,16,3),device='cuda',requires_grad=True))
    synthetic['sh'].data[:,0]=torch.tensor([.1,.3,-.2],device='cuda')
    raw=rasterize(model.g,cam,synthetic['xyz'].detach(),synthetic['cov'],synthetic['opacity'],synthetic['sh'])['render']
    (raw.square().mean()).backward()
    sg={n:None if v.grad is None else float(v.grad.norm()) for n,v in synthetic.items()}
    assert sg['xyz'] is None and all(sg[n]>0 for n in ['cov','opacity','sh'])
    # Exact A_late addition preserving every existing LR gradient.
    model.g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
    loss=(downsample(routed_render(model,cam)['render'],target.shape[-2:])-target).abs().mean()
    (loss+model.g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)).backward()
    named=all_named(model);before={n:None if p.grad is None else p.grad.clone() for n,p in named.items()}
    color=appearance(model);sr=.1*(routed_render(model,cam)['render']-teacher).abs().mean()
    for param in named.values():param.grad=None
    sr.backward(retain_graph=True)
    wanted=[None if param.grad is None else param.grad.clone() for param in color.values()]
    for n,param in named.items():param.grad=None if before[n] is None else before[n].clone()
    backward_prior(sr,model,'A_late',18000)
    appmax=0.
    for n,p0 in named.items():
        if n not in color:
            assert (before[n] is None and p0.grad is None) or torch.equal(before[n],p0.grad),n
    for (n,p0),d in zip(color.items(),wanted):
        expected=before[n] if d is None else d if before[n] is None else before[n]+d
        if expected is not None:appmax=max(appmax,float((p0.grad-expected).abs().max()))
    write_json(out/'appearance_addition.json',dict(maxabs=appmax,atol=1e-8,rtol=1e-5,reference='ordinary full SR backward, CUDA atomic reductions may differ'))
    for (n,p0),d in zip(color.items(),wanted):
        expected=before[n] if d is None else d if before[n] is None else before[n]+d
        if expected is not None:torch.testing.assert_close(p0.grad,expected,atol=1e-8,rtol=1e-5)
    # SR-only SGD with inherited per-group LR, explicitly no Adam momentum.
    model.g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
    xyz_before=effective_state(model,cam.time)['xyz'].detach().clone()
    (.1*(routed_render(model,cam,True)['render']-teacher).abs().mean()).backward()
    step_norm={}
    with torch.no_grad():
        for opt in [model.g.optimizer,model.child_optimizer]:
            for group in opt.param_groups:
                for param in group['params']:
                    if param.grad is not None:param.add_(param.grad,alpha=-group['lr'])
        drift=(effective_state(model,cam.time)['xyz']-xyz_before).norm(dim=-1)
        indirect=dict(median=float(drift.median()),max=float(drift.max()),mean=float(drift.mean()),nonzero=int((drift>0).sum()),optimizer='one SR-only SGD step at inherited group LRs, no momentum')
    write_json(out/'routing.json',dict(status='passed',zero_step=zero,traces=traces,synthetic=sg,appearance_addition_maxabs=appmax,indirect_center_motion=indirect,base_child_covered=132972))
    del model,r,states,named,color,raw;torch.cuda.empty_cache()
    def run(arm,resume,stop,name):
        dest=out/name
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--method',arm,'--manifest',str(p['manifest']),'--resume',str(resume),'--schedule',str(p['schedule']),'--protocol',str(OUT/'protocol.json'),'--lr-curve',str(p['lr_curve']),'--target-index',str(p['teacher']),'--stop',str(stop),'--out',str(dest),'--smoke']
        with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[];limits=protocol['engineering']
    for arm in ['C_joint','P_early','A_late']:
        run(arm,start,6004,arm+'_direct');run(arm,start,6002,arm+'_prefix')
        run(arm,out/(arm+'_prefix')/'checkpoint_6002.pt',6004,arm+'_resume');run(arm,start,6004,arm+'_repeat')
        x=load_model(out/(arm+'_direct')/'checkpoint_6004.pt',m);comparisons={}
        for label in ['resume','repeat']:
            y=load_model(out/(arm+'_'+label)/'checkpoint_6004.pt',m)
            assert x.checkpoint['samplers']==y.checkpoint['samplers']
            assert x.checkpoint['metadata']['draw_sha256']==y.checkpoint['metadata']['draw_sha256']
            maximum=tensor_diff((x.g.capture(),x.children.state_dict(),x.child_optimizer.state_dict()),(y.g.capture(),y.children.state_dict(),y.child_optimizer.state_dict()))
            with torch.no_grad():delta=render_model(x,cam)['render']-render_model(y,cam)['render']
            comparisons[label]=dict(tensor_maxabs=maximum,render_maxabs=float(delta.abs().max()),meanabs=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()))
            del y
        r,t=comparisons['resume'],comparisons['repeat']
        tests=dict(tensor=r['tensor_maxabs']<=max(limits['tensor_floor'],limits['replay_envelope']*t['tensor_maxabs']),mean=r['meanabs']<=limits['render_meanabs'],rms=r['rmse']<=limits['render_rmse'],rms_envelope=r['rmse']<=max(limits['render_rmse_floor'],limits['replay_envelope']*t['rmse']))
        rows.append(dict(arm=arm,comparisons=comparisons,tests=tests,passed=all(tests.values())))
        write_json(out/'replay.json',dict(rows=rows,limits=limits))
        assert all(tests.values()),(arm,tests)
        del x;torch.cuda.empty_cache()
    write_json(out/'complete.json',dict(status='passed_with_cuda_replay_envelope',rows=rows,routing='routing.json',gpu=torch.cuda.get_device_name(),fixture_updates=36,scientific_use=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'engineering_v1/failed.json',dict(traceback=traceback.format_exc()))
        raise
