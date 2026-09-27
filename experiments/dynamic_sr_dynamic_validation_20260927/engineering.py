"""At most 26 planned fixture updates; exact identities and old CUDA replay envelope."""
import subprocess
import traceback
import time
from dv_common import *
shared=setup()
import torch
import numpy as np
from routing import render_model,effective_state,backward_prior,selected,policy_info
from training_support import initial_identity,digest_state
from motion_model import load_model
from common import downsample
from n3dv_data import load_manifest
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
from resume_control import restore_global_rng,assert_state_equal
from train import rng

def maxdiff(a,b):
    values=[]
    def rec(x,y):
        assert type(x)==type(y),(type(x),type(y))
        if torch.is_tensor(x):
            assert x.shape==y.shape and x.dtype==y.dtype;values.append(float((x.double()-y.double()).abs().max()) if x.numel() else 0.)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:rec(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y)
            for u,v in zip(x,y):rec(u,v)
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
        else:assert x==y,(x,y)
    rec(a,b);return max(values,default=0.)

def main():
    torch.set_num_threads(4);out=OUT/'engineering';out.mkdir(exist_ok=False);assert read(OUT/'deployment_verified.json')['protocol_sha256']==sha(OUT/'protocol.json');p=read(OUT/'protocol.json');m=load_manifest(bound(p['manifest']));parent=bound(p['parent']);ev=shared.evaluator();started=time.time()
    source=sources();assert os.environ['CUDA_VISIBLE_DEVICES']==p['training']['physical_gpu']
    obs=next(o for o in m['observations'] if (o['camera_id'],o['frame_index'])==('cam02',40));cam=ev.render_camera(m,obs,0)
    lrpath=Path(m['_root'])/obs['lr_path'];lr=torch.from_numpy(ev.legacy.read_rgb(lrpath)).permute(2,0,1).cuda()
    idx=read(bound(p['teacher']));e=next(e for e in idx['entries'] if (e['camera'],e['frame'])==('cam02',40));tp=Path(m['_root'])/e['relative_path'];assert sha(tp)==e['sha256'];teacher=torch.from_numpy(ev.legacy.read_rgb(tp)).permute(2,0,1).cuda()
    audits=[];zero_identity=None;zero_rgb=None
    prior_route=OUT/'routing_audit.json';reuse=None
    previous_config=OUT/'engineering_lr_roundoff_failure/r1_J_joint_direct/config.json'
    if previous_config.exists():
        cfg=read(previous_config);reuse=read(prior_route);assert cfg['protocol_sha256']==sha(OUT/'protocol.json')
        for k,v in source.items():
            if k not in [str(HERE.relative_to(ROOT)/'train.py'),str(HERE.relative_to(ROOT)/'engineering.py')]:assert cfg['sources'][k]==v,k
        audits=reuse['rows'];assert [r['arm'] for r in audits]==['J_joint','A_sh','S_cov']
        assert all(r['zero_rgb_exact'] for r in audits) and all(r['isolated_center_check']['maxabs']==0 for r in audits[1:])
        write(out/'routing_reuse.json',dict(source=str(previous_config),source_sha256=sha(previous_config),route_audit_sha256=sha(prior_route),reason='Routing and dependencies unchanged; previous LR-table roundoff failure occurred before any Adam. Reuse passed derivative/SGD checks without repeating updates.'))
    for arm in ([] if reuse else ['J_joint','A_sh','S_cov']):
        model=load_model(parent,m);restore_global_rng(model.checkpoint['rng']);ident=initial_identity(model);rg=digest_state(rng())
        assert_state_equal(model.checkpoint['model'][12],model.g.optimizer.state_dict());assert_state_equal(model.checkpoint['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
        with torch.no_grad():
            debit('engineering','routing',aux=1);rgb=render_model(model,cam)['render'].clone()
        if zero_identity is None:zero_identity=(ident,rg);zero_rgb=rgb.clone()
        assert (ident,rg)==zero_identity and torch.equal(rgb,zero_rgb)
        policy=policy_info(model,arm);named=shared.all_named(model);chosen=selected(model,arm)
        for x in named.values():x.grad=None
        debit('engineering','routing',rgb=1);raw=render_model(model,cam)['render']
        loss=(downsample(raw,lr.shape[-2:])-lr).abs().mean()+model.g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight);loss.backward()
        before={n:None if x.grad is None else x.grad.clone() for n,x in named.items()}
        debit('engineering','routing',rgb=1);sr=p['training']['sr_weight']*(render_model(model,cam)['render']-teacher).abs().mean()
        for x in named.values():x.grad=None
        sr.backward(retain_graph=True);wanted={n:None if x.grad is None else x.grad.clone() for n,x in named.items()}
        for x in named.values():x.grad=None
        sr.backward(retain_graph=True)
        ordinary_replay={n:0. if wanted[n] is None else float((x.grad-wanted[n]).abs().max()) for n,x in named.items()}
        for n,x in named.items():x.grad=None if before[n] is None else before[n].clone()
        actual_increment=backward_prior(sr,model,arm,audit=True);checks=[]
        for n,x in named.items():
            if n not in chosen:
                expected=before[n];assert (expected is None and x.grad is None) or (expected is not None and x.grad is not None and torch.equal(x.grad,expected)),n
                delta=0.
            else:
                if arm=='J_joint':
                    # The production J call is exactly loss.backward(), with no gradient filter.
                    # Repeated CUDA derivatives are recorded descriptively, not mistaken for an identity gate.
                    expected=before[n] if wanted[n] is None else wanted[n] if before[n] is None else before[n]+wanted[n]
                    delta=0. if expected is None else float((x.grad-expected).abs().max())
                else:
                    increment=actual_increment[n]
                    expected=before[n] if increment is None else increment if before[n] is None else before[n]+increment
                    assert (expected is None and x.grad is None) or (expected is not None and x.grad is not None and torch.equal(x.grad,expected)),n
                    delta=0.
            checks.append(dict(name=n,allowed=n in chosen,LR_gradient_none=before[n] is None,SR_reference_none=wanted[n] is None,after_none=x.grad is None,maxabs=delta,ordinary_SR_replay_maxabs=ordinary_replay[n],reference='actual autograd.grad SR derivative from this call; exact accumulation' if arm!='J_joint' else 'unfiltered torch backward; repeated CUDA derivative descriptive only',exact_outside=n not in chosen))
        drift=None
        if arm!='J_joint':
            for x in named.values():x.grad=None
            with torch.no_grad():
                debit('engineering','routing',aux=1);centers=effective_state(model,cam.time)['xyz'].clone()
            debit('engineering','routing',rgb=1);sr=p['training']['sr_weight']*(render_model(model,cam)['render']-teacher).abs().mean();backward_prior(sr,model,arm)
            debit('engineering',arm+'_SR_only_SGD',updates=1)
            with torch.no_grad():
                for opt in [model.g.optimizer,model.child_optimizer]:
                    for group in opt.param_groups:
                        for x in group['params']:
                            if x.grad is not None:x.add_(x.grad,alpha=-group['lr'])
                debit('engineering','routing',aux=1);after=effective_state(model,cam.time)['xyz'];assert torch.equal(after,centers)
                drift=dict(maxabs=float((after-centers).abs().max()),optimizer='one disposable SR-only SGD; no Adam history',LR_geometry_still_enabled=True)
        audits.append(dict(arm=arm,zero_identity=ident,zero_global_rng_sha256=rg,zero_rgb_exact=True,policy=policy,gradient_checks=checks,isolated_center_check=drift))
        write(OUT/'routing_audit.json',dict(status='in_progress',rows=audits))
        del model,named,chosen,raw,sr,before,wanted;torch.cuda.empty_cache()
    def run(arm,repeat,resume,stop,name):
        dest=out/name;cmd=[sys.executable,'-u',str(HERE/'train.py'),'--method',arm,'--repeat',repeat,'--resume',str(resume),'--stop',str(stop),'--out',str(dest),'--engineering']
        with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        return dest/f'checkpoint_{stop}.pt'
    replays=[];limits=p['engineering']
    for repeat,arm in [('1','J_joint'),('1','A_sh'),('1','S_cov'),('2','S_cov')]:
        prefix=f'r{repeat}_{arm}';direct=run(arm,repeat,parent,6002,prefix+'_direct');mid=run(arm,repeat,parent,6001,prefix+'_prefix');resumed=run(arm,repeat,mid,6002,prefix+'_resume');repeated=run(arm,repeat,parent,6002,prefix+'_repeat')
        x=load_model(direct,m);comp={}
        for label,path in [('resume',resumed),('repeat',repeated)]:
            y=load_model(path,m);assert_state_equal(x.checkpoint['samplers'],y.checkpoint['samplers']);assert digest_state(x.checkpoint['rng'])==digest_state(y.checkpoint['rng']);assert x.checkpoint['metadata']['schedule_sha256']==y.checkpoint['metadata']['schedule_sha256']
            maximum=maxdiff((x.g.capture(),x.children.state_dict(),x.child_optimizer.state_dict()),(y.g.capture(),y.children.state_dict(),y.child_optimizer.state_dict()))
            with torch.no_grad():
                debit('engineering','replay_render',aux=2);delta=render_model(x,cam)['render']-render_model(y,cam)['render']
            comp[label]=dict(tensor_maxabs=maximum,meanabs=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()),samplers_exact=True,global_rng_exact=True)
            del y
        r,t=comp['resume'],comp['repeat'];tests=dict(tensor=r['tensor_maxabs']<=max(limits['tensor_floor'],limits['replay_envelope']*t['tensor_maxabs']),mean=r['meanabs']<=limits['render_meanabs'],rms=r['rmse']<=limits['render_rmse'],rms_envelope=r['rmse']<=max(limits['render_rmse_floor'],limits['replay_envelope']*t['rmse']))
        replays.append(dict(repeat=repeat,arm=arm,comparisons=comp,tests=tests));write(out/'replay.json',dict(rows=replays,limits=limits));assert all(tests.values()),(arm,repeat,tests)
        del x;torch.cuda.empty_cache()
    assert sources()==source
    write(OUT/'routing_audit.json',dict(status='passed',rows=audits,protocol_sha256=sha(OUT/'protocol.json'),sources=source))
    write(out/'complete.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),parent_sha256=p['parent']['sha256'],sources=source,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],fixtures=read(OUT/'budget.json'),replays=replays,seconds=time.time()-started,planned_updates=26))
if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'engineering/failed.json',dict(status='failed',traceback=traceback.format_exc()));raise
