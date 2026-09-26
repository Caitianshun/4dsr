"""Sixteen independent one-step probes; train LR/SwinIR only, never HR.

Each intervention restores U6000 and both Adam histories; no cumulative fitting.
HR-derived region masks, if desired, are joined by a separate diagnostic reader.
"""
import argparse
import os
import time
import numpy as np
from shared import *
from gradient_policy import group_name,appearance


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','checkpoint','teacher-index','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            s=os.fsdecode(args[0]);assert not (s.endswith('.png') and '/hr/' in s),'HR read in legal gradient probe'
    sys.addaudithook(guard)
    m=load_manifest(a.manifest);ev=evaluator();by={(e['camera'],e['frame']):e for e in read(a.teacher_index)['entries']}
    observations=[o for o in m['observations'] if o['camera_id'] in ['cam02','cam06','cam12','cam18'] and o['frame_index'] in [0,40,80,118]];assert len(observations)==16
    rows=[];started=time.time()
    for o in observations:
        camera=ev.render_camera(m,o,0);entry=by[o['camera_id'],o['frame_index']]
        def tensor(path,digest):
            assert sha256(path)==digest
            return torch.from_numpy(ev.legacy.read_rgb(path)).permute(2,0,1).cuda()
        lr=tensor(Path(m['_root'])/o['lr_path'],o['lr_sha256']);sr=tensor(Path(m['_root'])/entry['relative_path'],entry['sha256'])
        model=load_model(a.checkpoint,m,restore_rng=True);model.g.update_learning_rate(13201)
        named={n:v for n,v in all_named(model).items() if v.requires_grad};names=list(named);params=list(named.values())
        pred=render_model(model,camera)['render'];lr_loss=(downsample(pred,lr.shape[-2:])-lr).abs().mean();sr_loss=.1*(pred-sr).abs().mean()
        gl=torch.autograd.grad(lr_loss,params,retain_graph=True,allow_unused=True);gs=torch.autograd.grad(sr_loss,params,allow_unused=True)
        gradients={};groups={n:group_name(n) for n in names}
        for group in sorted(set(groups.values())):
            pairs=[(x,y) for n,x,y in zip(names,gl,gs) if groups[n]==group and x is not None and y is not None]
            ll=sum(float(x.double().square().sum()) for x,y in pairs);ss=sum(float(y.double().square().sum()) for x,y in pairs)
            dot=sum(float((x.double()*y.double()).sum()) for x,y in pairs)
            active=sum(int(((x!=0)&(y!=0)).sum()) for x,y in pairs);conflict=sum(int((x*y<0).sum()) for x,y in pairs)
            gradients[group]=dict(lr_norm=ll**.5,sr_norm=ss**.5,cosine=dot/(ll*ss)**.5 if ll*ss else None,sign_conflict_fraction=conflict/active if active else None,active_elements=active)
        grad_lr={n:None if x is None else x.detach().cpu() for n,x in zip(names,gl)}
        grad_sr={n:None if x is None else x.detach().cpu() for n,x in zip(names,gs)}
        base=pred.detach().clone();del pred,gl,gs,params,named,model;torch.cuda.empty_cache()
        maps={};interventions={}
        for arm in ['LR_only','joint','appearance']:
            model=load_model(a.checkpoint,m,restore_rng=True);model.g.update_learning_rate(13201)
            named={n:v for n,v in all_named(model).items() if v.requires_grad};color=appearance(model)
            before={n:p.detach().clone() for n,p in named.items()}
            for n,p0 in named.items():p0.grad=None if grad_lr[n] is None else grad_lr[n].to(p0.device).clone()
            model.g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight).backward()
            for n,p0 in named.items():
                d=grad_sr[n]
                if d is not None and (arm=='joint' or (arm=='appearance' and n in color)):
                    if p0.grad is None:p0.grad=d.to(p0.device).clone()
                    else:p0.grad.add_(d.to(p0.device))
            model.g.optimizer.step();model.child_optimizer.step()
            with torch.no_grad():
                after=render_model(model,camera)['render'];change=(after-base).abs().mean(0)
                maps[arm]=change.cpu().numpy();maps[arm+'_sr_error_delta']=((after-sr).abs().mean(0)-(base-sr).abs().mean(0)).cpu().numpy()
                update={group:sum(float((p0-before[n]).double().square().sum()) for n,p0 in named.items() if groups[n]==group)**.5 for group in gradients}
                interventions[arm]=dict(render_change_mean=float(change.mean()),lr_l1=float((downsample(after,lr.shape[-2:])-lr).abs().mean()),sr_l1=float((after-sr).abs().mean()),parameter_update_l2=update)
            del model,named,before,after;torch.cuda.empty_cache()
        path=a.out/f"{o['camera_id']}_{o['frame_index']:04d}.npz";np.savez_compressed(path,**maps)
        rows.append(dict(camera=o['camera_id'],frame=o['frame_index'],gradients=gradients,interventions=interventions,maps=path.name,maps_sha256=sha256(path),inputs=dict(lr_sha256=o['lr_sha256'],sr_sha256=entry['sha256'])))
        write_json(a.out/'progress.json',dict(completed=len(rows)));print('completed observation',len(rows),flush=True)
    write_json(a.out/'complete.json',dict(status='completed',rows=rows,checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),source_sha256=sha256(__file__),parameter_groups={n:group_name(n) for n in names},independent_updates=48,observations=16,gpu=torch.cuda.get_device_name(),seconds=time.time()-started,interpretation='Within-group LR/SR directions and actual one-step Adam response, not a cross-unit decomposition of physical error. LR_only still inherits historical Adam momentum; no cumulative training. No HR input.'))


if __name__=='__main__':main()
