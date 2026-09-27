"""Eight independent old/new RGB backward and named Adam contracts."""
import gc
import argparse
import time
import numpy as np
from context import *

def gradients(model,loss):
    named={n:p for n,p in all_named(model).items() if p.requires_grad}
    values=torch.autograd.grad(loss,list(named.values()),allow_unused=True)
    return {n:None if v is None else cpu(v) for n,v in zip(named,values)}

def snapshot(renderer,o,m,lr,target):
    p=paths();model=motion.load_model(p['start'],m,restore_rng=True)
    model.g.update_learning_rate(13201)
    cam=evalmod.render_camera(m,o,0);named=all_named(model)
    initial=dict(state=digest((model.g.capture(),model.children.state_dict(),model.child_optimizer.state_dict())),
                 rng=digest(rng()),buffers=digest(dict(model.children.named_buffers())))
    out={}
    pred=render(model,cam,renderer)['render']
    out['forward']=dict(rgb=cpu(pred))
    out['LR']=gradients(model,(downsample(pred,lr.shape[-2:])-lr).abs().mean())
    pred=render(model,cam,renderer)['render']
    out['SR']=gradients(model,.1*(pred-target).abs().mean())
    out['regularization']=gradients(model,model.g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight))
    model.g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
    pred=render(model,cam,renderer)['render']
    loss=(downsample(pred,lr.shape[-2:])-lr).abs().mean()
    reg=model.g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)
    (loss+reg).backward()
    pred=render(model,cam,renderer)['render'];(.1*(pred-target).abs().mean()).backward()
    out['merged']={n:None if v.grad is None else cpu(v.grad) for n,v in named.items()}
    model.g.optimizer.step();model.child_optimizer.step()
    out['parameters']={n:cpu(v) for n,v in named.items()}
    out['adam']={}
    for n,q in optimizer_named(model).items():
        for k,v in q['state'].items():
            if torch.is_tensor(v):out['adam'][n+'/'+k]=v
    with torch.no_grad():out['updated_render']={'rgb':cpu(render(model,cam,renderer)['render'])}
    out['identity']=initial
    del model,named,pred;gc.collect();torch.cuda.empty_cache()
    return out

def compare(a,b):
    rows={}
    for stage in a:
        if stage=='identity':continue
        assert set(a[stage])==set(b[stage])
        for name,x in a[stage].items():
            y=b[stage][name];key=stage+'::'+name
            if x is None or y is None:
                rows[key]=dict(stage=stage,name=name,none_equal=(x is None)==(y is None),maxabs=0. if x is y else None,reference_maxabs=0.)
                continue
            x=x.double();y=y.double();diff=x-y;xx=float(x.square().sum());yy=float(y.square().sum());dd=float(diff.square().sum())
            rows[key]=dict(stage=stage,name=name,group=policy.group_name(name.split('/')[0]),none_equal=True,
                maxabs=float(diff.abs().max()),reference_maxabs=max(float(x.abs().max()),float(y.abs().max())),
                l2=dd**.5,relative_l2=dd**.5/max(xx**.5,yy**.5,1e-30),
                cosine=float((x*y).sum())/max((xx*yy)**.5,1e-30) if xx*yy else None,
                left_l2=xx**.5,right_l2=yy**.5,numel=x.numel())
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delegated',action='store_true');a=parser.parse_args()
    p=paths();protocol=read(OUT/'protocol.json');dest=OUT/('renderer_delegated_v1' if a.delegated else 'renderer_backward_v1');dest.mkdir(exist_ok=False)
    choices=['legacy_direct_v1','legacy_direct_v1'] if a.delegated else protocol['renderer_choices']
    guard_training_images();torch.set_num_threads(4);m=load_manifest(p['manifest'])
    by={(e['camera'],e['frame']):e for e in read(p['teacher'])['entries']}
    observations=[o for o in m['observations'] if o['camera_id'] in protocol['probe_cameras'] and o['frame_index'] in protocol['parity_frames']]
    rows=[];start=time.time()
    for o in observations:
        tag=f"{o['camera_id']}_{o['frame_index']:04d}";d=dest/tag;d.mkdir()
        e=by[o['camera_id'],o['frame_index']];lp=Path(m['_root'])/o['lr_path'];tp=Path(m['_root'])/e['relative_path']
        assert sha256(lp)==o['lr_sha256'] and sha256(tp)==e['sha256']
        lr=image_tensor(lp);teacher=image_tensor(tp);snap=[]
        for renderer in choices:
            for repeat in range(2):snap.append(snapshot(renderer,o,m,lr,teacher))
        assert all(s['identity']==snap[0]['identity'] for s in snap)
        rr=[compare(snap[0],snap[1]),compare(snap[2],snap[3])];t=protocol['parity'];eps=np.finfo(np.float32).eps
        bounds={k:max(t['repeat_multiplier']*max(q[k]['maxabs'] or 0. for q in rr),
                       t['ulp_multiplier']*eps*max(q[k]['reference_maxabs'] for q in rr),t['absolute_floor']) for k in rr[0]}
        # This immutable bound is written before any cross-implementation comparison.
        write_json(d/'repeat_envelope.json',dict(registered_before_cross_difference=True,bounds=bounds,ordinary_repeats=rr,protocol_sha256=sha256(OUT/'protocol.json')))
        cross=compare(snap[0],snap[2]);failed=[]
        for k,v in cross.items():
            v['bound']=float(bounds[k]);v['passes']=bool(v['none_equal'] and v['maxabs'] is not None and v['maxabs']<=bounds[k])
            if not v['passes']:failed.append(k)
        write_json(d/'comparison.json',dict(cross=cross,failed=failed,identity=snap[0]['identity'],
            snapshot_hashes=[{k:digest(v) for k,v in s.items()} for s in snap]))
        row=dict(camera=o['camera_id'],frame=o['frame_index'],passed=not failed,failed_names=failed,
                 max_forward=max(v['maxabs'] or 0 for v in cross.values() if v['stage']=='forward'),source_path=str(d/'comparison.json'))
        rows.append(row);write_json(dest/'progress.json',dict(completed=len(rows),rows=rows));print(tag,not failed,len(failed),flush=True)
        del snap,rr,cross,lr,teacher;gc.collect()
    result=dict(status='passed' if all(r['passed'] for r in rows) else 'not_equivalent_within_registered_repeat_envelope',rows=rows,
        independent_updates=32,observations=8,renderer_ids=choices,modules=imports(),seconds=time.time()-start,
        gpu=torch.cuda.get_device_name(),checkpoint_sha256=sha256(p['start']),source_sha256=sha256(__file__),
        interpretation='Named gradients and Adam state, not just final PSNR; bounds fixed from within-entry ordinary repeats before cross comparison.')
    write_json(dest/'complete.json',result)
    if not a.delegated:write_json(OUT/'renderer_parity.json',result)

if __name__=='__main__':main()
