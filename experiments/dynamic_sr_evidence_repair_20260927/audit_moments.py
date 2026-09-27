"""Preserve raw bicubic/area moments and audit negative variance and gradients."""
import argparse
import copy
import math
import time
import numpy as np
import torch.nn.functional as F
from context import *

old=module('evidence_old_moments','experiments/dynamic_sr_prior_guidance_20260927/moment_renderer.py')

def reduce(image,operator):
    if operator=='area':return F.avg_pool2d(image[None],4,4)[0]
    if operator=='bicubic':return F.interpolate(image[None],size=(image.shape[1]//4,image.shape[2]//4),mode='bicubic',align_corners=False,antialias=True)[0]
    if operator=='hr':return image
    raise ValueError(operator)

def raw_stats(m,cfg,boundary=None):
    a,m1,m2=m;valid=a>cfg['alpha_valid'];z=m1/a.clamp_min(cfg['alpha_epsilon']);var=m2/a.clamp_min(cfg['alpha_epsilon'])-z.square()
    tol=cfg['variance_absolute_tolerance']+cfg['variance_relative_tolerance']*z.square()
    det=a*m2-m1.square()
    def summary(mask):
        n=int(mask.sum())
        def q(v):return [float(x) for x in torch.quantile(v[mask].double(),torch.tensor([0.,.01,.5,.99,1.],dtype=torch.float64)).tolist()] if n else None
        return dict(pixels=n,negative_variance_fraction=float((var[mask]<0).double().mean()) if n else None,
                    substantial_negative_fraction=float((var[mask]<-tol[mask]).double().mean()) if n else None,
                    variance_quantiles=q(var),determinant_quantiles=q(det),z_quantiles=q(z))
    out=dict(alpha_below_zero_fraction=float((a<0).double().mean()),alpha_above_one_fraction=float((a>1).double().mean()),
        alpha_min=float(a.min()),alpha_max=float(a.max()),valid=summary(valid))
    if boundary is not None:out['boundary_valid']=summary(valid&boundary)
    return out,dict(moments=m.numpy(),alpha=a.numpy(),expected_z=z.numpy(),variance_raw=var.numpy(),determinant=det.numpy(),valid=valid.numpy())

def main():
    parser=argparse.ArgumentParser()
    for k in ['manifest','checkpoint','protocol','out']:parser.add_argument('--'+k,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cfg=read(a.protocol)['moments'];m=load_manifest(a.manifest);started=time.time()
    observations=[o for o in m['observations'] if o['camera_id'] in read(a.protocol)['train_cameras'] and o['frame_index'] in read(a.protocol)['frames']]
    guard_training_images();model=motion.load_model(a.checkpoint,m);rows=[]
    with torch.no_grad():
        for o in observations:
            cam=evalmod.render_camera(m,o,0);s=policy.effective_state(model,cam.time);r=old.render_moments(cam,s['xyz'],s['cov'],s['opacity']);hr=r['moments']
            assert tuple(hr.shape[-2:])==(1008,1344)
            # Boundaries are diagnostic only and never define training reliability.
            small=reduce(hr,'area').cpu();az=small[1]/small[0].clamp_min(cfg['alpha_epsilon']);grad=(az[:,1:]-az[:,:-1]).abs();edge=torch.zeros_like(az,dtype=torch.bool);edge[:,1:]=grad>.02*az[:,1:].abs().clamp_min(1e-6);edge|=(small[0]>.01)&(small[0]<.95)
            arrays={};stats={}
            for mode in ['bicubic','area']:
                mm=reduce(hr,mode).cpu();st,ar=raw_stats(mm,cfg,edge);stats[mode]=st
                for k,v in ar.items():arrays[mode+'_'+k]=v
                mm64=reduce(hr.cpu().double(),mode);st64,_=raw_stats(mm64,cfg,edge);stats[mode+'_float64']=st64
                stats[mode]['float32_vs_float64_moment_maxabs']=float((mm.double()-mm64).abs().max())
            path=a.out/f"{o['camera_id']}_{o['frame_index']:04d}.npz";np.savez_compressed(path,**arrays)
            rows.append(dict(camera=o['camera_id'],frame=o['frame_index'],path=path.name,sha256=sha256(path),statistics=stats))
            print('moments',len(rows),flush=True)
    # Four actual-rasterizer fixtures, including separated sharp depth/alpha edges.
    original=evalmod.render_camera(m,observations[0],0);cam=copy.copy(original);cam.image_height=64;cam.image_width=64;inv=torch.linalg.inv(cam.world_view_transform.cuda())
    def fixture(zs,ops,xs=None,scale=.12):
        xs=[0.]*len(zs) if xs is None else xs
        v=torch.tensor([[x,0.,z,1.] for x,z in zip(xs,zs)],device='cuda').reshape(-1,4)
        xyz=(v@inv)[:,:3];cov=torch.eye(3,device='cuda')[None].repeat(len(zs),1,1)*scale**2
        return old.render_moments(cam,xyz,cov,torch.tensor(ops,device='cuda').reshape(-1,1))['moments']
    fixtures={}
    for name,z,op,x in [('single',[2.],[.5],None),('two_layers',[2.,4.],[.5,.4],None),('empty',[],[],None),('sharp_edge',[1.8,4.],[.99,.85],[-.17,.17])]:
        hr=fixture(z,op,x).detach().cpu();fixtures[name]={mode:raw_stats(reduce(hr,mode),cfg)[0] for mode in ['hr','bicubic','area']}
        fixtures[name]['float64']={mode:raw_stats(reduce(hr.double(),mode),cfg)[0] for mode in ['hr','bicubic','area']}
    def loss(v):
        z,op,ls=v.unbind();coord=torch.stack((z*0+.04,z*0+.025,z,z*0+1))[None];xyz=(coord@inv)[:,:3]
        far=(torch.tensor([[0.,0.,4.,1.]],device='cuda')@inv)[:,:3]
        cov=torch.cat((torch.eye(3,device='cuda')[None]*torch.exp(2*ls),torch.eye(3,device='cuda')[None]*.04))
        hr=old.render_moments(cam,torch.cat((xyz,far)),cov,torch.stack((op,op*0+.4))[:,None])['moments'];mm=reduce(hr,'area');a0,b,c=mm[:,7:9,7:9];depth=b/a0.clamp_min(cfg['alpha_epsilon']);var=c/a0.clamp_min(cfg['alpha_epsilon'])-depth.square()
        return (depth+.2*a0+.05*var).mean()
    v=torch.tensor([2.,.5,math.log(.2)],device='cuda',requires_grad=True);loss(v).backward();analytic=v.grad.tolist();fd=[]
    for i in range(3):
        plus=v.detach().clone();minus=plus.clone();plus[i]+=.001;minus[i]-=.001;fd.append(float((loss(plus)-loss(minus))/.002))
    derivative=[dict(parameter=k,autograd=x,finite_difference=y,passed=abs(x-y)<=max(.003,.02*abs(y))) for k,x,y in zip(['axial_center','opacity','logscale'],analytic,fd)]
    area_valid=all(r['statistics']['area']['valid']['substantial_negative_fraction']==0 for r in rows)
    write_json(a.out/'complete.json',dict(status='completed',rows=rows,fixtures=fixtures,derivatives=derivative,
        area_engineering_passed=area_valid and all(x['passed'] for x in derivative),
        normalization='No clamp of raw variance; statistics restricted to alpha>1e-6. Tolerance fixed in protocol.',
        rgb_unchanged=True,parameter_updates=0,checkpoint_sha256=sha256(a.checkpoint),protocol_sha256=sha256(a.protocol),source_sha256=sha256(__file__),
        gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),seconds=time.time()-started,modules=imports()))

if __name__=='__main__':main()
