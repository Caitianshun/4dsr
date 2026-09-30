"""SplatSuRe same-time selection maps from frozen legal U6000 geometry.

Exact native alpha-compositing contribution sums are recovered as derivatives
with respect to a dummy color channel. No model/optimizer is updated. The source
radius formula excludes +0.3 covariance dilation but retains its 0.1 eigenvalue
discriminant safeguard. The original map min-max fallback is retained explicitly.
"""
from dv_common import *
shared=setup()
import math
import argparse
import numpy as np
import torch
from motion_model import load_model,packed_covariance
from common import resized_camera
from n3dv_data import load_manifest


def scalar_render(camera,state,colors):
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    xyz=state['xyz']; view=camera.world_view_transform.to(xyz.device)
    settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),bg=torch.zeros(3,device=xyz.device),
        scale_modifier=1.,viewmatrix=view,projmatrix=camera.full_proj_transform.to(xyz.device),
        sh_degree=0,campos=camera.camera_center.to(xyz.device),prefiltered=False,debug=False)
    return GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=torch.zeros_like(xyz),
        shs=None,colors_precomp=colors,opacities=state['opacity'].contiguous(),scales=None,rotations=None,
        cov3D_precomp=packed_covariance(state['cov']))


def screen_radius(camera,state):
    xyz=state['xyz']; view=camera.world_view_transform.to(xyz.device)
    t=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)@view
    z=t[:,2]; safez=torch.where(z.abs()>1e-8,z,torch.full_like(z,1e-8))
    tanx,tany=math.tan(camera.FoVx/2),math.tan(camera.FoVy/2)
    fx,fy=camera.image_width/(2*tanx),camera.image_height/(2*tany)
    xz=(t[:,0]/safez).clamp(-1.3*tanx,1.3*tanx)
    yz=(t[:,1]/safez).clamp(-1.3*tany,1.3*tany)
    J=torch.zeros((len(xyz),2,3),device=xyz.device)
    J[:,0,0]=fx/safez;J[:,1,1]=fy/safez
    J[:,0,2]=-fx*xz/safez;J[:,1,2]=-fy*yz/safez
    R=view[:3,:3].T
    cov_cam=R[None]@state['cov']@R.T[None]
    c=J@cov_cam@J.transpose(1,2)
    a,b,d=c[:,0,0],c[:,0,1],c[:,1,1]
    mid=(a+d)/2; root=(mid.square()-(a*d-b.square())).clamp_min(.1).sqrt()
    return 3*(mid+root).clamp_min(0).sqrt(), c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames',nargs='+',type=int,default=list(range(0,120,2)))
    ap.add_argument('--out',type=Path,default=OUT/'legal/texture_maps');a=ap.parse_args()
    torch.set_num_threads(4);start=time.time();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    registry=read(OUT/'audit_manifest.json');p=registry['assets']
    m=load_manifest(bound(p['manifest']));ev=shared.evaluator()
    model=load_model(bound(p['parent']),m)
    obs=[o for o in m['observations'] if o['split']=='train'];cams=sorted(set(o['camera_id'] for o in obs))
    assert len(cams)==19 and not set(cams)&{'cam00','cam01'}
    entries=[]; rows=[]; forwards=0; backwards=0
    for frame in a.frames:
        items=sorted([o for o in obs if o['frame_index']==frame],key=lambda o:o['camera_id'])
        assert len(items)==19 and len({o['time'] for o in items})==1
        with torch.no_grad():state=shared.effective_state(model,items[0]['time'])
        n=len(state['xyz']);vmin=torch.full((n,),1e6,device='cuda');vmax=torch.full((n,),-1.,device='cuda')
        argmax=torch.full((n,),-1,device='cuda',dtype=torch.long);seen=torch.zeros(n,device='cuda',dtype=torch.long)
        cameras=[];contribs=[];radius_all=[]
        for j,o in enumerate(items):
            camera=ev.render_camera(m,o,j);w,h=m['resolutions']['lr'];camera=resized_camera(camera,h,w);cameras.append(camera)
            color=torch.zeros((n,3),device='cuda',requires_grad=True)
            rgb,radii,_=scalar_render(camera,state,color);forwards+=1
            weight=torch.autograd.grad(rgb[0].sum(),color)[0][:,0].detach();backwards+=1
            with torch.no_grad():radius,_=screen_radius(camera,state)
            assert torch.isfinite(weight).all() and (weight>=0).all()
            visible=(radii>0)&(weight>0)&torch.isfinite(radius)&(radius>0)
            seen+=visible.long();update=visible&(radius>vmax)
            vmax=torch.where(update,radius,vmax);argmax=torch.where(update,j,argmax)
            vmin=torch.where(visible& (radius<vmin),radius,vmin)
            contribs.append(weight);radius_all.append(radius)
        ratio=(vmax/vmin).clamp_min(1);score=torch.sigmoid((ratio-1.1)/.05);score[seen<3]=0
        for j,(o,camera) in enumerate(zip(items,cameras)):
            with torch.no_grad():
                colors=torch.stack((score,(argmax==j).float(),torch.ones_like(score)),1).contiguous()
                image,_,_=scalar_render(camera,state,colors);forwards+=1
                raw=1-image[0]+image[1];fallback=bool(raw.max()<.9)
                if fallback:raw=(raw-raw.min())/(raw.max()-raw.min()+1e-6)
                weight=raw.clamp(0,1)
            path=a.out/o['camera_id']/f'{frame:04d}.npy';path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,weight.cpu().numpy().astype(np.float32),allow_pickle=False)
            entries.append(dict(camera=o['camera_id'],frame=frame,**entry(path)))
            contribution=contribs[j];vis=contribution>0
            rows.append(dict(camera=o['camera_id'],frame=frame,mean_weight=float(weight.mean()),
                fraction_gt_half=float((weight>.5).float().mean()),alpha_mean=float(image[2].mean()),
                minmax_fallback=fallback,contributing_points=int(vis.sum()),
                contribution_weighted_true_radius=float((contribution*radius_all[j]).sum()/contribution.sum().clamp_min(1e-12)),
                under_three_view_contribution=float(contribution[seen<3].sum()/contribution.sum().clamp_min(1e-12))))
        print(json.dumps(dict(frame=frame,maps=len(entries),seconds=time.time()-start)),flush=True)
    write(a.out/'prior_index.json',dict(status='completed',entries=entries,rows=rows,
        information_boundary='legal_train_LR_and_U6000_only',parent=p['parent'],manifest=p['manifest'],
        fixed_strategy='offline U6000 per camera and time; never recompute while training',
        selection=dict(ratio_threshold=1.1,sigmoid_scale=.05,min_views=3,
            formula='clip(1-alpha_composite(score)+alpha_composite(is_largest_view),0,1)',
            fallback='original min-max only if map max < 0.9',normalization='none; full pixel loss mean',
            output_grid='native LR; nearest to HR as original train.py',
            radius='SplatSuRe subpixel radius before +0.3 dilation; retains discriminant floor 0.1'),
        confidence='no independent confidence applied; demand is not confidence',
        source='https://github.com/pranav-asthana/SplatSuRe',script=entry(Path(__file__)),
        gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        rgb_forward_equivalents=forwards,backward_color_contribution_passes=backwards,
        parameter_updates=0,seconds=time.time()-start))

if __name__=='__main__':main()
