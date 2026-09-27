"""Raw teacher/H diagnostics, independent HR evaluation and footprint evidence."""
import argparse
import csv
import math
import time
import numpy as np
from PIL import Image
from context import *
from renderer import render_model,RENDERER_ID
from fixed_model import bake,render_baked,load_baked

VERSION='evidence_probe_raw_v1'
def scalar_error(a,b):
    d=(a-b).double();return dict(l1=float(d.abs().mean()),mse=float(d.square().mean()))

def footprint(model,camera,regions,outfile):
    """Native 0.3 pixel covariance floor; actual alpha-composited base/child mass."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    n=model.base_count;x=model.xyz
    cov=torch.cat((motion.covariance(model.logscale[:n],model.quaternion[:n]),motion.covariance(model.logscale[n:],model.quaternion[n:])))
    V=camera.world_view_transform.cuda();t=x@V[:3,:3]+V[3,:3];z=t[:,2]
    tx=math.tan(camera.FoVx/2);ty=math.tan(camera.FoVy/2)
    xc=torch.clamp(t[:,0]/z,-1.3*tx,1.3*tx)*z;yc=torch.clamp(t[:,1]/z,-1.3*ty,1.3*ty)*z
    J=torch.zeros((len(x),2,3),device=x.device);fx=camera.image_width/(2*tx);fy=camera.image_height/(2*ty)
    J[:,0,0]=fx/z;J[:,1,1]=fy/z;J[:,0,2]=-fx*xc/z.square();J[:,1,2]=-fy*yc/z.square()
    R=V[:3,:3].T;cv=R[None]@cov@R.T[None];c2=J@cv@J.transpose(-1,-2)
    finite=torch.isfinite(c2).all(dim=(1,2));ev=torch.zeros((len(x),2),device=x.device);ev[finite]=torch.linalg.eigvalsh(c2[finite]);raw_axes=ev.clamp_min(0).sqrt();axes=(ev+.3).clamp_min(0).sqrt()
    clip=torch.cat((x,torch.ones_like(x[:,:1])),1)@camera.full_proj_transform.cuda();ndc=clip[:,:2]/clip[:,3:4]
    center=(ndc+1)*torch.tensor([camera.image_width,camera.image_height],device=x.device)/2-.5
    colors=torch.zeros((len(x),3),device=x.device);colors[:n,0]=1;colors[n:,1]=1;colors[:,2]=1
    settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),tanfovx=tx,tanfovy=ty,bg=torch.zeros(3,device=x.device),scale_modifier=1.,viewmatrix=V,projmatrix=camera.full_proj_transform.cuda(),sh_degree=model.degree,campos=camera.camera_center.cuda(),prefiltered=False,debug=False)
    mass,radii,_=GaussianRasterizer(raster_settings=settings)(means3D=x,means2D=torch.zeros_like(x),shs=None,colors_precomp=colors,opacities=model.opacity.sigmoid(),scales=None,rotations=None,cov3D_precomp=motion.packed_covariance(cov))
    sums=float((mass[0]+mass[1]-mass[2]).abs().max());visible=(radii>0)&finite
    np.savez_compressed(outfile,projected_center=center.cpu().numpy(),covariance_eigenvalues_before_floor=ev.cpu().numpy(),covariance_eigenvalues_native=(ev+.3).cpu().numpy(),axis_minor_major_hr=axes.cpu().numpy(),axis_minor_major_before_floor=raw_axes.cpu().numpy(),radii=radii.cpu().numpy(),base_count=n,alpha_mass_area=torch.nn.functional.avg_pool2d(mass[None],4,4)[0].cpu().numpy())
    rows=[]
    for label,box in {'full':[0,0,camera.image_width,camera.image_height],**regions}.items():
        x0,y0,x1,y1=box;mask=visible&(center[:,0]>=x0)&(center[:,0]<x1)&(center[:,1]>=y0)&(center[:,1]<y1)
        vals=mass[:,y0:y1,x0:x1].mean((1,2));row=dict(region=label,base_alpha_mass=float(vals[0]),child_alpha_mass=float(vals[1]),total_alpha_mass=float(vals[2]),group_sum_maxabs=sums)
        for kind,sel in [('base',torch.arange(len(x),device=x.device)<n),('child',torch.arange(len(x),device=x.device)>=n)]:
            v=mask&sel;row[kind+'_candidate_count']=int(v.sum());row[kind+'_minor_major_hr_quantiles']=torch.quantile(axes[v].double(),torch.tensor([.1,.5,.9,.99],device=x.device,dtype=torch.float64),dim=0).tolist() if v.any() else None
        rows.append(row)
    return dict(rows=rows,sha256=sha256(outfile),path=str(outfile),meaning='Axes are 1-sigma projected standard deviations in HR pixels; native +0.3 covariance floor included. Radius>0 denotes raster candidates, not unoccluded visibility. Alpha mass is exact front-to-back composited group contribution, not per-point visibility.')

def save_png(raw,path):
    Image.fromarray((evalmod.legacy.image_array(raw)*255).round().astype(np.uint8)).save(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['Shared40','Baked40','references'],required=True);a=p.parse_args()
    out=OUT/'probe_evaluation'/a.mode;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time();pa=paths();m=load_manifest(pa['manifest']);root=Path(m['_root']);cfg=read(OUT/'protocol.json')
    if a.mode=='references':
        prior=read(PRIOR/'methods.json')['methods'];oracle=read(ROOT/'output/dynamic_sr_controlled_headroom_20260926/final_v1/methods.json')['methods']
        jobs={'U6000':pa['start'],'shared12000':prior['shared_12000']['checkpoint'],'shared17550':prior['shared_17550']['checkpoint'],'C18000':prior['C_joint_18000']['checkpoint'],'O_privileged18000':oracle['O_HR_PRIVILEGED18000']['checkpoint']}
    else:jobs={a.mode+str(step):OUT/'fixed_time'/a.mode/f'checkpoint_{step}.pt' for step in cfg['fixed_time']['save_steps']}
    teachers={(x['camera'],x['frame']):x for x in read(pa['teacher'])['entries']};rois=read(pa['roi'])['regions_by_camera_xyxy_exclusive']
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False);results={}
    with torch.inference_mode():
        for name,checkpoint in jobs.items():
            dest=out/name;dest.mkdir();identity=sha256(checkpoint);baked=a.mode=='Baked40'
            model=load_baked(checkpoint) if baked else motion.load_model(checkpoint,m)
            if not baked:model.g._deformation.eval()
            fn=render_baked if baked else render_model
            # U6000 supports both the train16 historical panel and frame40 probe.
            obs=[o for o in m['observations'] if ((a.mode!='references' or name=='U6000') and o['frame_index']==40) or (a.mode=='references' and o['split']=='train' and o['camera_id'] in cfg['probe_cameras'] and o['frame_index'] in cfg['frames'])]
            rows=[];footprints=[]
            for i,o in enumerate(obs):
                cam=o['camera_id'];frame=o['frame_index'];camera=evalmod.render_camera(m,o,i);raw=fn(model,camera)['render'];assert torch.isfinite(raw).all();lr=image_tensor(root/o['lr_path']);assert sha256(root/o['lr_path'])==o['lr_sha256'];hp=detail.highpass(raw,lr.shape[-2:]);entry=teachers.get((cam,frame))
                row=dict(run_id=name,parent_run_id='U6000' if a.mode!='references' else 'historical',checkpoint_sha256=identity,renderer_id=RENDERER_ID if not baked else 'baked_same_native_rasterizer_v1',metric_version=VERSION,camera=cam,frame=frame,split=o['split'],source_path=str(checkpoint),privileged_training=name.startswith('O_privileged'))
                row['lr']=scalar_error(downsample(raw,lr.shape[-2:]),lr)
                # Legal teacher/H phase completes before reading evaluation HR.
                if entry:
                    tp=root/entry['relative_path'];assert sha256(tp)==entry['sha256'];teacher=image_tensor(tp);ht=detail.highpass(teacher,lr.shape[-2:]);row['teacher_rgb']=scalar_error(raw,teacher);row['teacher_H']=scalar_error(hp,ht);row['teacher_sha256']=entry['sha256']
                # Independent privileged evaluation; never fed to optimization.
                hrp=root/o['hr_path'];assert sha256(hrp)==o['hr_sha256'];hr=image_tensor(hrp);hrarr=evalmod.legacy.image_array(hr);pred=evalmod.legacy.image_array(raw);zero=np.zeros(hrarr.shape[:2],bool)
                row['hr']=evalmod.legacy.spatial_metrics(pred,hrarr,zero,metric)['full'];row['hr_H']=scalar_error(hp,detail.highpass(hr,lr.shape[-2:]));row['roi']={}
                if entry:row['teacher_hr_H']=scalar_error(ht,detail.highpass(hr,lr.shape[-2:]));row['teacher_hr']=evalmod.legacy.spatial_metrics(evalmod.legacy.image_array(teacher),hrarr,zero,metric)['full']
                for region,box in rois.get(cam,{}).items():
                    x0,y0,x1,y1=box;sl=(slice(y0,y1),slice(x0,x1));row['roi'][region]=evalmod.legacy.spatial_metrics(pred[sl],hrarr[sl],zero[sl],metric)['full']
                    row['roi'][region]['hr_H_l1']=float((hp[:,y0:y1,x0:x1]-detail.highpass(hr,lr.shape[-2:])[:,y0:y1,x0:x1]).abs().double().mean())
                    if entry:row['roi'][region]['teacher_H_l1']=float((hp[:,y0:y1,x0:x1]-ht[:,y0:y1,x0:x1]).abs().double().mean())
                if frame==40 and cam in ['cam00','cam01','cam02','cam06','cam12','cam18']:save_png(raw,dest/f'{cam}_{frame:04d}.png')
                if True:  # Save every registered frame40 and train16 observation.
                    fm=model if baked else bake(model,frame/300);fp=footprint(fm,camera,rois.get(cam,{}),dest/f'footprint_{cam}_{frame:04d}.npz');footprints.append(dict(camera=cam,frame=frame,**fp))
                    if not baked:del fm
                rows.append(row);print(name,cam,frame,flush=True)
            result=dict(rows=rows,footprints=footprints,checkpoint_sha256=identity,checkpoint=str(checkpoint));write_json(dest/'metrics.json',result);results[name]=result
            del model;torch.cuda.empty_cache()
    write_json(out/'complete.json',dict(status='completed',methods=results,metric_version=VERSION,highpass_operator=detail.OPERATOR,parameter_updates=0,gpu=torch.cuda.get_device_name(),source_sha256=sha256(__file__),seconds=time.time()-started,information_boundary='Predictions rendered from calibration/time before reading HR. Teacher RGB/H uses only frozen train19. HR and cam00/01 are evaluation-only. Fixed-time models have no temporal generalization claim.'))

if __name__=='__main__':main()
