"""Fixed eight-observation sampling and posed Gaussian structure diagnostics.

Depth moments and rendered contribution weights are observational summaries,
not geometric truth. Different topologies are compared on one fixed 3D grid.
"""
import argparse,gc,math,time,traceback
from structure_common import *
import numpy as np
import torch
import torch.nn.functional as F

Q=np.array([0.,.01,.1,.5,.9,.99,1.])
def quantiles(v,w=None):
 v=np.asarray(v).reshape(-1);ok=np.isfinite(v)
 if w is None:return np.quantile(v[ok],Q).tolist() if ok.any() else None
 w=np.asarray(w).reshape(-1);ok&=np.isfinite(w)&(w>0)
 if not ok.any():return None
 v,w=v[ok],w[ok];idx=np.argsort(v);v,w=v[idx],w[idx];cdf=(np.cumsum(w)-w*.5)/w.sum();return np.interp(Q,cdf,v).tolist()
def summary(v,w):return dict(count_quantiles=quantiles(v),contribution_quantiles=quantiles(v,w))

def moments_and_weights(cam,s,motion):
 from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
 xyz,cov,op=[s[k].detach() for k in ['xyz','cov','opacity']]
 z=(torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)@cam.world_view_transform.to(xyz))[:,2]
 colors=torch.stack((torch.ones_like(z),z,z.square()),1).contiguous().requires_grad_(True)
 settings=GaussianRasterizationSettings(image_height=int(cam.image_height),image_width=int(cam.image_width),tanfovx=math.tan(cam.FoVx/2),tanfovy=math.tan(cam.FoVy/2),bg=torch.zeros(3,device=xyz.device),scale_modifier=1.,viewmatrix=cam.world_view_transform.to(xyz),projmatrix=cam.full_proj_transform.to(xyz),sh_degree=0,campos=cam.camera_center.to(xyz),prefiltered=False,debug=False)
 image,radii,native=GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=torch.zeros_like(xyz),shs=None,colors_precomp=colors,opacities=op.contiguous(),scales=None,rotations=None,cov3D_precomp=motion.packed_covariance(cov))
 # RGB compositing is linear in the precomputed colors. Gradient of summed
 # first channel is exactly sum_pixels T_i alpha_i, under native cutoffs.
 weights=torch.autograd.grad(image[0].sum(),colors)[0][:,0].detach()
 return image.detach(),radii.detach(),native.detach(),weights

def moment_arrays(m):
 a,m1,m2=m;a_safe=a.clamp_min(1e-6);z=m1/a_safe;var=m2/a_safe-z.square();valid=a>1e-6;tol=1e-5+1e-5*z.square()
 arrays=dict(moments_raw=m.numpy(),alpha=a.numpy(),z_normalized=z.numpy(),variance_raw=var.numpy(),valid=valid.numpy())
 st=dict(valid_fraction=float(valid.float().mean()),alpha_quantiles=quantiles(a),z_valid_quantiles=quantiles(z[valid]),variance_valid_quantiles=quantiles(var[valid]),negative_variance_fraction=float((var[valid]<0).float().mean()),substantial_negative_variance_fraction=float((var[valid]<-tol[valid]).float().mean()),alpha_gt_one=float((a>1).float().mean()),alpha_lt_zero=float((a<0).float().mean()))
 return arrays,st

def mse(x,y):return float(np.mean((np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64))**2))
def measure(raw,gt,ev,metric):
 pred=ev.legacy.image_array(raw);q=ev.legacy.spatial_metrics(pred,gt,np.zeros(gt.shape[:2],bool),metric)['full'];return dict(psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],mse=mse(pred,gt))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=list(registry()));a=ap.parse_args();start=time.time();out=OUT/'structure';out.mkdir(parents=True,exist_ok=True)
 motion,ev=setup();from n3dv_data import load_manifest
 from common import resized_camera,downsample,UPSTREAM
 from utils.sh_utils import eval_sh
 import lpips
 torch.set_num_threads(4);p=read(OLD/'protocol.json');m=load_manifest(local(p['manifest']['path']));obs=[o for o in m['observations'] if o['camera_id'] in ['cam00','cam01'] and o['frame_index'] in [0,40,80,118]];assert len(obs)==8
 metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False);rows=[];cell=6.776811717207905/128
 write(out/'protocol.json',dict(observations=[(o['camera_id'],o['frame_index']) for o in obs],models=a.models,world_coordinates='same manifest camera/initialization conventions checked through checkpoint manifest_sha; no transform/ICP',grid_origin=[0,0,0],grid_cell_world_units=cell,grid_support='sparse unbounded integer floor(xyz/cell); visible radii>0 and actual alpha compositing contribution independently reported',moment_operator='native HR raw [A,M1,M2]; area 4x4 before normalization; no raw variance clamp',contribution='gradient d sum_pixel rendered_color0 / d per_Gaussian_color0, equals sum_pixel native T_i alpha_i',sample_weighting='count and true alpha-compositing contribution; opacity is not used as substitute',fixed_view_color='same camera/time SH color, with native lower clamp',information_boundary='All structure/sampling outputs diagnostic_only; HR model is not geometric truth',covariance_floor='native adds 0.3 I in output-pixel covariance at each canvas; no opacity compensation',two_hr='2HR raw rendering then non-overlap 2x2 area pooling; then metric clamp',lr_bicubic='native LR rendering clamp then bicubic align_corners=False antialias=True then clamp',source=sha(__file__)))
 for name in a.models:
  cp=registry()[name]['checkpoint'];assert sha(local(cp['path']))==cp['sha256'];model=motion.load_model(local(cp['path']),m);model.g._deformation.eval();cached_states={};modelout=out/name;modelout.mkdir(exist_ok=True)
  for o in obs:
   stem=f"{o['camera_id']}_{o['frame_index']:04d}";receipt=modelout/(stem+'.json')
   if receipt.exists():rows.append(read(receipt));continue
   with torch.no_grad():
    cam=ev.render_camera(m,o,0);frame=o['frame_index'];st=effective_state(model,cam.time,motion);t0=time.perf_counter();lr=motion.render_model(model,resized_camera(cam,252,336))['render'];torch.cuda.synchronize();lr_s=time.perf_counter()-t0
    t0=time.perf_counter();twice=motion.render_model(model,resized_camera(cam,2016,2688))['render'];torch.cuda.synchronize();two_s=time.perf_counter()-t0;area=F.avg_pool2d(twice[None],2,2)[0];del twice
    hr=torch.from_numpy(np.load(OUT/'observables'/name/(stem+'.npz'))['rgb_raw']).cuda();hr_lr=downsample(hr,(252,336));bic=F.interpolate(lr.clamp(0,1)[None],size=(1008,1344),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
    gt=ev.legacy.read_rgb(Path(m['_root'])/o['hr_path']);gtlr=ev.legacy.read_rgb(Path(m['_root'])/o['lr_path']);qualities={label:measure(x,gt,ev,metric) for label,x in [('direct_hr',hr),('lr_bicubic',bic),('two_hr_area',area)]}
    sampling=dict(quality=qualities,native_LR_vs_DHR_mse=mse(lr.clamp(0,1).cpu(),hr_lr.cpu()),native_LR_to_trueLR_mse=mse(lr.clamp(0,1).cpu().permute(1,2,0),gtlr),DHR_to_trueLR_mse=mse(hr_lr.cpu().permute(1,2,0),gtlr),render_seconds=dict(lr=lr_s,two_hr=two_s),two_hr_vs_direct_hr_mse=mse(area.clamp(0,1).cpu(),hr.clamp(0,1).cpu()))
   mom,radii,native,weights=moments_and_weights(cam,st,motion);torch.cuda.synchronize()
   with torch.no_grad():
    hrar,hrst=moment_arrays(mom.cpu());lrar,lrst=moment_arrays(F.avg_pool2d(mom[None],4,4)[0].cpu());w=weights.cpu().numpy();rad=radii.cpu().numpy();xyz=st['xyz'].cpu().numpy();cov=st['cov'].cpu().double();eig=torch.linalg.eigvalsh(cov).numpy();op=st['opacity'].cpu().numpy()[:,0];sh=st['sh'].cpu().numpy();dirs=F.normalize(st['xyz']-cam.camera_center.to(st['xyz'])[None],dim=1);color=(eval_sh(model.g.active_sh_degree,st['sh'].transpose(1,2),dirs)+.5).clamp_min(0).cpu().numpy();shenergy=np.sum(sh[:,1:]**2,axis=(1,2));scale=np.sqrt(np.maximum(eig,1e-30));visible=rad>0
    # A contribution sum parity proves the derivative weights describe the
    # native compositing used for these moments, not projected-opacity guesses.
    contribution_parity=abs(float(weights.sum())-float(mom[0].sum()))/max(float(mom[0].sum()),1e-10)
    attrs=dict(points=len(xyz),visible_points=int(visible.sum()),nonzero_contribution_points=int((w>0).sum()),contribution_weight_sum=float(w.sum()),alpha_pixel_sum=float(mom[0].sum()),weight_alpha_relative_error=contribution_parity,negative_cov_eigen_fraction=float((eig<0).mean()),opacity=summary(op,w),anisotropy=summary(scale[:,-1]/scale[:,0],w),screen_radius=summary(rad,w),sh_nonDC_energy=summary(shenergy,w),log_scale=[summary(np.log(scale[:,j]),w) for j in range(3)],covariance_eigenvalues=[summary(eig[:,j],w) for j in range(3)],fixed_view_rgb=[summary(color[:,j],w) for j in range(3)])
    vox=np.floor(xyz/cell).astype(np.int64);uniq,inv=np.unique(vox,axis=0,return_inverse=True);counts=np.bincount(inv,minlength=len(uniq));visible_counts=np.bincount(inv,weights=visible.astype(float),minlength=len(uniq));mass=np.bincount(inv,weights=w,minlength=len(uniq))
    samplepath=modelout/(stem+'_sampling.npz');save_npz(samplepath,lr_raw=lr.cpu().numpy(),two_hr_area_raw=area.cpu().numpy(),hr_D_lr=hr_lr.cpu().numpy())
    momentpath=modelout/(stem+'_moments.npz');save_npz(momentpath,**{'hr_'+k:v for k,v in hrar.items()},**{'area_'+k:v for k,v in lrar.items()},native_unnormalized_first_moment=native.cpu().numpy())
    pointpath=modelout/(stem+'_points.npz');save_npz(pointpath,contribution=w,radii=rad,fixed_view_rgb=color,voxel=uniq,voxel_count=counts,voxel_visible_count=visible_counts,voxel_contribution=mass)
    statepath=modelout/f'state_{frame:04d}.npz'
    if not statepath.exists():save_npz(statepath,xyz=xyz,covariance=st['cov'].cpu().numpy(),opacity=op,sh=sh)
    row=dict(model=name,camera=o['camera_id'],frame=frame,time=o['time'],checkpoint=cp,point_count=len(xyz),sampling=sampling,moments_hr=hrst,moments_area_lr=lrst,attributes=attrs,artifacts={kind:dict(path=str(path),sha256=sha(path)) for kind,path in [('sampling',samplepath),('moments',momentpath),('points',pointpath),('posed_state',statepath)]},information_boundary='diagnostic_only',hr_model_is_geometric_truth=False)
    write(receipt,row);rows.append(row);print(name,stem,'done',round(time.time()-start,1),flush=True)
   del st,hr,lr,area,mom,weights;torch.cuda.empty_cache()
  del model;gc.collect();torch.cuda.empty_cache()
 # Fixed-grid differences are descriptive distributions; no point IDs matched.
 comparisons=[]
 for o in obs:
  stem=f"{o['camera_id']}_{o['frame_index']:04d}"
  for other in ['LR6k','HR6k','U6000','r1_J1']:
   x=np.load(out/other/(stem+'_points.npz'));y=np.load(out/'r1_Async2'/(stem+'_points.npz'))
   xm={tuple(v):float(w) for v,w in zip(x['voxel'],x['voxel_contribution']) if w>0};ym={tuple(v):float(w) for v,w in zip(y['voxel'],y['voxel_contribution']) if w>0};keys=set(xm)|set(ym);xs=sum(xm.values());ys=sum(ym.values());l1=sum(abs(xm.get(k,0)/xs-ym.get(k,0)/ys) for k in keys);inter=set(xm)&set(ym)
   comparisons.append(dict(camera=o['camera_id'],frame=o['frame_index'],reference=other,current='r1_Async2',visible_mass_voxel_IoU=len(inter)/len(keys),normalized_contribution_mass_L1=l1,role='descriptive, not geometry accuracy'))
 write(out/'results.json',dict(status='completed',rows=rows,grid_comparisons=comparisons,parameter_updates=0,seconds=time.time()-start,gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_cuda_gb=torch.cuda.max_memory_allocated()/1e9))
 # Camera equal means within the fixed four frames; these are sample diagnostics.
 aggregate={}
 for name in a.models:
  rr=[r for r in rows if r['model']==name];agg={}
  for camera in ['cam00','cam01','both']:
   subset=[r for r in rr if camera=='both' or r['camera']==camera];agg[camera]={mode:{k:float(np.mean([r['sampling']['quality'][mode][k] for r in subset])) for k in ['psnr','ssim','lpips','mse']} for mode in ['direct_hr','lr_bicubic','two_hr_area']}
  aggregate[name]=agg
 write(out/'sampling_summary.json',dict(status='completed',aggregate=aggregate,note='fixed eight observations/model; not full120 main table; matched raw-render clamp arithmetic',num_observations=len(rows)))
 write(out/'complete.json',dict(status='completed',results_sha256=sha(out/'results.json'),observations=len(rows),parameter_updates=0,seconds=time.time()-start))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'structure/failed.json',dict(error=traceback.format_exc()));raise
