"""CPU-only posed-structure comparisons, projection support and ROI accounting."""
import csv,math
from structure_common import *
import numpy as np
from PIL import Image
import cv2
from scipy.stats import spearmanr
from audit_structure_sampling import quantiles

def rgb(p):return np.asarray(Image.open(p).convert('RGB')).astype(np.float32)/255.
def masked(v,m):return dict(count=int(m.sum()),median=float(np.median(v[m])) if m.any() else None,p90=float(np.quantile(v[m],.9)) if m.any() else None)
def main():
 out=OUT/'structure';m=read(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json');roi=read(OUT/'spectrum/roi_protocol.json')['regions_by_camera_xyxy_exclusive'];base=read(out/'results.json');rows=[];cross=[];parity=[];historical=list(csv.DictReader(open(OLD/'metrics_per_frame.csv')));lookup={(r['endpoint'],r['camera'],int(r['frame'])):r for r in historical};labels={'LR6k':'LR-direct-HRrender','HR6k':'HR-direct-6k','U6000':'U6000','r1_J1':'r1_J1','r1_Async2':'r1_Async2'}
 for name in registry():
  for cam in ['cam00','cam01']:
   for frame in [0,40,80,118]:
    stem=f'{cam}_{frame:04d}';mom=np.load(out/name/(stem+'_moments.npz'));z=mom['hr_z_normalized'];alpha=mom['hr_alpha'];variance=mom['hr_variance_raw'];valid=(alpha>.5)&np.isfinite(z)&(z>0);thickness=np.sqrt(np.maximum(variance,0))/np.maximum(z,1e-6);raw=np.load(OUT/'observables'/name/(stem+'.npz'))['rgb_raw'];pred=np.clip(raw,0,1).transpose(1,2,0);gt=rgb(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/hr'/cam/f'{frame:04d}.png');e2=(pred-gt)**2;err=e2.mean(2);old=lookup[labels[name],cam,frame];psnr=float(-10*np.log10(max(float(err.mean()),1e-12)));parity.append(dict(model=name,camera=cam,frame=frame,old_psnr=float(old['psnr']),new_psnr=psnr,abs_difference=abs(psnr-float(old['psnr']))))
    regions={'full':np.ones(z.shape,bool)};used=np.zeros(z.shape,bool)
    for rn,(x0,y0,x1,y1) in roi.get(cam,{}).items():
     mask=np.zeros(z.shape,bool);mask[y0:y1,x0:x1]=True;mask&=~used;regions[rn]=mask;used|=mask
    regions['rest']=~used
    point=np.load(out/name/(stem+'_points.npz'));state=np.load(out/name/f'state_{frame:04d}.npz');vis=point['radii']>0;eig=np.linalg.eigvalsh(state['covariance'].astype(np.float64));scale=np.sqrt(np.maximum(eig,1e-30));visattrs=dict(opacity=quantiles(state['opacity'][vis]),anisotropy=quantiles((scale[:,-1]/scale[:,0])[vis]),log_scale=[quantiles(np.log(scale[:,i])[vis]) for i in range(3)],covariance_eigenvalues=[quantiles(eig[:,i][vis]) for i in range(3)])
    values={}
    for rn,mask in regions.items():
     vv=mask&valid;values[rn]=dict(pixel_fraction=float(mask.mean()),mse_contribution=float(err[mask].sum(dtype=np.float64)/err.size),alpha_mean=float(alpha[mask].mean()),valid_alpha_depth_fraction=float(vv.sum()/mask.sum()),relative_ray_thickness=masked(thickness,vv),relative_thickness_gt_0p1_fraction=float(((thickness>.1)&vv).sum()/max(vv.sum(),1)),normalized_z=masked(z,vv))
    rows.append(dict(model=name,camera=cam,frame=frame,regions=values,visible_count_attribute_quantiles=visattrs))
    # Same-time cross-view: report support & depth layering, never across-time.
    target='cam01' if cam=='cam00' else 'cam00';tm=np.load(out/name/f'{target}_{frame:04d}_moments.npz');k=np.asarray(m['cameras'][cam]['K_hr']);kt=np.asarray(m['cameras'][target]['K_hr']);T=np.asarray(m['cameras'][target]['w2c'])@np.asarray(m['cameras'][cam]['c2w']);yy,xx=np.mgrid[0:z.shape[0]:4,0:z.shape[1]:4];zz=z[::4,::4];src=np.stack(((xx-k[0,2])*zz/k[0,0],(yy-k[1,2])*zz/k[1,1],zz),-1);dst=src@T[:3,:3].T+T[:3,3];zzp=dst[:,:,2];xp=dst[:,:,0]/np.maximum(zzp,1e-8)*kt[0,0]+kt[0,2];yp=dst[:,:,1]/np.maximum(zzp,1e-8)*kt[1,1]+kt[1,2]
    remap=lambda a:cv2.remap(a.astype(np.float32),xp.astype(np.float32),yp.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    ta=remap(tm['hr_alpha']);tz=remap(tm['hr_z_normalized']);tv=remap(tm['hr_variance_raw']);inside=(xp>=0)&(xp<z.shape[1]-1)&(yp>=0)&(yp<z.shape[0]-1)&(zzp>.2);support=inside&valid[::4,::4]&(ta>.5)&(tz>0);reldiff=(zzp-tz)/np.maximum(tz,1e-6);target_thick=np.sqrt(np.maximum(tv,0))/np.maximum(tz,1e-6)
    curves=[]
    for threshold in [.01,.05,.1,.2]:
     mask=support&(thickness[::4,::4]<threshold)&(target_thick<threshold);curves.append(dict(max_relative_ray_thickness=threshold,coverage=float(mask.mean()),relative_depth_disagreement=masked(np.abs(reldiff),mask),source_behind_target_fraction=float(((reldiff>.05)&mask).sum()/max(mask.sum(),1)),source_ahead_target_fraction=float(((reldiff<-.05)&mask).sum()/max(mask.sum(),1))))
    cross.append(dict(model=name,source_camera=cam,target_camera=target,frame=frame,sample_stride=4,in_bounds_positive_depth_fraction=float(inside.mean()),alpha_support_fraction=float(support.mean()),curves=curves,note='Normalized mean depth on potentially multilayer rays; mismatches include true visibility/occlusion and geometric inconsistency. No truth or metric-depth guarantee.'))
 # Depth shape comparison at identical pixels, no free scale/shift/ICP.
 depth=[]
 for cam in ['cam00','cam01']:
  for f in [0,40,80,118]:
   curr=np.load(out/'r1_Async2'/f'{cam}_{f:04d}_moments.npz')
   for ref in ['LR6k','HR6k','U6000','r1_J1']:
    old=np.load(out/ref/f'{cam}_{f:04d}_moments.npz');x=curr['area_z_normalized'];y=old['area_z_normalized'];valid=(curr['area_alpha']>.5)&(old['area_alpha']>.5)&(x>0)&(y>0);dx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3);ex=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3);ey=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3);den=np.sqrt((dx*dx+dy*dy)*(ex*ex+ey*ey));edge=valid&(den>1e-8);cos=(dx*ex+dy*ey)/np.maximum(den,1e-8)
    depth.append(dict(camera=cam,frame=f,reference=ref,current='r1_Async2',alpha_support_fraction=float(valid.mean()),unscaled_relative_Z_difference=masked(np.abs(x-y)/np.maximum(y,1e-6),valid),depth_Spearman=float(spearmanr(x[valid],y[valid]).statistic),depth_gradient_cosine_mean=float(cos[edge].mean()),note='Agreement of estimated model observables; HR reference is not geometry truth'))
 prior_status=read(ROOT/'output/dynamic_sr_20260920/mip2d_diagnostic/status.json')
 write(out/'postprocess.json',dict(status='completed',roi_rows=rows,same_time_crossview=cross,model_depth_comparisons=depth,native_RGB_parity=dict(rows=parity,max_abs_psnr_difference=max(r['abs_difference'] for r in parity)),prior_mip2d_diagnostic=prior_status,independent_raw_image_depth_crosscheck=dict(status='NA',reason='The required native/bicubic/HR image priors cover legal training cameras; fixed eight posed-model moments cover development cameras. No same-camera/time raw-image depth cache registered for these eight observations.'),point_ID_comparison=False,per_frame_alignment=False,parameter_updates=0,source_sha256=sha(__file__)))
 print('postprocess complete',max(r['abs_difference'] for r in parity),flush=True)
if __name__=='__main__':main()
