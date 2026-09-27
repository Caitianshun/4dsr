"""Stable float64 diagnostic from a projected ellipsoid factor, no covariance clamp."""
import math
import numpy as np
from context import *
from fixed_model import bake

def audit(model,baked,camera,old_file,out_file,regions):
    effective=model if baked else bake(model,float(camera.time))
    # Preserve the actual FP32 activated factor as input; subsequent diagnostic
    # projection is float64. SVD avoids subtracting near-equal covariance terms.
    A=motion.frame(effective.logscale,effective.quaternion).cpu().numpy().astype(np.float64)
    x=effective.xyz.cpu().numpy().astype(np.float64);V=camera.world_view_transform.cpu().numpy().astype(np.float64)
    t=x@V[:3,:3]+V[3,:3];z=t[:,2];tx=math.tan(camera.FoVx/2);ty=math.tan(camera.FoVy/2)
    xc=np.clip(t[:,0]/z,-1.3*tx,1.3*tx)*z;yc=np.clip(t[:,1]/z,-1.3*ty,1.3*ty)*z
    fx=camera.image_width/(2*tx);fy=camera.image_height/(2*ty);J=np.zeros((len(x),2,3))
    J[:,0,0]=fx/z;J[:,1,1]=fy/z;J[:,0,2]=-fx*xc/(z*z);J[:,1,2]=-fy*yc/(z*z)
    factor=J@V[:3,:3].T@A;finite=np.isfinite(factor).all((1,2));physical=np.full((len(x),2),np.nan)
    physical[finite]=np.linalg.svd(factor[finite],compute_uv=False)[:,::-1]
    native_axes=np.sqrt(physical**2+.3);old=np.load(old_file);center=old['projected_center'];candidate=(old['radii']>0)&finite;n=int(old['base_count'])
    np.savez_compressed(out_file,physical_axis_minor_major=physical,with_native_floor_axis_minor_major=native_axes,projected_factor_float64=factor,actual_factor_float32=A.astype(np.float32),xyz=x,center=center,radii=old['radii'],base_count=n)
    rows=[]
    for name,(x0,y0,x1,y1) in {'full':[0,0,camera.image_width,camera.image_height],**regions}.items():
        mask=candidate&(center[:,0]>=x0)&(center[:,0]<x1)&(center[:,1]>=y0)&(center[:,1]<y1);row=dict(region=name)
        for kind,sel in [('base',np.arange(len(x))<n),('child',np.arange(len(x))>=n)]:
            q=mask&sel;row[kind+'_count']=int(q.sum());row[kind+'_axes_median']=np.median(native_axes[q],axis=0).tolist() if q.any() else None
            row[kind+'_axes_quantiles_p10_p50_p90_p99']=np.quantile(native_axes[q],[.1,.5,.9,.99],axis=0).tolist() if q.any() else None
        rows.append(row)
    diff=np.abs(native_axes[candidate]-old['axis_minor_major_hr'][candidate]);neg=int((old['covariance_eigenvalues_before_floor'][candidate,0]<0).sum())
    return dict(path=str(out_file),sha256=sha256(out_file),old_path=str(old_file),old_sha256=sha256(old_file),rows=rows,
        old_negative_eigenvalue_candidates=neg,candidates=int(candidate.sum()),native_axis_abs_difference_quantiles=np.quantile(diff,[.5,.9,.99,1],axis=0).tolist(),
        source_sha256=sha256(__file__),note='FP32 actual activated 3D factor projected in float64; SVD singular values are physical axes. Native covariance floor added only for comparison. No training/rendering changes, no reconstructed clamped covariance.')
