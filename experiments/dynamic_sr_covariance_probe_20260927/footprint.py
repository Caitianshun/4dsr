"""Stable major-axis buckets measured by native front-to-back alpha contribution."""
import argparse
import math
from common_cov import *

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--checkpoint',type=Path,required=True);a=pa.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);reads,blocked=guard_images(p);tick=time.time();_,data,masks=load_data(p);model=fixed.load_baked(a.checkpoint);parent=initial(p);before=model_hash(model);rows=[];audits=[];shape=[];cksha=sha(a.checkpoint);pcov=cpu(covariance(parent)).double();cv=cpu(covariance(model)).double();relative=torch.linalg.matrix_norm(cv-pcov)/torch.linalg.matrix_norm(pcov).clamp_min(1e-30)
    rois={c:{n:[4*v for v in xy] for n,xy in r.items()} for c,r in p['rois'].items()};labels=['sigma<=1','1<sigma<=2','2<sigma<=4','sigma>4']
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    with torch.no_grad():
        for c in [p['a'],p['b']]:
            camera=data[c]['camera'];A=motion.frame(model.logscale,model.quaternion).cpu().numpy().astype(np.float64);x=model.xyz.cpu().numpy().astype(np.float64);V=camera.world_view_transform.cpu().numpy().astype(np.float64);t=x@V[:3,:3]+V[3,:3];z=t[:,2];tx=math.tan(camera.FoVx/2);ty=math.tan(camera.FoVy/2);xc=np.clip(t[:,0]/z,-1.3*tx,1.3*tx)*z;yc=np.clip(t[:,1]/z,-1.3*ty,1.3*ty)*z
            J=np.zeros((len(x),2,3));fx=camera.image_width/(2*tx);fy=camera.image_height/(2*ty);J[:,0,0]=fx/z;J[:,1,1]=fy/z;J[:,0,2]=-fx*xc/z**2;J[:,1,2]=-fy*yc/z**2
            factor=J@V[:3,:3].T@A;assert np.isfinite(factor).all();axes=np.linalg.svd(factor,compute_uv=False);major=np.sqrt(axes[:,0]**2+.3);minor=np.sqrt(axes[:,1]**2+.3);ratio=major/minor;
            for part,sl in [('base',slice(0,model.base_count)),('child',slice(model.base_count,None))]:
                visible=z[sl]>0;stats={'covariance_relative_frobenius':relative[sl].numpy(),'major_sigma':major[sl][visible],'minor_sigma':minor[sl][visible],'axis_ratio':ratio[sl][visible]}
                shape.append(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=cksha,camera=c,region='all_points_covariance_and_positive_z_axes',base_or_child=part,**{name+'_'+q:float(np.quantile(val,quant)) for name,val in stats.items() for q,quant in [('median',.5),('p95',.95)]}))
            buckets=np.digitize(major,[1,2,4],right=True);group=buckets+4*(np.arange(len(x))>=model.base_count)
            settings=GaussianRasterizationSettings(image_height=1008,image_width=1344,tanfovx=tx,tanfovy=ty,bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=camera.world_view_transform.cuda(),projmatrix=camera.full_proj_transform.cuda(),sh_degree=model.degree,campos=camera.camera_center.cuda(),prefiltered=False,debug=False)
            cov=torch.cat((motion.covariance(model.logscale[:model.base_count],model.quaternion[:model.base_count]),motion.covariance(model.logscale[model.base_count:],model.quaternion[model.base_count:])))
            def aux(color):
                COUNTS['auxiliary_forwards']+=1
                return GaussianRasterizer(raster_settings=settings)(means3D=model.xyz,means2D=torch.zeros_like(model.xyz),shs=None,colors_precomp=color,opacities=model.opacity.sigmoid(),scales=None,rotations=None,cov3D_precomp=motion.packed_covariance(cov))[0]
            parts=[]
            for start in [0,3,6]:
                color=torch.stack([torch.tensor(group==i,device='cuda',dtype=torch.float32) for i in range(start,start+3)],1);parts.append(aux(color))
            mass=torch.cat(parts,0)[:8];total=aux(torch.ones((len(x),3),device='cuda'))[0];difference=(mass.sum(0)-total).abs();assert float(difference.max())<=1e-5
            areas={'full':torch.ones_like(total,dtype=torch.bool),'M':masks[c]}
            for name,(x0,y0,x1,y1) in rois.get(c,{}).items():
                mask=torch.zeros_like(total,dtype=torch.bool);mask[y0:y1,x0:x1]=True;areas[name]=mask
            for region,mask in areas.items():
                denominator=float(total[mask].double().sum())
                for gi in range(8):
                    part='base' if gi<4 else 'child';partden=float(mass[(0 if gi<4 else 4):(4 if gi<4 else 8),:, :][:,mask].double().sum());num=float(mass[gi][mask].double().sum())
                    rows.append(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=cksha,camera=c,frame=40,region=region,base_or_child=part,part=part,low_coverage_fraction=float((total[mask]<.1).double().mean()),mean_alpha=float(total[mask].double().mean()),bucket=labels[gi%4],alpha_mass=num,total_alpha_mass=denominator,part_alpha_mass=partden,global_share=num/denominator if denominator else None,within_part_share=num/partden if partden else None,pixels=int(mask.sum()),point_count=int((group==gi).sum()),sigma_units='HR pixels,1sigma; stable FP64 projected actual FP32 factor plus native0.3 variance'))
            path=a.out/f'{c}_buckets.npz';np.savez_compressed(path,major_sigma=major,minor_sigma=minor,axis_ratio=ratio,bucket=buckets,part=np.arange(len(x))>=model.base_count,alpha_contribution=mass.cpu().numpy(),total_alpha=total.cpu().numpy());audits.append(dict(camera=c,path=str(path),sha256=sha(path),sum_maxabs=float(difference.max()),sum_meanabs=float(difference.double().mean())))
    assert model_hash(model)==before;csvwrite(a.out/'shape_statistics.csv',shape);csvwrite(a.out/'footprint_bins.csv',rows);write(a.out/'complete.json',dict(status='completed',rows=rows,audits=audits,model_hash=before,checkpoint=str(a.checkpoint),checkpoint_sha256=cksha,shape_statistics=shape,parameter_updates=0,seconds=time.time()-tick,counts=COUNTS,HR_reads=0,image_reads=reads,blocked=blocked,source_sha256=sources()))
if __name__=='__main__':main()
