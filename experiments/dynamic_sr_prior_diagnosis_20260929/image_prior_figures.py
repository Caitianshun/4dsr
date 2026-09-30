"""Export scientific evidence figures and numerical/cache integrity checks."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_prior_common import *
from scipy.ndimage import binary_dilation


def depth_maps(keys):
    # Only the existing legal B/scale/crop outputs and legal three-view tracks enter.
    geometry=read(OUT/'geometry/complete.json');support={}
    for r in geometry['rows']:
        z=np.load(ROOT/r['path'])
        for i,c in enumerate(r['cameras']):support[c,r['frame']]=z['xy'+str(i)][z['third_valid']]
    directory=OUT/'depth_diagnostic_maps';directory.mkdir(exist_ok=True);rows=[]
    for c,f in keys:
        z=np.load(OLD/'depth_legal'/f'{c}_{f:04d}.npz');base=z['lr'];cal=np.load(OLD/'depth_legal'/f'{c}_0000.npz')['lr'];den=max(float(np.quantile(cal,.75)-np.quantile(cal,.25)),1e-8);acc=np.zeros_like(base);count=np.zeros_like(base)
        for dy,dx in [(0,8),(8,0),(0,-8),(-8,0)]:
            bb=np.roll(base,(dy,dx),(0,1));bd=base-bb
            for field in ['scale','crop']:
                q=z[field];dd=q-np.roll(q,(dy,dx),(0,1));good=np.isfinite(dd)
                if dy>0:good[:dy]=False
                if dy<0:good[dy:]=False
                if dx>0:good[:,:dx]=False
                if dx<0:good[:,dx:]=False
                acc+=np.where(good,np.exp(-np.abs(bd-dd)/(.1*den)),0);count+=good
        confidence=np.divide(acc,count,out=np.zeros_like(acc),where=count>0);grid=np.zeros_like(base,dtype=bool);xy=support.get((c,f),np.empty((0,2)))
        if len(xy):
            ij=np.rint(xy).astype(int);ij[:,0]=ij[:,0].clip(0,335);ij[:,1]=ij[:,1].clip(0,251);grid[ij[:,1],ij[:,0]]=True
        nearby=binary_dilation(grid,iterations=3);path=directory/f'{c}_{f:04d}.npz';np.savez_compressed(path,LR_perturbation_confidence=confidence,LR_third_view_support_pixels=grid,LR_third_view_support_3px=nearby);rows.append(dict(camera=c,frame=f,path=str(path.relative_to(ROOT)),sha256=sha(path),mean_LR_perturbation_confidence=float(confidence.mean()),third_view_supported_pixel_fraction=float(grid.mean()),third_view_3px_fraction=float(nearby.mean())))
    c,f='cam12',40;z=np.load(OLD/'depth_legal'/f'{c}_{f:04d}.npz');q=np.load(directory/f'{c}_{f:04d}.npz');fig,axs=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,array,title in zip(axs,[z['lr'],q['LR_perturbation_confidence'],q['LR_third_view_support_3px']],['Raw B relative inverse-depth score','LR scale/crop perturbation confidence','Independent third-view support (3 px)']):
        im=ax.imshow(array,cmap='viridis');ax.set_title(title,fontsize=10);ax.axis('off');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('High resolution-stability confidence does not prove correct geometry');fig.savefig(OUT/'depth_confidence_example.png',dpi=170,bbox_inches='tight');plt.close(fig)
    write(directory/'complete.json',dict(status='completed',rows=rows,source='Existing legal depth cache and same-time legal LR three-view tracks only; no HR image/estimate read by this function.',confidence='Average over scale/crop and four8px neighbor offsets of exp(-abs(local_depth_difference_B - local_depth_difference_perturbed)/(0.1*camera_frame0_B_IQR)); no per-frame fit/minmax; missing crop/border neighbors excluded.',use='Diagnostic maps only, not formal training weights; perturbation agreement is stability, not correctness; third-view unmatched pixels unknown.'))
    csvwrite(OUT/'depth_confidence_coverage.csv',rows)


def reliability_summary():
    s=read(OUT/'summary.json');d=s['depth'];f=s['flow']['aggregate'];t=s['texture']['aggregate'];rows=[]
    for b in ['A','B','SwinIR']:
        rows.append(dict(prior='depth',source=b,HR_estimate_stability_Spearman=d['resolution_comparison'][b]['spearman_with_HR_estimate']['mean'],independent_LR_order_accuracy=d['third_view_geometry'][b]['inverse_depth_order_accuracy'],independent_support_3px_fraction=d['coverage']['mean_within_3px_fraction'],applicability='Sparse relative ordering / boundaries; metric-depth regression unsupported',cam01_lamp_wall_error_support='unknown: no same-location held-out observations in original92'))
    for b in ['A','B','Farneback']:
        for reg in ['static_proxy','motion_interior','visibility_boundary_proxy']:
            a=f[b][reg];rows.append(dict(prior='flow',source=b,region=reg,region_fraction=a['region_fraction']['mean'],LR_photo_improvement=a['LR_photometric_improvement']['mean'],LR_census_improvement=a['LR_census_improvement']['mean'],FB_residual_LRpx=a['forward_backward_LRpx']['mean'],applicability='Conditional local motion correspondence; no unconditional full-image flow loss',cam01_lamp_wall_error_support='unknown; dynamic subset does not explain static low-frequency wall error'))
    for b in ['B','SwinIR','Video7']:rows.append(dict(prior='texture',source=b,raw_image_PSNR92=t[b]['PSNR']['mean'],low_band_MSE=t[b]['MSE_low']['mean'],mid_band_MSE=t[b]['MSE_mid']['mean'],high_band_MSE=t[b]['MSE_high']['mean'],applicability='SwinIR remains best candidate details; chosen LR anchoring T must be trained/evaluated',cam01_lamp_wall_error_support='Prior quality alone cannot repair reconstruction support or geometry'))
    csvwrite(OUT/'prior_reliability.csv',rows)
    s['controller_decision_context']=dict(selected_G=False,reason='Controller complete development120 audit: Async2 cam01 low band is92.33% of MSE; lamp/wall ROI17.36% area contributes69.67% error. Current92 training-only prior observations provide no same-location verified support there. Local motion evidence remains positive but narrow; do not generalize as all priors failing.',controller_supplied_numbers_are_external_to_this_audit=True,formal_G_training_cache_built=False)
    s['artifacts']={p.name:sha(p) for p in OUT.glob('*.csv')};write(OUT/'summary.json',s)
    cards=OUT/'prior_cards.md';text=cards.read_text();marker='\n## 控制器最终适用性决定\n'
    text=text.split(marker)[0]+marker+'\n本轮G不进入正式训练：控制器的完整开发120观察诊断显示，cam01低频占其MSE的92.33%，灯具墙角ROI以17.36%面积贡献69.67%误差。当前原图92观察不含cam00/01，未给这一主要区域提供同位置独立支持；不能用少量可靠运动或全图HR相关率替代。局部运动证据仍有价值，保留为后续有明确动态错误覆盖时的候选。未生成G正式训练缓存。\n\n固定置信度图见depth_confidence_example.png、flow_spatial_example.png；覆盖曲线见flow_coverage_curves.png。depth_diagnostic_maps中的LR扰动一致性与第三视角支持分开存储，明确区分“稳定”和“有观测支持”。\n'
    cards.write_text(text)

def main():
    m,obs,keys=setup();rows=list(csv.DictReader((OUT/'flow_confidence_coverage.csv').open()));fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,region in enumerate(['motion_interior','visibility_boundary_proxy']):
        for j,metric in enumerate(['mean_LR_photo_improvement','mean_FB_LRpx']):
            for b,color in [('A','#2563eb'),('B','#dc2626'),('Farneback','#16a34a')]:
                xs=[.1,.25,.5,.75,1.];ys=[np.mean([float(r[metric]) for r in rows if r['branch']==b and r['region']==region and float(r['requested_coverage'])==x]) for x in xs];axs[i,j].plot(np.array(xs)*100,ys,'o-',label=b,color=color)
            axs[i,j].set_title(region.replace('_',' '));axs[i,j].set_xlabel('Retained confidence-ranked region (%)');axs[i,j].set_ylabel('LR photometric gain vs zero flow' if j==0 else 'Forward/backward residual (LR px)');axs[i,j].grid(alpha=.2)
            if j==0:axs[i,j].axhline(0,color='gray',linewidth=.8)
    axs[0,0].legend();fig.suptitle('Only LR observations define confidence; positive photometric gain is better');fig.savefig(OUT/'flow_coverage_curves.png',dpi=170);plt.close(fig)
    c,f,g='cam12',40,42;A=np.load(OUT/'flow_diagnostic_maps'/f'{c}_{f:04d}_{g:04d}.npz');src=imfloat(image(obs,c,f));dst=imfloat(image(obs,c,g));B=np.load(OUT/'flow_legal'/f'B_{c}_{f:04d}_{g:04d}.npz');F=resize_flow(B['forward'],(252,336));bw,v=warp(resize_flow(B['backward'],(252,336)),F);warped,_=warp(dst,F);mag=np.linalg.norm(F,axis=-1);fb=np.linalg.norm(F+bw,axis=-1);conf=np.exp(-fb/(.5+.05*mag))*np.exp(-np.abs(src-warped).mean(2)/.05)*v
    fig,axs=plt.subplots(1,4,figsize=(13,3.3),constrained_layout=True);axs[0].imshow(src);axs[0].set_title(f'Legal LR {c}: {f} -> {g}');im=axs[1].imshow(A['magnitude'],vmin=0,vmax=3,cmap='magma');axs[1].set_title('A flow magnitude (LR px)');fig.colorbar(im,ax=axs[1],shrink=.75);im=axs[2].imshow(conf,vmin=0,vmax=1,cmap='viridis');axs[2].set_title('B confidence from LR only');fig.colorbar(im,ax=axs[2],shrink=.75);labels=A['static_proxy'].astype(int)+A['motion_interior'].astype(int)*2+A['boundary_proxy'].astype(int)*3;from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors=['#999999','#245aa4','#f49a25','#b73785'];axs[3].imshow(labels,vmin=0,vmax=3,cmap=ListedColormap(colors));axs[3].set_title('LR-defined strata (diagnostic)');axs[3].legend(handles=[Patch(color=co,label=la) for co,la in zip(colors,['Unclassified','Static proxy','Motion interior','Boundary proxy'])],loc='lower right',fontsize=7,framealpha=.9)
    for ax in axs:ax.axis('off')
    fig.savefig(OUT/'flow_spatial_example.png',dpi=170);plt.close(fig)
    depth_maps(keys)
    reliability_summary()
    # Coordinate sanity: displacement and spatial grid both rescale; pad is not resize.
    z=np.empty((252,336,2),np.float32);z[...,0]=2;z[...,1]=-3;r=resize_flow(z,(1008,1344));rr=resize_flow(r,z.shape[:2]);coordinate_error=float(np.abs(rr-z).max());assert coordinate_error==0
    assert np.all(r[...,0]==8) and np.all(r[...,1]==-12)
    fi=read(OUT/'flow_index.json');assert len(fi['rows'])==48
    for record in fi['rows']:
        path=ROOT/record['path'];assert sha(path)==record['sha256'];a=np.load(path)
        for key in ['forward','backward']:assert a[key].dtype==np.float32 and np.isfinite(a[key]).all()
    depths(keys);texture=read(OUT/'texture_summary.json');assert texture['max_DCT_partition_error']<1e-7
    write(OUT/'integrity.json',dict(status='passed',depth_A_B_C_observations=92,flow_pairs=16,flow_predictions=96,flow_cache_hashes_verified=48,coordinate_constant_translation_roundtrip_max_error=coordinate_error,texture_DCT_max_abs_partition_error=texture['max_DCT_partition_error'],figures={p.name:sha(p) for p in OUT.glob('*.png')},source_sha256=sha(__file__),HR_training_weights=False,source_identifiers={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('image_prior_*.py')}))
if __name__=='__main__':main()
