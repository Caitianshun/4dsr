"""One registered image-prior diagnostic. All HR comparisons stay diagnostic-only.

A= native LR file/model-supported preprocessing; B= LR bicubic at HR size;
C= HR image. No per-frame depth alignment and no HR-derived training weights.
"""
import argparse, itertools, time
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from scipy.fft import dctn
from image_prior_common import *

def grad(im):
    if im.ndim==3:im=im.mean(2)
    return np.stack([cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)/8],-1)

def boundary_difference(a,b):
    ag=np.linalg.norm(grad(a),axis=-1);bg=np.linalg.norm(grad(b),axis=-1)
    ea=ag>=np.quantile(ag,.9);eb=bg>=np.quantile(bg,.9)
    da=distance_transform_edt(~ea);db=distance_transform_edt(~eb)
    precision=float((db[ea]<=1).mean());recall=float((da[eb]<=1).mean())
    return dict(boundary_proxy_f1_1LRpx=2*precision*recall/max(precision+recall,1e-12),boundary_proxy_symmetric_distance_LRpx=float((db[ea].mean()+da[eb].mean())/2))

def rank_order(a,b,seed=20260929):
    # Same deterministic pixel pairs across every input route, no fitted normalization.
    rng=np.random.default_rng(seed);a=a.ravel();b=b.ravel();i=rng.integers(len(a),size=16384);j=rng.integers(len(a),size=16384);x=a[i]-a[j];y=b[i]-b[j];v=np.isfinite(x)&np.isfinite(y)&(y!=0)
    return float((np.sign(x[v])==np.sign(y[v])).mean()) if v.any() else None

def fit_positive(a,b):
    x=a.ravel().astype(np.float64);y=b.ravel().astype(np.float64);scale=max(float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean())),1e-8)
    return dict(scale=scale,shift=float(y.mean()-scale*x.mean()))

def load_flow_index():
    index=read(OUT/'flow_index.json');cache={}
    for r in index['rows']:
        p=ROOT/r['path'];assert sha(p)==r['sha256'];z=np.load(p);cache[r['camera'],r['first'],r['second'],r['branch']]={k:v for k,v in z.items()}
    return index,cache

def legal_static(obs,c):
    ims=np.stack([imfloat(image(obs,c,f)) for f in [0,40,80,118]])
    return (np.sqrt(ims.var(0).mean(2))<.025)&(ims.max((0,3))<.98)&(ims.min((0,3))>.01)

def depth_audit(keys,data,identities,flow,obs):
    calibration_keys=[key for key in keys if key[1]==0]
    write(OUT/'depth_calibration_protocol.json',dict(status='registered_before_evaluation',calibration_keys=[list(k) for k in calibration_keys],evaluation_keys=[list(k) for k in keys if k[1]!=0],fit='One pooled positive scale+shift per A/B/SwinIR, fit only frame0 across19 training cameras using every fourth pixel. Frozen for all remaining times and cameras. No per-frame fits.',use='diagnostic_only; calibrates relative depth to HR estimator scores, not metric depth; never a training label'))
    target=np.stack([data[k]['C'][::4,::4] for k in calibration_keys]);ref_iqr=float(np.quantile(target,.75)-np.quantile(target,.25));fits={b:fit_positive(np.stack([data[k][b][::4,::4] for k in calibration_keys]),target) for b in ['A','B','SwinIR']}
    write(OUT/'depth_frozen_alignment.json',dict(coefficients=fits,hr_calibration_iqr=ref_iqr,protocol_sha256=sha(OUT/'depth_calibration_protocol.json')))
    rows=[]
    for key in keys:
        c,f=key;z=data[key];C=z['C']
        for b in ['A','B','SwinIR']:
            A=z[b];fit=fits[b];delta=fit['scale']*A+fit['shift']-C
            rows.append(dict(prior='depth',camera=c,frame=f,branch=b,split='calibration' if f==0 else 'evaluation',n_pixels=A.size,spearman_with_HR_estimate=corr(A,C),rank_agreement_with_HR_estimate=rank_order(A,C),raw_RMSE_to_HR_estimate=float(np.sqrt(np.mean((A-C)**2))),frozen_aligned_RMSE_to_HR_estimate=float(np.sqrt(np.mean(delta**2))),frozen_aligned_MAE_over_calibration_HR_IQR=float(np.abs(delta).mean()/ref_iqr),mean_raw_score=float(A.mean()),std_raw_score=float(A.std()),**boundary_difference(A,C)))
    csvwrite(OUT/'depth_image_comparison.csv',rows)
    geom_index=read(OUT/'geometry/complete.json');geometry=[];pooled={b:[] for b in ['A','B','C','SwinIR']};coverage=[]
    for r in geom_index['rows']:
        z=np.load(ROOT/r['path']);good=z['third_valid'];f=r['frame']
        for ci,c in enumerate(r['cameras']):
            points=z['xy'+str(ci)][good];depth=z['depths'][good,ci];n=len(points)
            grid=np.zeros((252,336),bool)
            if n:
                ix=np.rint(points).astype(int);ix[:,0]=ix[:,0].clip(0,335);ix[:,1]=ix[:,1].clip(0,251);grid[ix[:,1],ix[:,0]]=True
            coverage.append(dict(camera=c,frame=f,points=n,unique_pixel_fraction=float(grid.mean()),within_3px_fraction=float(binary_dilation(grid,iterations=3).mean())))
            if n<2:continue
            i,j=np.triu_indices(n,1);dist=np.linalg.norm(points[i]-points[j],axis=1);reldiff=np.abs(depth[i]-depth[j])/np.maximum(depth[i],depth[j]);sel=(dist>=4)&(dist<=80)&(reldiff>=.03);i=i[sel];j=j[sel];target=np.sign(1/depth[i]-1/depth[j])
            for b in ['A','B','C','SwinIR']:
                pred=sample(data[c,f][b],points);delta=pred[i]-pred[j];ok=np.isfinite(delta);correct=(np.sign(delta[ok])==target[ok]);pooled[b].extend(correct.tolist());wrongpoints=points.copy();wrongpoints[:,0]=(wrongpoints[:,0]+48)%336;wrong=sample(data[c,f][b],wrongpoints);wrongdelta=wrong[i]-wrong[j];wg=np.isfinite(wrongdelta)
                geometry.append(dict(prior='depth_geometry',camera=c,frame=f,branch=b,third_verified_points=n,local_order_pairs=int(ok.sum()),inverse_depth_order_accuracy=float(correct.mean()) if correct.size else None,opposite_direction_order_accuracy=float(1-correct.mean()) if correct.size else None,wrong_spatial_order_accuracy=float((np.sign(wrongdelta[wg])==target[wg]).mean()) if wg.any() else None,spearman_to_triangulated_inverse_depth=corr(pred,1/depth),mean_third_reprojection_LRpx=float(z['reprojection_error'][good,2].mean()) if n else None))
    csvwrite(OUT/'depth_geometry.csv',geometry);csvwrite(OUT/'geometry_coverage.csv',coverage)
    temporal=[];static={c:legal_static(obs,c) for c in ['cam02','cam06','cam12','cam18']}
    for c,f,g in PAIRS:
        fl=flow[c,f,g,'A']['forward'];back=flow[c,f,g,'A']['backward'];bw,valid=warp(back,fl);fb=np.linalg.norm(fl+bw,axis=-1);speed=np.linalg.norm(fl,axis=-1);v=valid&(fb<.5+.05*speed);st=v&static[c]&(speed<.25);moving=v&(speed>=.5)
        for b in ['A','B','C','SwinIR']:
            a=data[c,f][b];w,_=warp(data[c,g][b],fl);diff=np.abs(a-w);coef=fits.get(b,dict(scale=1,shift=0));cal_ref=data[c,0][b];den=max(float(np.quantile(cal_ref,.75)-np.quantile(cal_ref,.25)),1e-8)
            # Dynamic depth value change is not scored as temporal error; relative local order only.
            orders=[]
            for dy,dx in [(0,8),(8,0)]:
                sl=np.s_[:252-dy,:336-dx];sr=np.s_[dy:,dx:];mask=moving[sl]&moving[sr];d0=a[sl]-a[sr];d1=w[sl]-w[sr];mask&=np.abs(d0)>.01*den;orders.extend((np.sign(d0[mask])==np.sign(d1[mask])).tolist())
            temporal.append(dict(prior='depth_temporal',camera=c,frame=f,second=g,branch=b,static_proxy_fraction=float(st.mean()),static_raw_MAE_over_frame0_IQR=float(diff[st].mean()/den) if st.any() else None,static_frozen_aligned_MAE_over_calibration_HR_IQR=float((coef['scale']*diff[st]).mean()/ref_iqr) if st.any() else None,moving_support_fraction=float(moving.mean()),moving_local_order_agreement=float(np.mean(orders)) if orders else None,moving_order_pairs=len(orders),dynamic_depth_value_error=None))
    csvwrite(OUT/'depth_temporal.csv',temporal)
    summary=dict(status='completed',cache_identities=identities,observations=len(keys),calibration_observations=len(calibration_keys),evaluation_observations=len(keys)-len(calibration_keys),resolution_comparison={b:{metric:stats([r[metric] for r in rows if r['branch']==b and r['split']=='evaluation']) for metric in ['spearman_with_HR_estimate','rank_agreement_with_HR_estimate','frozen_aligned_MAE_over_calibration_HR_IQR','boundary_proxy_f1_1LRpx']} for b in ['A','B','SwinIR']},third_view_geometry={b:dict(order_pairs=len(vals),inverse_depth_order_accuracy=float(np.mean(vals)) if vals else None) for b,vals in pooled.items()},coverage=dict(mean_unique_pixel_fraction=float(np.mean([x['unique_pixel_fraction'] for x in coverage])),mean_within_3px_fraction=float(np.mean([x['within_3px_fraction'] for x in coverage]))),static_temporal={b:stats([r['static_raw_MAE_over_frame0_IQR'] for r in temporal if r['branch']==b]) for b in ['A','B','C','SwinIR']},scope='Relative hierarchy/boundaries only. Shared-network HR agreement is stability, not truth. Third-view supports are sparse textured locations. Depth drift checked only on legal static proxy; moving depth MAE intentionally NA.')
    write(OUT/'depth_summary.json',summary);return rows+geometry+temporal,summary

def census(im):
    gray=im.mean(2);out=[]
    for dy,dx in itertools.product([-1,0,1],repeat=2):
        if (dy,dx)==(0,0):continue
        out.append((gray>np.roll(gray,(dy,dx),(0,1))).astype(np.float32))
    return np.stack(out,-1)

def flow_audit(flow,index,obs):
    rows=[];curves=[];static={c:legal_static(obs,c) for c in ['cam02','cam06','cam12','cam18']};directory=OUT/'flow_diagnostic_maps';directory.mkdir(exist_ok=True)
    for c,f,g in PAIRS:
        for direction,first,second in [('forward',f,g),('backward',g,f)]:
            other='backward' if direction=='forward' else 'forward';src=imfloat(image(obs,c,first));dst=imfloat(image(obs,c,second));zero=np.abs(src-dst).mean(2);cen0=census(src);cen1=census(dst);czero=np.abs(cen0-cen1).mean(2);A=flow[c,f,g,'A'][direction];mag=np.linalg.norm(A,axis=-1);Abw,Av=warp(flow[c,f,g,'A'][other],A);Afb=np.linalg.norm(A+Abw,axis=-1);motion=mag>=.5;bound=binary_dilation(motion,iterations=2)^binary_erosion(motion,iterations=2);bound|=(np.linalg.norm(grad(mag),axis=-1)>.5)|(Afb>.5+.05*mag)
            interior=binary_erosion(motion,iterations=2)&~bound;sta=static[c]&(mag<.25)&~bound;unknown=~(bound|interior|sta);regions=dict(all=np.ones_like(sta),static_proxy=sta,motion_interior=interior,visibility_boundary_proxy=bound,unclassified=unknown)
            ref=resize_flow(flow[c,f,g,'C'][direction],(252,336));farneback=cv2.calcOpticalFlowFarneback(cv2.cvtColor((src*255).astype(np.uint8),cv2.COLOR_RGB2GRAY),cv2.cvtColor((dst*255).astype(np.uint8),cv2.COLOR_RGB2GRAY),None,.5,3,15,3,5,1.2,0);farback=cv2.calcOpticalFlowFarneback(cv2.cvtColor((dst*255).astype(np.uint8),cv2.COLOR_RGB2GRAY),cv2.cvtColor((src*255).astype(np.uint8),cv2.COLOR_RGB2GRAY),None,.5,3,15,3,5,1.2,0)
            for b in ['A','B','C','Farneback']:
                F=farneback if b=='Farneback' else resize_flow(flow[c,f,g,b][direction],(252,336));back=farback if b=='Farneback' else resize_flow(flow[c,f,g,b][other],(252,336));bw,valid=warp(back,F);yy,xx=np.indices(valid.shape);valid&=(xx>=1)&(xx<335)&(yy>=1)&(yy<251)&(xx+F[...,0]>=1)&(xx+F[...,0]<=334)&(yy+F[...,1]>=1)&(yy+F[...,1]<=250);warped,_=warp(dst,F);wc,_=warp(cen1,F);fb=np.linalg.norm(F+bw,axis=-1);photo=np.abs(src-warped).mean(2);ce=np.abs(cen0-wc).mean(2);confidence=np.exp(-fb/(.5+.05*np.linalg.norm(F,axis=-1)))*np.exp(-photo/.05)*valid;proxy=np.linalg.norm(F-ref,axis=-1);occ=fb>(.5+.05*np.linalg.norm(F,axis=-1));gain=zero-photo
                for region,mask in regions.items():
                    val=mask&valid;N=int(val.sum());row=dict(prior='flow',camera=c,frame=first,second=second,frame_gap=abs(second-first),direction=direction,branch=b,region=region,region_fraction=float(mask.mean()),in_bounds_fraction=float(valid[mask].mean()) if mask.any() else None,valid_pixels=N,proxy_EPE_to_HR_estimate_LRpx=float(proxy[val].mean()) if N else None,forward_backward_LRpx=float(fb[val].mean()) if N else None,fb_inconsistent_proxy_fraction=float(occ[val].mean()) if N else None,LR_photometric_warp_MAE=float(photo[val].mean()) if N else None,LR_photometric_zero_MAE=float(zero[val].mean()) if N else None,LR_photometric_improvement=float(gain[val].mean()) if N else None,LR_census_warp_MAE=float(ce[val].mean()) if N else None,LR_census_zero_MAE=float(czero[val].mean()) if N else None,LR_census_improvement=float((czero-ce)[val].mean()) if N else None,mean_LR_confidence=float(confidence[mask].mean()) if mask.any() else None)
                    if b!='Farneback':
                        native=flow[c,f,g,b][direction];refhr=flow[c,f,g,'C'][direction];proxyhr=np.linalg.norm(resize_flow(native,refhr.shape[:2])-refhr,axis=-1);hmask=cv2.resize(mask.astype(np.uint8),(1344,1008),interpolation=cv2.INTER_NEAREST)>0;row['proxy_EPE_HRgrid_HRpx']=float(proxyhr[hmask].mean()) if hmask.any() else None
                    rows.append(row)
                    if N:
                        # Coverage among this region, ranked only by LR self confidence.
                        ids=np.flatnonzero(val);order=ids[np.argsort(-confidence.ravel()[ids])]
                        for fraction in [.1,.25,.5,.75,1.]:
                            take=order[:max(1,int(len(order)*fraction))];curves.append(dict(camera=c,frame=first,second=second,direction=direction,branch=b,region=region,requested_coverage=fraction,full_image_coverage=len(take)/mask.size,region_coverage=len(take)/max(mask.sum(),1),confidence_min=float(confidence.ravel()[take].min()),mean_proxy_EPE_LRpx=float(proxy.ravel()[take].mean()),mean_FB_LRpx=float(fb.ravel()[take].mean()),mean_LR_photo_improvement=float(gain.ravel()[take].mean()),mean_LR_census_improvement=float((czero-ce).ravel()[take].mean())))
                if b=='A' and direction=='forward':
                    np.savez_compressed(directory/f'{c}_{f:04d}_{g:04d}.npz',confidence=confidence,magnitude=mag,fb=fb,photometric_gain=gain,valid=valid,static_proxy=sta,motion_interior=interior,boundary_proxy=bound,unknown=unknown)
        print('flow_audit',c,f,g,flush=True)
    csvwrite(OUT/'flow_comparison.csv',rows);csvwrite(OUT/'flow_confidence_coverage.csv',curves)
    metrics=['region_fraction','proxy_EPE_to_HR_estimate_LRpx','proxy_EPE_HRgrid_HRpx','forward_backward_LRpx','fb_inconsistent_proxy_fraction','LR_photometric_improvement','LR_census_improvement','mean_LR_confidence']
    aggregated={b:{reg:{met:stats([r.get(met) for r in rows if r['branch']==b and r['region']==reg and r.get(met) is not None]) for met in metrics} for reg in ['all','static_proxy','motion_interior','visibility_boundary_proxy']} for b in ['A','B','C','Farneback']}
    summary=dict(status='completed',pairs=16,directions=32,estimator=index['estimator'],weight_sha256=index['weight_sha256'],aggregate=aggregated,confidence_definition='exp(-FB/(0.5+0.05*|F|))*exp(-LR_photo_MAE/0.05)*in_bounds; branch A/B confidence uses only LR, C confidence diagnostic-only.',region_definition='Same masks for all branches, derived only from A native-LR RAFT. Motion>=.5LRpx; 2pixel morph boundaries plus flow-gradient>.5 or FB>.5+.05mag; eroded motion interior. Static proxy: four-time LR std<.025, unsaturated, mag<.25, not boundary. Remaining pixels unclassified. Masks are diagnostic stratification, not foreground truth.',independent_geometry_motion_support=None,geometry_motion_support_reason='Independent LR three-view support exists at40/80, but no four-view temporal 3D track validation; static geometry is not counted as proof of optical flow.',notes=['Proxy EPE to HR estimate is resolution stability, not ground-truth accuracy.','B/C network grid1008x1344; A252x336 pad to256x336. Both spatial resampling and displacement scaling used.','Photometric and census metrics evaluated on identical native LR observations for A/B/C; one-pixel source/target border excluded for3x3 census support; supports independent of HR reference.','FB inconsistency is an occlusion/failure proxy, not a true occlusion label.','Region aggregation is equal pair/direction means; coverage recorded separately.','Confidence maps remain diagnostic artifacts; not formal training caches.'])
    write(OUT/'flow_summary.json',summary);return rows,summary

def texture_audit(keys,obs):
    teacher={(r['camera'],r['frame']):r for r in read(TEACHER)['entries']};video={(r['camera'],r['frame']):r for r in read(VIDEO)['entries']};rows=[];identities=[];start=time.time()
    h,w=1008,1344;yy,xx=np.indices((h,w));freq=np.maximum(yy/h,xx/w);masks={'DC':freq==0,'low':(freq>0)&(freq<=.25),'mid':(freq>.25)&(freq<=.5),'high':freq>.5}
    for c,f in keys:
        target=imfloat(image(obs,c,f,'hr'));lr=image(obs,c,f);ups=imfloat(cv2.resize(lr,(w,h),interpolation=cv2.INTER_CUBIC));te=teacher[c,f];vp=video[c,f];tp=MANIFEST.parent/te['relative_path'];vpath=ROOT/vp['path'];assert sha(tp)==te['sha256'];assert sha(vpath)==vp['sha256'];sources={'B':ups,'SwinIR':imfloat(cv2.cvtColor(cv2.imread(str(tp)),cv2.COLOR_BGR2RGB)),'Video7':imfloat(cv2.cvtColor(cv2.imread(str(vpath)),cv2.COLOR_BGR2RGB))};identities.append(dict(camera=c,frame=f,HR_sha256=obs[c,f]['hr_sha256'],LR_sha256=obs[c,f]['lr_sha256'],SwinIR_path=str(tp),SwinIR_sha256=te['sha256'],Video7_path=str(vpath),Video7_sha256=vp['sha256']))
        gtg=grad(target);gtmag=np.linalg.norm(gtg,axis=-1);edge=gtmag>max(float(np.quantile(gtmag,.75)),.02);C=dctn(target,axes=(0,1),norm='ortho',workers=4);target_energy={k:float((C[mask]**2).sum()/(3*h*w)) for k,mask in masks.items()}
        for b,im in sources.items():
            delta=im-target;E=dctn(delta,axes=(0,1),norm='ortho',workers=4);cg=grad(im);mag=np.linalg.norm(cg,axis=-1);orient=(cg*gtg).sum(-1)/np.maximum(mag*gtmag,1e-9);bands={k:float((E[mask]**2).sum()/(3*h*w)) for k,mask in masks.items()};mse=float(np.mean(delta**2));bias=delta.mean((0,1));r=dict(prior='texture',camera=c,frame=f,branch=b,RGB_MSE=mse,PSNR=-10*np.log10(max(mse,1e-12)),RGB_bias_R=float(bias[0]),RGB_bias_G=float(bias[1]),RGB_bias_B=float(bias[2]),edge_orientation_cosine=float(orient[edge].mean()) if edge.any() else None,edge_gradient_energy_ratio=float((mag[edge]**2).sum()/max((gtmag[edge]**2).sum(),1e-12)),DCT_partition_error=float(abs(sum(bands.values())-mse)))
            for k,mask in masks.items():
                r['MSE_'+k]=bands[k];r['HR_energy_'+k]=target_energy[k];r['energy_ratio_'+k]=float(((C[mask]+E[mask])**2).sum()/max((C[mask]**2).sum(),1e-12))
            rows.append(r)
        if len(identities)%16==0:print('texture_audit',len(identities),time.time()-start,flush=True)
    csvwrite(OUT/'texture_comparison.csv',rows);write(OUT/'texture_input_index.json',identities)
    # Same-time, independently third-view-verified LR correspondences. Inherent view-dependent colors remain a limitation.
    cross=[]
    for r in read(OUT/'geometry/complete.json')['rows']:
        z=np.load(ROOT/r['path']);good=z['third_valid'];f=r['frame']
        if not good.any():continue
        for b in ['B','SwinIR','Video7','HR']:
            colors=[]
            for i,c in enumerate(r['cameras']):
                if b=='B':im=imfloat(cv2.resize(image(obs,c,f),(1344,1008),interpolation=cv2.INTER_CUBIC))
                elif b=='HR':im=imfloat(image(obs,c,f,'hr'))
                else:
                    p=MANIFEST.parent/teacher[c,f]['relative_path'] if b=='SwinIR' else ROOT/video[c,f]['path'];im=imfloat(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB))
                xy=(z['xy'+str(i)][good]+.5)*4-.5;colors.append(sample(im,xy))
            diff=np.stack([np.abs(colors[i]-colors[j]).mean(1) for i,j in [(0,1),(0,2),(1,2)]])
            cross.append(dict(prior='texture_crossview',cameras=','.join(r['cameras']),frame=f,branch=b,verified_points=int(good.sum()),mean_RGB_difference=float(np.nanmean(diff))))
    csvwrite(OUT/'texture_crossview.csv',cross)
    metrics=['RGB_MSE','PSNR','MSE_DC','MSE_low','MSE_mid','MSE_high','edge_orientation_cosine','edge_gradient_energy_ratio']
    summary=dict(status='completed',observations=len(keys),source_pngs_are_actual_training_targets=True,aggregate={b:{k:stats([r[k] for r in rows if r['branch']==b]) for k in metrics} for b in ['B','SwinIR','Video7']},crossview={b:stats([r['mean_RGB_difference'] for r in cross if r['branch']==b]) for b in ['B','SwinIR','Video7','HR']},max_DCT_partition_error=max(r['DCT_partition_error'] for r in rows),seconds=time.time()-start,notes=['Frequency energies use orthonormal RGB DCT and exact mutually-exclusive DC/<=.25/<=.5/>.5 index masks.','Sharper gradient energy does not establish closer texture; simultaneously report HR residual bands and gradient orientation.','Cross-view RGB differences measured only at legal independently third-view-verified sparse textured matches; view-dependent appearance and exposure can remain. HR is diagnostic reference, no weights generated.','Input B is OpenCV cubic from native uint8 LR; actual SwinIR/Video7 stored PNG values decoded FP32.'])
    write(OUT/'texture_summary.json',summary);return rows+cross,summary

def finalize(depth,flow,texture,allrows):
    motion=read(OUT/'motion_support/complete.json') if (OUT/'motion_support/complete.json').exists() else None
    if motion:
        flow['independent_geometry_motion_support']=motion
        flow['geometry_motion_support_reason']='Sparse LR six-image tracks validated in a third view at both times. Coverage and observed-motion counts disclosed; not dense 3D motion truth.'
        write(OUT/'flow_summary.json',flow)
        with (OUT/'flow_independent_motion_support.csv').open() as f:allrows+=list(csv.DictReader(f))
    csvwrite(OUT/'image_prior_audit.csv',allrows)
    summary=dict(status='completed',schema=1,scene='cook_spinach',manifest_sha256=sha(MANIFEST),depth=depth,flow=flow,texture=texture,training_weights_generated=False,HR_or_development_images_used_for_training=False,advice=dict(depth_G='Conditional relative-order or boundary-only candidate; sparse independent third-view coverage cannot support full-image metric-depth regression. Select only if reconstruction error overlaps supported structure.',flow_G='Conditional LR-confidence-weighted motion correspondence candidate; inspect motion-interior and visibility-boundary reliability/coverage, and reconstruction error overlap. No metric/3D flow truth. Do not apply full-image flow loss because near-zero static background dominates.',texture_T='SwinIR remains candidate detail source; B/SwinIR/Video7 color and frequency differences are diagnosed jointly. LR-only anchoring/confidence must be generated in separate legal process if selected.'),ordinary_missing=dict(DINO='NA: no compatible cache; not installed by design',true_depth='NA: no metric ground truth',dynamic_3D_motion_compensation='NA: no independently verified 3D trajectory',full_scene_validation='NA:92 image/16pair diagnostic subset only'),artifacts={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))},source_sha256=sha(__file__))
    write(OUT/'summary.json',summary)
    def v(x):return 'NA' if x is None else f'{x:.5f}'
    dr=depth['resolution_comparison'];dg=depth['third_view_geometry'];fa=flow['aggregate'];ta=texture['aggregate']
    lines=['# 原始图像先验诊断卡片','', '结论：三路缓存、固定校准、同刻三视角支持和分区光流可靠性已实测。HR估计仅衡量分辨率稳定性，不能作为深度或光流真值；本轮未生成任何HR驱动训练权重。','', '## 输入与边界','', '- 深度：92个训练观察。A原生LR文件经官方518预处理，B先bicubic到HR再预处理，C真实HR；三路网络实际均为518×686。因此A/B反映重采样路径，不是不同网络尺寸。复用B/C、scale/crop和SwinIR深度，补A。','- 深度只在19相机frame0拟合一次每来源的正scale＋shift，冻结后评估其余73观察。原始数值、排序、边界与固定校准误差均保留。','- 光流：4训练相机 × 4帧对 × 双向 × A/B/C，共96预测，固定现有torchvision RAFT-large C_T_SKHT_V2。A为252×336→pad256×336；B/C1008×1344。统一LR网格时resize并分别缩放dx、dy。Farneback为低成本参照。','- 全部输入仅来自训练相机，HR只走diagnostic_only。正式全训练缓存尚未构建；16对诊断不能成为正式光流训练全样本。','', '## 深度','', '| 来源 | 对HR估计Spearman（评价均值） | 排序一致率 | 固定校准MAE / HR校准IQR | 第三视角支持点深度排序正确率 |','| --- | ---: | ---: | ---: | ---: |']
    for b in ['A','B','SwinIR']:lines.append(f"| {b} | {v(dr[b]['spearman_with_HR_estimate']['mean'])} | {v(dr[b]['rank_agreement_with_HR_estimate']['mean'])} | {v(dr[b]['frozen_aligned_MAE_over_calibration_HR_IQR']['mean'])} | {v(dg[b]['inverse_depth_order_accuracy'])} |")
    lines+=['',f"四组相机三元组在0/40/80/118同刻匹配，以前两视角三角化，在第三视角独立对应验证。验证点本身平均只占图像 {depth['coverage']['mean_unique_pixel_fraction']*100:.3f}%，3像素邻域覆盖 {depth['coverage']['mean_within_3px_fraction']*100:.3f}%；缺少纹理/匹配处为未知。方向按三角化逆深度排序检验，不逐图min-max。",'', '可迁移性：支持相对层次或边界约束的局部候选；不能据全图相关高直接启用全图米制深度。静态区域时序误差只在合法LR四时刻稳定、低运动且前后向一致的代理区域计算。动态物体真实深度可能改变，只报动态局部排序，不把数值变化算作漂移。','', '## 光流','', '| LR来源 | 区域 | 占全图比例 | 对HR流proxy EPE（LR px） | FB一致差（LR px） | LR光度相对零流改善 | Census改善 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for b in ['A','B','Farneback']:
        for reg in ['static_proxy','motion_interior','visibility_boundary_proxy']:
            x=fa[b][reg];lines.append('| '+b+' | '+reg+' | '+' | '.join(v(x[k]['mean']) for k in ['region_fraction','proxy_EPE_to_HR_estimate_LRpx','forward_backward_LRpx','LR_photometric_improvement','LR_census_improvement'])+' |')
    lines+=['','光度/Census正改善表示相对零流更符合原始LR观测；proxy EPE只表示接近HR估计。分区完全来自LR，没有用HR流选可信像素。`flow_confidence_coverage.csv`给出连续置信度排序下10%—100%覆盖曲线；静态背景零流占比不能掩盖动态区域失败。显隐边界只是LR代理标签，并非真实遮挡标注。','', '可迁移性：若运动内部有正光度/Census收益且与重建关键误差重叠，可迁移为选择性的运动对应G；边界或不可靠区域不能无条件施加。第三视角静态几何并不证明跨时光流，本轮未声称独立三维运动真值。','', '## 纹理','', '| 来源 | 原始图像PSNR（92均值） | DC MSE | 低频MSE | 中频MSE | 高频MSE | 梯度方向一致性 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for b in ['B','SwinIR','Video7']:lines.append('| '+b+' | '+' | '.join(v(ta[b][k]['mean']) for k in ['PSNR','MSE_DC','MSE_low','MSE_mid','MSE_high','edge_orientation_cosine'])+' |')
    lines+=['', '所有数值直接读取冻结训练PNG，在FP32 RGB上计算。DCT互斥频带总和核验见texture_summary.json；不从展示图估计。跨视角RGB比较只使用独立第三视角支持的稀疏同刻点，并同时列真实HR自身差异；视角相关反射/曝光仍是限制。','', '## 本轮决策边界','', '先验是否改善3D重建仍需控制器将这些支持区域与重建误差相交，并由冻结同起点训练分支的PSNR、SSIM、LPIPS验证。当前证据不等于G必然有效，也不能把原始图像先验成绩放入新视角方法主表。DINO/度量深度/三维动态补偿无现成可靠资产，标NA，不临时扩展网络扫描。','', '来源：[Depth Anything V2官方实现](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py)；[torchvision RAFT官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)。缓存权重、代码和输入SHA256、网络网格、耗时、设备均在相应index/complete文件。']
    if motion:
        ma=motion['aggregate']['A'];mb=motion['aggregate']['B'];lines += ['', '## 独立跨视角运动补查', '', f"16对另用三相机×两时刻的六图SIFT循环匹配，每个时刻分别由前两视角三角化、第三视角验证。平均每对 {ma['all']['region_tracks']['mean']:.2f} 个轨迹，像素覆盖仅 {ma['all']['full_image_pixel_coverage']['mean']*100:.4f}%；实际位移≥0.5 LR像素平均仅 {ma['observed_motion_ge.5px']['region_tracks']['mean']:.3f} 个。A在这类稀疏运动轨迹EPE {ma['observed_motion_ge.5px']['EPE_to_LR_observed_track']['mean']:.4f}、B为 {mb['observed_motion_ge.5px']['EPE_to_LR_observed_track']['mean']:.4f} LR像素。SIFT自身有亚像素误差，不能当光流真值；这些稀疏支持仍未覆盖cam01大块错误。"]
    (OUT/'prior_cards.md').write_text('\n'.join(lines)+'\n')
    print('image_prior_audit_complete',len(allrows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','cache','geometry','audit','finalize'],default='audit');a=p.parse_args()
    if a.phase in ['all','cache']:
        from image_prior_cache import run_cache
        run_cache()
    if a.phase in ['all','geometry']:
        from image_prior_geometry import geometry
        geometry()
        from image_prior_motion_support import run as motion_run
        motion_run()
    if a.phase in ['all','audit']:
        m,obs,keys=setup();data,identity=depths(keys);index,flow=load_flow_index();allrows=[]
        rows,d=depth_audit(keys,data,identity,flow,obs);allrows+=rows
        rows,f=flow_audit(flow,index,obs);allrows+=rows
        rows,t=texture_audit(keys,obs);allrows+=rows
        finalize(d,f,t,allrows)
    if a.phase=='finalize':
        rows=[]
        for filename in ['depth_image_comparison.csv','depth_geometry.csv','depth_temporal.csv','flow_comparison.csv','texture_comparison.csv','texture_crossview.csv']:
            with (OUT/filename).open() as f:rows+=list(csv.DictReader(f))
        d=read(OUT/'depth_summary.json');_,obs,keys=setup();_,identity=depths(keys);d['cache_identities']=identity;write(OUT/'depth_summary.json',d)
        finalize(d,read(OUT/'flow_summary.json'),read(OUT/'texture_summary.json'),rows)
if __name__=='__main__':main()
