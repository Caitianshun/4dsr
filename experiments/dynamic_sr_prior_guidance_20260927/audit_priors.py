"""CPU depth evidence audit. HR comparisons are isolated, never training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.stats import rankdata
from scipy.spatial import cKDTree


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item()))
def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[ok].astype(np.float64);b=np.asarray(b)[ok].astype(np.float64)
    if len(a)<16 or a.var()<1e-6 or b.var()<1e-6:return None
    return float(np.corrcoef(a,b)[0,1])
def spearman(a,b):
    ok=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[ok];b=np.asarray(b)[ok]
    if len(a)<16 or a.var()<1e-6 or b.var()<1e-6:return None
    return corr(rankdata(a),rankdata(b))
def fit(a,b):
    valid=np.isfinite(a)&np.isfinite(b);x=a[valid].astype(np.float64);y=b[valid].astype(np.float64)
    if len(x)<16 or x.var()<1e-6:return None
    x=x[::4];y=y[::4];scale=np.cov(x,y,bias=True)[0,1]/x.var();return [float(scale),float(y.mean()-scale*x.mean())]
def summary(x):
    a=np.asarray([v for v in x if v is not None and np.isfinite(v)]);return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p10=float(np.quantile(a,.1)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)
def sample(image,xy):return cv2.remap(image.astype(np.float32),xy[:,0].astype(np.float32)[:,None],xy[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)[:,0]


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','legal','privileged','correspondences','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--moments',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);started=time.time()
    m=read(a.manifest);root=a.manifest.parent;protocol=read(a.protocol)
    obs={(o['camera_id'],o['frame_index']):o for o in m['observations']};cache={};hrcache={}
    for directory,dest in [(a.legal,cache),(a.privileged,hrcache)]:
        j=read(directory/'complete.json')
        for r in j['rows']:
            path=directory/r['path'];assert sha(path)==r['sha256'];dest[r['camera'],r['frame']]={k:v for k,v in np.load(path).items()}
    pool=[(y,x) for y in range(16,252-31,16) for x in range(16,336-31,16)]
    write(a.out/'fixed_patch_pool.json',dict(shape=[252,336],size=16,stride=16,patches=pool,protocol_sha256=sha(a.protocol)))
    support=np.load(a.correspondences);xyz=support['xyz'];base_good=(support['reprojection_error']<=.5)&(support['angle_degrees']>=2)&support['passes_spatial_filter']
    rows=[];geometry=[];coverage={};rank_correct=[];rank_wrong=[];valid_spearmans=[];hr_comp=[]
    for cam in m['splits']['train']:
        cal=m['cameras'][cam];cw=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T;uv=cw@np.asarray(cal['K_lr']).T;xy=uv[:,:2]/cw[:,2:]
        camera_pool=[];all_lr=[];all_hr=[]
        for f in [0,40,80,118]:all_lr.append(cache[cam,f]['lr']);all_hr.append(hrcache[cam,f]['hr'])
        fixed_alignment=fit(np.stack(all_lr),np.stack(all_hr))
        for f in [0,40,80,118]:
            z=cache[cam,f];h=hrcache[cam,f];o=obs[cam,f]
            lr=cv2.imread(str(root/o['lr_path']));assert sha(root/o['lr_path'])==o['lr_sha256']
            gray=cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
            internal=np.sqrt(gx*gx+gy*gy)<.08
            point_good=base_good&(support['frames']==f)&(support['camera_pair']==int(cam[3:])).any(1)&(cw[:,2]>0)&(xy[:,0]>=16)&(xy[:,0]<320)&(xy[:,1]>=16)&(xy[:,1]<236)
            points=xy[point_good];depth=cw[point_good,2]
            valid_points=np.isfinite(points).all(1);points=points[valid_points];depth=depth[valid_points]
            support_map=np.zeros((252,336),np.int32)
            if len(points):
                ij=np.floor(points).astype(int);np.add.at(support_map,(ij[:,1],ij[:,0]),1)
            moment=None
            if a.moments is not None:
                path=a.moments/f'{cam}_{f:04d}.npz'
                if path.exists():moment=np.load(path)
            accepted_area=0;stable_area=0
            for y,x in pool:
                sl=np.s_[y:y+16,x:x+16];u=z['lr'][sl];v=h['hr'][sl]
                rs=[spearman(u,z[k][sl]) for k in ['scale','crop']]
                valid_spearmans.extend(rs)
                stable=all(r is not None and r>=.90 for r in rs)
                stable_area+=256*stable
                area=internal[sl].mean();n=int(support_map[sl].sum())
                model_ok=False
                if moment is not None:
                    az=moment['alpha'][sl];dz=moment['expected_z'][sl];var=moment['variance_z'][sl]
                    model_ok=float(((az>=.9)&(var<=.02*np.maximum(dz*dz,1e-6))).mean())>=.5
                accepted=stable and area>=.5 and n>=2 and model_ok
                accepted_area+=256*accepted
                patch=dict(camera=cam,frame=f,y=y,x=x,pearson_hr_lr=corr(u,v),spearman_hr_lr=spearman(u,v),spearman_lr_scale=rs[0],spearman_lr_crop=rs[1],variance=float(u.var()),stable=stable,LR_interior_fraction=float(area),independent_support_points=n,model_support=model_ok,accepted=accepted)
                rows.append(patch)
            camera_pool.append(dict(frame=f,stable_fraction=stable_area/(252*336),trusted_fraction=accepted_area/(252*336),support_points=len(points),moment_available=moment is not None))
            for key in ['lr','sr']:
                pred=z['lr'] if key=='lr' else h['sr'];frame_alignment=fit(pred,h['hr']);fixed=fixed_alignment if key=='lr' else None
                iqr=max(float(np.quantile(h['hr'],.75)-np.quantile(h['hr'],.25)),1e-6)
                hr_comp.append(dict(camera=cam,frame=f,source=key,pearson=corr(pred,h['hr']),spearman=spearman(pred,h['hr']),frame_alignment=frame_alignment,frame_aligned_median_abs_over_iqr=None if frame_alignment is None else float(np.median(np.abs(frame_alignment[0]*pred+frame_alignment[1]-h['hr']))/iqr),camera_fixed_alignment=fixed,camera_fixed_median_abs_over_iqr=None if fixed is None else float(np.median(np.abs(fixed[0]*pred+fixed[1]-h['hr']))/iqr)))
            if len(points)>=10:
                tree=cKDTree(points);neighbors=tree.query(points,k=min(12,len(points)))[1];pairs=[]
                for i,ns in enumerate(neighbors):
                    for j in ns:
                        distance=np.linalg.norm(points[i]-points[j]);rel=abs(depth[i]-depth[j])/max(depth[i],depth[j])
                        if i<j and 4<=distance<=40 and rel>=.03:pairs.append((i,int(j)))
                if pairs:
                    ij=np.asarray(pairs);pred=sample(z['lr'],points);wrongxy=points.copy();wrongxy[:,0]=(wrongxy[:,0]+48)%336;wrong=sample(z['lr'],wrongxy)
                    target=np.sign(1/depth[ij[:,0]]-1/depth[ij[:,1]])
                    diff=pred[ij[:,0]]-pred[ij[:,1]];wdiff=wrong[ij[:,0]]-wrong[ij[:,1]]
                    good=np.isfinite(diff)&np.isfinite(wdiff);right=(np.sign(diff[good])==target[good]);bad=(np.sign(wdiff[good])==target[good]);rank_correct.extend(right.tolist());rank_wrong.extend(bad.tolist())
                    geometry.append(dict(camera=cam,frame=f,points=len(points),pairs=int(good.sum()),order_accuracy=float(right.mean()),wrong_accuracy=float(bad.mean()),relative_inverse_depth_pearson=corr(pred,1/depth)))
        coverage[cam]=dict(frames=camera_pool,mean_trusted_fraction=float(np.mean([x['trusted_fraction'] for x in camera_pool])),mean_stable_fraction=float(np.mean([x['stable_fraction'] for x in camera_pool])))
    # Legal LR optical-flow correspondence; independently record moving support.
    temporal=[]
    for cam in ['cam02','cam06','cam12','cam18']:
        reference=cache[cam,40]['lr'];iqr=max(float(np.quantile(reference,.75)-np.quantile(reference,.25)),1e-6)
        for fs in [[38,40,42],[78,80,82]]:
            pairrows=[]
            for f,g in zip(fs,fs[1:]):
                def gray(frame):return cv2.imread(str(root/obs[cam,frame]['lr_path']),cv2.IMREAD_GRAYSCALE)
                left,right=gray(f),gray(g)
                flow=cv2.calcOpticalFlowFarneback(left,right,None,.5,3,15,3,5,1.2,0);back=cv2.calcOpticalFlowFarneback(right,left,None,.5,3,15,3,5,1.2,0)
                yy,xx=np.indices(left.shape,dtype=np.float32);mapx=xx+flow[:,:,0];mapy=yy+flow[:,:,1]
                warped=cv2.remap(cache[cam,g]['lr'],mapx,mapy,cv2.INTER_LINEAR,borderValue=np.nan)
                bw=cv2.remap(back,mapx,mapy,cv2.INTER_LINEAR)
                valid=(np.linalg.norm(flow+bw,axis=-1)<1)&(mapx>=1)&(mapx<335)&(mapy>=1)&(mapy<251)&np.isfinite(warped)
                moving=valid&(np.linalg.norm(flow,axis=-1)>=.5)
                first=cache[cam,f]['lr'];free=fit(first[valid],warped[valid]);error=np.abs(first-warped)/iqr
                ordering=[]
                for dy,dx in [(0,8),(8,0)]:
                    base_slice=np.s_[:252-dy,:336-dx];other_slice=np.s_[dy:,dx:]
                    pair_valid=moving[base_slice]&moving[other_slice]
                    d0=first[base_slice]-first[other_slice];d1=warped[base_slice]-warped[other_slice]
                    pair_valid &= np.abs(d0)>=.01*iqr
                    ordering.extend((np.sign(d0[pair_valid])==np.sign(d1[pair_valid])).tolist())
                local_order=float(np.mean(ordering)) if len(ordering)>=32 else None
                pairrows.append(dict(first=f,second=g,valid_fraction=float(valid.mean()),moving_fraction=float(moving.mean()),fixed_camera_raw_median_abs_over_iqr=float(np.median(error[valid])),moving_median_abs_over_iqr=float(np.median(error[moving])) if moving.sum()>=16 else None,spearman=spearman(first[valid],warped[valid]),moving_spearman=spearman(first[moving],warped[moving]),moving_local_order_accuracy=local_order,moving_local_order_pairs=len(ordering),frame_free_alignment=free,frame_free_median_abs_over_iqr=float(np.median(np.abs(free[0]*first[valid]+free[1]-warped[valid]))/iqr) if free else None))
            passed=all(r['moving_fraction']>=.005 and r['moving_median_abs_over_iqr'] is not None and r['moving_median_abs_over_iqr']<=.05 and r['moving_spearman'] is not None and r['moving_spearman']>=.90 and r['moving_local_order_accuracy'] is not None and r['moving_local_order_accuracy']>=.90 for r in pairrows)
            temporal.append(dict(camera=cam,frames=fs,rows=pairrows,passes_registered_conservative_proxy=passed))
    order=float(np.mean(rank_correct)) if rank_correct else None;wrong=float(np.mean(rank_wrong)) if rank_wrong else None
    gates=dict(stability=summary(valid_spearmans)['median'] is not None and summary(valid_spearmans)['median']>=.90,coverage_cameras=sum(v['mean_trusted_fraction']>=.20 for v in coverage.values())>=12,geometry_cameras=len({r['camera'] for r in geometry}&{'cam02','cam06','cam12','cam18'})>=3,geometry_rank=order is not None and order>=.80 and order-wrong>=.10,temporal=sum(r['passes_registered_conservative_proxy'] for r in temporal)>=3)
    write(a.out/'patches.json',rows);write(a.out/'privileged_hr_comparison.json',hr_comp);write(a.out/'temporal.json',temporal)
    write(a.out/'complete.json',dict(status='completed',gates=gates,depth_admission_evidence_passed=all(gates.values()),weights_generated=False,depth_training_run=False,patch_pool_count=len(pool),patch_rows=len(rows),perturb_spearman=summary(valid_spearmans),hr_lr_spearman=summary([r['spearman'] for r in hr_comp if r['source']=='lr']),coverage=coverage,geometry=geometry,order_accuracy=order,wrong_order_accuracy=wrong,order_pairs=len(rank_correct),temporal_pass_clips=sum(r['passes_registered_conservative_proxy'] for r in temporal),notes=['No actual depth ground truth. Shared-network agreement can share bias.','Geometry support observed only at0/118 in this fixed audit;40/80 unsupported marked unknown, not inferred invisible.','Triangulations are mutual two-view matches, not independent ground-truth geometry. Fixed wrong spatial match preserves raw image statistics.','Temporal compares same camera raw scale (identity alignment) and discloses per-frame affine alignment; HR used only in separate comparison.','Coverage defined conservatively over all4 registered frames, not only frames having convenient support.','No HR-driven weights or labels generated.'],protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),seconds=time.time()-started))


if __name__=='__main__':main()
