"""Legal same-time LR three-view tracks; third view never triangulates points."""
import itertools,time
import numpy as np
import cv2
from image_prior_common import *

def matches(a,b):
    if a is None or b is None or min(len(a),len(b))<2:return {}
    matcher=cv2.BFMatcher(cv2.NORM_L2);ab=matcher.knnMatch(a,b,k=2);ba=matcher.knnMatch(b,a,k=2)
    f={x.queryIdx:x.trainIdx for x,y in ab if x.distance<.75*y.distance};r={x.queryIdx:x.trainIdx for x,y in ba if x.distance<.75*y.distance}
    return {i:j for i,j in f.items() if r.get(j)==i}

def geometry():
    start=time.time();m,obs,keys=setup();directory=OUT/'geometry';directory.mkdir(exist_ok=True)
    old=ROOT/'output/dynamic_sr_view_recovery_20260926/p0/support/prefilter_correspondences.npz';z=np.load(old)
    reuse=dict(path=str(old),sha256=sha(old),points=len(z['xyz']),frames=np.unique(z['frames']).tolist(),reprojection_error=stats(z['reprojection_error']),baseline_angle_degrees=stats(z['angle_degrees']),limitation='Existing two-view support, no third-view tracks; reused as inventory, never upgraded to independent three-view evidence.')
    write(directory/'reused_support.json',reuse)
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012,edgeThreshold=12);rows=[];sources={}
    for f in [0,40,80,118]:
        for triple in [('cam02','cam03','cam04'),('cam06','cam07','cam08'),('cam12','cam13','cam14'),('cam18','cam19','cam20')]:
            features=[]
            for c in triple:
                im=image(obs,c,f);k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None);features.append((np.array([x.pt for x in k],np.float64).reshape(-1,2),d));sources[str(MANIFEST.parent/obs[c,f]['lr_path'])]=obs[c,f]['lr_sha256']
            ab=matches(features[0][1],features[1][1]);ac=matches(features[0][1],features[2][1]);common=sorted(set(ab)&set(ac));
            x0=features[0][0][common];x1=features[1][0][[ab[i] for i in common]];x2=features[2][0][[ac[i] for i in common]]
            cams=[m['cameras'][c] for c in triple];Ps=[np.array(c['K_lr'])@np.array(c['w2c'])[:3] for c in cams]
            if len(common):
                q=cv2.triangulatePoints(Ps[0],Ps[1],x0.T,x1.T);xyz=(q[:3]/q[3:]).T;projs=[project(xyz,c) for c in cams];errors=np.stack([np.linalg.norm(p[0]-xy,axis=1) for p,xy in zip(projs,[x0,x1,x2])],1);depths=np.stack([p[1] for p in projs],1)
                centers=[np.array(c['c2w'])[:3,3] for c in cams];a=xyz-centers[0];b=xyz-centers[1];angle=np.degrees(np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1,1)))
                pair_valid=(depths>0).all(1)&np.isfinite(xyz).all(1)&(errors[:,:2].max(1)<1)&(angle>=2);valid=pair_valid&(errors[:,2]<1)
            else:
                xyz=np.empty((0,3));errors=np.empty((0,3));depths=np.empty((0,3));angle=np.empty(0);valid=pair_valid=np.empty(0,bool)
            p=directory/f'{triple[0]}_{triple[1]}_{triple[2]}_{f:04d}.npz';np.savez_compressed(p,xyz=xyz,xy0=x0,xy1=x1,xy2=x2,depths=depths,reprojection_error=errors,angle_degrees=angle,pair_valid=pair_valid,third_valid=valid)
            rows.append(dict(frame=f,cameras=list(triple),three_view_descriptor_tracks=len(common),two_view_valid=int(pair_valid.sum()),third_view_valid=int(valid.sum()),third_reprojection_all_pair_valid=stats(errors[pair_valid,2]),triangulation_angle=stats(angle[pair_valid]),baseline_world=float(np.linalg.norm(np.array(cams[0]['c2w'])[:3,3]-np.array(cams[1]['c2w'])[:3,3])),path=str(p.relative_to(ROOT)),sha256=sha(p)))
            print('geometry',triple,f,len(common),int(valid.sum()),flush=True)
    write(directory/'complete.json',dict(status='completed',rows=rows,sources=sources,manifest_sha256=sha(MANIFEST),reuse=reuse,seconds=time.time()-start,source_sha256=sha(__file__),parameters='SIFT2500 features contrast.012 edge12; mutual ratio.75 matches; source pair triangulation; positive depth all3, >=2degree baseline, max pair reprojection<1LRpx; third image independent descriptor correspondence check<1LRpx.',limitations=['Third-view correspondence is observed LR support, not metric ground truth.','Only textured matched pixels can be assessed; unmatched pixels are unknown.','All three images are same original frame, including40/80; no assumption of static motion.','No HR used for matching, triangulation, validity or support.']))
if __name__=='__main__':geometry()
