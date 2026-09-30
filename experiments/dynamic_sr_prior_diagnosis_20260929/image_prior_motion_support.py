"""Sparse legal LR motion checks with independent third-view validation at both times."""
import time
from image_prior_common import *
from image_prior_geometry import matches

def run():
    start=time.time();m,obs,keys=setup();directory=OUT/'motion_support';directory.mkdir(exist_ok=True);sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012,edgeThreshold=12);features={};rows=[];sources={}
    def feature(c,f):
        if (c,f) not in features:
            im=image(obs,c,f);k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None);features[c,f]=(np.array([x.pt for x in k],np.float64).reshape(-1,2),d);sources[str(MANIFEST.parent/obs[c,f]['lr_path'])]=obs[c,f]['lr_sha256']
        return features[c,f]
    def triangulate(xys,cams):
        Ps=[np.array(cal['K_lr'])@np.array(cal['w2c'])[:3] for cal in cams];q=cv2.triangulatePoints(Ps[0],Ps[1],xys[0].T,xys[1].T);xyz=(q[:3]/q[3:]).T;p=[project(xyz,cal) for cal in cams];err=np.stack([np.linalg.norm(a[0]-b,axis=1) for a,b in zip(p,xys)],1);depth=np.stack([a[1] for a in p],1);a=xyz-np.array(cams[0]['c2w'])[:3,3];b=xyz-np.array(cams[1]['c2w'])[:3,3];angle=np.degrees(np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1,1)));good=np.isfinite(xyz).all(1)&(depth>0).all(1)&(err.max(1)<1)&(angle>=2)
        return xyz,good,err
    for c,f,g in PAIRS:
        n=int(c[3:]);cs=[f'cam{x:02d}' for x in [n,n+1,n+2]];fs=[[feature(cam,t) for cam in cs] for t in [f,g]];cross=[(matches(z[0][1],z[1][1]),matches(z[0][1],z[2][1])) for z in fs];temporal=[matches(fs[0][i][1],fs[1][i][1]) for i in range(3)];ids=[]
        for i,j in temporal[0].items():
            if all(i in cross[0][a] and j in cross[1][a] and temporal[a+1].get(cross[0][a][i])==cross[1][a][j] for a in [0,1]):ids.append((i,j))
        if ids:
            xs=[]
            for t in [0,1]:
                ks=[ij[t] for ij in ids];xs.append([fs[t][0][0][ks],fs[t][1][0][[cross[t][0][i] for i in ks]],fs[t][2][0][[cross[t][1][i] for i in ks]]])
            cams=[m['cameras'][cam] for cam in cs];x,good0,err0=triangulate(xs[0],cams);y,good1,err1=triangulate(xs[1],cams);good=good0&good1;points=xs[0][0][good];q=xs[1][0][good];displacement=q-points;mag=np.linalg.norm(displacement,axis=1)
        else:
            points=q=displacement=np.empty((0,2));mag=np.empty(0);good=np.empty(0,bool)
        path=directory/f'{c}_{f:04d}_{g:04d}.npz';np.savez_compressed(path,source_points=points,target_points=q,observed_displacement=displacement,observed_magnitude=mag)
        for branch in ['A','B','C','Farneback']:
            if branch=='Farneback':
                im0=image(obs,c,f);im1=image(obs,c,g);F=cv2.calcOpticalFlowFarneback(cv2.cvtColor(im0,cv2.COLOR_RGB2GRAY),cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY),None,.5,3,15,3,5,1.2,0)
            else:
                folder='flow_diagnostic_only' if branch=='C' else 'flow_legal';F=resize_flow(np.load(OUT/folder/f'{branch}_{c}_{f:04d}_{g:04d}.npz')['forward'],(252,336))
            pred=sample(F,points) if len(points) else np.empty((0,2));epe=np.linalg.norm(pred-displacement,axis=1)
            for region,mask in [('all',np.ones(len(mag),bool)),('observed_static_lt.25px',mag<.25),('observed_motion_ge.5px',mag>=.5)]:
                rows.append(dict(prior='independent_motion_support',camera=c,frame=f,second=g,branch=branch,region=region,six_image_descriptor_tracks=len(ids),third_verified_temporal_tracks=len(points),region_tracks=int(mask.sum()),full_image_pixel_coverage=int(mask.sum())/(252*336),EPE_to_LR_observed_track=float(epe[mask].mean()) if mask.any() else None,observed_displacement_mean=float(mag[mask].mean()) if mask.any() else None,zero_flow_EPE=float(mag[mask].mean()) if mask.any() else None,improvement_vs_zero_flow=float((mag-epe)[mask].mean()) if mask.any() else None))
    csvwrite(OUT/'flow_independent_motion_support.csv',rows)
    result=dict(status='completed',pairs=16,parameters='SIFT mutual ratio.75 descriptor tracks across all three same-time views and both times; require source and target tracks pass >=2degree source pair angle, positive depth and <1LRpx reprojection all3. Triangulation excludes third view at each time.',sources=sources,source_sha256=sha(__file__),seconds=time.time()-start,aggregate={b:{region:{metric:stats([r[metric] for r in rows if r['branch']==b and r['region']==region and r[metric] is not None]) for metric in ['region_tracks','EPE_to_LR_observed_track','improvement_vs_zero_flow','full_image_pixel_coverage']} for region in ['all','observed_static_lt.25px','observed_motion_ge.5px']} for b in ['A','B','C','Farneback']},limitation='Sparse textured repeated-feature tracks only; subpixel SIFT localization is noisy, not flow ground truth. Counted as independent LR correspondence support, not proof for unmatched/dynamic boundary pixels. C remains HR diagnostic only.')
    write(directory/'complete.json',result);print(result['aggregate']['A'],flush=True)
if __name__=='__main__':run()
