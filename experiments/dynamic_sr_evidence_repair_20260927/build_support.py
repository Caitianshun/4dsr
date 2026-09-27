"""Fixed four-frame train-LR matching with held-out third-view track checks."""
import argparse
from collections import defaultdict, Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_evidence_repair_20260927'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item()))
data_spec=importlib.util.spec_from_file_location('n3dv_data',ROOT/'experiments/dynamic_sr_20260918/n3dv_data.py')
data=importlib.util.module_from_spec(data_spec);sys.modules['n3dv_data']=data;data_spec.loader.exec_module(data)
spec=importlib.util.spec_from_file_location('evidence_prepare18',ROOT/'experiments/dynamic_sr_20260918/prepare_n3dv.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)

def project(xyz,cal):
    xyz=np.atleast_2d(xyz);q=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T
    h=q@np.asarray(cal['K_lr']).T
    return h[:,:2]/q[:,2:3],q[:,2]

def triangulate(ca,cb,xa,xb):
    a=np.asarray(ca['K_lr'])@np.asarray(ca['w2c'])[:3];b=np.asarray(cb['K_lr'])@np.asarray(cb['w2c'])[:3]
    h=cv2.triangulatePoints(a,b,np.asarray(xa).reshape(-1,2).T,np.asarray(xb).reshape(-1,2).T).T
    return h[:,:3]/h[:,3:4]

def pair(a,b,fa,fb,cfg):
    pa,da=fa;pb,db=fb
    record=dict(features_a=len(pa),features_b=len(pb),mutual_matches=0,pose_inliers=0,triangulated=0,accepted=0)
    if da is None or db is None or len(da)<8 or len(db)<8:return [],[],record
    matcher=cv2.BFMatcher(cv2.NORM_L2);forward=matcher.knnMatch(da,db,k=2);reverse=matcher.knnMatch(db,da,k=2)
    rev={v[0].queryIdx:v[0].trainIdx for v in reverse if len(v)==2 and v[0].distance<cfg['ratio']*v[1].distance}
    mutual=[v[0] for v in forward if len(v)==2 and v[0].distance<cfg['ratio']*v[1].distance and rev.get(v[0].trainIdx)==v[0].queryIdx]
    record['mutual_matches']=len(mutual)
    if len(mutual)<8:return [],[],record
    ia=np.array([v.queryIdx for v in mutual]);ib=np.array([v.trainIdx for v in mutual]);xa=pa[ia];xb=pb[ib]
    _,mask=cv2.findFundamentalMat(xa,xb,cv2.FM_RANSAC,cfg['ransac_px'],cfg['ransac_confidence'])
    if mask is None or len(mask)!=len(xa):return [],[],record
    good=mask.ravel().astype(bool)&(prep.sampson_distance(prep.known_fundamental(a,b),xa,xb)<cfg['known_sampson_px']**2)
    ia,ib,xa,xb=ia[good],ib[good],xa[good],xb[good];record['pose_inliers']=len(ia)
    edges=list(zip(ia.tolist(),ib.tolist()))
    if not len(ia):return edges,[],record
    xyz=triangulate(a,b,xa,xb);uva,za=project(xyz,a);uvb,zb=project(xyz,b)
    error=np.maximum(np.linalg.norm(uva-xa,axis=1),np.linalg.norm(uvb-xb,axis=1))
    va=xyz-np.asarray(a['c2w'])[:3,3];vb=xyz-np.asarray(b['c2w'])[:3,3]
    angle=np.degrees(np.arccos(np.clip((va*vb).sum(1)/(np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1)),-1,1)))
    finite=np.isfinite(xyz).all(1)&(za>0)&(zb>0);record['triangulated']=int(finite.sum())
    valid=finite&(error<=cfg['source_reprojection_px'])&(angle>=cfg['angle_deg'])
    pts=[dict(ia=int(ia[i]),ib=int(ib[i]),xyz=xyz[i],error=float(error[i]),angle=float(angle[i])) for i in np.where(valid)[0]]
    record['accepted']=len(pts);record['nonfinite_or_negative']=int((~finite).sum());record['reprojection_rejected']=int((finite&(error>cfg['source_reprojection_px'])).sum());record['angle_rejected']=int((finite&(angle<cfg['angle_deg'])).sum())
    return edges,pts,record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT/'support_v1');a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    protocol=read(OUT/'protocol.json');cfg=protocol['geometry'];manifest=Path(protocol['manifest']['path']);m=read(manifest);assert sha(manifest)==protocol['manifest']['sha256']
    cams=protocol['train_cameras'];assert cams==m['splits']['train'];obs={(o['camera_id'],o['frame_index']):o for o in m['observations']};root=manifest.parent
    cv2.setNumThreads(4);cv2.setRNGSeed(20260918);started=time.time();tracks=[];two=[];pairrows=[];obsrows=[];inputs={}
    sift=cv2.SIFT_create(nfeatures=cfg['sift_nfeatures'],contrastThreshold=cfg['sift_contrast'],edgeThreshold=cfg['sift_edge'])
    for f in protocol['frames']:
        features={};counts={c:Counter() for c in cams};states={}
        for c in cams:
            o=obs[c,f];path=root/o['lr_path'];assert o['split']=='train'
            if not path.exists():states[c]='missing_input';features[c]=(np.empty((0,2)),None);continue
            assert sha(path)==o['lr_sha256'];inputs[str(path)]=o['lr_sha256'];im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            kp,desc=sift.detectAndCompute(im,None);xy=np.array([k.pt for k in kp],np.float64).reshape(-1,2);features[c]=(xy,desc)
            counts[c]['features']=len(kp);states[c]='no_features' if not len(kp) else 'matching_rejected'
        parent={};adj=defaultdict(set);sources={}
        def find(x):
            parent.setdefault(x,x)
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(x,y):
            u,v=find(x),find(y)
            if u!=v:parent[max(u,v)]=min(u,v)
        for ca,cb in itertools.combinations(cams,2):
            edges,pts,r=pair(m['cameras'][ca],m['cameras'][cb],features[ca],features[cb],cfg)
            pairrows.append(dict(camera_a=ca,camera_b=cb,frame=f,**r))
            for c in [ca,cb]:
                for k in ['mutual_matches','pose_inliers','triangulated','accepted']:counts[c][k]+=r[k]
            for ia,ib in edges:
                u,v=(ca,ia),(cb,ib);union(u,v);adj[u].add(v);adj[v].add(u)
            for point in pts:
                u,v=(ca,point['ia']),(cb,point['ib']);sources[u,v]=point
                two.append(dict(frame=f,camera_a=ca,camera_b=cb,**point))
        components=defaultdict(list)
        for n in parent:components[find(n)].append(n)
        rejects=Counter()
        for nodes in sorted(components.values(),key=lambda x:min(x)):
            nodes=sorted(nodes);seen=[n[0] for n in nodes]
            if len(set(seen))!=len(seen):
                rejects['conflicting_camera_feature_ids']+=1
                for c in set(seen):counts[c]['conflicting_track']+=1
                continue
            if len(nodes)<3:rejects['fewer_than_three_views']+=1;continue
            source=None;accepted_nodes=[]
            for u,v in itertools.combinations(nodes,2):
                point=sources.get((u,v))
                if point is None:continue
                valid=[]
                for w in sorted(adj[u]&adj[v]):
                    if w not in nodes or w[0] in [u[0],v[0]]:continue
                    uv,z=project(point['xyz'],m['cameras'][w[0]]);err=float(np.linalg.norm(uv[0]-features[w[0]][0][w[1]]))
                    if z[0]>0 and np.isfinite(err) and err<=cfg['third_reprojection_px']:valid.append((w,err))
                    else:counts[w[0]]['third_view_rejected']+=1
                if valid:source=(u,v,point);accepted_nodes=valid;break
            if source is None:
                rejects['no_source_pair_with_valid_third_view']+=1
                for c in seen:counts[c]['third_track_rejected']+=1
                continue
            u,v,point=source;xa=features[u[0]][0][u[1]];xb=features[v[0]][0][v[1]]
            perturb=np.array(list(itertools.product([-cfg['perturb_px'],cfg['perturb_px']],repeat=4)))
            xyzp=triangulate(m['cameras'][u[0]],m['cameras'][v[0]],xa+perturb[:,:2],xb+perturb[:,2:])
            key=f'{f}:'+','.join(f'{c}:{k}' for c,k in nodes);track_id=hashlib.sha256(key.encode()).hexdigest()[:24]
            verified=[u,v]+[w for w,e in accepted_nodes];projections={}
            for c,k in verified:
                uv,z=project(point['xyz'],m['cameras'][c]);_,zp=project(xyzp,m['cameras'][c]);unc=float(np.max(np.abs(zp-z[0]))) if np.isfinite(zp).all() and (zp>0).all() else None
                projections[c]=dict(feature_id=k,observed_xy=features[c][0][k].tolist(),projected_xy=uv[0].tolist(),z=float(z[0]),z_uncertainty=unc,role='source' if c in [u[0],v[0]] else 'held_out_validation')
                counts[c]['unique_tracks']+=1
                if c not in [u[0],v[0]]:counts[c]['heldout_validation_tracks']+=1
            tracks.append(dict(track_id=track_id,frame=f,xyz=point['xyz'].tolist(),source_pair=[u[0],v[0]],source_error=point['error'],source_angle=point['angle'],component_length=len(nodes),verified_length=len(verified),projections=projections))
        for c in cams:
            q=counts[c]
            if q['unique_tracks']:state='supported'
            elif states[c] in ['missing_input','no_features']:state=states[c]
            elif not q['pose_inliers']:state='matching_rejected'
            elif not q['accepted']:state='triangulation_rejected'
            else:state='third_view_rejected'
            obsrows.append(dict(camera=c,frame=f,status=state,**q))
        write(a.out/f'frame_{f:04d}.json',dict(rejections=dict(rejects),observations=obsrows[-19:],tracks_total=len(tracks)));print('completed frame',f,'tracks',len(tracks),flush=True)
    xyz=np.array([r['xyz'] for r in two]);dist=np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1);radius=max(float(np.quantile(dist,.98))*2,1e-6)
    np.savez_compressed(a.out/'full_pair_support.npz',xyz=xyz.astype(np.float32),reprojection_error=np.array([r['error'] for r in two]),angle_degrees=np.array([r['angle'] for r in two]),frames=np.array([r['frame'] for r in two]),camera_pair=np.array([[int(r['camera_a'][3:]),int(r['camera_b'][3:])] for r in two]),passes_spatial_filter=dist<=radius)
    write(a.out/'tracks.json',tracks);write(a.out/'observations.json',obsrows);write(a.out/'pairs.json',pairrows)
    result=dict(status='completed',unique_tracks=len(tracks),full_two_view_points=len(two),frames=protocol['frames'],inputs=inputs,
        protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),tracks_sha256=sha(a.out/'tracks.json'),pair_support_sha256=sha(a.out/'full_pair_support.npz'),
        observations=obsrows,seconds=time.time()-started,limitations=['One source pair per unique track; third views not fitted. Known calibration may have shared bias.','Direct point evidence is distinct from block area or 4-pixel propagation.','Uncertainty uses all 16 fixed +/-0.5 source-coordinate corners; nonpositive perturbed depths invalid.'],parameter_updates=0,uses_hr=False)
    write(a.out/'complete.json',result);write(OUT/'tracks_manifest.json',result)

if __name__=='__main__':main()
