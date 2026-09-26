"""Reconstruct omitted prefilter training-LR support, never change initialization."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
from prepare_n3dv import triangulate_pair,sha256,atomic_json

def project(xyz,cal):
    p=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T
    q=p@np.asarray(cal['K_lr']).T
    xy=q[:,:2]/np.where(np.abs(q[:,2:])>1e-10,q[:,2:],np.nan)
    valid=(p[:,2]>0)&np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<336)&(xy[:,1]>=0)&(xy[:,1]<252)
    return xy,valid

def coverage(xyz,cameras):
    out={}
    for c in ['cam00','cam01']:
        xy,valid=project(xyz,cameras[c]);xy=xy[valid]
        ij=np.floor(xy).astype(int);grid=np.zeros((252,336),bool);grid[ij[:,1],ij[:,0]]=True
        yy,xx=np.indices(grid.shape);border=(xx<33.6)|(xx>=302.4)|(yy<25.2)|(yy>=226.8)
        out[c]=dict(in_frustum=int(valid.sum()),lr_cells=int(grid.sum()),outer_cells=int((grid&border).sum()),
            top_cells=int((grid&(yy<25.2)).sum()),total_outer_cells=int(border.sum()),
            note='Projected sparse points, no target-image matching or occlusion proof')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);cv2.setRNGSeed(20260918);start=time.monotonic()
    m=json.loads(a.manifest.read_text());root=a.manifest.parent;cams=m['cameras'];train=m['splits']['train']
    assert set(train)=={f'cam{x:02d}' for x in range(2,21)}
    original=json.loads((root/'initialization/report.json').read_text());frames=original['source_frames']
    obs={(o['camera_id'],o['frame_index']):o for o in m['observations']};sources={};stable={}
    def rgb(c,f):
        o=obs[c,f];assert o['split']=='train';path=root/o['lr_path'];assert sha256(path)==o['lr_sha256'];sources[str(path)]=o['lr_sha256']
        return cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    for c in train:
        images=np.stack([rgb(c,f).astype(np.float32)/255 for f in [0,40,80,118]])
        stable[c]=(np.sqrt(images.var(0).mean(2))<.025)&(images.max((0,3))<.98)&(images.min((0,3))>.01)
    xyzs=[];errs=[];angles=[];background=[];pair_records=[];point_pairs=[];point_frames=[]
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012,edgeThreshold=12)
    for f in frames:
        features={}
        for c in train:
            im=rgb(c,f);k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None)
            features[c]=(np.asarray([q.pt for q in k],np.float64).reshape(-1,2),d,im)
        for ca,cb in itertools.combinations(train,2):
            pts,colors,error,record=triangulate_pair(cams[ca],cams[cb],features[ca],features[cb])
            record.update(camera_a=ca,camera_b=cb,frame=f);pair_records.append(record)
            if not len(pts):continue
            bg=np.ones(len(pts),bool)
            for c in [ca,cb]:
                xy,valid=project(pts,cams[c]);ij=np.rint(np.nan_to_num(xy)).astype(int);ij[:,0]=np.clip(ij[:,0],0,335);ij[:,1]=np.clip(ij[:,1],0,251)
                bg&=valid&stable[c][ij[:,1],ij[:,0]]
            v1=pts-np.asarray(cams[ca]['c2w'])[:3,3];v2=pts-np.asarray(cams[cb]['c2w'])[:3,3]
            angle=np.degrees(np.arccos(np.clip((v1*v2).sum(1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)),-1,1)))
            xyzs.append(pts);errs.append(error);angles.append(angle);background.append(bg)
            point_pairs.append(np.tile([int(ca[3:]),int(cb[3:])],(len(pts),1)));point_frames.append(np.full(len(pts),f))
        print('support frame',f,flush=True)
    xyz=np.concatenate(xyzs);error=np.concatenate(errs);angle=np.concatenate(angles);bg=np.concatenate(background)
    radii=np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1);radius=max(float(np.quantile(radii,.98))*2,1e-6);good=np.isfinite(radii)&(radii<=radius)
    kept=np.load(root/'initialization/points_lr.npz')['points']
    np.savez_compressed(a.out/'prefilter_correspondences.npz',xyz=xyz.astype(np.float32),reprojection_error=error.astype(np.float32),angle_degrees=angle.astype(np.float32),
        camera_pair=np.concatenate(point_pairs),frames=np.concatenate(point_frames),stable_unsaturated_train_proxy=bg,passes_spatial_filter=good)
    levels=dict(actual_train_correspondence=dict(points=len(xyz),support_cameras_per_point=2,median_reprojection_lr_px=float(np.median(error)),median_angle_deg=float(np.median(angle)),
        stable_unsaturated_proxy_points=int(bg.sum()),spatial_rejected=int((~good).sum()),spatial_rejected_stable=int((bg&~good).sum()),
        raw_coverage=coverage(xyz,cams),background_proxy_coverage=coverage(xyz[bg],cams),rejected_coverage=coverage(xyz[~good],cams),initial_kept_coverage=coverage(kept,cams)),
        model_only='Effective-state frusta/AABB and post-training projections are reported by drift audit; not real visibility',
        unknown='Saturation, no texture, occlusion, reflection and unmatched areas remain unknown. Absence of SIFT is NOT evidence of true invisibility.')
    atomic_json(a.out/'summary.json',dict(status='completed',levels=levels,pair_statistics=pair_records,
        original_report_sha256=sha256(root/'initialization/report.json'),initial_points_sha256=sha256(root/'initialization/points_lr.npz'),
        original_raw_count=original['raw_point_count'],recomputed_raw_count=len(xyz),same_raw_count=original['raw_point_count']==len(xyz),
        spatial_clip_radius=radius,original_voxel_size=original['voxel_size_world'],initial_kept_count=len(kept),source_sha256=sources,
        limitations='Two-view supports retained per point; no claimed multi-view tracks. Stable proxy is four legal LR times, not semantics. Projected target coverage does not establish target surface correspondence.',
        used_target_images=False,parameter_updates=0,seconds=time.monotonic()-start))

if __name__=='__main__':main()
