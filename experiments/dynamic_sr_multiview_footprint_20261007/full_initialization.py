"""Seed-specific pure-LR full-time sparse initialization, CPU only.

Derived train-ready manifests preserve the immutable full decoded manifest.
This prepares input points; it does not train any independent LR model prefix.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import itertools
import os
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
import torch
from fp_common import ROOT, OUT, read, write, sha, entry, local, module

LEGACY = ROOT/'experiments/dynamic_sr_20260918/prepare_n3dv.py'
sys.path.insert(0,str(LEGACY.parent))
from n3dv_data import load_manifest, load_initial_points, N3DVPreparedDataset
original=module('fp_full_original_lr_triangulation',LEGACY)
FRAMES=(0,150,299)
RULES=dict(method='known_calibration_SIFT_mutual_ratio_RANSAC_epipolar_DLT_v1',
 nfeatures=2500,contrastThreshold=.012,edgeThreshold=12,mutual_ratio=.8,
 RANSAC_threshold_LR_px=1.5,RANSAC_confidence=.999,known_pose_sampson_squared_LR_px=1.5**2,
 reprojection_error_maximum_LR_px=1.5,minimum_triangulation_angle_degrees=.75,
 camera_depth_positive=True,radius_clip='2*98%quantile_about_point_median',
 voxel_size='0.002*90%radius_quantile_after_outlier_filter',
 voxel_retention='minimum reprojection error per voxel, stable sort',
 minimum_points=256,maximum_points=60000,
 seed_usage='cv2 RANSAC RNG and optional maximum-point subsampling; no model parameters initialized here',
 topology_policy='input point count is not the eventual densified Gaussian count',
 frames=list(FRAMES),camera_boundary='exact manifest.splits.train; held-out cameras excluded by split, cam01 legal if train',
 HR_images_read=False,test_images_read=False,development_images_read=False,CUDA_calls=0)


def generate(manifest,seed,cpu_threads=2):
    manifest=local(manifest);m=load_manifest(manifest);root=Path(m['_root']);seed=int(seed)
    if not m.get('full_time_decode_verified') or m['frame_indices']!=list(range(300)):
        raise ValueError('Verified full 0..299 decoded manifest required')
    train_ids=list(m['splits']['train'])
    if not train_ids or set(train_ids)&set(m['splits']['test']+m['splits']['dev']):raise ValueError('Split overlap')
    if 'cam00' in train_ids:raise ValueError('Official held-out camera not legal')
    rows={(r['camera_id'],int(r['frame_index'])):r for r in m['observations'] if r['split']=='train'}
    if set(rows)!=set(itertools.product(train_ids,range(300))):raise ValueError('Complete training observation boundary mismatch')
    # Freeze canonical sources and original full manifest before opening pixels.
    identity=dict(original_full_manifest=entry(manifest),source=entry(Path(__file__)),
      legacy_triangulation=entry(LEGACY),data_loader=entry(LEGACY.parent/'n3dv_data.py'),seed=seed,rules=RULES,
      resolution_LR=m['resolutions']['lr'],train_cameras=train_ids,CPU_threads=cpu_threads,
      cv2=cv2.__version__,numpy=np.__version__,torch=torch.__version__)
    destination=root/f'initialization_seed_{seed}';destination.mkdir(parents=True,exist_ok=True)
    ready=root/f'manifest_train_ready_seed{seed}.json';complete=destination/'complete.json'
    if complete.exists():
        old=read(complete)
        if old['identity']!=identity:raise ValueError('Completed initialization identity changed')
        for ref in old['artifacts']:assert sha(local(ref['path']))==ref['sha256']
        return old
    if ready.exists():raise ValueError('Derived ready manifest exists without completion; preserve for explicit recovery')
    cv2.setNumThreads(cpu_threads);cv2.setRNGSeed(seed);np.random.seed(seed);torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(),'CPU-only preparation must not initialize CUDA'
    write(destination/'config.json',identity)
    snapshots=destination/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for p in (Path(__file__),LEGACY,LEGACY.parent/'n3dv_data.py'):
        q=snapshots/p.name
        if q.exists() and sha(q)!=sha(p):raise ValueError('Frozen initialization source changed')
        if not q.exists():shutil.copy2(p,q)
    t0=time.monotonic();sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012,edgeThreshold=12)
    all_points=[];all_colors=[];all_errors=[];point_frames=[];pairs=[];sources=[]
    expected_shape=tuple(reversed(m['resolutions']['lr']))
    for frame in FRAMES:
        features={}
        for camera in train_ids:
            row=rows[camera,frame];path=root/row['lr_path']
            # Use exact registered LR location/hash. No HR or test path is opened.
            if sha(path)!=row['lr_sha256']:raise ValueError(f'Full legal LR identity changed: {path}')
            bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2]!=expected_shape:raise ValueError('Registered LR shape mismatch')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            keypoints,descriptors=sift.detectAndCompute(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),None)
            features[camera]=(np.asarray([k.pt for k in keypoints],dtype=np.float64).reshape(-1,2),descriptors,rgb)
            sources.append(dict(camera=camera,frame=frame,features=len(keypoints),**entry(path)))
        for a,b in itertools.combinations(train_ids,2):
            xyz,color,error,record=original.triangulate_pair(m['cameras'][a],m['cameras'][b],features[a],features[b],ratio=.8)
            record['frame_index']=frame;pairs.append(record)
            if len(xyz):
                all_points.append(xyz);all_colors.append(color);all_errors.append(error)
                point_frames.append(np.full(len(xyz),frame,dtype=np.int32))
        print({'stage':'full_LR_triangulation','scene':m['scene'],'seed':seed,'frame':frame,'raw_points':sum(len(x) for x in all_points)},flush=True)
    write(destination/'triangulation_diagnostics.json',dict(status='completed_before_spatial_filter',identity=identity,
      sources=sources,pair_statistics=pairs,used_HR=False,used_heldout_pixels=False,CUDA_calls=0))
    if not all_points:raise ValueError('No reliable full legal LR points; do not substitute HR/held-out data')
    xyz,rgb,error,frames=np.concatenate(all_points),np.concatenate(all_colors),np.concatenate(all_errors),np.concatenate(point_frames)
    raw_count=len(xyz);center=np.median(xyz,axis=0);radii=np.linalg.norm(xyz-center,axis=1)
    clip_radius=max(float(np.quantile(radii,.98))*2,1e-6)
    keep=np.isfinite(radii)&(radii<=clip_radius);xyz,rgb,error,frames=xyz[keep],rgb[keep],error[keep],frames[keep]
    extent=max(float(np.quantile(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1),.9)),1e-4);voxel=extent*.002
    order=np.argsort(error,kind='stable');_,unique=np.unique(np.floor(xyz[order]/voxel).astype(np.int64),axis=0,return_index=True)
    indices=order[unique]
    if len(indices)>60000:indices=np.random.default_rng(seed).choice(indices,60000,replace=False)
    xyz,rgb,error,frames=xyz[indices],rgb[indices],error[indices],frames[indices]
    if len(xyz)<256:raise ValueError(f'Only {len(xyz)} reliable points; preserve failed legal initialization')
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():raise ValueError('Nonfinite point cloud')
    points=destination/'points_lr.npz';ply=destination/'points_lr.ply';report=destination/'report.json'
    np.savez_compressed(points,points=xyz.astype(np.float32),colors=rgb.astype(np.float32),
      normals=np.zeros_like(xyz,dtype=np.float32),reprojection_error_lr_px=error.astype(np.float32),source_frame_index=frames)
    original.write_ply(ply,xyz,rgb)
    stat=dict(status='completed_pure_LR_point_initialization_only',identity=identity,
      train_cameras=train_ids,cam01_included_if_legal='cam01' in train_ids,source_frames=list(FRAMES),source_images=sources,
      raw_point_count=raw_count,point_count=len(xyz),voxel_size_world=voxel,clip_radius_world=clip_radius,
      reprojection_error_median_LR_pixels=float(np.median(error)),points_per_source_frame=dict(Counter(str(int(f)) for f in frames)),
      pair_statistics=pairs,CPU_seconds=time.monotonic()-t0,
      model_prefix_trained=False,independent_model_seed_completed=False,used_HR=False,used_heldout_pixels=False,
      official_COLMAP_initialization=False,CUDA_calls=0,
      limitation='Known-camera LR triangulation is a disclosed sparse initialization adaptation; multi-time points may mix moving surfaces. Each seed needs its own newly trained LR prefix.')
    write(report,stat)
    derived={k:copy.deepcopy(v) for k,v in m.items() if not k.startswith('_')}
    derived['initialization']=dict(npz_path=str(points.relative_to(root)),ply_path=str(ply.relative_to(root)),
      report_path=str(report.relative_to(root)),point_count=len(xyz),npz_sha256=sha(points),ply_sha256=sha(ply),report_sha256=sha(report),
      train_only_lr=True,seed=seed,official_COLMAP_initialization=False)
    derived['original_full_manifest']=entry(manifest)
    derived['full_initialization_source']=entry(Path(__file__))
    derived['independent_seed_registration']=dict(seed=seed,LR_model_prefix_trained=False,
      same_seed_baseline_and_method_share_LR_prefix=True,different_seeds_must_train_independent_LR_prefixes=True)
    write(ready,derived)
    # CPU consumer acceptance. load_training's camera adapter allocates CUDA, so
    # only its exact CPU dataset and initial-point loaders are accepted here.
    accepted=load_manifest(ready);initial=load_initial_points(accepted)
    assert initial['points'].shape==(len(xyz),3) and initial['colors'].shape==(len(xyz),3)
    dataset=N3DVPreparedDataset(accepted,'train','lr',device='cpu',cache=False)
    for index in (0,len(dataset)//2,len(dataset)-1):
        observation=dataset[index]
        assert observation['camera_id'] in train_ids and tuple(observation['image'].shape[1:])==expected_shape
    assert not torch.cuda.is_initialized()
    receipt=dict(status='completed_train_ready_manifest_and_initial_points_CPU_acceptance',identity=identity,
      derived_manifest=entry(ready),point_count=len(xyz),CPU_seconds=time.monotonic()-t0,
      accepted_load_manifest=True,accepted_load_initial_points=True,accepted_CPU_train_dataset=True,
      load_training_CUDA_camera_creation_executed=False,legal_train_observations=len(dataset),CUDA_calls=0,
      model_prefix_trained=False,artifacts=[entry(p) for p in (ready,points,ply,report,destination/'triangulation_diagnostics.json')])
    write(complete,receipt);print({'status':receipt['status'],'scene':m['scene'],'seed':seed,'points':len(xyz),'seconds':receipt['CPU_seconds']},flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[20261007,20261008]);p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    if a.cpu_threads not in (1,2):raise ValueError('CPU initialization bound is at most two threads')
    receipts=[generate(a.manifest,s,a.cpu_threads) for s in a.seeds]
    target=OUT/'full_data_readiness'/('initialization_'+read(a.manifest)['scene']+'_index.json')
    write(target,dict(status='completed_requested_seed_point_initializations_prefix_models_pending',
      original_full_manifest=entry(a.manifest),source=entry(Path(__file__)),seeds=a.seeds,
      completed_initializations=[r['derived_manifest'] for r in receipts],receipts=[entry(local(r['derived_manifest']['path']).parent/f'initialization_seed_{s}/complete.json') for s,r in zip(a.seeds,receipts)],
      CPU_seconds=sum(r['CPU_seconds'] for r in receipts),CUDA_calls=0,trained_LR_prefixes=0))

if __name__=='__main__':main()
