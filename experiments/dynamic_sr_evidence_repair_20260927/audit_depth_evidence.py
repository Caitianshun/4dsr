"""Versioned support/moment funnel; final W is train-LR-only and conservative."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_evidence_repair_20260927';PRIOR=ROOT/'output/dynamic_sr_prior_guidance_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item()))
spec=importlib.util.spec_from_file_location('evidence_prior_stats',ROOT/'experiments/dynamic_sr_prior_guidance_20260927/audit_priors.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def project(xyz,cal):
    q=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T;h=q@np.asarray(cal['K_lr']).T
    return h[:,:2]/q[:,2:3],q[:,2]

def rankings(points,depth,unc,prior,cfg):
    if len(points)<cfg['min_tracks']:return dict(tracks=len(points),pairs=0,eligible=False)
    pairs=sorted(cKDTree(points).query_pairs(cfg['point_pair_distance'][1]));ret=[]
    for i,j in pairs:
        distance=np.linalg.norm(points[i]-points[j]);delta=abs(depth[i]-depth[j]);relative=delta/max(depth[i],depth[j])
        if distance>=cfg['point_pair_distance'][0] and relative>=cfg['relative_depth_difference']:ret.append((i,j))
    if not ret:return dict(tracks=len(points),pairs=0,eligible=False)
    ij=np.array(ret);prediction=old.sample(prior,points);wrongxy=points.copy();wrongxy[:,0]=(wrongxy[:,0]+cfg['wrong_shift'])%336;wrong=old.sample(prior,wrongxy)
    target=np.sign(1/depth[ij[:,0]]-1/depth[ij[:,1]]);a=prediction[ij[:,0]]-prediction[ij[:,1]];b=wrong[ij[:,0]]-wrong[ij[:,1]]
    valid=np.isfinite(a)&np.isfinite(b);certain=(np.abs(depth[ij[:,0]]-depth[ij[:,1]])>unc[ij[:,0]]+unc[ij[:,1]])&valid
    def result(mask):
        n=int(mask.sum());return dict(pairs=n,correct=int((np.sign(a[mask])==target[mask]).sum()),wrong_correct=int((np.sign(b[mask])==target[mask]).sum()),accuracy=float((np.sign(a[mask])==target[mask]).mean()) if n else None,wrong_accuracy=float((np.sign(b[mask])==target[mask]).mean()) if n else None,eligible=n>=cfg['min_pairs'])
    return dict(tracks=len(points),unfiltered=result(valid),uncertainty_filtered=result(certain),eligible=bool(certain.sum()>=cfg['min_pairs']))

def main():
    p=read(OUT/'protocol.json');cfg=p['depth'];geom=p['geometry'];mcfg=p['moments'];manifest=Path(p['manifest']['path']);m=read(manifest);obs={(o['camera_id'],o['frame_index']):o for o in m['observations']};root=manifest.parent
    dest=OUT/'depth_evidence_v1';dest.mkdir(exist_ok=False);(dest/'weights').mkdir();started=time.time();cv2.setNumThreads(4)
    caches={}
    for r in read(PRIOR/'remote_a100/depth_legal/complete.json')['rows']:
        f=PRIOR/'remote_a100/depth_legal'/r['path'];assert sha(f)==r['sha256'];caches[r['camera'],r['frame']]={k:v for k,v in np.load(f).items()}
    moments=OUT/'remote_a100/moments_v1';receipt=read(moments/'complete.json')
    for r in receipt['rows']:assert sha(moments/r['path'])==r['sha256']
    tracks=read(OUT/'support_v1/tracks.json');support_obs={(r['camera'],r['frame']):r for r in read(OUT/'support_v1/observations.json')}
    pool=[(y,x) for y in range(cfg['border'],252-31,cfg['patch_stride']) for x in range(cfg['border'],336-31,cfg['patch_stride'])];assert len(pool)==cfg['patches_per_observation']
    full=np.load(OUT/'support_v1/full_pair_support.npz');xyz=full['xyz'];versions=['full_pairs_old_clamped','full_pairs_raw_bicubic','full_pairs_area','dedup_sources_area','third_verified_area'];funnel=[];patchrows=[];rankrows=[];valid_corr=[];weights=[];content=[]
    for c in p['train_cameras']:
        cal=m['cameras'][c];xyall,zall=project(xyz,cal);lrtime=[]
        for f in p['frames']:lrtime.append(cv2.imread(str(root/obs[c,f]['lr_path'])).astype(np.float32)/255)
        change=np.sqrt(np.stack(lrtime).var(0).mean(2))>=.025
        for f in p['frames']:
            o=obs[c,f];lrpath=root/o['lr_path'];assert sha(lrpath)==o['lr_sha256'];gray=cv2.imread(str(lrpath),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8;internal=np.hypot(gx,gy)<cfg['LR_gradient_threshold'];prior=caches[c,f]
            mo=np.load(moments/f'{c}_{f:04d}.npz');legacy=np.load(PRIOR/f'remote_a100/u6000_moments/{c}_{f:04d}.npz')
            def mask(mode):
                a=mo[mode+'_alpha'];z=mo[mode+'_expected_z'];v=mo[mode+'_variance_raw'];tol=mcfg['variance_absolute_tolerance']+mcfg['variance_relative_tolerance']*z*z
                return np.isfinite(z)&np.isfinite(v)&(a>=mcfg['reference_alpha'])&(a<=1+mcfg['variance_absolute_tolerance'])&(v>=-tol)&(v<=mcfg['reference_var_ratio']*z*z)&(z>0)
            oldmask=(legacy['alpha']>=mcfg['reference_alpha'])&(legacy['variance_z']<=mcfg['reference_var_ratio']*np.maximum(legacy['expected_z']**2,1e-6))
            moment_masks={'bicubic':mask('bicubic'),'area':mask('area')}
            base=full['passes_spatial_filter']&(full['frames']==f)&(full['camera_pair']==int(c[3:])).any(1)&(zall>0)&np.isfinite(xyall).all(1)&(xyall[:,0]>=16)&(xyall[:,0]<320)&(xyall[:,1]>=16)&(xyall[:,1]<236)
            rawpts=xyall[base];verified=[t for t in tracks if t['frame']==f and c in t['projections']];sourcepts=[t['projections'][c]['projected_xy'] for t in verified if c in t['source_pair']]
            pts=np.array([t['projections'][c]['projected_xy'] for t in verified]).reshape(-1,2);depth=np.array([t['projections'][c]['z'] for t in verified]);unc=np.array([t['projections'][c]['z_uncertainty'] if t['projections'][c]['z_uncertainty'] is not None else np.inf for t in verified])
            inside=np.isfinite(pts).all(1)&(pts[:,0]>=16)&(pts[:,0]<320)&(pts[:,1]>=16)&(pts[:,1]<236)&(depth>0);pts,depth,unc=pts[inside],depth[inside],unc[inside]
            rank=rankings(pts,depth,unc,prior['lr'],geom);rankrows.append(dict(camera=c,frame=f,**rank))
            maps={}
            for key,v in [('pairs',rawpts),('sources',np.asarray(sourcepts).reshape(-1,2)),('third',pts)]:
                supportmap=np.zeros((252,336),np.int32);v=v[np.isfinite(v).all(1)&(v[:,0]>=0)&(v[:,0]<336)&(v[:,1]>=0)&(v[:,1]<252)]
                if len(v):ij=np.floor(v).astype(int);np.add.at(supportmap,(ij[:,1],ij[:,0]),1)
                maps[key]=supportmap
            near=np.zeros((252,336),bool)
            if len(pts):
                yy,xx=np.indices(near.shape);dist,_=cKDTree(pts).query(np.c_[xx.ravel()+.5,yy.ravel()+.5]);near=dist.reshape(near.shape)<=cfg['W_radius']
            stable=np.zeros((252,336),bool);counts={v:Counter(candidate=len(pool)) for v in versions};selected={v:np.zeros((252,336),bool) for v in versions}
            for y,x in pool:
                sl=np.s_[y:y+16,x:x+16];rs=[old.spearman(prior['lr'][sl],prior[k][sl]) for k in ['scale','crop']];valid_corr.extend(rs);ok=all(r is not None and r>=cfg['stability'] for r in rs);stable[sl]=ok;inter=internal[sl].mean()>=cfg['LR_interior_fraction']
                row=dict(camera=c,frame=f,y=y,x=x,stable=ok,LR_interior_fraction=float(internal[sl].mean()),spearman_scale=rs[0],spearman_crop=rs[1])
                for v in versions:
                    key='sources' if v=='dedup_sources_area' else 'third' if v=='third_verified_area' else 'pairs';supportok=maps[key][sl].sum()>=cfg['min_support_points'];model=oldmask if v=='full_pairs_old_clamped' else moment_masks['bicubic' if v=='full_pairs_raw_bicubic' else 'area'];modelok=model[sl].mean()>=cfg['model_valid_fraction']
                    accepted=ok and inter and supportok and modelok;selected[v][sl]=accepted
                    for k,val in [('stable',ok),('interior',ok and inter),('support',ok and inter and supportok),('moment',accepted),('final',accepted)]:counts[v][k]+=int(val)
                    row[v]=bool(accepted)
                patchrows.append(row)
            W=selected['third_verified_area']&internal&near&moment_masks['area']
            for y,x in pool:
                sl=np.s_[y:y+16,x:x+16]
                if W[sl].mean()<cfg['W_patch_fraction']:W[sl]=False
            wp=dest/'weights'/f'{c}_{f:04d}.npz';np.savez_compressed(wp,W=W);weights.append(dict(camera=c,frame=f,path=str(wp),sha256=sha(wp),fraction=float(W.mean()),nonzero=int(W.sum())))
            content.append(dict(camera=c,frame=f,nonzero_W=int(W.sum()),LR_change_W=int((W&change).sum())))
            for v in versions:
                funnel.append(dict(version=v,**support_obs[c,f],**{k+'_blocks':counts[v][k] for k in ['candidate','stable','interior','support','moment','final']},selected_block_fraction=float(selected[v].mean()),direct_support_pixel_fraction=float((maps['sources' if v=='dedup_sources_area' else 'third' if v=='third_verified_area' else 'pairs']>0).mean()),third_verified_direct_support_pixel_fraction=float((maps['third']>0).mean()),direct_support_version='sources' if v=='dedup_sources_area' else 'third' if v=='third_verified_area' else 'pairs',nonzero_W_fraction=float(W.mean()) if v=='third_verified_area' else None,source_path=str(dest/'patches.json')))
    write(dest/'patches.json',patchrows);write(dest/'rankings.json',rankrows);write(dest/'weights_manifest.json',weights);write(dest/'coverage_content.json',content)
    with (OUT/'support_funnel.csv').open('w') as f:
        keys=sorted({k for r in funnel for k in r});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(funnel)
    byversion={v:{c:float(np.mean([r['selected_block_fraction'] for r in funnel if r['version']==v and r['camera']==c])) for c in p['train_cameras']} for v in versions};wcoverage={c:float(np.mean([r['fraction'] for r in weights if r['camera']==c])) for c in p['train_cameras']}
    eligible=[r['uncertainty_filtered'] for r in rankrows if r.get('eligible')];unfiltered=[r['unfiltered'] for r in rankrows if r.get('unfiltered',{}).get('eligible')]
    def aggregate(rs):
        return dict(observations=len(rs),equal_observation_accuracy=float(np.mean([r['accuracy'] for r in rs])) if rs else None,equal_observation_wrong=float(np.mean([r['wrong_accuracy'] for r in rs])) if rs else None,pairs=sum(r['pairs'] for r in rs),pair_weighted_accuracy=sum(r['correct'] for r in rs)/sum(r['pairs'] for r in rs) if rs else None)
    rank=aggregate(eligible);sensitivity=aggregate(unfiltered);oldresult=read(PRIOR/'prior_audit_v2/complete.json');temporal=oldresult['temporal_pass_clips'];corr=old.summary(valid_corr)
    supported={r['camera'] for r in rankrows if r.get('eligible')}&set(cfg['geometry_spaced_cameras'])
    gates=dict(stability=corr['median']>=cfg['stability'],block_coverage=sum(v>=cfg['coverage'] for v in byversion['third_verified_area'].values())>=cfg['min_cameras'],geometry_cameras=len(supported)>=cfg['min_geometry_cameras'],rank=bool(rank['observations'] and rank['equal_observation_accuracy']>=cfg['order_accuracy'] and rank['equal_observation_accuracy']-rank['equal_observation_wrong']>=cfg['wrong_margin']),temporal=temporal>=cfg['min_temporal_clips'],area_engineering=receipt['area_engineering_passed'],actual_W_coverage=sum(v>=cfg['coverage'] for v in wcoverage.values())>=cfg['min_cameras'])
    schedule=read(Path(p['schedule']['path']));wby={(r['camera'],r['frame']):r['nonzero'] for r in weights};exposure=Counter()
    for step in range(6001,12001):
        li=schedule['rows'][step-1][0];camera,frame=schedule['record_keys'][li]
        if wby.get((camera,frame),0)>0:exposure[f'{camera}/{frame}']+=1
    gates['depth_step_support']=sum(exposure.values())>=cfg['min_nonzero_training_steps']
    result=dict(status='completed',gates=gates,passed=all(gates.values()),unrun_reason=[k for k,v in gates.items() if not v],versions=byversion,actual_W_coverage=wcoverage,main_rank=rank,unfiltered_rank=sensitivity,old_reported=dict(max_camera_coverage=max(v['mean_trusted_fraction'] for v in oldresult['coverage'].values()),order_accuracy=oldresult['order_accuracy'],wrong_order_accuracy=oldresult['wrong_order_accuracy']),stability=corr,temporal_pass_clips=temporal,nonzero_depth_steps=sum(exposure.values()),nonzero_exposure=dict(exposure),other_1064_observations_W=0,weights_are_final_training_weights=False,raw_cache_sha256=sha(moments/'complete.json'),protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),seconds=time.time()-started,
        note='Diagnostic W only. No training unless every gate plus P0 engineering passes; original 4-frame denominator retained. Pair sharing prevents independent-binomial inference.')
    write(dest/'complete.json',result);write(OUT/'depth_admission.json',result)

from collections import Counter
if __name__=='__main__':main()
