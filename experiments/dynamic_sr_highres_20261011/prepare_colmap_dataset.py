"""CPU-only COLMAP video adapter. metadata/Z1 first; prepare rejects a failed gate.

Published sparse points are used only for calibration checks, never initialization.
All stored training inputs are exact bicubic-AA x4 LR from the selected HR grid.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, math, re, struct, subprocess, time, traceback
from pathlib import Path
from common import *

def models(cfg):
    loader=module('highres_colmap_loader', ROOT/'third_party/4dgs_843d5ac_runtime/scene/colmap_loader.py')
    raw=local(cfg['raw_directory']); d=raw/cfg['distorted_model']; u=raw/cfg['pinhole_model']
    return loader, raw, loader.read_intrinsics_binary(d/'cameras.bin'), loader.read_extrinsics_binary(d/'images.bin'), loader.read_intrinsics_binary(u/'cameras.bin'), loader.read_extrinsics_binary(u/'images.bin')

def scale_K(K, s):
    import numpy as np
    return np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1.]]) @ K

def model_K(cam):
    import numpy as np
    if cam.model=='PINHOLE': fx,fy,cx,cy=cam.params
    elif cam.model=='SIMPLE_PINHOLE': fx,cx,cy=cam.params; fy=fx
    elif cam.model=='OPENCV': fx,fy,cx,cy=cam.params[:4]
    else: raise ValueError(f'Unsupported camera model {cam.model}')
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],dtype=np.float64)

def metadata(cfg):
    import numpy as np
    loader,raw,dc,di,uc,ui=models(cfg)
    dn={i.name:i for i in di.values()}; result={}
    for im in ui.values():
        n=int(re.search(r'(\d+)',Path(im.name).stem).group(1))
        src=dn[im.name]; a,b=dc[src.camera_id],uc[im.camera_id]
        if a.model!='OPENCV': raise ValueError('Distorted calibration must be OPENCV')
        if (a.width,a.height)!=tuple(cfg['raw_size']): raise ValueError('Calibration/video raw grid differs')
        if not np.allclose(im.qvec,src.qvec,atol=1e-10) or not np.allclose(im.tvec,src.tvec,atol=1e-10):
            raise ValueError('Distorted/pinhole extrinsics differ')
        K=model_K(b); W,H=cfg['full_crop']
        x,y=(0,0) if cfg['crop_rule']=='upper_left' else (math.floor(K[0,2]-W/2),math.floor(K[1,2]-H/2))
        if x<0 or y<0 or x+W>b.width or y+H>b.height: raise ValueError('Crop outside pinhole image')
        crop=np.array([[1.,0,-x],[0,1.,-y],[0,0,1.]])
        w=np.eye(4); w[:3,:3]=loader.qvec2rotmat(im.qvec); w[:3,3]=im.tvec
        video=raw/cfg['video_directory']/cfg['video_rule'].format(number=n)
        if not video.is_file(): raise FileNotFoundError(video)
        result[n]=dict(original_number=n, image_name=im.name, video=str(video), raw_width=a.width,raw_height=a.height,
            K_raw=model_K(a).tolist(), K_pinhole=K.tolist(),K_full=(crop@K).tolist(),w2c=w.tolist(),c2w=np.linalg.inv(w).tolist(),
            near_far_metadata=None, near_far_reason='not read by training loader; culling derives near from legal LR points',
            undistortion=dict(model=a.model,coefficients=a.params[4:].tolist(),pinhole_size=[b.width,b.height],crop_offset=[x,y],full_crop=[W,H]),
            colmap_image_id=im.id, colmap_intrinsics_id=im.camera_id)
    expected=21 if cfg['dataset']=='DeskGames' else 36
    if set(result)!=set(range(expected)): raise ValueError('Incomplete published camera numbers')
    train=sorted(set(result)-set(cfg['test_original']))
    centers={n:np.array(result[n]['c2w'])[:3,3] for n in result}
    main=cfg['main_test_original']; diag=min(train,key=lambda n:(np.linalg.norm(centers[n]-centers[main]),n))
    others=[n for n in train if n!=diag]
    order=[main,diag,*others]
    mapping={n:f'cam{i:02d}' for i,n in enumerate(order)}
    center_rule=None
    if cfg['dataset']=='DeskGames':
        center=np.mean(list(centers.values()),axis=0)
        directions={n:np.array(result[n]['c2w'])[:3,2] for n in result}
        direction=np.mean(list(directions.values()),axis=0); direction/=np.linalg.norm(direction)
        rank=sorted(result,key=lambda n:(np.linalg.norm(centers[n]-center),math.acos(np.clip(directions[n]@direction,-1,1)),n))
        center_rule=dict(selected_original=rank[0],distances={str(n):float(np.linalg.norm(centers[n]-center)) for n in result})
        if rank[0]!=main: raise ValueError(f'Central-camera rule selects {rank[0]}, preregistered {main}; stop and report')
    return result,mapping,diag,center_rule

def remap_maps(cam):
    import cv2, numpy as np
    u=cam['undistortion']; K=np.array(cam['K_raw']); P=np.array(cam['K_pinhole']); size=tuple(u['pinhole_size'])
    maps=cv2.initUndistortRectifyMap(K,np.array(u['coefficients']),np.eye(3),P,size,cv2.CV_32FC1)
    gx=np.linspace(0,size[0]-1,50).round().astype(int); gy=np.linspace(0,size[1]-1,50).round().astype(int)
    yy,xx=np.meshgrid(gy,gx,indexing='ij'); pts=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3)@np.linalg.inv(P).T
    projected,_=cv2.projectPoints(pts,np.zeros(3),np.zeros(3),K,np.array(u['coefficients']))
    sampled=np.stack([maps[0][yy,xx],maps[1][yy,xx]],-1).reshape(-1,2)
    error=np.abs(projected.reshape(-1,2)-sampled).max()
    if error>.01: raise ValueError(f'Undistortion map error {error}')
    return maps,float(error)

def transform_frame(frame,cam,rung,maps=None):
    import cv2
    maps=maps or remap_maps(cam)[0]
    u=cam['undistortion']; x,y=u['crop_offset']; W,H=u['full_crop']
    f=cv2.remap(frame,*maps,interpolation=cv2.INTER_LANCZOS4)[y:y+H,x:x+W]
    if f.shape[:2]!=(H,W): raise ValueError('Crop shape differs')
    return cv2.resize(f,(W//2,H//2),interpolation=cv2.INTER_AREA) if rung=='half' else f

def downsample(hr):
    import numpy as np,torch,torch.nn.functional as F
    a=torch.from_numpy(hr.copy()).permute(2,0,1).float()[None]/255
    t=F.interpolate(a,size=(hr.shape[0]//4,hr.shape[1]//4),mode='bicubic',antialias=True,align_corners=False).clamp(0,1)
    return t[0].permute(1,2,0).mul(255).round().byte().numpy()

def png(path,a):
    import cv2
    ok,b=cv2.imencode('.png',a,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not ok: raise ValueError('PNG encode failure')
    p=Path(path); mkdir(p.parent)
    with p.open('xb') as f:f.write(b.tobytes())
    p.chmod(0o664);return sha(p)

def read_frame(video,frame):
    import cv2
    params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,params);cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,a=cap.read(); pos=cap.get(cv2.CAP_PROP_POS_FRAMES);cap.release()
    if not ok or abs(pos-(frame+1))>.1: raise ValueError('Frame seek failed')
    return a

def points_dict(path):
    import numpy as np
    result={}
    with Path(path).open('rb') as f:
        count=struct.unpack('<Q',f.read(8))[0]
        for _ in range(count):
            p=struct.unpack('<QdddBBBd',f.read(43));result[p[0]]=np.array(p[1:4]);n=struct.unpack('<Q',f.read(8))[0];f.seek(n*8,1)
    return result

def z1(cfg,cameras,mapping,rung='full'):
    """Geometric reprojection plus correct/wrong video mapping patch NCC, training views only."""
    import cv2,numpy as np
    loader,raw,dc,di,uc,ui=models(cfg); pts=points_dict(raw/cfg['pinhole_model']/'points3D.bin')
    errors=[]; per={}; samples={}; point_sets={}; images={i.name:i for i in ui.values()}
    train=[n for n in cameras if n not in cfg['test_original']]
    frames={};wrong={}; checks={}
    for n in train:
        c=cameras[n]; im=images[c['image_name']]; K=np.array(c['K_full']); R=np.array(c['w2c'])
        if rung=='half':K=scale_K(K,.5)
        keys=[(int(p),xy) for p,xy in zip(im.point3D_ids,im.xys) if p in pts]
        xyz=np.array([pts[p] for p,_ in keys]); pc=xyz@R[:3,:3].T+R[:3,3]; uv=pc[:,:2]/pc[:,2,None]@K[:2,:2].T+K[:2,2]
        obs=np.array([xy for _,xy in keys])-c['undistortion']['crop_offset']
        if rung=='half':obs=(obs+.5)*.5-.5
        e=np.linalg.norm(uv-obs,axis=1)
        errors.extend(e.tolist());per[str(n)]=dict(points=len(e),median=float(np.median(e)),p90=float(np.quantile(e,.9)))
        samples[n]={p:xy for (p,_),xy in zip(keys,obs)};point_sets[n]=set(samples[n])
        maps,checks[str(n)]=remap_maps(c)
        frames[n]=cv2.cvtColor(transform_frame(read_frame(c['video'],0),c,rung,maps),cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Deliberately wrong one-position mapping, using the same target camera geometry.
        wrong_n=(n+1)%len(cameras)
        wrong[n]=cv2.cvtColor(transform_frame(read_frame(cameras[wrong_n]['video'],0),c,rung,maps),cv2.COLOR_BGR2GRAY).astype(np.float32)
    centers={n:np.array(cameras[n]['c2w'])[:3,3] for n in train};pairs=set()
    for a in train:
        b=min((n for n in train if n!=a),key=lambda n:np.linalg.norm(centers[n]-centers[a]));pairs.add(tuple(sorted((a,b))))
    def patch(im,xy):
        x,y=np.rint(xy).astype(int)
        if x<5 or y<5 or x+5>=im.shape[1] or y+5>=im.shape[0]:return None
        p=im[y-5:y+6,x-5:x+6].ravel();p=p-p.mean();norm=np.linalg.norm(p)
        return p/norm if norm>10 else None
    correct=[]; incorrect=[];pair_stats=[]
    for a,b in sorted(pairs):
        good=[];bad=[]
        for p in sorted(point_sets[a]&point_sets[b])[::max(1,len(point_sets[a]&point_sets[b])//1000)]:
            x=patch(frames[a],samples[a][p]);y=patch(frames[b],samples[b][p]);z=patch(wrong[b],samples[b][p])
            if x is not None and y is not None and z is not None:good.append(float(x@y));bad.append(float(x@z))
        correct.extend(good);incorrect.extend(bad)
        pair_stats.append(dict(pair=[a,b],n=len(good),NCC_median=float(np.median(good)) if good else None,wrong_NCC_median=float(np.median(bad)) if bad else None))
    med=float(np.median(errors));p90=float(np.quantile(errors,.9));nc=float(np.median(correct)) if correct else -1.;nw=float(np.median(incorrect)) if incorrect else -1.
    passed=med<=.5 and p90<=1.5 and nc>=.7 and nc>nw+.05
    return dict(schema='highres_Z1_v1',rung=rung,pixel_unit='prepared HR pixel at selected rung',passed=passed,reprojection=dict(median_px=med,p90_px=p90,per_camera=per),NCC=dict(median=nc,wrong_mapping_median=nw,required_minimum=.7,minimum_margin=.05,pairs=pair_stats),undistortion_map_max_error_px=checks,official_points_used_for_initialization=False,training_views_only=True,limitation='Unwarped 11x11 NCC is sensitive to viewpoint, photometry and motion; a failure blocks GPU chains pending disclosed diagnosis, not proof calibration is wrong.')

def select_window(cfg,cameras,diag):
    import cv2,numpy as np,torch
    torch.set_num_threads(1);cv2.setNumThreads(1)
    c=cameras[diag];maps,_=remap_maps(c);params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else [];cap=cv2.VideoCapture(c['video'],cv2.CAP_FFMPEG,params)
    prev=None;energy=[];count=0
    while True:
        ok,f=cap.read()
        if not ok:break
        a=downsample(transform_frame(f,c,'half',maps)).astype(np.int16)
        energy.append(0. if prev is None else float((np.max(np.abs(a-prev),axis=2)>8).mean()));prev=a;count+=1
    cap.release()
    if count!=cfg['frames']:raise ValueError(f'Window selector decoded {count}/{cfg["frames"]}')
    scored=[]
    for start in cfg['window_candidates']:
        values=energy[start+1:start+300]
        sub=[float(np.mean(energy[start+i*30+(1 if i==0 else 0):start+(i+1)*30])) for i in range(10)]
        scored.append(dict(start=start,mean_motion_energy=float(np.mean(values)),ten_subwindow_mean=sub,all_ten_have_motion=all(x>0 for x in sub)))
    eligible=[s for s in scored if s['all_ten_have_motion']]
    if not eligible:raise ValueError('No candidate has motion in all ten subwindows')
    chosen=max(eligible,key=lambda x:(x['mean_motion_energy'],-x['start']))
    return dict(selected_start=chosen['start'],selected_original_camera=diag,selection_resolution='half rung LR; identical chosen window shared by both rungs to avoid time confound',threshold_uint8=8,metric='fraction of pixels with maximum channel consecutive difference >8',source_decoded_count=count,candidates=scored,test_views_read=False)

def packet_audit(video,cfg,path):
    import numpy as np
    ffprobe=ROOT/'scripts/ffprobe_local.sh'
    cmd=[str(ffprobe),'-v','error','-threads','1','-select_streams','v:0','-show_streams','-show_packets','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:packet=pts_time','-of','json',str(video)]
    p=json.loads(subprocess.check_output(cmd));write(path,p)
    stream=p['streams'][0];pts=sorted(float(x['pts_time']) for x in p['packets']);expected=np.arange(cfg['frames'])/cfg['fps']
    if [stream['width'],stream['height']]!=cfg['raw_size'] or len(pts)!=cfg['frames'] or not np.allclose(pts,expected,atol=1e-6,rtol=0):raise ValueError('Packet count/grid/PTS differs from preregistration')
    return dict(packet_count=len(pts),PTS_max_error_seconds=float(np.max(np.abs(pts-expected))),stream=stream,source=entry(video),packet_record=entry(path))

def prepare_camera(cfg,cam,camera_id,start,rungs,destinations,extra=False):
    import cv2,torch
    torch.set_num_threads(1);cv2.setNumThreads(1);begin=time.monotonic();maps,error=remap_maps(cam)
    first_dest=Path(destinations[rungs[0]])
    audit=packet_audit(cam['video'],cfg,first_dest/f'{camera_id}_packets.json');params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []; cap=cv2.VideoCapture(cam['video'],cv2.CAP_FFMPEG,params)
    rows={r:[] for r in rungs};count=0
    while True:
        ok,f=cap.read()
        if not ok:break
        if start<=count<start+300:
            index=count-start
            for rung in rungs:
                hr=transform_frame(f,cam,rung,maps);lr=downsample(hr);dest=Path(destinations[rung]);hp=f'hr/{camera_id}/{index:04d}.png';lp=f'lr/{camera_id}/{index:04d}.png'
                stored=extra or camera_id=='cam00' or (camera_id=='cam01' and index%30==15)
                rows[rung].append(dict(camera_id=camera_id,frame_index=index,original_frame_index=count,source_PTS_seconds=count/cfg['fps'],time=index/300.,split='test' if extra or camera_id=='cam00' else 'train',hr_path=hp,hr_sha256=png(dest/hp,hr) if stored else None,lr_path=lp,lr_sha256=png(dest/lp,lr)))
        count+=1
    cap.release()
    if count!=cfg['frames'] or any(len(v)!=300 for v in rows.values()):raise ValueError(f'Actual decoded frames {count}, expected {cfg["frames"]}')
    if entry(cam['video'])!=audit['source']:raise ValueError('Video changed during decode')
    receipt=dict(status='completed_actual_full_video_decode_selected_window',actual_decoded_count=count,selected_frame_count=300,original_start=start,seconds=time.monotonic()-begin,undistortion_map_error_px=error,**audit)
    for rung in rungs:write(Path(destinations[rung])/f'{camera_id}_decode_receipt.json',dict(decode=receipt,observations=rows[rung]))
    return camera_id,rows,receipt

def prepare(cfg,rungs,out_root,workers):
    import numpy as np
    cameras,mapping,diag,center=metadata(cfg); diagroot=mkdir(OUT/'zero_checks'/cfg['id'])
    gates={}
    for rung in rungs:
        target=diagroot/f'z1_{rung}.json'
        gate=read(target) if target.exists() else z1(cfg,cameras,mapping,rung)
        write(target,gate);gates[rung]=target
        if not gate['passed']:raise RuntimeError(f'Z1 failed: {target}; no data/GPU chain started')
    window_path=diagroot/'window_selection.json';window=read(window_path) if window_path.exists() else select_window(cfg,cameras,diag);write(window_path,window)
    destinations={r:Path(out_root)/cfg['rungs'][r] for r in rungs}
    extras={r:Path(out_root)/(cfg['rungs'][r]+'_extra_test') for r in rungs} if cfg['dataset']=='Long360' else {}
    for d in [*destinations.values(),*extras.values()]:
        if d.exists():raise FileExistsError(f'Preserve completed/partial destination {d}')
    disk=disk_guard(out_root,12 if cfg['dataset']=='Long360' else 10)
    for d in [*destinations.values(),*extras.values()]:mkdir(d)
    for r,d in destinations.items():write(d/'window_selection.json',window);write(d/'camera_mapping.json',dict(mapping={str(n):c for n,c in mapping.items()},diagnostic_original=diag,main_test_original=cfg['main_test_original'],additional_tests=cfg['test_original'][1:],central_rule=center));write(d/'dataset_spec.json',cfg)
    allrows={r:[] for r in rungs};extrrows={r:[] for r in rungs};receipts=[];started=time.monotonic()
    try:
        with cf.ProcessPoolExecutor(workers) as pool:
            jobs={}
            for n,c in cameras.items():
                extra=n in cfg['test_original'] and n!=cfg['main_test_original'];cid=f'extra{n:02d}' if extra else mapping[n]
                f=pool.submit(prepare_camera,cfg,c,cid,window['selected_start'],rungs,{r:str(d) for r,d in (extras if extra else destinations).items()},extra);jobs[f]=extra
            for f in cf.as_completed(jobs):
                cid,rows,receipt=f.result();receipts.append(receipt)
                for r in rungs:(extrrows if jobs[f] else allrows)[r].extend(rows[r])
                print(json.dumps(dict(stage='camera_prepared',dataset=cfg['id'],camera=cid,seconds=receipt['seconds'])),flush=True)
        for rung,dest in destinations.items():
            condition=cfg['rungs'][rung];W,H=CONDITIONS[condition];cams={}
            for n,cid in mapping.items():
                c=dict(cameras[n],camera_id=cid);K=np.array(c['K_full']);K=scale_K(K,.5) if rung=='half' else K
                c['K_hr']=K.tolist();c['K_lr']=scale_K(K,.25).tolist();cams[cid]=c
            observations=sorted(allrows[rung],key=lambda x:(x['camera_id'],x['frame_index']))
            m=dict(schema='n3dv_dynamic_sr_pilot_v1',scene=condition,dataset_scene=cfg['scene'],source=cfg['source'],scale=4,raw_directory=cfg['raw_directory'],frame_indices=list(range(300)),time_convention='selected_window_frame_index/300; original PTS retained',resolutions=dict(hr=[W,H],lr=[W//4,H//4]),cameras=cams,splits=dict(train=sorted(set(cams)-{'cam00'}),dev=[],test=['cam00']),observations=observations,official_split=cfg['dataset']=='Long360',complete_background_preserved=True,full_time_decode_verified=True,role='development',initialization=dict(status='not_prepared; train-LR-only triangulation required'),frozen_SR_cache=dict(status='not_prepared'),source_sha256=sha(__file__),original_start=window['selected_start'],source_fps=cfg['fps'],preparation_seconds=time.monotonic()-started,decode_audit=receipts)
            write(dest/'manifest.json',m)
            stored=[o['hr_path'] for o in observations if o['hr_sha256'] is not None]
            write(dest/'hr_storage.json',dict(schema='highres_sparse_HR_materialization_v1',materialized_hr_paths=stored,not_materialized_hr_paths=[o['hr_path'] for o in observations if o['hr_sha256'] is None],not_materialized_hr_sha256=None,train_HR_supervision=False))
            write(dest/'complete.json',dict(status='completed_highres_full_time_LR_sparse_HR_inputs',manifest=entry(dest/'manifest.json'),z1=entry(gates[rung]),window=entry(dest/'window_selection.json'),all_camera_LR_count=len(observations),materialized_HR_count=len(stored),disk_guard=disk,CPU_seconds=time.monotonic()-started,GPU_calls=0))
            if extras:
                em=dict(m,cameras={},splits=dict(train=[],dev=[],test=[f'extra{n:02d}' for n in cfg['test_original'][1:]]),observations=sorted(extrrows[rung],key=lambda x:(x['camera_id'],x['frame_index'])))
                for n in cfg['test_original'][1:]:
                    c=dict(cameras[n],camera_id=f'extra{n:02d}');K=scale_K(np.array(c['K_full']),.5);c['K_hr']=K.tolist();c['K_lr']=scale_K(K,.25).tolist();em['cameras'][c['camera_id']]=c
                write(extras[rung]/'manifest_extra_test.json',em)
        return destinations
    except BaseException:
        for d in destinations.values():write(d/f'failure_{time.time_ns()}.json',dict(status='failed_preparation_preserved',error=traceback.format_exc(),automatic_retry=False))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);p.add_argument('--rungs',nargs='+',choices=['half','full'],default=['half']);p.add_argument('--out-root',type=Path,default=DATA/'prepared');p.add_argument('--workers',type=int,default=4);p.add_argument('--metadata-only',action='store_true');p.add_argument('--z1-only',action='store_true');p.add_argument('--z1-rung',choices=['half','full'],default='full');p.add_argument('--select-window-only',action='store_true');a=p.parse_args();cfg=read(a.dataset)
    for r in a.rungs:
        if r not in cfg['rungs']:raise ValueError('Rung not authorized in dataset configuration')
    cameras,mapping,diag,center=metadata(cfg)
    if a.metadata_only:
        print(json.dumps(dict(cameras=cameras,mapping=mapping,diagnostic_original=diag,central_rule=center),ensure_ascii=False));return
    if a.z1_only:
        target=OUT/'zero_checks'/cfg['id']/f'z1_{a.z1_rung}.json';result=z1(cfg,cameras,mapping,a.z1_rung);write(target,result);print(json.dumps(result));return
    if a.select_window_only:
        target=OUT/'zero_checks'/cfg['id']/'window_selection.json';result=select_window(cfg,cameras,diag);write(target,result);print(json.dumps(result));return
    prepare(cfg,a.rungs,local(a.out_root),a.workers)
if __name__=='__main__':main()
