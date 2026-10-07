"""CPU-only official full-time video audit, acquisition and isolated preparation.

Existing short-window records are preserved. No GPU/SR model is invoked. The
unseen scenes are acquired from actual official release assets; preparation is
not method selection and no confirmation-camera metric is computed here.
"""
from __future__ import annotations
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from fp_common import ROOT, OUT, read, write, entry, sha, module

BASE = ROOT/'data/dynamic_sr/full_time_20261007'
DATA_OUT = OUT/'full_data_readiness'
N3DV_REPO = 'https://github.com/facebookresearch/Neural_3D_Video'
MEET_REPO = 'https://github.com/AlgoHunt/StreamRF'
RELEASE_API = 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0'
CONFIRMATION_SCENES = ('coffee_martini','flame_steak')
SCENES = ('cook_spinach','cut_roasted_beef','meetroom_discussion','meetroom_vrheadset')+CONFIRMATION_SCENES
OFFICIAL = dict(
    n3dv=dict(repository=N3DV_REPO,split='cam00 test, all other supplied synchronized camera streams train',
        video_order='sorted existing camNN.mp4; missing stream IDs remain absent; each pose row follows sorted existing videos',
        temporal_protocol='all 300 synchronized frames 0..299 for these six selected/available assets; verify decoded count, not infer from short window',
        source_urls=[N3DV_REPO+'/blob/main/README.md', MEET_REPO+'/blob/main/README.md']),
    meetroom=dict(repository=MEET_REPO,split='author configs meetroom_init/full llffhold=100 => sorted index0 (cam00) test; all other 12 train',
        video_order='numeric raw cam_N.mp4 to zero-padded camNN; poses follow numeric ordering',
        temporal_protocol='author README --frame_end 300 --fps 30; prepare_dataset.py defaults 300',
        source_urls=[MEET_REPO+'/blob/main/README.md',MEET_REPO+'/blob/main/prepare_dataset.py',MEET_REPO+'/blob/main/configs/meetroom_full.json']),
    resolutions=dict(n3dv=dict(hr=[1344,1008],lr=[336,252],
        operation='raw2704x2028 area to1352x1014 then crop x4..1347,y3..1010; same historical explicit camera transform'),
        meetroom=dict(hr=[1280,720],lr=[320,180],operation='native raw HR')), 
    degradation='torch bicubic antialias=True align_corners=False, clamp[0,1], uint8 round; scale4',
    independence='Cook/Cut/Meet discussion/vrheadset have already been used for development; coffee_martini/flame_steak reserved before any metric inspection',
    scope='two held confirmation scenes are early external evidence, not a complete official N3DV paper main table',
    synchronization_limit='matching complete file PTS and supplied official synchronization do not independently prove physical shutter synchronization')


def raw_path(scene):
    if scene.startswith('meetroom_'):return ROOT/'data/dynamic_sr/meetroom_raw'/scene.removeprefix('meetroom_')
    old=ROOT/'data/dynamic_sr/n3dv_raw'/scene
    return old if old.exists() else BASE/'raw'/scene


def official_snapshots():
    import requests
    dest=DATA_OUT/'official_sources';dest.mkdir(parents=True,exist_ok=True)
    urls={
        'n3dv_README.md':'https://raw.githubusercontent.com/facebookresearch/Neural_3D_Video/main/README.md',
        'streamrf_README.md':'https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/README.md',
        'streamrf_prepare_dataset.py':'https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/prepare_dataset.py',
        'streamrf_meetroom_full.json':'https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/configs/meetroom_full.json',
        'streamrf_meetroom_init.json':'https://raw.githubusercontent.com/AlgoHunt/StreamRF/main/configs/meetroom_init.json'}
    refs=[]
    for name,url in urls.items():
        p=dest/name
        if not p.exists():
            # GitHub Contents API is an official fallback when the raw host
            # times out. Preserve actual endpoint and content identity.
            try:
                r=requests.get(url,timeout=(10,20));r.raise_for_status();content=r.content;actual=url
            except requests.RequestException:
                parts=url.split('raw.githubusercontent.com/',1)[1].split('/')
                actual='https://api.github.com/repos/'+parts[0]+'/'+parts[1]+'/contents/'+'/'.join(parts[3:])+'?ref='+parts[2]
                r=requests.get(actual,timeout=(10,30));r.raise_for_status()
                payload=r.json()
                if payload.get('encoding')!='base64':raise ValueError('Unexpected official Contents API encoding')
                content=base64.b64decode(payload['content'])
            p.write_bytes(content)
            write(p.with_suffix(p.suffix+'.receipt.json'),dict(requested_url=url,actual_url=actual,**entry(p)))
        refs.append(dict(url=url,**entry(p)))
    p=dest/'n3dv_release_v1.0.json'
    if not p.exists():
        r=requests.get(RELEASE_API,timeout=60);r.raise_for_status();write(p,r.json())
    refs.append(dict(url=RELEASE_API,**entry(p)))
    write(dest/'index.json',dict(status='fetched_primary_sources',sources=refs,policy=OFFICIAL,
         queried_unix=time.time(),source_sha256=sha(__file__)))
    return read(p)


def acquire(scene,release):
    if scene not in CONFIRMATION_SCENES:raise ValueError('Only pre-registered two new confirmation assets may be acquired')
    asset=next(a for a in release['assets'] if a['name']==scene+'.zip')
    archives=BASE/'archives';archives.mkdir(parents=True,exist_ok=True)
    archive=archives/asset['name'];receipt=archive.with_suffix('.receipt.json');t0=time.monotonic()
    if not archive.exists():
        partial=archive.with_suffix('.zip.part')
        # curl verifies HTTPS; resumable single connection, no simultaneous broad dataset download.
        subprocess.run(['curl','--fail','--location','--continue-at','-','--retry','3',
            '--connect-timeout','15','--max-time','3600','--silent','--show-error',
            '--header','Accept: application/octet-stream',asset['url'],'--output',str(partial)],check=True)
        if partial.stat().st_size!=asset['size']:raise ValueError('Official asset byte size mismatch')
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:raise ValueError('Official ZIP CRC failed')
        partial.replace(archive)
    if archive.stat().st_size!=asset['size']:raise ValueError('Existing archive size mismatch')
    h=sha(archive)
    if receipt.exists() and read(receipt)['archive']['sha256']!=h:raise ValueError('Acquired archive changed')
    destination=BASE/'raw'/scene;destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        videos=[n for n in z.namelist() if Path(n).name.startswith('cam') and n.endswith('.mp4')]
        poses=[n for n in z.namelist() if Path(n).name=='poses_bounds.npy']
        if not videos or len(poses)!=1:raise ValueError('Unexpected official archive contents')
        names={}
        for member in videos+poses:
            name=Path(member).name
            if name in names:raise ValueError('Ambiguous official archive basename')
            names[name]=member;p=destination/name;size=z.getinfo(member).file_size
            if p.exists():
                if p.stat().st_size!=size:raise ValueError('Partial extracted source; retain and repair separately')
            else:
                temporary=p.with_suffix(p.suffix+'.part')
                with z.open(member) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
                temporary.replace(p)
    files=[dict(**entry(p),bytes=p.stat().st_size) for p in sorted(destination.glob('*')) if p.is_file()]
    result=dict(status='completed_actual_official_raw_scene_acquired',scene=scene,official_asset_id=asset['id'],
        official_url=asset['browser_download_url'],actual_download_API=asset['url'],
        transport='official GitHub asset API redirect to release-assets.githubusercontent.com; direct github.com timed out and was archived',official_bytes=asset['size'],archive=entry(archive),
        archive_CRC_verified=True,raw_files=files,seconds=time.monotonic()-t0,
        confirmation_role='reserved before any validation metric or visual method comparison',
        decoded_complete_time=False,SR_cache_ready=False,GPU_calls=0,formal_parameter_updates=0)
    if not receipt.exists():write(receipt,result)
    else:result=read(receipt)
    write(DATA_OUT/(scene+'_acquisition.json'),result);print(json.dumps(dict(stage='official_scene_acquired',scene=scene,cameras=len(videos),seconds=result['seconds'])),flush=True)
    return result


def inventory():
    rows=[]
    historical=read(ROOT/'output/dynamic_sr_confidence_geometry_20261006/future_benchmark_readiness.json')
    for scene in SCENES:
        raw=raw_path(scene);videos=sorted(raw.glob('*.mp4')) if raw.exists() else []
        prepared=BASE/'prepared'/scene/'manifest.json'
        full=read(prepared) if prepared.exists() else None
        old=next((r for r in historical['scenes'] if r['scene']==scene),None)
        rows.append(dict(scene=scene,role='held_confirmation' if scene in CONFIRMATION_SCENES else 'already_used_development',
            raw_directory=str(raw.relative_to(ROOT)),raw_available=bool(videos and (raw/'poses_bounds.npy').exists()),
            raw_video_count=len(videos),raw_video_bytes=sum(p.stat().st_size for p in videos),
            videos=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in videos],
            prior_short_window=None if old is None else dict(manifest=old['manifest'],frames=old['prepared_frames'],cameras=old['prepared_camera_count'],splits=old['splits']),
            full_time_manifest=None if full is None else entry(prepared),
            full_decode_and_inputs_ready=bool(full and full.get('full_time_decode_verified')),
            formal_SR_cache_ready=False,independent_LR_prefix_seeds_ready=False,
            missing=([] if full else ['complete 0..299 packet/PTS and actual decoding checks','isolated full-time LR/HR manifest and hashes'])+['legal full-time complete SR cache','independent LR prefix per full-scene seed'],
            raw_size_or_download_seconds='unknown until acquired' if not videos else 'listed actual bytes'))
    confirmed=[r['scene'] for r in rows if r['role']=='held_confirmation' and r['raw_available']]
    result=dict(status='inventory_complete_full_preparation_in_progress',created_unix=time.time(),scenes=rows,
        official_protocol=OFFICIAL,at_least_two_actual_confirmation_raw_scenes_registered=len(confirmed)>=2,
        actual_confirmation_scenes=confirmed,full_paper_benchmark_ready=False,
        short_window_equals_complete_time=False,GPU_calls=0,training_updates=0,
        source_sha256=sha(__file__),historical_inventory=entry(ROOT/'output/dynamic_sr_confidence_geometry_20261006/future_benchmark_readiness.json'),
        disk_free_bytes=shutil.disk_usage(ROOT).free)
    write(DATA_OUT/'inventory.json',result);return result


def _packet_audit(video,dest):
    import numpy as np
    ffprobe=ROOT/'scripts/ffprobe_local.sh'
    if not ffprobe.exists():raise ValueError('Project ffprobe wrapper missing')
    command=[str(ffprobe),'-v','error','-threads','1','-select_streams','v:0',
        '-show_streams','-show_packets','-show_entries',
        'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time:packet=pts_time,dts_time',
        '-of','json',str(video)]
    data=json.loads(subprocess.check_output(command));write(dest,data)
    stream,=data['streams'];pts=sorted(float(p['pts_time']) for p in data['packets'])
    if len(pts)!=300 or not np.allclose(pts,np.arange(300)/30,atol=1e-6):
        raise ValueError(f'Official selected-scene 300-frame/30fps PTS mismatch: {video}')
    return dict(stream=stream,packet_count=len(pts),sorted_PTS_max_error_seconds=float(np.max(np.abs(np.asarray(pts)-np.arange(300)/30))),
        PTS_sort_reason='codec packet order contains B-frame reordering; display PTS sorted for frame-index correspondence',
        packet_audit=entry(dest),ffprobe_wrapper=entry(ffprobe))


def prepare(scene):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads()!=1 else None
    raw=raw_path(scene)
    if not raw.exists():raise ValueError(f'Raw scene unavailable: {scene}')
    dest=BASE/'prepared'/scene;dest.mkdir(parents=True,exist_ok=True)
    target=dest/'manifest.json'
    if target.exists():
        old=read(target)
        if old.get('source_sha256')!=sha(__file__):raise ValueError('Full prepared source changed; use another version')
        for row in old['observations']:
            for kind in ('lr','hr'):
                if sha(dest/row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Full prepared cache byte corruption')
        return old
    meeting=scene.startswith('meetroom_')
    sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
    if meeting:parser=module('full_meet_parser',ROOT/'experiments/dynamic_sr_20260919/prepare_meetroom.py').parse_meetroom_cameras
    else:parser=module('full_n3dv_parser',ROOT/'experiments/dynamic_sr_20260918/n3dv_data.py').parse_n3dv_cameras
    cameras=parser(raw);ordered=sorted(cameras);audit=[];observations=[];t0=time.monotonic()
    for c in ordered:
        cal=cameras[c];video=Path(cal['video']);receipt=dest/(c+'_decode_receipt.json')
        if receipt.exists():
            r=read(receipt)
            if r['video']['sha256']!=sha(video) or r['source_sha256']!=sha(__file__):raise ValueError('Full decode restart identity mismatch')
            for row in r['observations']:
                for kind in ('lr','hr'):
                    if sha(dest/row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Completed full decode camera changed')
            observations+=r['observations'];audit.append(r['decode']);continue
        packets=_packet_audit(video,dest/(c+'_packets.json'));tcam=time.monotonic()
        for kind in ('lr','hr'):(dest/kind/c).mkdir(parents=True,exist_ok=True)
        params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
        cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,params)
        if not cap.isOpened():raise ValueError(f'Cannot decode {video}')
        rows=[];frame_number=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame_number>=300:raise ValueError('More frames than official selected-scene300 protocol; do not silently use all decoded frames')
                if frame.shape[:2]!=(cal['raw_height'],cal['raw_width']):raise ValueError('Calibration/raw resolution mismatch')
                if meeting:hr=frame
                else:hr=cv2.resize(frame,(1352,1014),interpolation=cv2.INTER_AREA)[3:1011,4:1348].copy()
                size=(180,320) if meeting else (252,336)
                t=torch.from_numpy(hr.copy()).permute(2,0,1).float().div_(255)[None]
                low=F.interpolate(t,size=size,mode='bicubic',antialias=True,align_corners=False).clamp_(0,1)
                lr=low[0].permute(1,2,0).mul_(255).round_().byte().numpy()
                row=dict(camera_id=c,frame_index=frame_number,time=frame_number/300.0,split='test' if c=='cam00' else 'train')
                for kind,array in (('hr',hr),('lr',lr)):
                    p=dest/kind/c/f'{frame_number:04d}.png'
                    # Retry a partial unregistered camera in this independent
                    # new destination without touching historical images.
                    if not cv2.imwrite(str(p),array,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise ValueError('PNG write failure')
                    row[kind+'_path']=str(p.relative_to(dest));row[kind+'_sha256']=sha(p)
                rows.append(row);frame_number+=1
        finally:cap.release()
        if frame_number!=300:raise ValueError(f'Actual decoder returned {frame_number}/300 frames: {video}')
        da=dict(camera=c,full_actual_decoded_frames=frame_number,seconds=time.monotonic()-tcam,**packets,
            physical_sync_independently_verified=False)
        record=dict(status='completed_actual_full_video_decode',video=entry(video),source_sha256=sha(__file__),
            observations=rows,decode=da,CPU_threads=1,GPU_calls=0)
        write(receipt,record);observations+=rows;audit.append(da)
        print(json.dumps(dict(stage='complete_camera_decode',scene=scene,camera=c,frames=frame_number,seconds=da['seconds'])),flush=True)
    m=dict(schema='n3dv_dynamic_sr_pilot_v1',scene=scene,source=MEET_REPO if meeting else N3DV_REPO,
        scale=4,raw_directory=str(raw),poses_bounds_sha256=sha(raw/'poses_bounds.npy'),
        frame_indices=list(range(300)),time_convention='original_frame_index / 300.0',
        resolutions=dict(hr=[1280,720] if meeting else [1344,1008],lr=[320,180] if meeting else [336,252]),
        cameras=cameras,splits=dict(train=[c for c in ordered if c!='cam00'],dev=[],test=['cam00']),
        observations=observations,official_split=True,additional_cam01_dev_holdout=False,
        old_short_window_not_overwritten=True,complete_background_preserved=True,
        full_time_decode_verified=True,all_packet_PTS_equal_registered_30fps=True,
        role='held_confirmation' if scene in CONFIRMATION_SCENES else 'already_used_development',
        confirmation_metrics_or_method_visual_comparison_performed=False,
        source_sha256=sha(__file__),official_policy=OFFICIAL,decode_audit=audit,
        preparation_seconds=time.monotonic()-t0,CPU_threads=1,GPU_calls=0,parameter_updates=0,
        initialization=dict(status='not_prepared; legal LR-only prefix initialization per full-scene seed required'),
        frozen_SR_cache=dict(status='not_prepared; full legal training LR only; root will allocate GPU'))
    write(target,m)
    write(DATA_OUT/(scene+'_full_preparation.json'),dict(status='completed_full_time_LR_HR_inputs',manifest=entry(target),
        scene=scene,cameras=len(cameras),frames=300,observations=len(observations),seconds=m['preparation_seconds'],GPU_calls=0))
    inventory();return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('--acquire',nargs='*',default=[])
    p.add_argument('--prepare',nargs='*',default=[]);p.add_argument('--execution-name',choices=['existing_development','confirmation']);a=p.parse_args();DATA_OUT.mkdir(parents=True,exist_ok=True)
    if a.acquire:
        release=official_snapshots();inventory()
        for s in a.acquire:acquire(s,release);inventory()
    for s in a.prepare:prepare(s)
    result=inventory();print(json.dumps(dict(status=result['status'],actual_confirmation_scenes=result['actual_confirmation_scenes'])),flush=True)

if __name__=='__main__':
    began=time.time();DATA_OUT.mkdir(parents=True,exist_ok=True)
    label=sys.argv[sys.argv.index('--execution-name')+1] if '--execution-name' in sys.argv else None
    if label not in (None,'existing_development','confirmation'):raise ValueError('Unknown execution name')
    state_path=DATA_OUT/('execution_state_'+label+'.json' if label else 'execution_state.json')
    write(state_path,dict(status='running',started_unix=began,command=sys.argv,source_sha256=sha(__file__),GPU_calls=0))
    try:
        main()
    except BaseException as e:
        write(state_path,dict(status='failed_saved_partial_outputs',started_unix=began,finished_unix=time.time(),command=sys.argv,error=repr(e),source_sha256=sha(__file__),GPU_calls=0))
        raise
    else:
        write(state_path,dict(status='completed_requested_CPU_data_preparation',started_unix=began,finished_unix=time.time(),command=sys.argv,source_sha256=sha(__file__),GPU_calls=0))
