"""Separate privileged diagnostic HR; re-degraded LR must match prepared PNG hash."""
import argparse, concurrent.futures as cf, time, traceback
from pathlib import Path
from common import *
import prepare_colmap_dataset as prep
def camera_job(manifest,camera):
    import cv2,torch,hashlib
    cv2.setNumThreads(1);torch.set_num_threads(1);m=read(manifest);root=Path(manifest).parent;c=m['cameras'][camera];condition=m['scene'];rung='full' if condition=='dgF_cube' else 'half';dest=DATA/'hr_full'/condition
    rows={o['frame_index']:o for o in m['observations'] if o['camera_id']==camera};maps,_=prep.remap_maps(c);cap=cv2.VideoCapture(c['video'],cv2.CAP_FFMPEG);cap.set(cv2.CAP_PROP_POS_FRAMES,m['original_start']);hashes={};started=time.monotonic()
    for frame in range(300):
        ok,a=cap.read()
        if not ok:raise ValueError('HR re-decode ended early')
        hr=prep.transform_frame(a,c,rung,maps);lr=prep.downsample(hr);ok,b=cv2.imencode('.png',lr,[cv2.IMWRITE_PNG_COMPRESSION,3])
        if not ok or hashlib.sha256(b.tobytes()).hexdigest()!=rows[frame]['lr_sha256']:raise ValueError(f'HR LR roundtrip differs: {camera} {frame}')
        target=dest/rows[frame]['hr_path'];hashes[str(frame)]=prep.png(target,hr)
        if rows[frame]['hr_sha256'] and hashes[str(frame)]!=rows[frame]['hr_sha256']:raise ValueError('Sparse diagnostic HR differs')
    cap.release();return camera,dict(frames=300,hr_sha256=hashes,seconds=time.monotonic()-started)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--workers',default=4,type=int);a=p.parse_args();manifest=local(a.manifest);m=read(manifest);dest=DATA/'hr_full'/m['scene']
    if dest.exists():raise FileExistsError('Preserve existing HR expansion')
    disk_guard(dest,45);mkdir(dest);started=time.monotonic();results={}
    try:
        with cf.ProcessPoolExecutor(a.workers) as pool:
            futures=[pool.submit(camera_job,manifest,c) for c in m['splits']['train']]
            for f in cf.as_completed(futures):
                camera,r=f.result();results[camera]=r;print(camera,flush=True)
        write(dest/'hr_full_storage.json',dict(schema='highres_hr_full_storage_v1',manifest=entry(manifest),purpose='HRref/oracleHR privileged diagnostics only; never legal method supervision',cameras=results,source=entry(__file__),seconds=time.monotonic()-started,LR_roundtrip_all_frames_SHA_equal=True,GPU_calls=0))
    except BaseException:write(dest/f'failure_{time.time_ns()}.json',dict(status='failed_preserved',error=traceback.format_exc()));raise
if __name__=='__main__':main()
