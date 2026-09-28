"""Actual quantized teacher PNG quality on fixed train76; no 3D renders or updates."""
import argparse
import time
from dv_common import *
import numpy as np
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['single_image','repeat7','video7'],required=True);a=ap.parse_args()
    p=read(OLD/'protocol.json');assert os.environ['CUDA_VISIBLE_DEVICES']==p['evaluation']['physical_gpu']
    shared=setup();ev=shared.evaluator();torch.set_num_threads(4)
    from n3dv_data import load_manifest
    from detail_loss import highpass
    manifest=load_manifest(bound(p['manifest']));root=Path(manifest['_root']);start=time.monotonic()
    index=bound(p['teacher']) if a.mode=='single_image' else OUT/'priors'/a.mode/'prior_index.json'
    by={(e['camera'],e['frame']):e for e in read(index)['entries']}
    observations=sorted([o for o in manifest['observations'] if o['split']=='train' and o['frame_index'] in [0,40,80,118]],key=lambda o:(o['camera_id'],o['frame_index']))
    assert len(observations)==76
    dest=OUT/'teacher_quality'/a.mode;dest.mkdir(parents=True,exist_ok=False)
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for o in observations:
            e=by[o['camera_id'],o['frame_index']];path=root/e['relative_path'] if a.mode=='single_image' else local(e['path'])
            assert sha(path)==e['sha256'];target=ev.legacy.read_rgb(path)
            hrpath=root/o['hr_path'];assert sha(hrpath)==o['hr_sha256'];hr=ev.legacy.read_rgb(hrpath)
            lrpath=root/o['lr_path'];assert sha(lrpath)==o['lr_sha256'];lr=ev.legacy.read_rgb(lrpath)
            q=ev.legacy.spatial_metrics(target,hr,np.zeros(hr.shape[:2],bool),metric)['full']
            t=torch.from_numpy(target).permute(2,0,1).cuda();h=torch.from_numpy(hr).permute(2,0,1).cuda()
            hh=float((highpass(t,lr.shape[:2])-highpass(h,lr.shape[:2])).abs().double().mean())
            low=ev.lr_reprojection_metrics(t,lr,np.zeros(hr.shape[:2],bool))['full']
            rows.append(dict(mode=a.mode,camera=o['camera_id'],frame=o['frame_index'],psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],teacher_to_HR_H=hh,lr_l1=low['l1'],target_sha256=e['sha256']))
    metrics=['psnr','ssim','lpips','teacher_to_HR_H','lr_l1']
    cameras={c:{k:float(np.mean([r[k] for r in rows if r['camera']==c])) for k in metrics} for c in p['evaluation']['train_cameras']}
    write(dest/'metrics.json',dict(mode=a.mode,index=entry(index),rows=rows,by_camera=cameras,aggregate={k:float(np.mean([v[k] for v in cameras.values()])) for k in metrics},
        script=entry(Path(__file__)),metric_source=entry(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),seconds=time.monotonic()-start,rgb_renders=0,updates=0,
        interpretation='Actual quantized training target PNG vs training HR; explanatory only, not a quality gate or training weight'))

if __name__=='__main__':main()
