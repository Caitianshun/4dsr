"""Exactly 19 x four fixed observations; no train-view masks or optical flow."""
import argparse
import time
import csv
from dv_common import *
shared=setup()
import torch
import numpy as np
from motion_model import load_model,render_model
from n3dv_data import load_manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    p=read(OUT/'protocol.json');ev=shared.evaluator();m=load_manifest(bound(p['manifest']));torch.set_num_threads(4);start=time.monotonic()
    observations=sorted([o for o in m['observations'] if o['split']=='train' and o['frame_index'] in p['evaluation']['train_frames']],key=lambda o:(o['camera_id'],o['frame_index']))
    assert len(observations)==76;allowed={str(local(e['path']).resolve()):e for e in p['evaluation_files'] if e['camera'] in p['evaluation']['train_cameras']};opens={}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        f=Path(os.fsdecode(args[0]))
        if f.suffix.lower() not in ['.png','.jpg','.jpeg','.npy','.npz']:return
        key=str(f.resolve());assert key in allowed,('train76 unregistered image/cache',key);opens[key]=opens.get(key,0)+1
    sys.addaudithook(guard)
    model=load_model(a.checkpoint,m);model.g._deformation.eval();before=shared.initial_identity(model)
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for i,o in enumerate(observations):
            cam=ev.render_camera(m,o,i);raw=render_model(model,cam)['render'];assert torch.isfinite(raw).all()
            images={}
            for role in ['lr','hr']:
                path=Path(m['_root'])/o[role+'_path'];assert sha(path)==o[role+'_sha256'];assert str(path.resolve()) in allowed
                images[role]=ev.legacy.read_rgb(path)
            empty=np.zeros(raw.shape[-2:],bool);q=ev.legacy.spatial_metrics(ev.legacy.image_array(raw),images['hr'],empty,metric)['full'];low=ev.lr_reprojection_metrics(raw,images['lr'],empty)['full']
            rows.append(dict(camera=o['camera_id'],frame=o['frame_index'],psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],lr_l1=low['l1'],lr_psnr=low['psnr'],hr_sha256=o['hr_sha256'],lr_sha256=o['lr_sha256']))
    def mean(rr):return {k:float(np.mean([r[k] for r in rr])) for k in ['psnr','ssim','lpips','lr_l1','lr_psnr']}
    cameras={c:mean([r for r in rows if r['camera']==c]) for c in p['evaluation']['train_cameras']}
    assert shared.initial_identity(model)==before
    write(a.out/'metrics.json',dict(status='completed',rows=rows,by_camera=cameras,aggregate=mean(list(cameras.values())),train16=mean([r for r in rows if r['camera'] in p['evaluation']['train16_cameras']]),checkpoint_sha256=sha(a.checkpoint),protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),metric_source_sha256=sha(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),image_reads=opens,rgb_forwards=76,parameter_updates=0,seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),note='Raw float render; quality clamp matches original metric, LR uses original D. No teacher, mask, flow or extra train HR frames read.'))
    write(a.out/'complete.json',dict(status='completed_evaluation',metrics_sha256=sha(a.out/'metrics.json'),observations=76,checkpoint_sha256=sha(a.checkpoint)))
if __name__=='__main__':main()
