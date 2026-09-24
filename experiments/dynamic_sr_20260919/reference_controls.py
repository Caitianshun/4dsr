"""Generic reference controls with explicit target-view information boundaries."""
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
OLD=Path(__file__).resolve().parents[1]/'dynamic_sr_20260918'
sys.path.insert(0,str(OLD))
from common import sha256,load_checkpoint,render_image,image_tensor
from n3dv_data import load_manifest,N3DVPreparedDataset,observation_to_4dgs_camera
from evaluate import aggregate,prepare_cache,read_rgb,spatial_metrics,temporal_metrics,full_flow,write_json,write_rgb
from generate_prior import build_model,infer,DEFAULT_CHECKPOINT,DEFAULT_NETWORK,CHECKPOINT_SHA256,NETWORK_SHA256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['real_lr','render_lr','train_prior'],required=True)
    p.add_argument('--checkpoint');p.add_argument('--prior-cameras',default='cam02,cam04,cam08,cam12')
    a=p.parse_args();out=Path(a.out)
    if (out/'metrics.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    m=load_manifest(a.manifest);root=Path(m['_root']);w,h=m['resolutions']['hr']
    split='train' if a.mode=='train_prior' else 'test'
    obs=sorted([o for o in m['observations'] if o['split']==split and
                (a.mode!='train_prior' or o['camera_id'] in a.prior_cameras.split(','))],
               key=lambda o:(o['camera_id'],o['frame_index']))
    data=N3DVPreparedDataset(m,split,'lr');data.observations=obs
    if split=='test':
        cache,ci=prepare_cache(m,obs,SimpleNamespace(dynamic_threshold=.025,flow_scale=.5,cache_dir=None))
        mask=np.array(Image.open(cache/'dynamic_mask.png'))>0
    else:mask=np.zeros((h,w),bool);ci=None
    g=None
    if a.mode=='render_lr':
        if not a.checkpoint:raise ValueError('render_lr needs checkpoint')
        g,_,_,_=load_checkpoint(a.checkpoint)
    sr=None
    if a.mode!='train_prior':
        assert sha256(DEFAULT_CHECKPOINT)==CHECKPOINT_SHA256 and sha256(DEFAULT_NETWORK)==NETWORK_SHA256
        sys.path.insert(0,str(OLD/'vendor'));sr=build_model(DEFAULT_NETWORK,DEFAULT_CHECKPOINT,'cuda')
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    rows={'bicubic':[],'swinir':[]};previous={};oldgt=None;started=time.monotonic();sources=[]
    with torch.inference_mode():
        for i,o in enumerate(obs):
            hrpath=root/o['hr_path'];lrpath=root/o['lr_path'];gt=read_rgb(hrpath)
            if g is not None:
                camera=observation_to_4dgs_camera(data[i],i)
                raw=render_image(g,camera)['render'].clamp(0,1)
                arr=raw.mul(255).round().byte().permute(1,2,0).cpu().numpy()
            else:arr=np.array(Image.open(lrpath).convert('RGB'))
            x=torch.from_numpy(arr.copy()).permute(2,0,1).cuda().float()/255
            bic=F.interpolate(x[None],size=(h,w),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
            if a.mode=='train_prior':
                prior=root/'sr_swinir_x4'/o['camera_id']/Path(o['lr_path']).name
                srpred=read_rgb(prior)
            else:srpred=infer(sr,arr,'cuda',0,32).astype(np.float32)/255
            predictions={'bicubic':bic.permute(1,2,0).cpu().numpy(),'swinir':srpred}
            if split=='test' and i:
                with np.load(cache/ci['flow_pairs'][i-1]['file']) as f:flow,valid=full_flow(f['backward'],f['valid'],h,w)
            for mode,pred in predictions.items():
                row=dict(camera_id=o['camera_id'],frame_index=o['frame_index'],spatial=spatial_metrics(pred,gt,mask,metric))
                if split=='test' and i:row['temporal']=temporal_metrics(previous[mode],pred,oldgt,gt,flow,valid,mask)
                rows[mode].append(row);previous[mode]=pred
                if split=='test':write_rgb(out/mode/f'{o["frame_index"]:04d}.png',pred)
            oldgt=gt
            sources.append(dict(camera_id=o['camera_id'],frame_index=o['frame_index'],hr_sha256=sha256(hrpath),
                lr_sha256=sha256(lrpath) if g is None else None,
                prior_sha256=sha256(prior) if a.mode=='train_prior' else None))
    result=dict(scene=m['scene'],mode=a.mode,manifest_sha256=sha256(a.manifest),script_sha256=sha256(__file__),
        checkpoint_sha256=sha256(a.checkpoint) if g is not None else None,
        admissible_novel_view_baseline=a.mode=='render_lr',target_view_lr_used=a.mode=='real_lr',
        used_for_training=False,hr_used_only_for_evaluation=True,
        note='Real target-view LR SR is a diagnostic reference, not a mathematical upper bound or novel-view competitor.',
        cache_key=ci['cache_key'] if ci else None,sources=sources,
        modes={k:dict(rows=v,aggregate=aggregate(v,'spatial'),temporal_aggregate=aggregate(v,'temporal')) for k,v in rows.items()},
        elapsed_s=time.monotonic()-started)
    write_json(out/'metrics.json',result)
    print({k:v['aggregate']['full'] for k,v in result['modes'].items()},flush=True)

if __name__=='__main__':main()
