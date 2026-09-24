"""New ZJU dynamic LR reconstruction; no old person template or HR RGB loss."""
import argparse, json, os, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
from common import new_model, seed_all, render_image, resized_camera, downsample, save_checkpoint, save_image, sha256, write_json, UPSTREAM
from n3dv_data import load_manifest, load_initial_points
from run_experiment import load_training
from PIL import Image


def weighted_l1(pred,target,mask):
    error=(pred-target).abs().mean(0,keepdim=True)
    return .8*(error*mask).sum()/mask.sum().clamp_min(1)+.2*(error*(1-mask)).sum()/(1-mask).sum().clamp_min(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--coarse-steps',type=int,default=1000);p.add_argument('--fine-steps',type=int,default=6000)
    p.add_argument('--max-points',type=int,default=60000);p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args();torch.set_num_threads(4);seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'source.py')
    m=load_manifest(args.manifest);records,cameras=load_training(m);data=Path(m['_root'])
    obs={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    masks=[torch.from_numpy(np.array(Image.open(data/obs[(r['camera_id'],r['frame_index'])]['mask_lr_path'])).copy()).float()[None].div(255) for r in records]
    init=load_initial_points(m);centers=np.stack([np.asarray(m['cameras'][c]['c2w'])[:3,3] for c in m['splits']['train']])
    g,h,o,extent=new_model(init['points'],init['colors'],centers)
    w,hh=m['resolutions']['hr'];cams=[resized_camera(c,hh,w) for c in cameras]
    rng=random.Random(args.seed);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    config=dict(**vars(args),manifest_sha256=sha256(args.manifest),gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=sha256(__file__),initial_points=len(g._xyz),mask_input='Official foreground mask supplied to all methods; LR area mask for loss',
        initialization='Train LR SIFT and supplied calibration only; no SMPL or prior scene weights',
        loss='0.8 foreground mean L1 + 0.2 background mean L1; HR-size render then bicubicAA LR; no HR RGB read')
    write_json(out/'config.json',config)
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for stage,steps in [('coarse',args.coarse_steps),('fine',args.fine_steps)]:
            g.training_setup(o)
            for step in range(1,steps+1):
                g.update_learning_rate(step)
                if step%1000==0:g.oneupSHdegree()
                idx=rng.randrange(len(records));target=records[idx]['image'].cuda();mask=masks[idx].cuda()
                package=render_image(g,cams[idx],stage);pred=downsample(package['render'],target.shape[-2:])
                photo=weighted_l1(pred,target,mask);loss=photo
                if stage=='fine':loss=loss+g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
                if not torch.isfinite(loss):raise FloatingPointError((stage,step))
                loss.backward()
                with torch.no_grad():
                    # Accumulate gradients before the same established clone/split operation.
                    active=300<step<min(3500,steps-500)
                    if active:
                        v=package['visibility_filter'];g.max_radii2D[v]=torch.maximum(g.max_radii2D[v],package['radii'][v])
                        g.add_densification_stats(package['viewspace_points'].grad,v)
                    g.optimizer.step();g.optimizer.zero_grad(set_to_none=True)
                    if active and step%100==0 and len(g._xyz)<args.max_points:
                        g.densify(.0002,.005,extent,None,5,5,str(out),step,stage)
                if step==1 or step%100==0 or step==steps:
                    mse=((pred.detach().clamp(0,1)-target)**2).mean(0,keepdim=True)
                    row=dict(stage=stage,step=step,loss=float(photo),foreground_lr_psnr=float(-10*torch.log10((mse*mask).sum()/mask.sum().clamp_min(1))),
                        points=len(g._xyz),elapsed_s=time.monotonic()-start,peak_gb=torch.cuda.max_memory_allocated()/1e9)
                    log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if stage=='fine':
                meta=dict(scene=m['scene'],stage='new_zju_lr_parent',step=steps,coarse_steps=args.coarse_steps,
                    manifest=args.manifest,manifest_sha=sha256(args.manifest),seed=args.seed,extent=extent,args=vars(args),
                    elapsed_s=time.monotonic()-start,points=len(g._xyz),parameter_updates=args.coarse_steps+steps)
                save_checkpoint(out/'checkpoint_final.pt',g,h,o,meta)
    with torch.inference_mode():
        for index in [0,len(records)//2,len(records)-1]:
            rendered=render_image(g,cams[index])['render'];save_image(out/f'train_preview_{index}.png',rendered)
    write_json(out/'complete.json',dict(**meta,status='completed',checkpoint_sha256=sha256(out/'checkpoint_final.pt'),
        peak_gb=torch.cuda.max_memory_allocated()/1e9,no_hr_rgb_read=True,template_used=False))


if __name__=='__main__':main()
