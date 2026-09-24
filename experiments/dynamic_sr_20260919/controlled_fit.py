"""Matched interventions on duration, teacher weight/quality and point capacity.

All branches start from the same sampling-adapted checkpoint, preserving Adam.
HR targets in the oracle branch are privileged diagnostics, not SR baselines.
"""
from __future__ import annotations
import argparse, collections, json, os, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

OLD = Path(__file__).resolve().parents[1] / 'dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
from common import (UPSTREAM, downsample, image_tensor, load_checkpoint, mse_psnr,
                    render_image, resized_camera, save_checkpoint as _save_checkpoint, seed_all,
                    sha256, write_json as _write_json)
from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera
from run_experiment import load_training
from evaluate import spatial_metrics, image_array, read_rgb, write_rgb


def persist(path):
    """Make completed artifacts durable before advancing the queue/checkpoint."""
    with open(path,'rb') as f:os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_json(path,value):
    _write_json(path,value);persist(path)


def save_checkpoint(path,*args):
    _save_checkpoint(path,*args);persist(path)


@torch.no_grad()
def bounded_densify(g, extent, max_points=220000, per_event=20000):
    """Official clone/split with a hard net growth budget; no opacity pruning.

    Each selected original contributes exactly +1 net point. Selection uses
    accumulated mean screen-space gradients among visibly nonzero candidates.
    This rank-based capacity control intentionally relaxes the old threshold
    so increasing the budget actually adds points at HR sampling resolution.
    Called AFTER Adam.step: existing parameter updates are not discarded.
    Old point moments are retained; new point moments are zero (upstream).
    """
    before = len(g._xyz)
    score = torch.nan_to_num(g.xyz_gradient_accum / g.denom, nan=0.).flatten()
    candidates = torch.where(score > 1e-8)[0]
    count = min(max_points-before, per_event, len(candidates))
    if count <= 0:
        return dict(before=before, after=before, selected=0)
    ids = candidates[torch.topk(score[candidates], count, sorted=False).indices]
    selected = torch.zeros((before, 1), device='cuda')
    selected[ids] = 1.
    # The scalar value encodes selection only. Clone and split use disjoint
    # size conditions; their actual child initialization is unmodified.
    g.densify_and_clone(selected, .5, extent)
    g.densify_and_split(selected, .5, extent)
    assert len(g._xyz) == before + count <= max_points
    for group in g.optimizer.param_groups:
        for p in group['params']:
            assert torch.isfinite(p).all(), group['name']
            st = g.optimizer.state.get(p, {})
            if 'exp_avg' in st:
                assert st['exp_avg'].shape == p.shape
    return dict(before=before, after=len(g._xyz), selected=count,
                min_selected_score=float(score[ids].min()))


@torch.no_grad()
def fit_diagnostic(g, manifest, prior_cameras, out, step, metric):
    """Same fixed train observations; HR never determines train gradients/masks."""
    data = N3DVPreparedDataset(manifest, 'train', 'lr')
    frames = sorted({o['frame_index'] for o in data.observations})
    frames = [frames[i] for i in [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]]
    data.observations = [o for o in data.observations if o['camera_id'] in prior_cameras and o['frame_index'] in frames]
    root = Path(manifest['_root'])
    w,h = manifest['resolutions']['hr']
    mask = np.zeros((h,w),bool)
    rows = []
    for i, obs in enumerate(data.observations):
        rec = data[i]
        cam = resized_camera(observation_to_4dgs_camera(rec,i),h,w)
        raw = render_image(g,cam)['render']
        pred = image_array(raw)
        hr = read_rgb(root/obs['hr_path'])
        prior = read_rgb(root/'sr_swinir_x4'/obs['camera_id']/Path(obs['lr_path']).name)
        lr_psnr = mse_psnr(downsample(raw,rec['image'].shape[-2:]),rec['image'].cuda())[1]
        row = dict(camera_id=obs['camera_id'],frame_index=obs['frame_index'],lr_psnr=lr_psnr,
                   render_hr=spatial_metrics(pred,hr,mask,metric)['full'],
                   render_prior=spatial_metrics(pred,prior,mask,metric)['full'])
        rows.append(row)
        if step in [0,6000,18000]:
            write_rgb(out/f'train_renders_{step}'/obs['camera_id']/f'{obs["frame_index"]:04d}.png',pred)
    keys=['psnr','ssim','lpips_alex','mse']
    result=dict(step=step,rows=rows,aggregate={kind:{k:float(np.mean([r[kind][k] for r in rows])) for k in keys}
                  for kind in ['render_hr','render_prior']},lr_psnr=float(np.mean([r['lr_psnr'] for r in rows])))
    write_json(out/f'fit_{step}.json',result)
    return result['aggregate']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--teacher',choices=['none','sr','hr'],default='sr')
    p.add_argument('--weight',type=float,default=.1)
    p.add_argument('--dense',action='store_true')
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--milestones',default='1200,3000,6000')
    p.add_argument('--prior-cameras',default='cam02,cam06,cam12,cam18')
    p.add_argument('--seed',type=int,default=20260919)
    p.add_argument('--skip-fit',action='store_true')
    args=p.parse_args()
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    seed_all(args.seed)
    m=load_manifest(args.manifest)
    g,h,o,ck=load_checkpoint(args.checkpoint)
    records,cameras=load_training(m)
    w,hh=m['resolutions']['hr']
    hr_cams=[resized_camera(c,hh,w) for c in cameras]
    prior_cams=set(args.prior_cameras.split(','))
    ids=[i for i,r in enumerate(records) if r['camera_id'] in prior_cams]
    assert ids
    obs={(x['camera_id'],x['frame_index']):x for x in m['observations']}
    paths={}
    for i in ids:
        r=records[i]; v=obs[(r['camera_id'],r['frame_index'])]
        path=Path(m['_root'])/(v['hr_path'] if args.teacher=='hr' else str(Path('sr_swinir_x4')/r['camera_id']/Path(v['lr_path']).name))
        if args.teacher!='none': assert path.exists(),path
        paths[i]=path
    config=dict(**vars(args),parent_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),
                parent_metadata=ck['metadata'],initial_points=len(g._xyz),sh_degree=g.active_sh_degree,
                optimizer_preserved=True,scheduler_offset=ck['metadata']['step'],
                scheduler_max_steps=o.position_lr_max_steps,
                information_boundary='HR used for gradients ONLY for teacher=hr oracle; diagnostic HR never enters training otherwise',
                device=torch.cuda.get_device_name(),torch=torch.__version__)
    source_files=[Path(__file__),OLD/'common.py',OLD/'run_experiment.py',OLD/'n3dv_data.py',
                  UPSTREAM/'scene/gaussian_model.py']
    (out/'sources').mkdir()
    config['sources']=[]
    for i,f in enumerate(source_files):
        dst=out/'sources'/f'{i}_{f.name}';shutil.copy2(f,dst)
        config['sources'].append(dict(path=str(f),sha256=sha256(f),copy=str(dst)))
    write_json(out/'config.json',config)
    metric=None
    if not args.skip_fit:
        import lpips
        metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    milestones={int(x) for x in args.milestones.split(',')}|{args.steps}
    lr_rng=random.Random(args.seed+177);sr_rng=random.Random(args.seed+211)
    exposure=collections.Counter();cache={};started=time.monotonic();train_seconds=0.;diagnostic_seconds=0.
    g.xyz_gradient_accum.zero_();g.denom.zero_();g.max_radii2D.zero_()
    log=open(out/'training.jsonl','w',buffering=1)
    for step in range(1,args.steps+1):
        tick=time.monotonic()
        g.update_learning_rate(ck['metadata']['step']+step)
        li,si=lr_rng.randrange(len(records)),sr_rng.choice(ids)
        target=records[li]['image'].cuda()
        g.optimizer.zero_grad(set_to_none=True)
        pkg=render_image(g,hr_cams[li]);pred=downsample(pkg['render'],target.shape[-2:])
        lr_loss=(pred-target).abs().mean()
        reg=g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
        (lr_loss+reg).backward()
        if args.dense:
            with torch.no_grad():g.add_densification_stats(pkg['viewspace_points'].grad,pkg['visibility_filter'])
        teacher_loss=torch.zeros((),device='cuda')
        if args.teacher!='none':
            if si not in cache:cache[si]=image_tensor(paths[si],device='cpu')
            tpkg=render_image(g,hr_cams[si])
            teacher_loss=(tpkg['render']-cache[si].cuda()).abs().mean()
            (teacher_loss*args.weight).backward()
            exposure[f'{records[si]["camera_id"]}/{records[si]["frame_index"]}']+=1
            if args.dense:
                with torch.no_grad():g.add_densification_stats(tpkg['viewspace_points'].grad,tpkg['visibility_filter'])
        if not torch.isfinite(lr_loss+reg+teacher_loss):raise FloatingPointError(step)
        g.optimizer.step()
        growth=None
        if args.dense and 1000<=step<=3500 and step%500==0:
            growth=bounded_densify(g,ck['metadata']['extent'])
        torch.cuda.synchronize()
        train_seconds+=time.monotonic()-tick
        if step==1 or step%100==0:
            row=dict(step=step,lr_l1=float(lr_loss),teacher_l1=float(teacher_loss),reg=float(reg),
                     points=len(g._xyz),train_s=train_seconds,wall_s=time.monotonic()-started,
                     peak_gb=torch.cuda.max_memory_allocated()/1e9,
                     learning_rates={x['name']:x['lr'] for x in g.optimizer.param_groups},growth=growth)
            log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        if step in milestones:
            tick=time.monotonic()
            meta=dict(scene=m['scene'],stage='controlled_fit',step=ck['metadata']['step']+step,
                      intervention_step=step,extent=ck['metadata']['extent'],args=vars(args),
                      parent_sha=config['parent_sha256'],manifest_sha=config['manifest_sha256'],
                      elapsed_s=time.monotonic()-started,train_s=train_seconds,points=len(g._xyz))
            save_checkpoint(out/f'checkpoint_{step}.pt',g,h,o,meta)
            if not args.skip_fit:
                agg=fit_diagnostic(g,m,prior_cams,out,step,metric)
                print(json.dumps(dict(event='fit',step=step,aggregate=agg)),flush=True)
            diagnostic_seconds+=time.monotonic()-tick
    (out/'checkpoint_final.pt').symlink_to(f'checkpoint_{args.steps}.pt')
    write_json(out/'exposure.json',dict(exposure))
    write_json(out/'complete.json',dict(**meta,diagnostic_s=diagnostic_seconds,wall_s=time.monotonic()-started))

if __name__=='__main__':main()
