"""Freeze a pure-LR trajectory cache and calibrate one mild motion weight."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from soft_model import ROOT, MOTION, FrozenReference, child_centers, frame, make_model, render_with_centers
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
from common import load_checkpoint, sha256, write_json, named_parameters, downsample, image_tensor, resized_camera
from n3dv_data import load_manifest
from run_experiment import load_training


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['manifest','checkpoint','selection','out','prior-cameras']:p.add_argument('--'+k,required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);started=time.monotonic()
    m=load_manifest(args.manifest);g,h,o,ck=load_checkpoint(args.checkpoint)
    parent_sha,manifest_sha,selection_sha=map(sha256,[args.checkpoint,args.manifest,args.selection])
    assert ck['metadata']['manifest_sha']==manifest_sha
    assert ck['metadata'].get('mode')=='lr_integrated', 'Reference must be the pure LR integrated parent'
    selection=torch.load(args.selection,map_location='cpu',weights_only=False)
    assert selection['parent_sha256']==parent_sha and selection['manifest_sha256']==manifest_sha
    ids=selection['selected_ids'].long().cuda()
    # Save selected canonical geometry before ordinary split removes those rows.
    parent_values=[x[ids].detach().clone() for x in [g._xyz,g._scaling,g._rotation,g._opacity,g.get_features]]
    parent_max_axis=g._scaling.detach().exp().amax(-1)
    finite_scale=parent_max_axis[torch.isfinite(parent_max_axis)&(parent_max_axis>0)]
    lower=torch.quantile(finite_scale,.05)
    length=parent_max_axis[ids].clamp_min(lower).repeat_interleave(2)
    model=make_model(g,h,o,ck,m,'ordinary_split',selection)
    delta0=model.children.offset().detach().clone()
    frame_times={int(r['frame_index']):float(r['time']) for r in m['observations'] if r['split']=='train'}
    frames=sorted(frame_times);assert frames==list(range(0,120,2)) and 60 in frames
    mu,axes,centers=[],[],[]
    # Network frozen for reference cache generation, then re-enabled unchanged for calibration.
    parameters=list(named_parameters(g).values())
    flags=[x.requires_grad for x in parameters]
    for x in parameters:x.requires_grad_(False)
    with torch.no_grad():
        for f in frames:
            times=torch.full((len(ids),1),frame_times[f],device='cuda')
            z=g._deformation(*parent_values,times)
            a=frame(z[1],z[2]);b=z[0].repeat_interleave(2,0)+(a.repeat_interleave(2,0)@delta0.unsqueeze(-1)).squeeze(-1)
            mu.append(z[0].cpu());axes.append(a.cpu());centers.append(b.cpu())
    for x,flag in zip(parameters,flags):x.requires_grad_(flag)
    centers=torch.stack(centers)
    valid=torch.isfinite(centers).all(0).all(-1)&torch.isfinite(length.cpu())&(length.cpu()>0)
    if 'counts' in selection:valid &= (selection['counts'][ids.cpu()]>0).repeat_interleave(2)
    assert bool(valid.any())
    metadata=dict(schema=1,scene=m['scene'],parent=str(Path(args.checkpoint).resolve()),parent_sha256=parent_sha,
        manifest_sha256=manifest_sha,selection_sha256=selection_sha,reference_frame=60,
        reference_time=frame_times[60],reference='Entire frozen pure LR parent network + canonical geometry; never SR-updated',
        source_sha256=sha256(__file__),soft_model_sha256=sha256(Path(__file__).with_name('soft_model.py')),
        length_rule='max exp(canonical parent logscale), lower bounded by q05 of all original finite positive parent maxima; no factor2',
        length_lower_bound=float(lower),parent_count=len(ids),children=len(delta0),valid_children=int(valid.sum()),
        selected_visibility='ignore only never observed in original 32-view selection; no per-time occlusion gate',
        gpu=torch.cuda.get_device_name(),parameter_updates=0)
    payload=dict(metadata=metadata,frames=frames,times=[frame_times[f] for f in frames],
        parent_centers=torch.stack(mu),parent_frames=torch.stack(axes),reference_child_centers=centers,
        child_length=length.cpu(),valid_children=valid,delta0=delta0.cpu(),
        initial_offset_raw=model.children.offset_raw.detach().cpu(),child_parent_ids=model.children.parent_ids.cpu())
    path=out/'reference.pt';torch.save(payload,path)
    reference=FrozenReference(path)
    reference.validate(model,parent_sha,selection_sha,manifest_sha)
    cache_seconds=time.monotonic()-started
    records,cameras=load_training(m);lookup={(r['camera_id'],int(r['frame_index'])):i for i,r in enumerate(records)}
    w,hgt=m['resolutions']['hr'];rows=[];data_sq=motion_sq=0.0
    # Eight fixed training observations, selected before inspecting any model quality.
    for camera in args.prior_cameras.split(','):
        for f in [0,118]:
            i=lookup[camera,f];cam=resized_camera(cameras[i],hgt,w)
            teacher=Path(m['_root'])/'sr_swinir_x4'/camera/f'{f:04d}.png'
            package=render_with_centers(model,cam);b=package['child_centers']
            br=child_centers(model,reference.reference_time)
            image=package['render'];lr=records[i]['image'].cuda()
            data=(downsample(image,lr.shape[-2:])-lr).abs().mean()+.1*(image-image_tensor(teacher)).abs().mean()
            motion=reference.loss(b,br,f)
            gd=torch.autograd.grad(data,b,retain_graph=True)[0]
            gm=torch.autograd.grad(motion,b)[0]
            gd_sq=float(gd.detach().double().square().sum());gm_sq=float(gm.detach().double().square().sum())
            assert all(torch.isfinite(x).all() for x in [gd,gm])
            data_sq+=gd_sq;motion_sq+=gm_sq
            rows.append(dict(camera=camera,frame=f,data_loss=float(data),motion_loss=float(motion),
                data_gradient_norm=gd_sq**.5,motion_gradient_norm=gm_sq**.5,teacher_sha256=sha256(teacher)))
    dn,mn=data_sq**.5,motion_sq**.5
    calibration=dict(status='calibrated',observations=rows,data_gradient_norm=dn,motion_gradient_norm=mn,
        gradient_tensor='same rendered current-time child centers; full 3D xyz; no screen-space or parameter gradient',
        rule='lambda=.1*sqrt(sum observation data_gradient_norm^2)/sqrt(sum observation motion_gradient_norm^2)',
        data_loss='same-observation full LR L1+.1 frozen SR L1, excluding original regularization',
        training_application='one motion term per iteration on SR sampled time; not a claim about total training parameter gradient ratio',
        target_ratio=.1,warmup_steps=300,reference_sha256=sha256(path),cache_bytes=path.stat().st_size,
        cache_build_seconds=cache_seconds,parameter_updates=0,numerical_correction=False)
    if not (mn>1e-12 and dn>0):
        calibration.update(status='inactive_or_degenerate_gradient',lambda_motion=None)
        write_json(out/'calibration.json',calibration)
        raise RuntimeError('Motion gradient degenerate; no epsilon division or formal run')
    lam=.1*dn/mn
    if not (0<lam<1e3):
        calibration.update(status='anomalous_scale_requires_review',lambda_motion=lam)
        write_json(out/'calibration.json',calibration)
        raise RuntimeError('Unexpected weight magnitude; no automatic large-lambda launch')
    calibration.update(lambda_motion=lam,calibrated_ratio=lam*mn/dn,elapsed_s=time.monotonic()-started,
        manifest_sha256=manifest_sha,parent_sha256=parent_sha,selection_sha256=selection_sha)
    write_json(out/'calibration.json',calibration)
    write_json(out/'complete.json',dict(status='completed_reference_and_calibration',**metadata,
        lambda_motion=lam,reference_sha256=calibration['reference_sha256'],cache_bytes=path.stat().st_size,
        elapsed_s=calibration['elapsed_s'],calibration_sha256=sha256(out/'calibration.json')))
    print(json.dumps(calibration),flush=True)


if __name__=='__main__':main()
