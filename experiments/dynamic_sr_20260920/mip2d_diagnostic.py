#!/usr/bin/env python3
"""Evaluation-only normalized 2D Gaussian pixel filter on frozen checkpoints.

The installed rasterizer still adds 0.3 I. We compensate opacity outside it,
using post-deformation covariance and the same projection convention. This is
NOT full Mip-Splatting: there is no 3D filter, retraining, or CUDA modification.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from output_swap import (PROJECT, OLD, ORIGINAL, UPSTREAM, load_manifest,
    N3DVPreparedDataset, observation_to_4dgs_camera, resized_camera,
    load_checkpoint, render_image, final_outputs, render_outputs, downsample,
    image_array, read_rgb, file_sha, write_json, write_rgb,
    GaussianRasterizationSettings, GaussianRasterizer)
from evaluate import ssim_map_rgb, metric_psnr
from diff_gaussian_rasterization import _C

FRAMES = [0, 40, 80, 118]
SCENES = ['cook_spinach', 'meetroom_discussion']
ROOT = PROJECT / 'output/dynamic_sr_20260920/mip2d_diagnostic'


def covariance_3d(scales, rotations):
    # CUDA receives already-normalized quaternions and does not renormalize.
    r, x, y, z = rotations.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
        2*(x*y+r*z), 1 - 2*(x*x+z*z), 2*(y*z-r*x),
        2*(x*z-r*y), 2*(y*z+r*x), 1 - 2*(x*x+y*y)], -1).reshape(-1, 3, 3)
    L = R * scales[:, None, :]
    return L @ L.transpose(-1, -2)


def projected_covariance(outputs, camera):
    xyz, scales, rotations = outputs[:3]
    V = camera.world_view_transform.to(xyz)
    t = xyz @ V[:3, :3] + V[3, :3]
    tanx, tany = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    focalx = camera.image_width / (2*tanx)
    focaly = camera.image_height / (2*tany)
    # Mirror CUDA clamp at +/- 1.3 tan(FOV/2). Culled near-plane points get
    # neutral compensation below and are never used to form a diagnostic.
    z = torch.where(t[:, 2].abs() > 1e-8, t[:, 2], torch.full_like(t[:, 2], 1e-8))
    x = (t[:, 0]/z).clamp(-1.3*tanx, 1.3*tanx) * z
    y = (t[:, 1]/z).clamp(-1.3*tany, 1.3*tany) * z
    J = torch.zeros(len(xyz), 2, 3, device=xyz.device, dtype=xyz.dtype)
    J[:, 0, 0], J[:, 0, 2] = focalx/z, -focalx*x/(z*z)
    J[:, 1, 1], J[:, 1, 2] = focaly/z, -focaly*y/(z*z)
    A = J @ V[:3, :3].T
    C = A @ covariance_3d(scales, rotations) @ A.transpose(-1, -2)
    return C, t


def compensation(outputs, camera):
    # Float64 determinants avoid cancellation on nearly singular projected
    # ellipses; values near zero are explicitly reported, never interpreted
    # as exact reproduction of Mip's CUDA epsilon heuristics.
    C, t = projected_covariance(tuple(v.double() for v in outputs), camera)
    a, b, d = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    det0 = (a*d-b*b).clamp_min(0)
    det1 = ((a+.3)*(d+.3)-b*b).clamp_min(1e-30)
    coeff = (det0/det1).clamp(0, 1).sqrt()
    coeff = torch.where(t[:, 2] > .2, coeff, torch.ones_like(coeff))
    assert torch.isfinite(coeff).all()
    return coeff.to(outputs[0])[:, None], det0


def raw_render_and_compensation(outputs, camera, degree):
    """Read actual CUDA conic; source-bound geometry layout, no buffer edits."""
    empty = torch.Tensor([])
    _, baseline, _, radii, geom, _, _ = _C.rasterize_gaussians(
        torch.zeros(3,device='cuda'), outputs[0], empty, outputs[3],
        outputs[1],outputs[2],1.,empty,camera.world_view_transform.cuda(),
        camera.full_proj_transform.cuda(),math.tan(camera.FoVx/2),
        math.tan(camera.FoVy/2),int(camera.image_height),int(camera.image_width),
        outputs[4],degree,camera.camera_center.cuda(),False,False)
    n=len(outputs[0]);offset=0;base=geom.data_ptr()
    # GeometryState::fromChunk: depths float; clamped bool[3]; radii int;
    # means2D float2; cov3D float[6]; conic_opacity float4. All align 128.
    for byte_count in [n*4,n*3,n*4,n*8,n*24]:
        offset=((base+offset+127)//128)*128-base
        offset+=byte_count
    offset=((base+offset+127)//128)*128-base
    assert offset+n*16 <= geom.numel()
    conic=geom[offset:offset+n*16].view(torch.float32).reshape(n,4)
    valid=radii>0
    assert int(valid.sum())>0
    assert torch.equal(conic[valid,3],outputs[3][valid,0])
    K=conic[valid,:3].double()
    assert torch.isfinite(K).all()
    detK=K[:,0]*K[:,2]-K[:,1].square()
    assert (detK>0).all()
    # det(C)/det(C+kI) = det(I-k*(C+kI)^-1). This avoids reconstructing
    # the covariance and subtracting nearly equal large matrix entries.
    ratio=1-.3*(K[:,0]+K[:,2])+.09*detK
    assert float(ratio.min()) > -2e-5 and float(ratio.max()) <= 1.000001, dict(
        ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),
        invalid_count=int(((ratio < -2e-5)|(ratio>1.000001)).sum()),
        worst_conic=K[ratio.argmin()].tolist(),n_visible=int(valid.sum()),
        width=camera.image_width,height=camera.image_height)
    coeff=torch.ones((n,1),device='cuda')
    coeff[valid,0]=ratio.clamp(0,1).sqrt().float()
    # Independent analytic projection remains a validation, not the source
    # of production coefficients. It cannot cause CUDA support changes.
    analytic,_=compensation(outputs,camera)
    diff=(coeff[valid]-analytic[valid]).abs()
    audit=dict(visible_gaussians=int(valid.sum()),
        conic_opacity_field_exact=True,raw_negative_ratio_count=int((ratio<0).sum()),
        coefficient_vs_analytic_max_abs=float(diff.max()),
        coefficient_vs_analytic_rmse=float(diff.square().mean().sqrt()))
    assert audit['coefficient_vs_analytic_rmse'] < 2e-5, audit
    return baseline,coeff,audit


def render_plain(outputs, camera, degree, cov_precomp=None, white=False):
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2),
        bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
        viewmatrix=camera.world_view_transform.cuda(),
        projmatrix=camera.full_proj_transform.cuda(), sh_degree=degree,
        campos=camera.camera_center.cuda(), prefiltered=False, debug=False)
    color = torch.ones(len(outputs[0]), 3, device='cuda') if white else None
    result, _, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=outputs[0], means2D=torch.zeros_like(outputs[0]),
        shs=None if white else outputs[4], colors_precomp=color,
        opacities=outputs[3], scales=None if cov_precomp is not None else outputs[1],
        rotations=None if cov_precomp is not None else outputs[2],
        cov3D_precomp=cov_precomp)
    return result


def projection_probe(camera):
    # Independent one-Gaussian raster test, all three color channels white.
    # Covers off-axis projection and rotated anisotropic covariance, not
    # just a centered isotropic Gaussian where transposes could cancel.
    V = camera.world_view_transform.to(device='cuda', dtype=torch.float64)
    records = []
    for pos in ([.12, -.07, 2.], [-.25, .11, 3.], [.05, .03, 1.5]):
        xyz_cam = torch.tensor([pos], device='cuda', dtype=torch.float64)
        xyz = ((xyz_cam - V[3, :3]) @ torch.linalg.inv(V[:3, :3])).float()
        scales = torch.tensor([[.012, .023, .017]], device='cuda')
        rot = torch.tensor([[.8, .2, -.3, .4]], device='cuda')
        rot = rot/rot.norm(dim=1, keepdim=True)
        outputs = (xyz, scales, rot, torch.full((1,1), .4, device='cuda'),
                   torch.zeros((1, 16, 3), device='cuda'))
        actual = render_plain(outputs, camera, 3, white=True)[0]
        C, _ = projected_covariance(outputs, camera)
        C = C[0] + .3 * torch.eye(2, device='cuda')
        homogeneous = torch.cat([xyz, torch.ones(1,1,device='cuda')], -1)
        clip = homogeneous @ camera.full_proj_transform.cuda()
        ndc = clip[0,:2] / (clip[0,3]+1e-7)
        mean = ((ndc+1)*torch.tensor([camera.image_width,camera.image_height], device='cuda')-1)/2
        yy, xx = torch.meshgrid(torch.arange(camera.image_height, device='cuda'),
                                torch.arange(camera.image_width, device='cuda'), indexing='ij')
        delta = torch.stack([xx-mean[0], yy-mean[1]], -1)
        exponent = -.5*torch.einsum('hwi,ij,hwj->hw',delta,torch.linalg.inv(C),delta)
        expected = .4*exponent.exp()
        expected = torch.where(expected >= 1/255, expected, torch.zeros_like(expected))
        error = float((actual-expected).abs().max())
        records.append(dict(camera_coordinate=pos,max_abs_rgb_delta=error))
        assert error < 2e-5, records[-1]
    return records


def metrics(pred, gt, lpips_model):
    p, g = image_array(pred), image_array(gt)
    mse = float(np.mean((p-g)**2))
    result = dict(mse=mse,psnr=metric_psnr(mse),
                  ssim=float(ssim_map_rgb(p,g)[5:-5,5:-5].mean()))
    result['lpips_alex'] = float(lpips_model(pred.clamp(0,1)[None]*2-1,gt.clamp(0,1)[None]*2-1))
    return result


@torch.no_grad()
def run(smoke=False):
    import lpips
    begin=time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=False)
    metric=lpips.LPIPS(net='alex').cuda().eval()
    records=[]
    meta=dict(script_sha256=file_sha(__file__), gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
        intervention='Frozen checkpoint, post-deformation opacity times sqrt(det C / det(C + 0.3I)); no 3D filter, no optimization, no CUDA edit.',
        kernel_variance_pixel_squared=.3, frames=FRAMES, camera='cam00',
        checkpoint_branch='sr_w01/checkpoint_6000.pt',
        outputs_are_diagnostic_not_retrained_mip=True,
        covariance_and_nearplane='Actual CUDA conic_opacity read from source-defined geomBuffer; radii>0 only. Coefficient from det(I-0.3K), K=CUDA inverse filtered covariance. No Mip epsilon clipping heuristic.',
        rasterizer_source_sha256=file_sha(UPSTREAM/'submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.cu'),
        renderer_source_sha256=file_sha(UPSTREAM/'gaussian_renderer/__init__.py'),scenes={})
    meta['geometry_layout_source_sha256']=file_sha(UPSTREAM/'submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu')
    for scene in SCENES[:1] if smoke else SCENES:
        cfg=json.loads((ORIGINAL/f'{scene}_sr_w01/config.json').read_text())
        manifest=load_manifest(cfg['manifest'])
        assert file_sha(cfg['manifest']) == cfg['manifest_sha256']
        cp=ORIGINAL/f'{scene}_sr_w01/checkpoint_6000.pt'
        model,_,_,ck=load_checkpoint(cp)
        model._deformation.eval()
        data=N3DVPreparedDataset(manifest,'test','lr')
        data.observations=[o for o in data.observations if o['frame_index'] in (FRAMES[:1] if smoke else FRAMES)]
        smeta=dict(checkpoint=str(cp),checkpoint_sha256=file_sha(cp),
            manifest=cfg['manifest'],manifest_sha256=file_sha(cfg['manifest']),
            baseline_parity=[],cuda_conic_audit=[],projection_probes=[],input_hashes=[])
        for i in range(len(data)):
            item=data[i]
            obs=data.observations[i]
            camera=observation_to_4dgs_camera(item,i)
            lr=item['image'].cuda()
            arr=read_rgb(Path(manifest['_root'])/obs['hr_path'])
            hr=torch.from_numpy(arr).permute(2,0,1).cuda()
            w,h=manifest['resolutions']['hr']
            assert tuple(hr.shape[-2:])==(h,w)
            outputs=final_outputs(model,camera.time)
            if i==0:
                smeta['projection_probes']=projection_probe(camera)
            smeta['input_hashes'].append(dict(frame=item['frame_index'],lr=file_sha(Path(manifest['_root'])/obs['lr_path']),hr=file_sha(Path(manifest['_root'])/obs['hr_path'])))
            scene_outputs={}
            for factor in [1,2,4]:
                cam=resized_camera(camera,camera.image_height*factor,camera.image_width*factor)
                baseline,coeff,conic_audit=raw_render_and_compensation(outputs,cam,model.active_sh_degree)
                expected=render_image(model,cam)['render']
                delta=float((baseline-expected).abs().max())
                smeta['baseline_parity'].append(dict(frame=item['frame_index'],factor=factor,max_abs_rgb_delta=delta))
                assert delta==0., smeta['baseline_parity'][-1]
                smeta['cuda_conic_audit'].append(dict(frame=item['frame_index'],factor=factor,**conic_audit))
                altered=list(outputs);altered[3]=outputs[3]*coeff
                compensated=render_plain(altered,cam,model.active_sh_degree)
                for name,pred in [('original',baseline),('normalized_2d',compensated)]:
                    pred_lr=downsample(pred,lr.shape[-2:]) if factor>1 else pred
                    row=dict(scene=scene,frame=item['frame_index'],factor=factor,mode=name,
                        lr_metrics=metrics(pred_lr,lr,metric),
                        hr_metrics=metrics(pred,hr,metric) if factor==4 else None)
                    records.append(row)
                    scene_outputs[name,factor]=pred_lr
                    if item['frame_index']==40:
                        write_rgb(ROOT/scene/f'{name}_{factor}x_frame40.png',image_array(pred))
                smeta.setdefault('coefficient_summary',[]).append(dict(frame=item['frame_index'],factor=factor,
                    quantiles=[float(x) for x in torch.quantile(coeff.flatten(),torch.tensor([0,.1,.5,.9,1.],device='cuda'))],
                    negative_ratio_clamped=conic_audit['raw_negative_ratio_count']))
            for name in ['original','normalized_2d']:
                for factor in [1,2]:
                    mse=float((scene_outputs[name,factor]-scene_outputs[name,4]).square().mean())
                    records.append(dict(scene=scene,frame=item['frame_index'],mode=name,factor=factor,
                        consistency_to_same_mode_4x=dict(mse=mse,psnr=metric_psnr(mse))))
        meta['scenes'][scene]=smeta
        del model,ck,data
        gc.collect();torch.cuda.empty_cache()
    summary={}
    for scene in SCENES[:1] if smoke else SCENES:
        summary[scene]={}
        for mode in ['original','normalized_2d']:
            summary[scene][mode]={}
            for factor in [1,2,4]:
                rows=[r for r in records if r['scene']==scene and r['mode']==mode and r['factor']==factor and 'lr_metrics' in r]
                summary[scene][mode][str(factor)]={key:{k:float(np.mean([r[key][k] for r in rows])) for k in ['mse','psnr','ssim','lpips_alex']} for key in ['lr_metrics']+(['hr_metrics'] if factor==4 else [])}
    meta['wall_seconds']=time.monotonic()-begin
    meta['peak_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
    write_json(ROOT/'metrics.json',dict(metadata=meta,rows=records,summary=summary))
    print(json.dumps(dict(summary=summary,wall_seconds=meta['wall_seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args().smoke)
