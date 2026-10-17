"""LR stage with the physical (pixel-area) forward model (Claude, 4dsr-7c, 2026-10-17). Wraps cullretrain.py; nothing
is modified (module-attribute patches only).

  physlr_lrstage.py lr <cullretrain lr arguments>     (both --coarse-only and the fine replay)

Every LR-stage render, coarse and fine, is made at the HR camera of the observation (same pose, K_hr, HR size; the author
renderer, stage and deformation unchanged) and reduced by the exact integer-factor box average (F.avg_pool2d) before the
colour affine and the L1 loss against the training LR: the camera model of physlr.py area4 scenes (LR pixel = mean of the
HR pixels it covers). The registered LR stage instead renders directly at the LR grid (point sampling of the splats with
the rasterizer's 0.3 px dilation), which is not the area model. Densification statistics come from the HR renders (NDC
screen gradients, resolution-invariant to first order); thresholds and schedules are the author's.
physlr_lrstage.json in --out records the patch and source hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CULL = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
if str(CULL) not in sys.path:
    sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (sets up runtime paths, imports torch and the author code)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STATE = dict(manifest=None, renders=0, factor=None)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_plan_scene = cr.plan_scene


def plan_scene(plan):
    manifest_path, manifest = _plan_scene(plan)
    STATE["manifest"] = manifest
    return manifest_path, manifest


def hr_observation_camera(item, index):
    """HR geometry for the observation; original_image stays the training LR (the loss target)."""
    if STATE["manifest"] is None:
        raise RuntimeError("manifest not captured before camera construction")
    return cr.native.hr_camera(item, index, STATE["manifest"])


def area_reduce(rgb, size):
    h, w = rgb.shape[-2:]
    fy, fx = h // int(size[0]), w // int(size[1])
    if fy != fx or fy * int(size[0]) != h or fx * int(size[1]) != w:
        raise ValueError(f"area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}")
    STATE["factor"] = fy
    return F.avg_pool2d(rgb[None], fy)[0]


def lr_iteration(g, h, opt, pipe, background, cameras, camera_ids, stage, iteration, extent, out, render,
                 affine, prune_gate, counts):
    """cr.lr_iteration with the render made at HR and box-reduced to the LR target; same order of operations."""
    g.update_learning_rate(iteration)
    if iteration % 1000 == 0:
        g.oneupSHdegree()
    packages, images, targets = [], [], []
    for camera, cid in zip(cameras, camera_ids):
        packet = render(camera, g, pipe, background, stage=stage, cam_type='dynerf')
        packages.append(packet)
        target = camera.original_image.to(background.device)
        image = area_reduce(packet['render'], target.shape[-2:])
        STATE["renders"] += 1
        image = image if affine is None else affine.apply(image, cid)
        images.append(image.unsqueeze(0))
        targets.append(target.unsqueeze(0))
    prediction = torch.cat(images, dim=0)
    target = torch.cat(targets, dim=0)
    image_loss = (prediction - target[:, :3]).abs().mean()
    loss = image_loss
    if stage == 'fine' and h.time_smoothness_weight != 0:
        loss = loss + g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
    if opt.lambda_dssim != 0:
        from utils.loss_utils import ssim
        loss = loss + opt.lambda_dssim * (1 - ssim(prediction, target))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite LR loss')
    loss.backward()
    statistics = cr.prefix.combine_batch_statistics(packages, torch)
    applied = iteration < opt.iterations
    with torch.no_grad():
        cr.topology(g, opt, stage, iteration, extent, out, statistics, prune_gate, counts)
        if applied:
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
    if affine is not None and applied:
        affine.step(iteration)
    return dict(loss=float(loss.detach()), L1=float(image_loss.detach()), stage=stage, iteration=iteration,
                Adam_applied=applied, points=int(g.get_xyz.shape[0]), batch_size=len(cameras))


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != 'lr':
        raise SystemExit('physlr_lrstage.py only wraps the lr subcommand')
    cr.plan_scene = plan_scene
    cr.observation_to_4dgs_camera = hr_observation_camera
    cr.lr_iteration = lr_iteration
    cr.main(argv)
    out = Path(argv[argv.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    record = dict(patch='LR-stage renders at the HR camera, exact box reduction to the LR grid before affine and L1',
                  box_factor=STATE["factor"], renders=STATE["renders"], argv=argv,
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, cr.__file__)})
    (out / 'physlr_lrstage.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physlr_lrstage=True, box_factor=STATE["factor"], renders=STATE["renders"])), flush=True)


if __name__ == '__main__':
    main()
