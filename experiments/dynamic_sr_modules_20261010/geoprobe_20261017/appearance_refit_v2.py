"""Frozen-geometry appearance refit probe (Claude, session 4dsr-e9, 2026-10-17). New file; cullretrain is imported, never
modified. Diagnostic only (some arms read training HR); not a method.
v2 (copy of experiments/dynamic_sr_geoprobe_20261017/appearance_refit.py placed under a registered sync prefix) adds
--init source: keep the source's SH, SH-deformation head and colour affine (warm start; legal when the source is legal) -- the
"appearance re-projection" of a trained model onto the LR data under the correct operator. --eval-out is optional (remote hosts
cannot run cullretrain eval; score with group_eval.py instead).

Question (user, 10-17): does the base model's geometry gate how much HR detail the LR views (multi-view de-aliasing under the
pixel-area protocol) and an SR signal can put into the model?  For a source checkpoint, the geometry -- Gaussian centres,
scales, rotations, opacities, the deformation trunk (HexPlane grid + feature MLP) and the position/scale/rotation/opacity
heads -- is loaded and frozen.  The appearance -- per-Gaussian SH (DC reset to grey, rest to 0) and the SH-deformation head
(no_dshs=False in the author config: an additive, time-dependent SH offset; re-initialised, last layer 0) -- is refit from
scratch with one supervision:
  lr       LR closure (1-d) L1 + d (1-SSIM) on all three views of each same-time Bsync row; operator area (4x4 box) or bicaa.
  hr       the HR image of each of the three views (oracle; reads training HR, SHA256-checked against the manifest).
  teacher  frozen SwinIR teacher of the Bsync registry on all three views (weight --lam) + LR closure on all three.
--sh-degree 0 keeps the SH rest at 0 and renders with degree 0 (no view-dependent colour; the SH head still gives a
time-dependent DC offset). A per-training-camera colour affine (identity init, cullretrain.CameraAffine) is applied to the
renders in every loss, as in the SR stage. Saves checkpoint_<steps>.pt (cullretrain_sr_v1, plan of --plan-from) and runs
cullretrain eval on it (cam00 all 300 frames, post-hoc cull).

  appearance_refit.py --source CK --plan-from CK --baseline-sr BSYNC --supervision {lr,hr,teacher} [--operator area]
                      [--sh-degree 3] [--steps 3000] --out DIR --eval-out DIR --arm NAME
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'
if str(CULL) not in sys.path:
    sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (sets up the pinned runtime paths)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

native, rt, losses = cr.native, cr.rt, cr.losses


def read_hr(path, digest):
    from PIL import Image
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'HR image hash differs from the manifest: {path}')
    with Image.open(io.BytesIO(data)) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def area_degradation(rgb, size):
    h, w = rgb.shape[-2:]
    f = h // int(size[0])
    if f * int(size[0]) != h or f * int(size[1]) != w:
        raise ValueError(f'area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}')
    return F.avg_pool2d(rgb[None], f)[0].clamp(0, 1)


def photometric(pred, target, dssim, ssim):
    l1 = (pred - target).abs().mean()
    if not dssim:
        return l1
    return (1 - dssim) * l1 + dssim * (1 - ssim(pred[None], target[None]))


def run(a):
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    started = time.time()
    a.plan_from = a.plan_from or a.source
    plan = torch.load(a.plan_from, map_location='cpu', weights_only=False)['plan']
    payload = torch.load(a.source, map_location='cuda', weights_only=False)
    source_affine = payload.get('affine')
    src_plan = payload['plan']
    for key in ('ModelHiddenParams',):
        if plan['configuration'][key] != src_plan['configuration'][key]:
            raise ValueError(f'{key} of the source differs from the plan')
    if plan['configuration']['ModelParams']['sh_degree'] != src_plan['configuration']['ModelParams']['sh_degree']:
        raise ValueError('sh_degree differs')
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    base = cr.read(Path(a.baseline_sr) / 'config.json')
    if cr.rel(manifest_path) != base['data']['manifest']['path']:
        raise ValueError('Bsync registry belongs to another manifest')
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(a.seed)
    g = GaussianModel(sh, h)
    g._deformation = g._deformation.cuda()
    g.restore(payload['model'], opt)
    g._deformation_accum = payload['deformation_accum']
    del payload
    # ---- freeze geometry, reset appearance
    for name in ('_xyz', '_scaling', '_rotation', '_opacity'):
        getattr(g, name).requires_grad_(False)
    for p in g._deformation.parameters():
        p.requires_grad_(False)
    head = g._deformation.deformation_net.shs_deform
    if a.init == 'grey':
        head[1].reset_parameters()
        torch.nn.init.zeros_(head[3].weight)
        torch.nn.init.zeros_(head[3].bias)
    for p in head.parameters():
        p.requires_grad_(True)
    if a.init == 'grey':
        g._features_dc = torch.nn.Parameter(torch.zeros_like(g._features_dc).detach(), requires_grad=True)
        g._features_rest = torch.nn.Parameter(torch.zeros_like(g._features_rest).detach(), requires_grad=a.sh_degree > 0)
    else:
        g._features_dc = torch.nn.Parameter(g._features_dc.detach().clone(), requires_grad=True)
        g._features_rest = torch.nn.Parameter(g._features_rest.detach().clone(), requires_grad=a.sh_degree > 0)
    g.active_sh_degree = min(a.sh_degree, g.active_sh_degree) if a.init == 'source' else a.sh_degree
    groups = [dict(params=[g._features_dc], lr=a.lr_dc, init=a.lr_dc, name='f_dc')]
    if a.sh_degree > 0:
        groups.append(dict(params=[g._features_rest], lr=a.lr_dc / 20, init=a.lr_dc / 20, name='f_rest'))
    groups.append(dict(params=list(head.parameters()), lr=a.lr_head, init=a.lr_head, name='shs_head'))
    adam = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    affine = cr.CameraAffine(manifest['splits']['train'], lr_init=a.affine_lr, lr_final=a.affine_lr, max_steps=a.steps)
    if a.init == 'source' and source_affine is not None:
        affine.load_state_dict(source_affine)
        for group in affine.optimizer.param_groups:
            group['lr'] = a.affine_lr
    operator = area_degradation if a.operator == 'area' else losses.degradation
    # ---- data: registered same-time three-view rows
    dataset = cr.N3DVPreparedDataset(manifest, 'train', 'lr', device='cpu', cache=False)
    table = cr.read(ROOT / base['dependencies']['schedule']['path'])
    rows = table[base['definition']['schedule']]
    record_keys = [tuple(k) for k in table['record_keys']]
    if a.steps > len(rows):
        raise ValueError(f'steps {a.steps} > schedule rows {len(rows)}')
    root = Path(manifest['_root'])
    train = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations'] if o['split'] == 'train'}
    teacher_paths = None
    if a.supervision == 'teacher':
        index = cr.read(ROOT / base['dependencies']['teacher']['path'])
        teacher_paths = {(e['camera'], int(e['frame'])): native.safe_child(root, e['relative_path']) for e in index['entries']}

    def fetch(i):
        row = rows[i]
        items = [dataset[j] for j in row]
        keys = [record_keys[j] for j in row]
        hrs = [read_hr(native.safe_child(root, train[k]['hr_path']), train[k]['hr_sha256']) for k in keys] \
            if a.supervision == 'hr' else None
        teachers = [cr.read_teacher(teacher_paths[k]) for k in keys] if teacher_paths is not None else None
        return row, items, keys, hrs, teachers

    prefetch = cr.Prefetch(fetch, list(range(a.steps)), workers=a.io_workers)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else \
        torch.zeros(3, dtype=torch.float32, device='cuda')
    record = dict(kind='appearance_refit', arm=a.arm, args=vars(a), source=cr.entry(Path(a.source)),
                  source_points=int(g.get_xyz.shape[0]), plan_from=cr.entry(Path(a.plan_from)),
                  baseline_sr=cr.rel(a.baseline_sr), manifest=cr.entry(manifest_path), host=__import__('socket').gethostname(),
                  gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  HR_training_pixels_read=a.supervision == 'hr' or 'HRref' in str(a.source),
                  sources={cr.rel(p): rt.sha(p) for p in (Path(__file__), Path(cr.__file__))}, started_unix=started)
    rt.write_json(out / 'run.json', record)
    log = (out / 'trajectory.jsonl').open('a', buffering=1)
    torch.cuda.reset_peak_memory_stats()
    try:
        for cursor in range(1, a.steps + 1):
            tick = time.time()
            frac = (cursor - 1) / max(a.steps - 1, 1)
            for group in adam.param_groups:
                group['lr'] = group['init'] * (a.lr_final_ratio ** frac)
            row, items, keys, hrs, teacher_cpu = prefetch.get(cursor - 1)
            if len({k[0] for k in keys}) != 3 or len({int(k[1]) for k in keys}) != 1:
                raise ValueError('Registered same-time batch differs')
            cameras = [native.hr_camera(obs, i, manifest) for i, obs in zip(row, items)]
            state = native.native_effective_state(g, cameras[0].time, torch)
            rgbs = [native.native_rgb(c, g, state, pipe, background)['render'] for c in cameras]
            rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
            terms = {}
            if a.supervision in ('lr', 'teacher'):
                lrs = [obs['image'].cuda() for obs in items]
                terms['lr'] = sum(photometric(operator(rgb, lr.shape[-2:]), lr, a.dssim, ssim) for rgb, lr in zip(rgbs, lrs)) / 3
            if a.supervision == 'hr':
                terms['hr'] = sum(photometric(rgb, t.cuda(), a.dssim, ssim) for rgb, t in zip(rgbs, hrs)) / 3
            if a.supervision == 'teacher':
                terms['teacher'] = a.lam * sum(photometric(rgb, t.cuda(), a.dssim, ssim) for rgb, t in zip(rgbs, teacher_cpu)) / 3
            loss = sum(terms.values())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite objective')
            adam.zero_grad(set_to_none=True)
            loss.backward()
            adam.step()
            affine.step(cursor)
            if cursor % 20 == 0 or cursor == 1:
                log.write(json.dumps(dict(cursor=cursor, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()},
                                          seconds=time.time() - tick)) + '\n')
            del rgbs, state, loss, terms
        prefetch.close()
        g.optimizer.zero_grad(set_to_none=True)
        path = cr.save(out / f'checkpoint_{a.steps:05d}.pt', cr.SCHEMA_SR, g, plan,
                       dict(affine=affine.state_dict(), metadata=dict(cursor=a.steps, arm=a.arm, probe='appearance_refit')))
        result = dict(status='completed', seconds=time.time() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                      checkpoint=cr.entry(path), affine=affine.summary())
        rt.write_json(out / 'complete.json', dict(record, **result))
    except BaseException:
        prefetch.close()
        rt.write_json(out / f'failure_{time.time_ns()}.json', dict(record, error=traceback.format_exc()))
        raise
    finally:
        log.close()
    del g, adam, affine
    torch.cuda.empty_cache()
    if a.eval_out:
        summary = cr.run_eval(SimpleNamespace(checkpoint=str(path), plan=None, out=a.eval_out, label=a.arm + '_cam00_posthoc_cull',
                                              cull=True, frames=None, cpu_threads=a.cpu_threads))
        print(json.dumps(dict(arm=a.arm, psnr=summary['test']['psnr'], ssim=summary['test']['ssim'], lpips=summary['test']['lpips'])),
              flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='checkpoint whose geometry is frozen')
    p.add_argument('--plan-from', help='checkpoint whose embedded plan (scene/manifest/config) is used (default: --source)')
    p.add_argument('--baseline-sr', required=True, help='Bsync registry dir of the supervision manifest (schedule, teacher)')
    p.add_argument('--supervision', choices=('lr', 'hr', 'teacher'), required=True)
    p.add_argument('--operator', choices=('area', 'bicaa'), default='area')
    p.add_argument('--sh-degree', type=int, choices=(0, 3), default=3)
    p.add_argument('--init', choices=('grey', 'source'), default='grey')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--lr-dc', type=float, default=0.01)
    p.add_argument('--lr-head', type=float, default=1e-3)
    p.add_argument('--lr-final-ratio', type=float, default=0.1)
    p.add_argument('--affine-lr', type=float, default=1e-3)
    p.add_argument('--dssim', type=float, default=0.2)
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--io-workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--eval-out')
    p.add_argument('--arm', required=True)
    run(p.parse_args(argv))


if __name__ == '__main__':
    main()
