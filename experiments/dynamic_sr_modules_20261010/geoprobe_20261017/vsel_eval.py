"""Held-out-camera selection signal for per-scene configuration choice (Claude, session 4dsr-e9, 2026-10-18). Inference only. New
file; ens_eval_view / cullretrain / the pinned 4DGaussians runtime are imported, never modified.

User 10-18: the goal is per-scene optimisation; different parameters per scene are allowed. The checkpoints scored here were trained
with modkit8.py --holdout-camera CAM, so CAM (a training camera of the manifest) took no part in the SR stage. CAM is rendered with
the identity camera model (as the test view; its colour affine, calib and fields were never trained) on --frames, and scored
against three targets:
  teacher  CAM's teacher image of the registry (a function of CAM's LR only): the selection signal, no HR involved;
  lr       CAM's LR image; the render is reduced by the exact 4x4 area operator of the area4 protocol (as calibsr_v2): no HR;
  hr       CAM's HR image, only when every scored HR frame is present: diagnostic, never used by the selection rule.
Per target: psnr_odd (equal-frame mean PSNR of the odd-indexed frames of --frames) and apsnr (the same frames after a global 3x4
colour affine fitted on the even-indexed frames, float64 normal equations, as eval_full; CAM's colour gauge is arbitrary). For the
teacher target also the FFT band PSNR of the unaligned odd frames (low r < 0.125, mid < 0.25, high >= 0.25 cycles / HR px; band MSE
= band energy / (3 (hw)^2), as eval_full). Writes summary.json in --out.

  vsel_eval.py --checkpoint LABEL=rel/ck.pt [...] --camera CAM --baseline-sr DIR --out rel/dir [--frames 0 2 4 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CULL = HERE.parents[1] / 'dynamic_sr_cull_retrain_20261009'
VAR = CULL / 'variance_20261014'
for _p in (CULL, VAR, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cullretrain as cr  # noqa: E402
try:
    import ens_eval_view as ens_eval  # noqa: E402
except ImportError:
    import ens_eval  # noqa: E402
import full_evaluate_registered_v2 as ev  # noqa: E402
import full_refine_registered_v2 as native  # noqa: E402

ROOT = rt.ROOT
BANDS = (0.125, 0.25)


def area4(x, size):
    """x 3 x H x W -> exact integer-factor box average to size (h, w), as calibsr_v2.area_degradation"""
    h, w = x.shape[-2:]
    fy, fx = h // int(size[0]), w // int(size[1])
    if fy != fx or fy * int(size[0]) != h or fx * int(size[1]) != w:
        raise ValueError(f'area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}')
    return F.avg_pool2d(x[None], fy)[0]


def new_acc(device):
    return dict(G=torch.zeros(4, 4, dtype=torch.float64, device=device), B=torch.zeros(4, 3, dtype=torch.float64, device=device))


def accumulate(acc, x, y):
    """normal equations of the 3x4 colour affine x -> y (3 x h x w each)"""
    a = x.reshape(3, -1).double()
    b = y.reshape(3, -1).double()
    A = torch.cat([a, torch.ones_like(a[:1])], 0)
    acc['G'] += A @ A.T
    acc['B'] += A @ b.T


def solve(acc):
    return torch.linalg.solve(acc['G'], acc['B']).T.float()          # 3 x 4


def colour(M, x):
    return torch.einsum('ij,jhw->ihw', M[:, :3], x) + M[:, 3].view(3, 1, 1)


def psnr(pred, target):
    return -10 * math.log10(max(float((pred.clamp(0, 1) - target).pow(2).mean()), 1e-12))


def band_psnrs(pred, target):
    r = (pred.clamp(0, 1) - target).double()
    h, w = r.shape[-2:]
    e = torch.fft.fft2(r).abs().pow(2).sum(0)
    fy = torch.fft.fftfreq(h, device=r.device, dtype=torch.float64)
    fx = torch.fft.fftfreq(w, device=r.device, dtype=torch.float64)
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    masks = (rad < BANDS[0], (rad >= BANDS[0]) & (rad < BANDS[1]), rad >= BANDS[1])
    return [-10 * math.log10(max(float(e[m].sum()) / (3 * (h * w) ** 2), 1e-12)) for m in masks]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--baseline-sr', required=True, help='registry Bsync directory (its config names the teacher index)')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=list(range(0, 300, 2)))
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed evaluation preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    base = json.loads((ROOT / a.baseline_sr / 'config.json').read_text())
    tpath = ROOT / base['dependencies']['teacher']['path']
    tindex = json.loads(tpath.read_text())
    teacher_rel = {(e['camera'], int(e['frame'])): e['relative_path'] for e in tindex['entries'] if e['camera'] == a.camera}
    even, odd = a.frames[0::2], a.frames[1::2]
    started = time.time()
    result = dict(kind='vsel_eval', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  teacher_index=cr.entry(tpath), sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(ens_eval.__file__), Path(cr.__file__))},
                  models={})

    def load(path):
        return torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(path))).permute(2, 0, 1).float().cuda()

    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        manifest = m['manifest']
        if a.camera not in manifest['splits']['train']:
            raise ValueError(f'{a.camera} is not a training camera of {label}')
        root = Path(manifest['_root'])
        obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == a.camera}
        has_hr = all((root / obs[f]['hr_path']).exists() for f in a.frames)
        names = ['teacher', 'lr'] + (['hr'] if has_hr else [])

        def targets(f):
            t = dict(teacher=load(root / teacher_rel[(a.camera, f)]), lr=load(root / obs[f]['lr_path']))
            if has_hr:
                t['hr'] = load(root / obs[f]['hr_path'])
            return t

        def render(f):
            o = obs[f]
            with torch.no_grad():
                cam = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                return native.native_rgb(cam, m['g'], st, m['pipe'], m['background'])['render'].float()

        acc = {n: new_acc('cuda') for n in names}
        for f in even:
            x, t = render(f), targets(f)
            for n in names:
                accumulate(acc[n], area4(x, t[n].shape[-2:]) if n == 'lr' else x, t[n])
        M = {n: solve(acc[n]) for n in names}
        per = {n: dict(psnr_odd=[], apsnr=[]) for n in names}
        bands = []
        for f in odd:
            x, t = render(f), targets(f)
            for n in names:
                y = area4(x, t[n].shape[-2:]) if n == 'lr' else x
                per[n]['psnr_odd'].append(psnr(y, t[n]))
                per[n]['apsnr'].append(psnr(colour(M[n], y), t[n]))
            bands.append(band_psnrs(x, t['teacher']))
        row = {f'{n}_{k}': float(np.mean(v)) for n in names for k, v in per[n].items()}
        row.update(teacher_low_psnr=float(np.mean([b[0] for b in bands])), teacher_mid_psnr=float(np.mean([b[1] for b in bands])),
                   teacher_high_psnr=float(np.mean([b[2] for b in bands])), hr_scored=has_hr,
                   colour_affines={n: M[n].tolist() for n in names})
        result['models'][label] = dict(info=m['info'], **row)
        print(json.dumps({label: {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, 's': round(time.time() - started)}), flush=True)
        del m
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
