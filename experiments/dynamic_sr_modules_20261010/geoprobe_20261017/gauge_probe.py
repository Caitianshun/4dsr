"""Test-view registration gauge probe (Claude, session 4dsr-e9, 2026-10-18). Diagnostic only: it reads the test camera's HR, which no
method may use. New file; ens_eval_view / cullretrain / the pinned 4DGaussians runtime are imported, never modified.

Question: the per-camera smooth warp field (modkit2 --warp-field) aligns the scene to the training cameras' mean camera (the field's
mean over cameras is removed; the test view is rendered with the identity). If the test camera's own residual distortion differs from
that mean, the field model is penalised at the test view even when its geometry is better, as with the -0.35 frame offset of Cook07's
cam00 for the time offsets (4dsr-7c, 10-17 19:05). For each model this probe renders the test view on --frames (default every 10th
frame), then fits on the even-indexed frames and scores on the odd-indexed ones:
  colour     a global 3x4 colour affine (normal equations; as eval_full's apsnr);
  warp G     a smooth displacement field on a G x G grid (HR px, bilinear align_corners, applied with bicubic grid_sample and border
             padding exactly as modkit2) jointly with the 3x4 colour affine, Adam on the MSE, for G in --grids.
PSNR is computed on the interior (--border px excluded on every side, the same pixels for every variant, so the numbers are below
eval_full's whole-image PSNR) over all pixels and over static pixels (eval_full's dynamic mask: HR frames 0, 5, 10, ... median
background, max_rgb |HR - B| > 0.06, dilated 5 px). Writes summary.json in --out.

Reading: when the field model's deficit against the calib model shrinks after a test-camera warp, part of it is a test-view
registration gauge (an evaluation ceiling that no legal method can remove); when it persists, the field model's reconstruction is
worse for this view.

  gauge_probe.py --checkpoint LABEL=rel/ck.pt [...] --view cam00 --out rel/dir [--frames ...] [--grids 4 8] [--iters 400]
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
TAU, DILATE = 0.06, 5


def warp(img, d):
    """img n x 3 x h x w, d 2 x G x G (HR px): out(p) = img(p + d(p)), as modkit2 ExtAffine.warp_field"""
    n, _, h, w = img.shape
    py, px = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32) + 0.5,
                            torch.arange(w, device=img.device, dtype=torch.float32) + 0.5, indexing='ij')
    up = F.interpolate(d[None], size=(h, w), mode='bilinear', align_corners=True)[0]
    grid = torch.stack([(px + up[0]) * (2 / w) - 1, (py + up[1]) * (2 / h) - 1], -1)
    return F.grid_sample(img, grid[None].expand(n, -1, -1, -1), mode='bicubic', padding_mode='border', align_corners=False)


def colour(img, M):
    """img n x 3 x h x w, M 3 x 4"""
    return torch.einsum('ij,njhw->nihw', M[:, :3], img) + M[:, 3].view(1, 3, 1, 1)


def fit_colour(x, y, inner):
    """least-squares 3x4 colour affine from renders x to HR y (n x 3 x h x w) on the interior, float64 normal equations"""
    a = x[..., inner[0], inner[1]].permute(1, 0, 2, 3).reshape(3, -1).double()
    b = y[..., inner[0], inner[1]].permute(1, 0, 2, 3).reshape(3, -1).double()
    A = torch.cat([a, torch.ones_like(a[:1])], 0)
    G = A @ A.T
    return torch.linalg.solve(G, A @ b.T).T.float()      # 3 x 4


def psnrs(pred, hr, inner, static):
    """equal-frame mean PSNR over the interior, and pooled static PSNR"""
    e = (pred.clamp(0, 1) - hr).pow(2).sum(1)[:, inner[0], inner[1]]        # n x h' x w'
    per = [-10 * math.log10(max(float(v) / 3, 1e-12)) for v in e.mean((1, 2))]
    s = static[:, inner[0], inner[1]]
    st = float(e[s].sum()) / max(3 * int(s.sum()), 1)
    return float(np.mean(per)), -10 * math.log10(max(st, 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=list(range(0, 300, 10)))
    p.add_argument('--grids', type=int, nargs='*', default=[4, 8])
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--border', type=int, default=16)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    result = dict(kind='gauge_probe', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(ens_eval.__file__), Path(cr.__file__))}, models={})
    hr = static = inner = None
    even = odd = None
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        obs = {int(o['frame_index']): o for o in ens_eval.view_observations(m['manifest'], a.view, label)}
        root = Path(m['manifest']['_root'])
        if hr is None:
            hrs = [legacy.read_rgb(root / obs[f]['hr_path']) for f in a.frames]
            bg = np.median(np.stack([legacy.read_rgb(root / obs[f]['hr_path']) for f in range(0, 300, 5)]), axis=0)
            masks = []
            for x in hrs:
                d = torch.from_numpy(np.abs(x - bg).max(axis=2) > TAU)[None, None].float()
                masks.append(F.max_pool2d(d, 2 * DILATE + 1, 1, DILATE)[0, 0] > 0)
            hr = torch.from_numpy(np.stack(hrs)).permute(0, 3, 1, 2).float().cuda()
            static = (~torch.stack(masks)).cuda()
            h, w = hr.shape[-2:]
            b = a.border
            inner = (slice(b, h - b), slice(b, w - b))
            even = list(range(0, len(a.frames), 2))
            odd = list(range(1, len(a.frames), 2))
            result['hr_shape'] = [h, w]
            result['static_share'] = float(static.float().mean())
        with torch.no_grad():
            imgs = []
            for f in a.frames:
                o = obs[f]
                cam = native.hr_camera(ev.camera_item(m['manifest'], o, torch), f, m['manifest'])
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                imgs.append(native.native_rgb(cam, m['g'], st, m['pipe'], m['background'])['render'].float())
            x = torch.stack(imgs)
        row = {}
        row['identity'] = psnrs(x[odd], hr[odd], inner, static[odd])
        M = fit_colour(x[even], hr[even], inner)
        row['colour'] = psnrs(colour(x[odd], M), hr[odd], inner, static[odd])
        for g in a.grids:
            d = torch.zeros(2, g, g, device='cuda', requires_grad=True)
            Mv = M.clone().requires_grad_(True)
            opt = torch.optim.Adam([dict(params=[d], lr=0.02), dict(params=[Mv], lr=2e-4)])
            xe, he = x[even], hr[even]
            for it in range(a.iters):
                loss = (colour(warp(xe, d), Mv) - he)[..., inner[0], inner[1]].pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                row[f'warp{g}'] = psnrs(colour(warp(x[odd], d), Mv), hr[odd], inner, static[odd])
                row[f'warp{g}_field_rms_px'] = float(d.pow(2).sum(0).mean().sqrt())
                row[f'warp{g}_field_max_px'] = float(d.pow(2).sum(0).sqrt().max())
        result['models'][label] = dict(info=m['info'], **{k: (dict(psnr=v[0], static_psnr=v[1]) if isinstance(v, tuple) else v)
                                                          for k, v in row.items()})
        print(json.dumps({label: {k: ([round(t, 3) for t in v] if isinstance(v, tuple) else round(v, 3)) for k, v in row.items()},
                          's': round(time.time() - started)}), flush=True)
        del m, x
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
