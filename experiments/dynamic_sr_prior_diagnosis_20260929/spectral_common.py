"""Auditable CPU spectral arithmetic and linear observation backprojection.

No renderer or training code is imported. HR is accepted only by diagnostic
functions; anchored_target's numerical inputs are exclusively LR and teacher.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929/spectrum'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)

def rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.

def bands(h, w):
    # Integer comparisons avoid approximate cutoff membership.
    ky = np.arange(h)[:, None]; kx = np.arange(w)[None, :]
    low_inc = (4 * ky <= h) & (4 * kx <= w)
    mid_inc = (2 * ky <= h) & (2 * kx <= w)
    dc = np.zeros((h, w), bool); dc[0, 0] = True
    result = {'dc': dc, 'low': low_inc & ~dc,
              'mid': mid_inc & ~low_inc, 'high': ~mid_inc}
    assert np.all(sum(m.astype(int) for m in result.values()) == 1)
    return result

def transform(x): return dctn(x.astype(np.float64), axes=(0, 1), norm='ortho', workers=1)
def inverse(x): return idctn(x, axes=(0, 1), norm='ortho', workers=1)

Q = np.array([[1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)],
              [1 / math.sqrt(2), -1 / math.sqrt(2), 0],
              [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]])

def spectral_budget(pred, gt, raw=None):
    h, w = gt.shape[:2]; n = 3 * h * w
    r, g = transform(pred), transform(gt); err = pred.astype(np.float64) - gt.astype(np.float64)
    e = r - g; masks = bands(h, w); mse = float(np.square(err).mean())
    gt_structure = float(np.square(g).sum() - np.square(g[0, 0]).sum())
    row = {'mse': mse, 'psnr_from_frame_mse': -10 * math.log10(max(mse, 1e-12))}
    for name, m in masks.items():
        er = float(np.square(e[m]).sum() / n); re = float(np.square(r[m]).sum()); ge = float(np.square(g[m]).sum())
        row.update({name + '_mse': er, name + '_error_share': er / mse if mse else 0.,
                    name + '_gt_structure_share': ge / gt_structure if name != 'dc' and gt_structure else None,
                    name + '_render_to_gt_energy': re / ge if ge else None,
                    name + '_coefficient_cosine': float((r[m] * g[m]).sum() / math.sqrt(re * ge)) if re * ge else None})
    row['parseval_abs_residual'] = abs(sum(row[k + '_mse'] for k in masks) - mse)
    q = err @ Q.T
    for i in range(3): row[f'q{i}_mse'] = float(np.square(q[..., i]).sum() / n)
    row['q_chroma_mse'] = row['q1_mse'] + row['q2_mse']
    row['color_additivity_abs_residual'] = abs(sum(row[f'q{i}_mse'] for i in range(3)) - mse)
    for i, c in enumerate('rgb'): row[c + '_mean_bias'] = float(err[..., i].mean())
    raw = pred if raw is None else raw
    row.update(raw_out_of_range_fraction=float(((raw < 0) | (raw > 1)).mean()),
               eval_saturated_channel_fraction=float(((pred <= 0) | (pred >= 1)).mean()),
               eval_white_pixel_fraction=float((pred >= 1).all(axis=2).mean()))
    assert row['parseval_abs_residual'] < 1e-11
    assert row['color_additivity_abs_residual'] < 1e-11
    return row, (r, g, masks)

def region_budget(pred, gt, boxes):
    h, w = gt.shape[:2]; n = 3 * h * w
    used = np.zeros((h, w), bool); sq = np.square(pred.astype(np.float64) - gt.astype(np.float64)).sum(axis=2)
    results = []
    for name, (x0, y0, x1, y1) in boxes.items():
        m = np.zeros_like(used); m[y0:y1, x0:x1] = True; m &= ~used; used |= m
        count = int(m.sum()); contribution = float(sq[m].sum() / n)
        results.append(dict(region=name, pixels=count, area_fraction=count / (h * w),
                            mse_contribution=contribution, within_region_mse=float(sq[m].mean() / 3) if count else None))
    m = ~used
    results.append(dict(region='rest', pixels=int(m.sum()), area_fraction=float(m.mean()),
                        mse_contribution=float(sq[m].sum() / n), within_region_mse=float(sq[m].mean() / 3)))
    assert abs(sum(r['mse_contribution'] for r in results) - sq.sum() / n) < 1e-11
    return results

def oracle_images(pred, transformed):
    r, g, masks = transformed
    for name, mask in [('replace_dc', masks['dc']), ('replace_low_non_dc', masks['low']),
                       ('replace_mid_high', masks['mid'] | masks['high'])]:
        coeff = r.copy(); coeff[mask] = g[mask]
        yield name, inverse(coeff)
    q = pred.astype(np.float64) @ Q.T; gq = inverse(g) @ Q.T
    q[..., 1:] = gq[..., 1:]
    yield 'replace_chroma_q1q2', q @ Q

def d0(x, size):
    return F.interpolate(x, size=size, mode='bicubic', align_corners=False, antialias=True)

def norm_protocol(hr_size=(1008, 1344), lr_size=(252, 336)):
    """Exact finite-grid separable operator norm via the actual torch kernels.

    Each identity image exposes one 1D interpolation matrix. Eigenvalues of
    A A^T give its spectral norm; separability is independently checked.
    """
    from scipy.linalg import eigvalsh
    h, w = hr_size; lh, lw = lr_size
    yy = torch.eye(h, dtype=torch.float64)[None, None]
    xx = torch.eye(w, dtype=torch.float64)[None, None]
    ay = d0(yy, (lh, h))[0, 0].numpy()
    ax = d0(xx, (w, lw))[0, 0].numpy().T
    ny = float(eigvalsh(ay @ ay.T, subset_by_index=(lh - 1, lh - 1))[0])
    nx = float(eigvalsh(ax @ ax.T, subset_by_index=(lw - 1, lw - 1))[0])
    rng = np.random.default_rng(20260929)
    u = torch.from_numpy(rng.normal(size=(1, 1, h, w))).requires_grad_(True)
    v = torch.from_numpy(rng.normal(size=(1, 1, lh, lw)))
    du = d0(u, lr_size); adj = torch.autograd.grad(du, u, v)[0]
    inner_res = abs(float((du * v).sum() - (u * adj).sum()))
    sep_res = float(np.max(np.abs(du.detach().numpy()[0, 0] - ay @ u.detach().numpy()[0, 0] @ ax.T)))
    assert inner_res < 1e-8 and sep_res < 1e-10
    return dict(hr_size=list(hr_size), lr_size=list(lr_size), norm_squared=ny * nx,
                eta=0.95 / (ny * nx), steps=8, safety_factor=0.95,
                norm_method='largest eigenvalues of actual separable finite-grid D0 matrices; no fitted data or HR content',
                adjoint_inner_product_abs_residual=inner_res, separability_max_abs_residual=sep_res,
                D0='torch.interpolate bicubic align_corners=False antialias=True, no clamp, no uint8 rounding',
                actual_D='clamp(D0,0,1); dataset also rounds to uint8; exact consistency with quantized LR is not asserted',
                clamp='only after 8 linear steps for usable target; both raw and clamped closure reported')

def anchored_target(teacher, lr, protocol):
    """Only legal teacher/LR arrays accepted. Exact autograd adjoint, 8 steps."""
    x = torch.from_numpy(np.ascontiguousarray(teacher.transpose(2, 0, 1)))[None].double()
    target = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1)))[None].double()
    trace = []
    for k in range(protocol['steps']):
        x = x.detach().requires_grad_(True); e = d0(x, target.shape[-2:]) - target
        trace.append(float(e.square().mean().detach()))
        grad = torch.autograd.grad(d0(x, target.shape[-2:]), x, e)[0]
        x = x - protocol['eta'] * grad
    raw = x.detach()[0].permute(1, 2, 0).numpy()
    return raw, trace

def closure(x, lr):
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].double()
    low = d0(t, lr.shape[:2]).numpy()[0].transpose(1, 2, 0)
    return {'d0_mse': float(np.square(low - lr).mean()),
            'd0_l1': float(np.abs(low - lr).mean()),
            'actual_D_mse': float(np.square(np.clip(low, 0, 1) - lr).mean()),
            'actual_D_l1': float(np.abs(np.clip(low, 0, 1) - lr).mean())}
