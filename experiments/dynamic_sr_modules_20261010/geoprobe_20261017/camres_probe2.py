"""Per-camera error structure v2: photometric fields, time offsets and local registration (Claude, session 4dsr-e9, 2026-10-17).
Diagnostic only. Reads training LR (never training HR); cam00 LR and HR only for the labelled test-view diagnostics below.
New file; camres_probe.py (v1) stays byte-frozen on the remotes; cullretrain / ens_eval_view are imported, never modified.

Each checkpoint is loaded as in evaluation (post-hoc cull, TF32 off). For every training camera c and frame t in --frames the model
is rendered at HR, the checkpoint's own calibration warp W_c (calibsr_v2 checkpoints) and colour affine A_c are applied as in
training, the result is reduced by the exact 4x4 box and compared with the LR image: r_t = box(A_c(W_c(render_t))) - LR_t.
cam00 (test view, identity warp and colour) is treated the same way against its own LR (test-LR diagnostic).

(1) Persistent photometric field F_c = Gaussian low-pass (sigma --sigma LR px) of the frame-mean residual.
    The v1 numbers (rms, persist_frac, persist_lf_frac, lf_gain_db, affine_refit_frac) are repeated for continuity.
    shared_frac: share of |F_c|^2 explained by the leave-one-out mean field of the other training cameras (per-channel alpha);
    knn_frac: the same with the mean field of the 2 nearest training cameras (camera centres), alpha = 1;
    radial_frac: per-channel least-squares fit c0 + c1 r^2 + c2 r^4.
    Test-view transfer (legal, training cameras only): the upsampled training mean field (alpha = 1, and alpha = median training
    alpha) or the 2-NN field is subtracted from the cam00 HR renders, scored against cam00 HR on the frames; the same with
    cam00's own field F_00 is the test-LR oracle bound (diagnostic, not a method).
(2) Time offset: residual energy (all pixels, and moving pixels = top --move-q share per frame of |LR_t - median_t LR|) of
    renders at time + k dt for k in --dts (dt = the camera's median frame spacing); parabola through the best k and its
    neighbours -> k*, moving-pixel energy at k* relative to k = 0, in dB.
(3) Local registration: Gauss-Newton Lucas-Kanade (4 iterations, shift field bilinear between cell centres) per cell of a
    --cells grid at LR scale (normal equations summed over channels, frames and static pixels): the shift (LR px) that would
    bring the render onto the LR, i.e. render(x + d) ~ LR(x), the same sign as the calibsr_v2 (tx, ty). The 5-term global model of calibsr_v2 full (tx, ty, s, theta, k1; first order, displacement
    at the half-diagonal R) is fitted to the cells (weights = trace of the normal matrix). Reported in HR px: rms of the cells, of
    the global fit and of the remainder (what one global warp per camera cannot absorb); split-half (even / odd frames)
    correlation of the remainder as a noise check.
Writes summary.json and fields.npz (F_c at 1/8 LR, cell shifts) in --out.

  camres_probe2.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--frames ...] [--sigma 6] [--dts -1 -0.5 -0.25 0 0.25 0.5 1]
                   [--cells 8 6] [--move-q 0.05]
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
for _p in (CULL, VAR):
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


def blur(x, sigma):
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sigma) ** 2)
    k = k / k.sum()
    y = x.reshape(-1, 1, *x.shape[-2:])
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))
    return y.reshape(x.shape)


def colour_fn(aff):
    """cullretrain.CameraAffine.apply with the checkpoint's deltas (identity for cameras outside the list, e.g. cam00)"""
    if not aff or 'delta' not in aff:
        return lambda img, cam: img
    names = list(aff['names'])
    d_all = aff['delta'].float()
    d_all = d_all - d_all.mean(0)

    def apply(img, cam):
        if cam not in names:
            return img
        d = d_all[names.index(cam)].to(img.device)
        return torch.einsum('ij,jhw->ihw', torch.eye(3, device=img.device) + d[:, :3], img) + d[:, 3, None, None]
    return apply


def warp_fn(aff):
    """calibsr_v2 CalibAffine.warp_image reproduced with the checkpoint's warp (identity without one, or outside the list)"""
    if not aff or 'warp' not in aff:
        return lambda img, cam: img
    names = list(aff['names'])
    w_all = aff['warp'].float()
    w_all = w_all - w_all.mean(0)
    n = w_all.shape[1]

    def apply(img, cam):
        if cam not in names:
            return img
        d = w_all[names.index(cam)].to(img.device)
        h, w = img.shape[-2:]
        ys = torch.arange(h, device=img.device, dtype=torch.float32) + 0.5 - h / 2
        xs = torch.arange(w, device=img.device, dtype=torch.float32) + 0.5 - w / 2
        py, px = torch.meshgrid(ys, xs, indexing='ij')
        R = math.hypot(h / 2, w / 2)
        tx, ty = d[0], d[1]
        if n == 5:
            s, th, k1 = d[2] / R, d[3] / R, d[4] / R
            radial = 1 + k1 * (px * px + py * py) / (R * R)
            c, sn = torch.cos(th), torch.sin(th)
            qx = (1 + s) * (c * px - sn * py) * radial + tx
            qy = (1 + s) * (sn * px + c * py) * radial + ty
        else:
            qx, qy = px + tx, py + ty
        grid = torch.stack([(qx + w / 2) * (2 / w) - 1, (qy + h / 2) * (2 / h) - 1], -1)
        return F.grid_sample(img[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]
    return apply


def frac_explained(f, pred, fit_alpha):
    """share of |f|^2 removed by alpha * pred (per-channel least-squares alpha, or alpha = 1); returns (frac, alphas)"""
    if fit_alpha:
        al = (f * pred).sum((1, 2)) / pred.pow(2).sum((1, 2)).clamp_min(1e-20)
    else:
        al = torch.ones(f.shape[0], device=f.device)
    rem = f - al[:, None, None] * pred
    return 1 - float(rem.pow(2).sum()) / max(float(f.pow(2).sum()), 1e-20), [float(v) for v in al]


def radial_frac(f):
    h, w = f.shape[-2:]
    ys = (torch.arange(h, device=f.device, dtype=torch.float64) + 0.5 - h / 2)
    xs = (torch.arange(w, device=f.device, dtype=torch.float64) + 0.5 - w / 2)
    py, px = torch.meshgrid(ys, xs, indexing='ij')
    r2 = (px * px + py * py) / (h * h / 4 + w * w / 4)
    X = torch.stack([torch.ones_like(r2), r2, r2 * r2], -1).reshape(-1, 3)
    tot, rem = 0.0, 0.0
    for c in range(f.shape[0]):
        y = f[c].double().reshape(-1)
        sol = torch.linalg.lstsq(X, y[:, None]).solution
        rem += float((X @ sol - y[:, None]).pow(2).sum())
        tot += float(y.pow(2).sum())
    return 1 - rem / max(tot, 1e-20)


def lk_accumulate(I, r, static, cells, acc):
    """add one frame's Lucas-Kanade normal equations per cell; I, r: 3 x h x w (render, render - LR); static: h x w bool"""
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    gx[:, :, 1:-1] = (I[:, :, 2:] - I[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (I[:, 2:, :] - I[:, :-2, :]) / 2
    valid = static.clone()
    valid[:2, :] = valid[-2:, :] = False
    valid[:, :2] = valid[:, -2:] = False
    m = valid.float()[None]
    terms = torch.stack([(gx * gx * m).sum(0), (gx * gy * m).sum(0), (gy * gy * m).sum(0),
                         -(gx * r * m).sum(0), -(gy * r * m).sum(0), m[0]], 0).double().cpu()  # 6 x h x w
    nx, ny = cells
    h, w = I.shape[-2:]
    ey = np.linspace(0, h, ny + 1).round().astype(int)
    ex = np.linspace(0, w, nx + 1).round().astype(int)
    for j in range(ny):
        for i in range(nx):
            acc[j, i] += terms[:, ey[j]:ey[j + 1], ex[i]:ex[i + 1]].sum((1, 2))


def lk_cells(downs, lrs, statics, cells, idx, iters=4):
    """Gauss-Newton per-cell shifts d (LR px, 2 x ny x nx) with render(x + d(x)) ~ LR(x), d(x) bilinear between cell centres.
    Iterating removes the bias of the central-difference gradient (the fixed point is sum g r = 0). Returns (d, last normal sums)."""
    nx, ny = cells
    h, w = lrs[0].shape[-2:]
    dev = lrs[0].device
    d = torch.zeros(2, ny, nx, dtype=torch.float64)
    py, px = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(w, device=dev, dtype=torch.float32) + 0.5, indexing='ij')
    for _ in range(iters):
        fld = F.interpolate(d.float()[None].to(dev), size=(h, w), mode='bilinear', align_corners=False)[0]
        grid = torch.stack([(px + fld[0]) * (2 / w) - 1, (py + fld[1]) * (2 / h) - 1], -1)
        acc = torch.zeros(ny, nx, 6, dtype=torch.float64)
        for i in idx:
            wv = F.grid_sample(downs[i][None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]
            lk_accumulate(wv, wv - lrs[i], statics[i], cells, acc)
        for j in range(ny):
            for i in range(nx):
                q = acc[j, i]
                A = torch.tensor([[q[0], q[1]], [q[1], q[2]]], dtype=torch.float64)
                A = A + 1e-6 * max(float(A.trace()), 1e-12) * torch.eye(2, dtype=torch.float64)
                d[:, j, i] += torch.linalg.solve(A, torch.stack([q[3], q[4]]))
    return d, acc


def lk_fit(d, acc, h, w):
    """weighted 5-term global fit (calibsr_v2 full, first order) of the cell shifts; returns dict in HR px"""
    ny, nx = d.shape[1:]
    ey = np.linspace(0, h, ny + 1)
    ex = np.linspace(0, w, nx + 1)
    R = math.hypot(h / 2, w / 2)
    cells, wts, rows = [], [], []
    for j in range(ny):
        for i in range(nx):
            px = (ex[i] + ex[i + 1]) / 2 - w / 2
            py = (ey[j] + ey[j + 1]) / 2 - h / 2
            r2 = (px * px + py * py) / (R * R)
            rows.append([[1, 0, px / R, -py / R, px * r2 / R], [0, 1, py / R, px / R, py * r2 / R]])
            cells.append([float(d[0, j, i]), float(d[1, j, i])])
            wts.append(float(acc[j, i, 0] + acc[j, i, 2]))
    cells = np.array(cells)
    wv = np.array(wts) / max(np.mean(wts), 1e-20)
    D = np.array(rows).reshape(-1, 5)
    W = np.repeat(wv, 2)
    sol, *_ = np.linalg.lstsq(D * np.sqrt(W)[:, None], cells.reshape(-1) * np.sqrt(W), rcond=None)
    fit = (D @ sol).reshape(-1, 2)
    rem = cells - fit
    wrms = lambda v: float(np.sqrt((wv[:, None] * v * v).sum() / (2 * wv.sum())))
    return dict(cells_lr=cells.tolist(), weights=wv.tolist(), global_hr_at_R=(4 * sol).tolist(), cell_rms_hr=4 * wrms(cells),
                fit_rms_hr=4 * wrms(fit), rem_rms_hr=4 * wrms(rem), rem=rem)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=[15, 45, 75, 105, 135, 165, 195, 225, 255, 285])
    p.add_argument('--sigma', type=float, default=6.0)
    p.add_argument('--dts', type=float, nargs='*', default=[-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
    p.add_argument('--cells', type=int, nargs=2, default=[8, 6])
    p.add_argument('--move-q', type=float, default=0.05)
    a = p.parse_args()
    if 0 not in a.dts:
        raise SystemExit('--dts must contain 0')
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    k0 = a.dts.index(0)
    result = dict(kind='camres_probe2', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models={})
    arrays = {}
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        aff = payload.get('affine')
        colour, warp = colour_fn(aff), warp_fn(aff)
        has_warp = bool(aff and 'warp' in aff)
        del payload
        manifest = m['manifest']
        root = Path(manifest['_root'])
        train = list(manifest['splits']['train'])
        cams = train + ['cam00']
        per, fields, centres, hr00 = {}, {}, {}, {}
        for cam in cams:
            obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == cam}
            tl = sorted(float(o['time']) for o in obs.values())
            dt = float(np.median(np.diff(tl)))
            lrs = [torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / obs[f]['lr_path']))).permute(2, 0, 1).float().cuda()
                   for f in a.frames]
            med = torch.stack(lrs).median(0).values
            moving = []
            for lr in lrs:
                dev = (lr - med).abs().sum(0)
                moving.append(dev >= torch.quantile(dev.flatten()[::7], 1 - a.move_q))
            h, w = lrs[0].shape[-2:]
            e_all = np.zeros(len(a.dts))
            e_mov = np.zeros(len(a.dts))
            n_mov = 0
            rs, downs, statics = [], [], []
            for fi, f in enumerate(a.frames):
                o = obs[f]
                with torch.no_grad():
                    camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                    if cam not in centres:
                        cc = getattr(camera, 'camera_center', None)
                        cc = camera.world_view_transform.inverse()[3, :3] if cc is None else cc
                        centres[cam] = [float(v) for v in cc.detach().cpu()]
                    mv = moving[fi]
                    n_mov += int(mv.sum())
                    for ki, k in enumerate(a.dts):
                        st = native.native_effective_state(m['g'], float(o['time']) + k * dt, torch)
                        img = native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float()
                        img = colour(warp(img, cam), cam)
                        fy = img.shape[-2] // h
                        down = F.avg_pool2d(img[None], fy)[0].clamp(0, 1)
                        e = (down - lrs[fi]).pow(2).sum(0)
                        e_all[ki] += float(e.mean())
                        e_mov[ki] += float(e[mv].sum())
                        if ki == k0:
                            r = down - lrs[fi]
                            rs.append(r)
                            downs.append(down)
                            statics.append(~(F.max_pool2d(mv.float()[None, None], 5, 1, 2)[0, 0] > 0))
                            if cam == 'cam00':
                                hr00[f] = img.clamp(0, 1)
            r = torch.stack(rs)
            tot = float(r.pow(2).mean())
            pm = r.mean(0)
            lf = blur(pm, a.sigma)
            x = torch.stack(downs).permute(1, 0, 2, 3).reshape(3, -1)
            y = torch.stack(lrs).permute(1, 0, 2, 3).reshape(3, -1)
            X = torch.cat([x, torch.ones(1, x.shape[1], device=x.device)], 0)
            sol = torch.linalg.lstsq(X.T.double(), y.T.double()).solution.float()
            refit = (X.T @ sol).T - y
            frac_lf = float(lf.pow(2).mean()) / max(tot, 1e-20)
            e_mov = e_mov / max(n_mov, 1)
            e_all = e_all / len(a.frames)
            b = int(np.argmin(e_mov))
            kstar, estar = a.dts[b], e_mov[b]
            if 0 < b < len(a.dts) - 1:
                xs3, ys3 = np.array(a.dts[b - 1:b + 2]), e_mov[b - 1:b + 2]
                c2, c1, c0 = np.polyfit(xs3, ys3, 2)
                if c2 > 0:
                    kstar = float(np.clip(-c1 / (2 * c2), xs3[0], xs3[-1]))
                    estar = float(c2 * kstar ** 2 + c1 * kstar + c0)
            nf = len(a.frames)
            lk = lk_fit(*lk_cells(downs, lrs, statics, a.cells, range(nf)), h, w)
            halves = [lk_fit(*lk_cells(downs, lrs, statics, a.cells, range(q, nf, 2)), h, w)['rem'].reshape(-1) for q in (0, 1)]
            split = float(np.corrcoef(halves[0], halves[1])[0, 1]) if np.std(halves[0]) > 0 and np.std(halves[1]) > 0 else None
            per[cam] = dict(rms=tot ** 0.5, closure_psnr=-10 * float(np.log10(max(tot, 1e-20))),
                            persist_frac=float(pm.pow(2).mean()) / max(tot, 1e-20), persist_lf_frac=frac_lf,
                            lf_gain_db=-10 * float(np.log10(max(1 - frac_lf, 1e-12))),
                            affine_refit_frac=1 - float(refit.pow(2).mean()) / max(tot, 1e-20),
                            radial_frac=radial_frac(lf),
                            time=dict(dt=dt, ks=a.dts, e_all=e_all.tolist(), e_mov=e_mov.tolist(), k_star=kstar,
                                      mov_gain_db=10 * float(np.log10(e_mov[k0] / max(estar, 1e-20))), moving_px=n_mov),
                            reg=dict(global_hr_at_R=lk['global_hr_at_R'], cell_rms_hr=lk['cell_rms_hr'], fit_rms_hr=lk['fit_rms_hr'],
                                     rem_rms_hr=lk['rem_rms_hr'], rem_split_corr=split))
            fields[cam] = lf
            arrays[f'{label}/{cam}/field'] = F.avg_pool2d(lf[None], 8)[0].cpu().numpy()
            arrays[f'{label}/{cam}/cells'] = np.array(lk['cells_lr'])
            print(json.dumps(dict(model=label, cam=cam, rms=round(per[cam]['rms'], 5), lf=round(frac_lf, 3),
                                  k_star=round(kstar, 3), mov_gain_db=round(per[cam]['time']['mov_gain_db'], 3),
                                  reg_cell=round(lk['cell_rms_hr'], 3), reg_rem=round(lk['rem_rms_hr'], 3), split=split,
                                  s=round(time.time() - started))), flush=True)
            del lrs, rs, downs, statics
        # shared / neighbour structure of the persistent fields (training cameras only)
        cen = {c: torch.tensor(v) for c, v in centres.items()}
        mean_all = torch.stack([fields[c] for c in train]).mean(0)
        alphas = []
        for c in train:
            loo = (mean_all * len(train) - fields[c]) / (len(train) - 1)
            fr, al = frac_explained(fields[c], loo, True)
            fr1, _ = frac_explained(fields[c], loo, False)
            others = sorted((float((cen[o] - cen[c]).norm()), o) for o in train if o != c)[:2]
            knn = torch.stack([fields[o] for _, o in others]).mean(0)
            per[c].update(shared_frac=fr, shared_frac_alpha1=fr1, shared_alpha=al, knn_frac=frac_explained(fields[c], knn, False)[0],
                          knn=[o for _, o in others])
            alphas.append(al)
        med_alpha = torch.tensor(np.median(np.array(alphas), 0), dtype=torch.float32, device=mean_all.device)
        nn00 = [o for _, o in sorted((float((cen[o] - cen['cam00']).norm()), o) for o in train)[:2]]
        knn00 = torch.stack([fields[o] for o in nn00]).mean(0)
        preds = dict(mean_alpha1=mean_all, mean_alpha_med=med_alpha[:, None, None] * mean_all, knn2=knn00,
                     own_field_test_lr_oracle=fields['cam00'])
        per['cam00'].update(shared_frac=frac_explained(fields['cam00'], mean_all, True)[0],
                            shared_frac_alpha1=frac_explained(fields['cam00'], mean_all, False)[0],
                            knn_frac=frac_explained(fields['cam00'], knn00, False)[0], knn=nn00)
        transfer = {k: [] for k in ['none', *preds]}
        o00 = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == 'cam00'}
        for f, img in hr00.items():
            hr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o00[f]['hr_path']))).permute(2, 0, 1).float().cuda()
            transfer['none'].append(float((img - hr).pow(2).mean()))
            for k, fld in preds.items():
                up = F.interpolate(fld[None], size=img.shape[-2:], mode='bilinear', align_corners=False)[0]
                transfer[k].append(float(((img - up).clamp(0, 1) - hr).pow(2).mean()))
        tr = {k: float(np.mean([-10 * np.log10(max(v, 1e-12)) for v in vals])) for k, vals in transfer.items()}
        tr_delta = {k: tr[k] - tr['none'] for k in preds}
        med = lambda key: float(np.median([per[c][key] for c in train]))
        result['models'][label] = dict(
            per_camera=per, info=m['info'], has_warp=has_warp, centres=centres,
            train_median={k: med(k) for k in ('rms', 'closure_psnr', 'persist_frac', 'persist_lf_frac', 'lf_gain_db',
                                               'affine_refit_frac', 'radial_frac', 'shared_frac', 'shared_frac_alpha1', 'knn_frac')},
            time_train=dict(k_star_median=float(np.median([per[c]['time']['k_star'] for c in train])),
                            k_star_abs_median=float(np.median([abs(per[c]['time']['k_star']) for c in train])),
                            mov_gain_db_median=float(np.median([per[c]['time']['mov_gain_db'] for c in train])),
                            mov_gain_db_max=float(max(per[c]['time']['mov_gain_db'] for c in train))),
            reg_train=dict(cell_rms_hr_median=float(np.median([per[c]['reg']['cell_rms_hr'] for c in train])),
                           rem_rms_hr_median=float(np.median([per[c]['reg']['rem_rms_hr'] for c in train])),
                           rem_split_corr_median=float(np.median([per[c]['reg']['rem_split_corr'] or 0 for c in train]))),
            cam00_transfer_psnr=tr, cam00_transfer_delta_db=tr_delta)
        print(json.dumps({label: dict(train_median=result['models'][label]['train_median'], time=result['models'][label]['time_train'],
                                      reg=result['models'][label]['reg_train'], transfer=tr_delta)}), flush=True)
        del m, fields, hr00
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    np.savez_compressed(out / 'fields.npz', **arrays)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
