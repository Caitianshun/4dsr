"""Per-camera temporal camera model, v3: rolling shutter and exposure blur (Claude, session 4dsr-e9, 2026-10-17). Diagnostic only.
Reads training LR (never training HR); cam00 LR only for a labelled test-LR diagnostic. New file; camres_probe2.py (byte-frozen)
supplies the colour / warp helpers; cullretrain / ens_eval_view are imported, never modified.

camres_probe2.py found per-camera time offsets of up to one frame. N3DV was captured with GoPro cameras, which read the sensor out
row by row (rolling shutter) and integrate over an exposure time; both act on moving content only. For each checkpoint and camera,
the render at time t + k dt (k in --ks, dt = median frame spacing) gets the checkpoint's calibsr warp and colour affine, is reduced
by the exact 4x4 box and compared with the LR image on moving pixels (top --move-q share per frame of |LR_t - median_t LR|):
  global     k* of the whole image (parabola through the best grid point and its neighbours), as camres_probe2;
  rolling    k*_b per horizontal band b of --bands bands; slope of k*_b against the band centre (row / H - 0.5), weighted by the
             band's moving-pixel count, in frames per image height (> 0: lower rows are later); moving-pixel residual with the
             per-band k*_b vs the global k* (dB gain);
  exposure   mean of the renders on the k grid within [k0 - W/2, k0 + W/2] around the grid point k0 nearest to the global k*, for
             W in --widths (frames); moving-pixel residual vs W = 0 (dB gain). A box-shaped exposure on the k grid.
Static pixels do not depend on time; the whole-image residual is reported as a check.
Writes summary.json in --out.

  camres_probe3.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--frames ...] [--ks ...] [--bands 4] [--widths 0 0.5 1]
"""
from __future__ import annotations

import argparse
import json
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
import camres_probe2 as p2  # noqa: E402

ROOT = rt.ROOT


def parabola_min(xs, ys):
    """grid argmin refined by a parabola through it and its neighbours (clipped to the neighbours)"""
    b = int(np.argmin(ys))
    if 0 < b < len(xs) - 1:
        x3, y3 = np.array(xs[b - 1:b + 2], float), np.array(ys[b - 1:b + 2], float)
        c2, c1, c0 = np.polyfit(x3, y3, 2)
        if c2 > 0:
            k = float(np.clip(-c1 / (2 * c2), x3[0], x3[-1]))
            return k, float(c2 * k * k + c1 * k + c0), False
    return float(xs[b]), float(ys[b]), b in (0, len(xs) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=[15, 45, 75, 105, 135, 165, 195, 225, 255, 285])
    p.add_argument('--ks', type=float, nargs='*', default=[-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5])
    p.add_argument('--bands', type=int, default=4)
    p.add_argument('--widths', type=float, nargs='*', default=[0, 0.5, 1.0])
    p.add_argument('--move-q', type=float, default=0.05)
    a = p.parse_args()
    if 0 not in a.ks or sorted(a.ks) != a.ks:
        raise SystemExit('--ks must be sorted and contain 0')
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    ks = list(a.ks)
    result = dict(kind='camres_probe3', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(p2.__file__), Path(cr.__file__))}, models={})
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        aff = payload.get('affine')
        colour, warp = p2.colour_fn(aff), p2.warp_fn(aff)
        del payload
        manifest = m['manifest']
        root = Path(manifest['_root'])
        cams = list(manifest['splits']['train']) + ['cam00']
        per = {}
        for cam in cams:
            obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == cam}
            tl = sorted(float(o['time']) for o in obs.values())
            dt = float(np.median(np.diff(tl)))
            lrs = [torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / obs[f]['lr_path']))).permute(2, 0, 1).float().cuda()
                   for f in a.frames]
            med = torch.stack(lrs).median(0).values
            h, w = lrs[0].shape[-2:]
            edges = np.linspace(0, h, a.bands + 1).round().astype(int)
            centres = [((edges[b] + edges[b + 1]) / 2) / h - 0.5 for b in range(a.bands)]
            e_band = np.zeros((a.bands, len(ks)))          # moving-pixel squared error per band and k
            n_band = np.zeros(a.bands)
            e_all = np.zeros(len(ks))
            e_exp = {W: 0.0 for W in a.widths}
            downs_cache = []
            moving = []
            for fi, f in enumerate(a.frames):
                o = obs[f]
                lr = lrs[fi]
                dev = (lr - med).abs().sum(0)
                mv = dev >= torch.quantile(dev.flatten()[::7], 1 - a.move_q)
                moving.append(mv)
                with torch.no_grad():
                    camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                    downs = []
                    for ki, k in enumerate(ks):
                        st = native.native_effective_state(m['g'], float(o['time']) + k * dt, torch)
                        img = native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float()
                        img = colour(warp(img, cam), cam)
                        fy = img.shape[-2] // h
                        down = F.avg_pool2d(img[None], fy)[0].clamp(0, 1)
                        downs.append(down)
                        e = (down - lr).pow(2).sum(0)
                        e_all[ki] += float(e.mean())
                        for b in range(a.bands):
                            sl = slice(edges[b], edges[b + 1])
                            e_band[b, ki] += float(e[sl][mv[sl]].sum())
                    downs_cache.append(torch.stack(downs).half())
                for b in range(a.bands):
                    sl = slice(edges[b], edges[b + 1])
                    n_band[b] += float(mv[sl].sum())
            e_mov = e_band.sum(0) / max(n_band.sum(), 1)
            kg, eg, edge_g = parabola_min(ks, e_mov)
            # rolling shutter: per-band optimum
            kb, eb, wb = [], [], []
            for b in range(a.bands):
                if n_band[b] < 50:
                    kb.append(None); eb.append(None); wb.append(0.0)
                    continue
                kk, ee, edge = parabola_min(ks, e_band[b] / n_band[b])
                kb.append(kk); eb.append(ee); wb.append(float(n_band[b]))
            idx = [b for b in range(a.bands) if kb[b] is not None]
            slope = None
            if len(idx) >= 2:
                x = np.array([centres[b] for b in idx]); y = np.array([kb[b] for b in idx]); wt = np.array([wb[b] for b in idx])
                xm, ym = (wt * x).sum() / wt.sum(), (wt * y).sum() / wt.sum()
                den = (wt * (x - xm) ** 2).sum()
                slope = float((wt * (x - xm) * (y - ym)).sum() / den) if den > 0 else None
            # moving-pixel residual with per-band optimum vs global optimum (interpolated on the grid)
            interp = lambda arr, k: float(np.interp(k, ks, arr))
            e_global = sum(interp(e_band[b], kg) for b in range(a.bands)) / max(n_band.sum(), 1)
            e_perband = sum((interp(e_band[b], kb[b]) if kb[b] is not None else interp(e_band[b], kg)) for b in range(a.bands)) / max(n_band.sum(), 1)
            # exposure: box mean of renders on the grid around k0 (grid point nearest to kg)
            k0 = int(np.argmin([abs(k - kg) for k in ks]))
            for W in a.widths:
                sel = [i for i, k in enumerate(ks) if abs(k - ks[k0]) <= W / 2 + 1e-9]
                tot = 0.0
                for fi in range(len(a.frames)):
                    avg = downs_cache[fi][sel].float().mean(0)
                    e = (avg - lrs[fi]).pow(2).sum(0)
                    tot += float(e[moving[fi]].sum())
                e_exp[W] = tot / max(n_band.sum(), 1)
            per[cam] = dict(dt=dt, ks=ks, e_mov=e_mov.tolist(), e_all=(e_all / len(a.frames)).tolist(), k_global=kg, k_global_at_edge=edge_g,
                            band_centres=centres, band_k=kb, band_moving_px=wb, rs_slope_frames_per_height=slope,
                            rs_gain_db=10 * float(np.log10(max(e_global, 1e-20) / max(e_perband, 1e-20))),
                            exposure_k0=ks[k0], exposure_gain_db={str(W): 10 * float(np.log10(max(e_exp[a.widths[0]], 1e-20) / max(e_exp[W], 1e-20)))
                                                                 for W in a.widths})
            print(json.dumps(dict(model=label, cam=cam, k=round(kg, 3), edge=edge_g, slope=None if slope is None else round(slope, 3),
                                  band_k=[None if v is None else round(v, 2) for v in kb], rs_gain=round(per[cam]['rs_gain_db'], 3),
                                  exp_gain={k: round(v, 3) for k, v in per[cam]['exposure_gain_db'].items()}, s=round(time.time() - started)),
                             ), flush=True)
            del lrs, downs_cache, moving
        train = [c for c in cams if c != 'cam00']
        slopes = [per[c]['rs_slope_frames_per_height'] for c in train if per[c]['rs_slope_frames_per_height'] is not None]
        result['models'][label] = dict(
            per_camera=per, info=m['info'],
            train=dict(rs_slope_median=float(np.median(slopes)) if slopes else None,
                       rs_slope_positive_share=float(np.mean([s > 0 for s in slopes])) if slopes else None,
                       rs_gain_db_median=float(np.median([per[c]['rs_gain_db'] for c in train])),
                       exposure_gain_db_median={str(W): float(np.median([per[c]['exposure_gain_db'][str(W)] for c in train])) for W in a.widths},
                       exposure_gain_db_max={str(W): float(max(per[c]['exposure_gain_db'][str(W)] for c in train)) for W in a.widths}))
        print(json.dumps({label: result['models'][label]['train'], 'cam00': dict(slope=per['cam00']['rs_slope_frames_per_height'],
                          exp=per['cam00']['exposure_gain_db'])}), flush=True)
        del m
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
