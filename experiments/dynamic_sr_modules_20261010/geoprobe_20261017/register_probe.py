"""Sub-pixel registration probe of trained 4DGS models (Claude, session 4dsr-e9, 2026-10-17). New file; cullretrain is imported,
never modified. Diagnostic only (reads HR of training views).

Question: the idealised multi-view simulation (tmp/claude_beyond_sr_20261017) shows that the pixel-area protocol's multi-view
headroom collapses once registration errors reach ~0.4 HR px. How large is the residual misregistration of our trained models on
their own training views, and is it per-camera systematic (calibration: principal point / rotation / distortion) or local
(geometry)? For every model, training camera (+ cam00) and frame, the raw render (no colour affine) is compared with the HR image
by a Hann-weighted Lucas-Kanade shift estimate (gain/offset terms, 3 warping iterations) on luma patches that are static
across the probed frames and textured. CPU check (tmp/claude_beyond_sr_20261017/scripts/test_register_cpu3.py): unblurred
shifts recovered to ~0.01-0.02 px median abs error; with a render-like Gaussian blur (sigma 1.2) on the moving image the
magnitude is under-estimated by ~13% (conservative). cv2.phaseCorrelate failed the same check (sign errors), so it is not used.

  register_probe.py --manifest M --model LABEL=CKPT [...] --frames 0 100 200 --out DIR [--patch 96 --stride 48]
Writes DIR/patches_<label>.npz (camera, frame, x, y, dx, dy, response = LK fit R^2, texture, static_frac) and DIR/summary.json.
Shift convention (checked by the built-in self-test): (dx, dy) > 0 means the render content sits right/below the HR content.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'
if str(CULL) not in sys.path:
    sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

native, ev, rt = cr.native, cr.ev, cr.rt


def read_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255


def luma(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)


def fft_shift(img, dx, dy):
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fx * dx + fy * dy))))


def patch_grid(h, w, p, s, m=8):
    return [(x, y) for y in range(m, h - p - m + 1, s) for x in range(m, w - p - m + 1, s)]


MARGIN = 8


def lk(ref, mov, x, y, p, win, iters=3, m=MARGIN):
    """Displacement d of mov relative to ref (mov(x) ~ a*ref(x-d)+b) on the patch [y:y+p, x:x+p]; returns (dx, dy, fit R^2)."""
    d = np.zeros(2)
    r0 = ref[y - m:y + p + m, x - m:x + p + m].astype(np.float32)
    gx = (cv2.Sobel(r0, cv2.CV_32F, 1, 0, ksize=3) / 8)
    gy = (cv2.Sobel(r0, cv2.CV_32F, 0, 1, ksize=3) / 8)
    mv = mov[y:y + p, x:x + p].reshape(-1)
    yy, xx = np.mgrid[0:p, 0:p].astype(np.float32)
    wv = win.reshape(-1)
    r2 = 0.0
    for _ in range(iters):
        mapx = (xx + m - d[0]).astype(np.float32)
        mapy = (yy + m - d[1]).astype(np.float32)
        rw = cv2.remap(r0, mapx, mapy, cv2.INTER_CUBIC).astype(np.float64).reshape(-1)
        gxw = cv2.remap(gx, mapx, mapy, cv2.INTER_CUBIC).astype(np.float64).reshape(-1)
        gyw = cv2.remap(gy, mapx, mapy, cv2.INTER_CUBIC).astype(np.float64).reshape(-1)
        A = np.stack([rw, np.ones_like(rw), -gxw, -gyw], -1)
        sol, *_ = np.linalg.lstsq(A * wv[:, None], mv * wv, rcond=None)
        a = sol[0]
        d = d + sol[2:] / (a if abs(a) > 1e-6 else 1e-6)
        res = (mv - A @ sol) * wv
        tot = (mv - np.average(mv, weights=wv)) * wv
        r2 = 1 - float(res @ res) / max(float(tot @ tot), 1e-12)
    return d[0], d[1], r2


def correlate(ref, mov, grid, p, window):
    return np.asarray([lk(ref, mov, x, y, p, window) for x, y in grid], np.float64)


def self_test(gray, grid, p, window):
    """Shift an HR luma image by a known sub-pixel amount and check sign and accuracy of the estimator."""
    rows = []
    for dx, dy in ((0.3, 0.0), (0.0, -0.2), (0.15, 0.1)):
        moved = fft_shift(gray, dx, dy)
        est = correlate(gray, moved, grid, p, window)
        good = est[:, 2] > 0.5
        rows.append(dict(true=[dx, dy], median_est=[float(np.median(est[good, 0])), float(np.median(est[good, 1]))],
                         mad=[float(np.median(np.abs(est[good, 0] - dx))), float(np.median(np.abs(est[good, 1] - dy)))],
                         n=int(good.sum())))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', action='append', required=True, help='LABEL=checkpoint (repeatable)')
    p.add_argument('--frames', type=int, nargs='+', default=[0, 100, 200])
    p.add_argument('--patch', type=int, default=96)
    p.add_argument('--stride', type=int, default=48)
    p.add_argument('--static-tol', type=float, default=4 / 255)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    manifest_path = ROOT / a.manifest
    manifest = cr.load_manifest(manifest_path)
    root = Path(manifest['_root'])
    cams = list(manifest['splits']['train']) + ['cam00']
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations'] if o['split'] in ('train', 'test')}
    w, h = manifest['resolutions']['hr']
    P = a.patch
    window = cv2.createHanningWindow((P, P), cv2.CV_64F)
    grid = patch_grid(h, w, P, a.stride)
    # ---- HR luma, static masks and textures per camera (independent of the model)
    gt, meta = {}, {}
    for c in cams:
        frames = [read_rgb(native.safe_child(root, obs[(c, f)]['hr_path'])) for f in a.frames]
        grays = [luma(x) for x in frames]
        med = np.median(np.stack(grays), 0)
        dev = np.max(np.abs(np.stack(grays) - med[None]), 0)
        static = cv2.blur((dev < a.static_tol).astype(np.float32), (5, 5)) > 0.99
        gx = cv2.Sobel(med, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(med, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        sf = np.array([static[y:y + P, x:x + P].mean() for x, y in grid])
        tex = np.array([grad[y:y + P, x:x + P].mean() for x, y in grid])
        gt[c] = grays
        meta[c] = dict(static_frac=sf, texture=tex)
    test = self_test(gt[cams[0]][0], grid, P, window)
    summary = dict(kind='register_probe', args=vars(a), manifest=cr.entry(manifest_path), cameras=cams, frames=a.frames,
                   patch=P, stride=a.stride, grid=len(grid), self_test=test, host=__import__('socket').gethostname(),
                   gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                   sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models={})
    rt.write_json(out / 'summary.json', summary)
    for spec in a.model:
        label, ck = spec.split('=', 1)
        ck = ROOT / ck
        plan = torch.load(ck, map_location='cpu', weights_only=False)['plan']
        g, _ = cr.load_for_render(ck, plan)
        h_, opt, pipe, sh = cr.configuration(plan)
        background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else \
            torch.zeros(3, dtype=torch.float32, device='cuda')
        rec = {k: [] for k in ('camera', 'frame', 'x', 'y', 'dx', 'dy', 'response', 'texture', 'static_frac', 'psnr')}
        for ci, c in enumerate(cams):
            for fi, f in enumerate(a.frames):
                o = obs[(c, f)]
                with torch.no_grad():
                    cam = native.hr_camera(ev.camera_item(manifest, o, torch), 0, manifest)
                    state = native.native_effective_state(g, float(o['time']), torch)
                    raw = native.native_rgb(cam, g, state, pipe, background)['render'].detach().clamp(0, 1)
                pred = luma(raw.permute(1, 2, 0).cpu().numpy())
                est = correlate(gt[c][fi], pred, grid, P, window)
                mse = float(np.mean((pred - gt[c][fi]) ** 2))
                n = len(grid)
                rec['camera'] += [ci] * n
                rec['frame'] += [f] * n
                rec['x'] += [x for x, _ in grid]
                rec['y'] += [y for _, y in grid]
                rec['dx'] += est[:, 0].tolist()
                rec['dy'] += est[:, 1].tolist()
                rec['response'] += est[:, 2].tolist()
                rec['texture'] += meta[c]['texture'].tolist()
                rec['static_frac'] += meta[c]['static_frac'].tolist()
                rec['psnr'] += [-10 * np.log10(max(mse, 1e-12))] * n
        arr = {k: np.asarray(v) for k, v in rec.items()}
        np.savez_compressed(out / f'patches_{label}.npz', cameras=np.asarray(cams), **arr)
        keep = (arr['static_frac'] >= 0.95) & (arr['response'] > 0.5) & (arr['texture'] >= np.quantile(arr['texture'], 0.5)) \
            & (np.hypot(arr['dx'], arr['dy']) < 3)
        per_cam = {}
        for ci, c in enumerate(cams):
            m = keep & (arr['camera'] == ci)
            if m.sum() < 5:
                per_cam[c] = dict(n=int(m.sum()))
                continue
            dx, dy = arr['dx'][m], arr['dy'][m]
            mx, my = float(np.median(dx)), float(np.median(dy))
            per_cam[c] = dict(n=int(m.sum()), median=[mx, my], median_norm=float(np.hypot(mx, my)),
                              local_mad=[float(np.median(np.abs(dx - mx))), float(np.median(np.abs(dy - my)))],
                              rms_norm=float(np.sqrt(np.mean(dx ** 2 + dy ** 2))))
        train = [c for c in cams if c != 'cam00']
        meds = np.array([per_cam[c]['median'] for c in train if 'median' in per_cam[c]])
        summary['models'][label] = dict(checkpoint=cr.entry(ck), points=int(g.get_xyz.shape[0]), kept_patches=int(keep.sum()),
                                        per_camera=per_cam,
                                        train_systematic_rms_per_axis=float(np.sqrt(np.mean(meds ** 2))) if len(meds) else None,
                                        train_local_mad_median=float(np.median([per_cam[c]['local_mad'] for c in train if 'local_mad' in per_cam[c]]))
                                        if len(meds) else None)
        rt.write_json(out / 'summary.json', summary)
        del g
        torch.cuda.empty_cache()
        print(json.dumps(dict(model=label, kept=int(keep.sum()), sys_rms=summary['models'][label]['train_systematic_rms_per_axis'],
                              local_mad=summary['models'][label]['train_local_mad_median'],
                              cam00=per_cam.get('cam00'))), flush=True)
    summary['seconds'] = time.time() - started
    rt.write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
