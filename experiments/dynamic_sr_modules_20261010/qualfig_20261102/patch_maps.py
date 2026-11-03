"""Window-level PSNR / SSIM / LPIPS of every method's cam00 renders against HR, for choosing zoom-in crops (Claude 4dsr-ac,
2026-11-02; rules in tmp/claude_qualfig_fps_20261102/prereg_crops.md, written before this ran).

Per frame and method (8-bit PNG renders vs the HR PNG, as the main-table scorer reads them):
  E = per-pixel squared error averaged over RGB; S = SSIM map of the main-table helper (11x11 Gaussian sigma 1.5, reflect-101
  border, C1 = 0.01^2, C2 = 0.03^2, RGB mean), here as a separable torch convolution (checked against the cv2 helper on frame 0
  of every scene); L = official LPIPS-Alex v0.1 spatial map (full-image context) and the scalar LPIPS of the main table.
  Windows: squares of 128 and 192 px, stride 32, at least 16 px from the border; window PSNR from the mean of E, window SSIM and
  LPIPS = means of S and L. Full-frame PSNR / SSIM (5-px border excluded) / scalar LPIPS per frame validate the renders against
  the main table.
HR descriptors per window: mean gradient magnitude of the HR luminance (central differences) and the dynamic-pixel share
(eval_full: background = median of HR frames 0, 5, ...; dynamic if max_rgb |HR - B| > 0.06, dilated by 5 px).
Re-running appends methods not yet in <Scene>_area4.npz (HR descriptors reused); directories whose frames this account cannot read are skipped.
Methods: ours and wu from the spec (main-table PNG directories) plus every renders/<Scene>_area4/<method>/ with a complete
render_index.json. Output: <out>/<Scene>_area4.npz and <Scene>_area4.json.
  patch_maps.py --spec SPEC.json --renders RENDERS_DIR --out OUT_DIR --torch-home TH [--scenes Cook07,...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / 'experiments/dynamic_sr_modules_20261010/area4_baselines_20261024/frozen'
WINS, STRIDE, MARGIN = (128, 192), 32, 16


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_rgb(path, np):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).astype(np.float32) / 255.0


def gauss_kernel(torch):
    x = torch.arange(11, dtype=torch.float64) - 5
    k = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
    return (k / k.sum()).float()


def ssim_map(x, y, k, F):
    """x, y: [1,3,H,W] in [0,1]. Same arithmetic as evaluate.ssim_map_rgb (cv2.GaussianBlur 11x11 sigma 1.5 reflect-101)."""
    def filt(z):
        z = F.pad(z, (5, 5, 5, 5), mode='reflect')
        z = F.conv2d(z, k.view(1, 1, 1, 11).expand(3, 1, 1, 11), groups=3)
        return F.conv2d(z, k.view(1, 1, 11, 1).expand(3, 1, 11, 1), groups=3)
    ux, uy = filt(x), filt(y)
    vx, vy, cov = filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy
    val = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / ((ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2))
    return val.mean(dim=1)[0]


def windows(m, F):
    """m: [H,W] map -> {win: [gy,gx]} window means (origins MARGIN + STRIDE*k)."""
    c = m[MARGIN:m.shape[0] - MARGIN, MARGIN:m.shape[1] - MARGIN][None, None]
    return {w: F.avg_pool2d(c, w, STRIDE)[0, 0] for w in WINS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--renders', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--torch-home', required=True)
    p.add_argument('--scenes', default=None)
    a = p.parse_args()
    os.environ['TORCH_HOME'] = str(ROOT / a.torch_home)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import lpips
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.hub.set_dir(str(ROOT / a.torch_home / 'hub'))
    sys.path.insert(0, str(FROZEN))
    import prepare_suite as PS
    model = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    wh = hashlib.sha256()
    for k_, v in sorted(model.state_dict().items()):
        wh.update(k_.encode())
        wh.update(v.detach().cpu().numpy().tobytes())
    if wh.hexdigest() != PS.LPIPS_STATE_SHA256:
        raise SystemExit('LPIPS weights differ from the main-table scorer')
    import cv2
    k = gauss_kernel(torch).cuda()
    spec = json.loads((ROOT / a.spec).read_text())
    scenes = a.scenes.split(',') if a.scenes else list(spec)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pool = cf.ThreadPoolExecutor(8)
    for scene in scenes:
        s = spec[scene]
        prev_meta = json.loads((out / f'{scene}_area4.json').read_text()) if (out / f'{scene}_area4.json').exists() else None
        prev = dict(np.load(out / f'{scene}_area4.npz')) if prev_meta else {}
        began = time.time()
        mp = ROOT / s['manifest']
        manifest = json.loads(mp.read_text())
        test = sorted([o for o in manifest['observations'] if o['camera_id'] == 'cam00'], key=lambda o: int(o['frame_index']))
        hr_paths = [mp.parent / o['hr_path'] for o in test]
        methods = {'ours': ROOT / s['ours']['renders'], 'wu': ROOT / s['wu']['renders']}
        rd = ROOT / a.renders / f'{scene}_area4'
        if rd.is_dir():
            for d in sorted(rd.iterdir()):
                ri = d / 'render_index.json'
                if not ri.exists():
                    continue
                recs = json.loads(ri.read_text())['records']
                # complete only: our re-renders write local_render.json / render_index.json last; 4dsr-a4's downloads write fetch.json last
                done = (d / 'local_render.json').exists() or (d / 'fetch.json').exists() or d.name == 'sr4d'
                if done and len(recs) == len(test) and all((d / r['path']).exists() for r in recs):
                    if all(os.access(d / r['path'], os.R_OK) for r in (recs[0], recs[len(recs) // 2], recs[-1])):
                        methods[d.name] = d
                    else:  # e.g. symlinks into another job's received/ that this service account cannot read
                        print(json.dumps(dict(scene=scene, unreadable=d.name)), flush=True)
                else:
                    print(json.dumps(dict(scene=scene, incomplete=d.name)), flush=True)
        have = set(prev_meta['methods']) if prev_meta else set()
        names = [m for m in methods if m not in have]
        if not names:
            print(json.dumps(dict(skip=scene, have=sorted(have))), flush=True)
            continue
        need_hr = prev_meta is None
        # HR descriptors (first pass of a scene only; appended methods reuse them)
        bg = np.median(np.stack([read_rgb(hr_paths[f], np) for f in range(0, len(test), 5)]), axis=0) if need_hr else None
        bg_t = torch.from_numpy(bg).cuda() if need_hr else None
        arrays, full = {}, {m: [] for m in names}
        grad_w, dyn_w = {w: [] for w in WINS}, {w: [] for w in WINS}
        ssim_check = None
        nxt = pool.submit(lambda f: [read_rgb(hr_paths[f], np)] + [read_rgb(methods[m] / f'cam00/{f:04d}.png', np) for m in names], 0)
        with torch.no_grad():
            for f in range(len(test)):
                imgs = nxt.result()
                if f + 1 < len(test):
                    nxt = pool.submit(lambda f: [read_rgb(hr_paths[f], np)] + [read_rgb(methods[m] / f'cam00/{f:04d}.png', np) for m in names], f + 1)
                gt = torch.from_numpy(imgs[0]).cuda()
                if need_hr:
                    lum = gt.mean(dim=2)
                    gx = torch.zeros_like(lum)
                    gy = torch.zeros_like(lum)
                    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) / 2
                    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) / 2
                    dyn = ((gt - bg_t).abs().amax(dim=2) > 0.06).float()[None, None]
                    dyn = (F.max_pool2d(dyn, 11, 1, 5)[0, 0] > 0).float()
                    for w, v in windows(torch.sqrt(gx ** 2 + gy ** 2), F).items():
                        grad_w[w].append(v.cpu().numpy())
                    for w, v in windows(dyn, F).items():
                        dyn_w[w].append(v.cpu().numpy())
                ty = gt.permute(2, 0, 1)[None]
                for m, arr in zip(names, imgs[1:]):
                    pr = torch.from_numpy(arr).cuda()
                    tx = pr.permute(2, 0, 1)[None]
                    e = ((pr - gt) ** 2).mean(dim=2)
                    sm = ssim_map(tx, ty, k, F)
                    if f == 0 and m == names[0]:
                        filt = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
                        x_, y_ = arr, imgs[0]
                        ux, uy = filt(x_), filt(y_)
                        vx, vy, cov = filt(x_ * x_) - ux * ux, filt(y_ * y_) - uy * uy, filt(x_ * y_) - ux * uy
                        ref = (((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / ((ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2))).mean(axis=2)
                        ssim_check = float(np.abs(ref - sm.cpu().numpy()).max())
                    model.spatial = False
                    lp = float(model(tx * 2 - 1, ty * 2 - 1).item())
                    model.spatial = True
                    lmap = model(tx * 2 - 1, ty * 2 - 1)[0, 0]
                    model.spatial = False
                    mse = float(e.mean().item())
                    full[m].append([-10 * math.log10(max(mse, 1e-12)), float(sm[5:-5, 5:-5].mean().item()), lp])
                    for w, v in windows(e, F).items():
                        arrays.setdefault(f'{m}__mse_{w}', []).append(v.cpu().numpy().astype(np.float32))
                    for w, v in windows(sm, F).items():
                        arrays.setdefault(f'{m}__ssim_{w}', []).append(v.cpu().numpy().astype(np.float16))
                    for w, v in windows(lmap, F).items():
                        arrays.setdefault(f'{m}__lpips_{w}', []).append(v.cpu().numpy().astype(np.float16))
        save = dict(prev)
        save.update({k_: np.stack(v) for k_, v in arrays.items()})
        if need_hr:
            for w in WINS:
                save[f'hr__grad_{w}'] = np.stack(grad_w[w]).astype(np.float32)
                save[f'hr__dyn_{w}'] = np.stack(dyn_w[w]).astype(np.float16)
        for m in names:
            save[f'{m}__full'] = np.asarray(full[m], dtype=np.float64)
        tmpf = out / f'{scene}_area4.tmp.npz'
        np.savez_compressed(tmpf, **save)
        tmpf.replace(out / f'{scene}_area4.npz')
        H, W = imgs[0].shape[:2]
        run = dict(methods=names, ssim_torch_vs_cv2_max_abs_frame0=dict(method=names[0], value=ssim_check), gpu=torch.cuda.get_device_name(),
                   seconds=time.time() - began, script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)))
        meta = prev_meta or dict(scene=scene, manifest=s['manifest'], hr_size=[W, H], windows=list(WINS), stride=STRIDE, margin=MARGIN,
                                 origins={str(w): dict(x=list(range(MARGIN, W - MARGIN - w + 1, STRIDE)), y=list(range(MARGIN, H - MARGIN - w + 1, STRIDE))) for w in WINS},
                                 methods={}, full_mean={}, lpips_state_sha256=wh.hexdigest(), runs=[])
        meta['methods'].update({m: str(methods[m].relative_to(ROOT)) for m in names})
        meta['full_mean'].update({m: dict(zip(('psnr', 'ssim', 'lpips'), np.asarray(full[m]).mean(axis=0).tolist())) for m in names})
        meta.setdefault('runs', []).append(run)
        meta['seconds'] = run['seconds']
        (out / f'{scene}_area4.json').write_text(json.dumps(meta, indent=1) + '\n')
        print(json.dumps(dict(scene=scene, methods=names, seconds=round(run['seconds']), ssim_check=ssim_check,
                              full={m: [round(x, 3) for x in meta['full_mean'][m].values()] for m in names})), flush=True)


if __name__ == '__main__':
    main()
