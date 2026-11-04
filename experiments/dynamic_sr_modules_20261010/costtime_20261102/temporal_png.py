"""Temporal-consistency read-out of a scored cam00 render directory (Claude, session 4dsr-a4, 2026-11-02).

User 11-02: the main text needs a temporal-consistency comparison of all methods. This scores the same 8-bit PNG renders that the
shared frozen scorer (score_views_area4.py / score_views_area2.py) scored for the main table, so every method is read out from
identical bytes. Definitions follow perframe_20261021/eval_resid.py, the project's existing temporal read-out (unchanged):
  for each consecutive pair (f - 1, f) of cam00:  d_R = R(f) - R(f - 1),  d_H = HR(f) - HR(f - 1)
  tPSNR      = -10 log10( pooled mean (d_R - d_H)^2 )                    over the whole image (RGB in [0, 1])
  tPSNR_dyn  = the same inside dyn(f) | dyn(f - 1);  dyn = geoprobe_20261017/eval_full.dyn_mask: max_c |HR - B| > TAU dilated by
               DILATE px, B = per-pixel median of HR frames 0, 5, 10, ... (TAU / DILATE read from eval_full.py, checked)
  energy     = mean |d_R| / mean |d_H| (whole image and dynamic region): 1 = as much frame-to-frame change as HR,
               > 1 = flicker, < 1 = temporal smoothing
  tLP        (optional, --lpips) = mean over pairs of | LPIPS(R(f-1), R(f)) - LPIPS(HR(f-1), HR(f)) |  (TecoGAN tLP), AlexNet
               LPIPS v0.1 with the frozen scorer's cached weights (same state-dict SHA256 check); HR-pair values are cached per
               manifest in --hr-cache.
Every render PNG and HR file is SHA256-checked against render_index.json / the manifest. CPU by default.
  temporal_png.py --manifest M.json --render-dir DIR --label NAME --out OUT.json [--lpips --torch-home T --hr-cache C.json]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_FULL = ROOT / 'experiments/dynamic_sr_modules_20261010/geoprobe_20261017/eval_full.py'
PREP = ROOT / 'deployment/dynamic_lr_expansion_20261022/prepare_suite.py'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def eval_full_constants():
    tree = ast.parse(EVAL_FULL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = [t.id for t in node.targets[0].elts if isinstance(t, ast.Name)]
            if names == ['TAU', 'DILATE']:
                return tuple(ast.literal_eval(node.value))
    raise ValueError('TAU, DILATE not found in eval_full.py')


def lpips_state_sha256():
    tree = ast.parse(PREP.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LPIPS_STATE_SHA256' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('LPIPS_STATE_SHA256 not found')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--render-dir', required=True, type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--camera', default='cam00')
    p.add_argument('--lpips', action='store_true')
    p.add_argument('--torch-home', type=Path, default=ROOT / 'weight/static_sr_wu_20261022/evaluation/torch')
    p.add_argument('--hr-cache', type=Path)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(a.threads)
    tau, dilate = eval_full_constants()

    def read_rgb(path):
        with Image.open(path) as im:
            return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0

    def dyn_mask(hr, bg):   # eval_full.dyn_mask, same operations
        d = torch.from_numpy(np.abs(hr - bg).max(axis=2) > tau)[None, None].float()
        k = 2 * dilate + 1
        return (F.max_pool2d(d, k, 1, dilate)[0, 0] > 0).numpy()

    m = json.loads(a.manifest.read_text())
    root = a.manifest.parent
    obs = sorted((o for o in m['observations'] if o['camera_id'] == a.camera), key=lambda o: int(o['frame_index']))
    frames = [int(o['frame_index']) for o in obs]
    if frames != list(range(len(frames))):
        raise ValueError('cam00 frames are not consecutive from 0')
    index = json.loads((a.render_dir / 'render_index.json').read_text())
    if index.get('manifest_sha256') not in (None, sha(a.manifest)):
        raise ValueError('render index belongs to another manifest')
    recs = {int(r['frame']): r for r in index['records'] if r['camera_id'] == a.camera}
    missing = [f for f in frames if f not in recs]
    if missing:
        raise ValueError(f'{len(missing)} frames without a render, e.g. {missing[:5]}')

    hr_paths = [root / o['hr_path'] for o in obs]
    bg = np.median(np.stack([read_rgb(hr_paths[f]) for f in range(0, len(frames), 5)]), axis=0)

    model = None
    hr_cache = {}
    if a.lpips:
        os.environ['TORCH_HOME'] = str(a.torch_home)
        torch.hub.set_dir(str(a.torch_home / 'hub'))
        import lpips
        model = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).eval()
        model.requires_grad_(False)
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        if h.hexdigest() != lpips_state_sha256():
            raise ValueError('LPIPS weights differ from the frozen scorer')
        if a.hr_cache and a.hr_cache.exists():
            c = json.loads(a.hr_cache.read_text())
            if c['manifest_sha256'] == sha(a.manifest) and c['camera'] == a.camera:
                hr_cache = {int(k): v for k, v in c['pairs'].items()}

    def lp(x, y):
        tx = torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1
        ty = torch.from_numpy(y).permute(2, 0, 1)[None] * 2 - 1
        with torch.inference_mode():
            return float(model(tx, ty).item())

    t0 = time.monotonic()
    acc = dict(se=0.0, n=0, se_d=0.0, n_d=0, er=0.0, eh=0.0, er_d=0.0, eh_d=0.0, pairs=0, tlp=0.0)
    prev = None
    new_hr = {}
    for f, o in zip(frames, obs):
        rp = a.render_dir / recs[f]['path']
        if sha(rp) != recs[f]['render_sha256']:
            raise ValueError(f'render changed: {rp}')
        if sha(hr_paths[f]) != o['hr_sha256']:
            raise ValueError(f'HR changed: {hr_paths[f]}')
        r, h = read_rgb(rp), read_rgb(hr_paths[f])
        if r.shape != h.shape:
            raise ValueError('render / HR size differ')
        dm = dyn_mask(h, bg)
        if prev is not None:
            r0, h0, m0 = prev
            dr, dh = r - r0, h - h0
            e = ((dr - dh) ** 2).sum(axis=2)
            mm = dm | m0
            acc['se'] += float(e.sum()); acc['n'] += e.size * 3
            acc['se_d'] += float(e[mm].sum()); acc['n_d'] += int(mm.sum()) * 3
            acc['er'] += float(np.abs(dr).sum()); acc['eh'] += float(np.abs(dh).sum())
            acc['er_d'] += float(np.abs(dr)[mm].sum()); acc['eh_d'] += float(np.abs(dh)[mm].sum())
            acc['pairs'] += 1
            if model is not None:
                lh = hr_cache.get(f)
                if lh is None:
                    lh = new_hr[f] = lp(h0, h)
                acc['tlp'] += abs(lp(r0, r) - lh)
        prev = (r, h, dm)
    ps = lambda se, n: float(-10 * np.log10(se / n))
    res = dict(schema='temporal_png_v1', label=a.label, camera=a.camera, pairs=acc['pairs'],
               tPSNR=ps(acc['se'], acc['n']), tPSNR_dyn=ps(acc['se_d'], acc['n_d']),
               energy=acc['er'] / acc['eh'], energy_dyn=acc['er_d'] / acc['eh_d'],
               dyn_fraction=acc['n_d'] / acc['n'],
               tLP=(acc['tlp'] / acc['pairs']) if model is not None else None,
               manifest=dict(path=str(a.manifest), sha256=sha(a.manifest)), render_dir=str(a.render_dir),
               render_index_sha256=sha(a.render_dir / 'render_index.json'), method=index.get('method'),
               dyn_mask=dict(tau=tau, dilate=dilate, background='median of HR frames 0,5,10,...', eval_full_sha256=sha(EVAL_FULL)),
               script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)), seconds=time.monotonic() - t0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    if model is not None and a.hr_cache and new_hr:
        c = dict(manifest_sha256=sha(a.manifest), camera=a.camera, pairs={**{str(k): v for k, v in hr_cache.items()},
                                                                          **{str(k): v for k, v in new_hr.items()}})
        tmp = a.hr_cache.with_suffix('.tmp')
        tmp.write_text(json.dumps(c))
        tmp.replace(a.hr_cache)
    print(json.dumps({k: res[k] for k in ('label', 'pairs', 'tPSNR', 'tPSNR_dyn', 'energy', 'energy_dyn', 'tLP', 'seconds')}))


if __name__ == '__main__':
    main()
