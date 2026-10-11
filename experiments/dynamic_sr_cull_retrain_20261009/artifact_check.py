"""Held-out-view artifact check from saved cam00 preview floats (frames 0/150/299), CPU only.

For each run: e = render - HR, per-pixel RGB L2 magnitude, Gaussian blur (sigma in HR pixels), then
  area_frac = fraction of pixels whose blurred error exceeds --thresh (large, smooth, localized errors),
  p999      = 99.9th percentile of the blurred error.
Static view-dependent colour artifacts (seen by some training camera, wrong only from cam00) show up as a
large area_frac; ordinary edge/detail error stays below the threshold after blurring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import rt

FRAMES = (0, 150, 299)


def hr_images(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    obs = {int(o['frame_index']): o for o in m['observations'] if o['camera_id'] == 'cam00'}
    return [np.asarray(Image.open(Path(manifest_path).parent / obs[f]['hr_path']).convert('RGB')).astype(np.float32) / 255 for f in FRAMES]


def score(run_dir, hrs, sigma, thresh):
    vals = []
    for f, hr in zip(FRAMES, hrs):
        p = Path(run_dir) / 'previews' / f'cam00_{f:04d}.npz'
        if not p.exists():
            p = Path(run_dir) / 'floats' / f'cam00_{f:04d}.npz'
        if not p.exists():
            return None
        img = np.load(p)['rgb_raw']
        img = img.transpose(1, 2, 0) if img.shape[0] == 3 else img
        err = np.sqrt(((np.clip(img, 0, 1) - hr) ** 2).sum(-1))
        blur = gaussian_filter(err, sigma)
        vals.append((float((blur > thresh).mean()), float(np.percentile(blur, 99.9))))
    return dict(area_frac=float(np.mean([v[0] for v in vals])), p999=float(np.mean([v[1] for v in vals])),
                per_frame=[dict(frame=f, area_frac=a, p999=q) for f, (a, q) in zip(FRAMES, vals)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sigma', type=float, default=12.0)
    p.add_argument('--thresh', type=float, default=0.12)
    p.add_argument('--out', default=str(rt.ROOT / 'output/dynamic_sr_cull_retrain_20261009/summary/artifact_check.json'))
    a = p.parse_args()
    from tasks import registry
    reg = registry()
    W = rt.ROOT / 'output/dynamic_sr_cull_retrain_20261009/wave1'
    result = {}
    for seed in sorted({k.rsplit('_', 1)[0] for k in reg}):
        e = reg[f'{seed}_Bsync']
        manifest = rt.ROOT / json.loads((rt.ROOT / e['sr'] / 'config.json').read_text())['data']['manifest']['path']
        hrs = hr_images(manifest)
        runs = {'official_Bsync': rt.ROOT / e['eval'], 'posthoc_Bsync': W / 'posthoc' / f'{seed}_Bsync_cull'}
        for d in sorted((W / 'eval').glob(f'{seed}_*')) + sorted((W / 'eval_repeat').glob(f'{seed}_*')):
            runs[d.name[len(seed) + 1:] + ('' if d.parent.name == 'eval' else '')] = d
        result[seed] = {}
        for name, d in runs.items():
            s = score(d, hrs, a.sigma, a.thresh)
            if s is not None:
                result[seed][name] = s
        line = ' '.join(f"{n}:{v['area_frac']*100:.2f}%" for n, v in result[seed].items())
        print(seed, line)
    rt.write_json(a.out, dict(sigma=a.sigma, thresh=a.thresh, frames=FRAMES, results=result))


if __name__ == '__main__':
    main()
