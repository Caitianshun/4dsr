"""tOF and tLP of every method on one scene's cam00 test video (Claude 4dsr-41, 2026-11-06; companion of temporal_png.py).

User 11-06: express temporal consistency with the metrics used in the literature. Same renders, HR frames and dynamic mask as
temporal_png.py (the 11-03 temporal table); every render PNG and HR file is SHA256-checked against render_index.json / the manifest.
Definitions follow the official TecoGAN metrics.py (Chu et al., TOG 2020):
  flow   = cv2.calcOpticalFlowFarneback(grey(f - 1), grey(f), None, 0.5, 3, 15, 3, 5, 1.2, 0), grey = cv2.COLOR_RGB2GRAY of the uint8 frame
  tOF    = mean over pairs of mean_p || flow_HR(p) - flow_R(p) ||_2              (whole image; tOF_dyn: p in dyn(f) | dyn(f - 1))
  tLP    = mean over pairs of | LPIPS(R(f-1), R(f)) - LPIPS(HR(f-1), HR(f)) |    (AlexNet LPIPS v0.1, the frozen scorer's weights)
dyn = temporal_png.py's mask (max_c |HR - B| > TAU, dilated by DILATE px, B = per-pixel median of HR frames 0, 5, 10, ...).
Render directories are those recorded in tmp/claude_cvpr_gap_20261102/temporal/<Scene>_area4/<method>.json. CPU only.
  temporal_tof_tlp.py --scene Cook07 --out OUT.json [--threads 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LONG = dict(Cook07='cook_spinach', Cut07='cut_roasted_beef', Flame07='flame_steak', Coffee07='coffee_martini',
            Discussion07='meetroom_discussion', VRHeadset07='meetroom_vrheadset')
METHODS = ['wu', 'yang', '4c4d', 'multi4d', 'localdygs', 'cem', 'sr4d', 'splatsure', 'iesrgs', 'sequence_matters', 'threedsr', 'ours']
TEMPORAL = ROOT / 'tmp/claude_cvpr_gap_20261102/temporal'
FARNEBACK = (None, 0.5, 3, 15, 3, 5, 1.2, 0)


def load_temporal_png():
    spec = importlib.util.spec_from_file_location('temporal_png', Path(__file__).resolve().parent / 'temporal_png.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, choices=sorted(LONG))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='smoke test: first N frames only (result marked partial)')
    p.add_argument('--torch-home', type=Path, default=ROOT / 'weight/static_sr_wu_20261022/evaluation/torch')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    tp = load_temporal_png()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    tau, dilate = tp.eval_full_constants()

    manifest = ROOT / 'data/dynamic_sr/full_time_20261007/prepared' / f'{LONG[a.scene]}_area4' / 'manifest_train_ready_seed20261007.json'
    m = json.loads(manifest.read_text())
    obs = sorted((o for o in m['observations'] if o['camera_id'] == 'cam00'), key=lambda o: int(o['frame_index']))
    frames = [int(o['frame_index']) for o in obs]
    if frames != list(range(300)):
        raise ValueError('cam00 frames are not 0..299')
    hr_paths = [manifest.parent / o['hr_path'] for o in obs]

    dirs, recs = {}, {}
    for meth in METHODS:
        t = json.loads((TEMPORAL / f'{a.scene}_area4' / f'{meth}.json').read_text())
        d = ROOT / t['render_dir']
        if tp.sha(d / 'render_index.json') != t['render_index_sha256']:
            raise ValueError(f'render index changed since the 11-03 temporal table: {d}')
        idx = json.loads((d / 'render_index.json').read_text())
        recs[meth] = {int(r['frame']): r for r in idx['records'] if r['camera_id'] == 'cam00'}
        if sorted(recs[meth]) != frames:
            raise ValueError(f'{meth}: cam00 frames incomplete')
        dirs[meth] = d

    os.environ['TORCH_HOME'] = str(a.torch_home)
    torch.hub.set_dir(str(a.torch_home / 'hub'))
    import lpips
    model = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).eval()
    model.requires_grad_(False)
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    if h.hexdigest() != tp.lpips_state_sha256():
        raise ValueError('LPIPS weights differ from the frozen scorer')

    def read_u8(path, expect):
        if tp.sha(path) != expect:
            raise ValueError(f'file changed: {path}')
        with Image.open(path) as im:
            return np.asarray(im.convert('RGB'), dtype=np.uint8)

    def lp(x, y):
        tx = torch.from_numpy(x.astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1
        ty = torch.from_numpy(y.astype(np.float32) / 255).permute(2, 0, 1)[None] * 2 - 1
        with torch.inference_mode():
            return float(model(tx, ty).item())

    def dyn_mask(hr, bg):
        d = torch.from_numpy(np.abs(hr.astype(np.float32) / 255 - bg).max(axis=2) > tau)[None, None].float()
        return (F.max_pool2d(d, 2 * dilate + 1, 1, dilate)[0, 0] > 0).numpy()

    def flow(g0, g1):
        return cv2.calcOpticalFlowFarneback(g0, g1, *FARNEBACK)

    bg = np.median(np.stack([read_u8(hr_paths[f], obs[f]['hr_sha256']).astype(np.float32) / 255 for f in range(0, 300, 5)]), axis=0)
    t0 = time.monotonic()
    acc = {meth: dict(tof=0.0, tof_dyn=0.0, tlp=0.0) for meth in METHODS}
    prev = None
    pairs = 0
    for f in (frames[:a.limit] if a.limit else frames):
        hr = read_u8(hr_paths[f], obs[f]['hr_sha256'])
        cur = dict(hr=hr, hr_grey=cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY), dyn=dyn_mask(hr, bg), r={}, r_grey={})
        for meth in METHODS:
            r = read_u8(dirs[meth] / recs[meth][f]['path'], recs[meth][f]['render_sha256'])
            if r.shape != hr.shape:
                raise ValueError(f'{meth}: render / HR size differ')
            cur['r'][meth], cur['r_grey'][meth] = r, cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            fh = flow(prev['hr_grey'], cur['hr_grey'])
            lh = lp(prev['hr'], cur['hr'])
            mm = prev['dyn'] | cur['dyn']
            for meth in METHODS:
                diff = np.sqrt(((fh - flow(prev['r_grey'][meth], cur['r_grey'][meth])) ** 2).sum(axis=-1))
                acc[meth]['tof'] += float(diff.mean())
                acc[meth]['tof_dyn'] += float(diff[mm].mean())
                acc[meth]['tlp'] += abs(lp(prev['r'][meth], cur['r'][meth]) - lh)
            pairs += 1
        prev = cur
    res = dict(schema='temporal_tof_tlp_v1', scene=a.scene, camera='cam00', pairs=pairs, partial=bool(a.limit),
               methods={meth: dict(tOF=v['tof'] / pairs, tOF_dyn=v['tof_dyn'] / pairs, tLP=v['tlp'] / pairs, tLP100=100 * v['tlp'] / pairs,
                                   render_dir=str(dirs[meth].relative_to(ROOT))) for meth, v in acc.items()},
               farneback=dict(args=list(FARNEBACK[1:]), grey='cv2.COLOR_RGB2GRAY of uint8 RGB', source='TecoGAN metrics.py'),
               lpips='AlexNet v0.1, frozen scorer weights (state SHA256 checked)',
               dyn_mask=dict(tau=tau, dilate=dilate, background='median of HR frames 0,5,10,...'),
               manifest=dict(path=str(manifest.relative_to(ROOT)), sha256=tp.sha(manifest)),
               script=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=tp.sha(__file__)), seconds=time.monotonic() - t0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({meth: {k: round(v[k], 4) for k in ('tOF', 'tOF_dyn', 'tLP100')} for meth, v in res['methods'].items()}))


if __name__ == '__main__':
    main()
