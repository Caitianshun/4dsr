"""cam00 dynamic / static region PSNR of trained checkpoints (Claude, session 4dsr-e9, 2026-10-17). Evaluation only.
New file; cullretrain / ens_eval_view are imported, never modified. Same render path as group_eval_bands.py (post-hoc cull,
TF32 off, identity colour for the test view).

Motivation: the per-camera time-offset module acts on moving content only, a few percent of the pixels, so its effect is
diluted in whole-image PSNR. The region masks come from the test view's HR frames (evaluation-side, the same for every model):
background B = per-pixel median of the HR frames 0, 5, 10, ...; a pixel of frame t is dynamic when max_rgb |HR_t - B| > --tau,
dilated by --dilate px; the rest is static. Reported per model (pooled MSE over the frames): PSNR full / dynamic / static and
the dynamic pixel share. per_frame.csv keeps the frame rows.

  region_eval.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--view cam00] [--frames ...] [--tau 0.06] [--dilate 5]
"""
from __future__ import annotations

import argparse
import csv
import json
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--frames', type=int, nargs='*', default=list(range(0, 300, 3)))
    p.add_argument('--tau', type=float, default=0.06)
    p.add_argument('--dilate', type=int, default=5)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed evaluation preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    specs = [c.split('=', 1) for c in a.checkpoint]
    models = [ens_eval.load_model(lab, path, True) for lab, path in specs]
    for m in models:
        m['obs'] = ens_eval.view_observations(m['manifest'], a.view, m['label'])
    ref = models[0]
    for m in models[1:]:
        if [o['hr_sha256'] for o in m['obs']] != [o['hr_sha256'] for o in ref['obs']]:
            raise ValueError('HR identity differs between manifests: ' + m['label'])
    root = Path(ref['manifest']['_root'])
    read = lambda o: torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['hr_path']))).permute(2, 0, 1).float()
    bg = torch.stack([read(ref['obs'][f]) for f in range(0, 300, 5)]).median(0).values.cuda()
    acc = {m['label']: dict(full=0.0, dyn=0.0, sta=0.0) for m in models}
    n_full = n_dyn = n_sta = 0
    rows = []
    k = 2 * a.dilate + 1
    for i, f in enumerate(a.frames):
        o0 = ref['obs'][f]
        hr = read(o0).cuda()
        dyn = (hr - bg).abs().amax(0) > a.tau
        dyn = F.max_pool2d(dyn.float()[None, None], k, 1, a.dilate)[0, 0] > 0
        sta = ~dyn
        nd, ns = int(dyn.sum()), int(sta.sum())
        n_dyn += 3 * nd
        n_sta += 3 * ns
        n_full += hr.numel()
        row = dict(frame=f, dyn_share=nd / dyn.numel())
        with torch.no_grad():
            for m in models:
                o = m['obs'][f]
                cam = native.hr_camera(ev.camera_item(m['manifest'], o, torch), f, m['manifest'])
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                img = native.native_rgb(cam, m['g'], st, m['pipe'], m['background'])['render'].float().clamp(0, 1)
                e = (img - hr).pow(2)
                s_full, s_dyn = float(e.sum()), float(e[:, dyn].sum())
                acc[m['label']]['full'] += s_full
                acc[m['label']]['dyn'] += s_dyn
                acc[m['label']]['sta'] += s_full - s_dyn
                row[f"{m['label']}_dyn_psnr"] = -10 * np.log10(max(s_dyn / max(3 * nd, 1), 1e-12))
                row[f"{m['label']}_psnr"] = -10 * np.log10(max(s_full / hr.numel(), 1e-12))
        rows.append(row)
        if i % 20 == 0:
            print(json.dumps(dict(frame=f, dyn_share=round(row['dyn_share'], 4), seconds=round(time.time() - started, 1))), flush=True)
    summary = {}
    for lab, q in acc.items():
        summary[lab] = dict(psnr_full=-10 * np.log10(q['full'] / n_full), psnr_dyn=-10 * np.log10(q['dyn'] / max(n_dyn, 1)),
                            psnr_static=-10 * np.log10(q['sta'] / max(n_sta, 1)))
    record = dict(kind='region_eval', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models=[m['info'] for m in models],
                  dyn_pixel_share=n_dyn / n_full, frames=len(a.frames), summary=summary, seconds=time.time() - started)
    with (out / 'per_frame.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(record, indent=1))
    print(json.dumps({k: {q: round(v, 3) for q, v in s.items()} for k, s in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
