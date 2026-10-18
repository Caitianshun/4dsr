"""Per-camera residual structure of trained models on their training views (Claude, session 4dsr-e9, 2026-10-17).
Diagnostic only; reads training LR (never training HR). New file; cullretrain / ens_eval_view are imported, never modified.

Question (user, 10-17 ~11:20): does every camera still carry its own error (a per-camera "error code" a model could learn)?
For each checkpoint and each training camera, the model is rendered at frames --frames (HR, post-hoc cull as in evaluation),
the checkpoint's own per-camera colour affine is applied, the render is reduced by the exact 4x4 box (area4 protocol) and
compared with the camera's LR image: r_t = box(A_c(render)) - LR. Per camera:
  rms                total residual rms over frames;
  persist_frac       energy of the frame-mean residual (persistent, camera-specific pattern) / total energy;
  persist_lf_frac    energy of the frame-mean residual after a Gaussian low-pass (sigma --sigma LR px) / total energy:
                     the share a smooth per-camera colour field could remove at most;
  lf_gain_db         -10 log10(1 - persist_lf_frac): the corresponding train-view LR-closure PSNR gain bound;
  affine_refit_frac  share of the residual a fresh per-camera 3x4 colour affine (least squares, fitted on the frames)
                     would still remove (should be ~0 for a converged affine).
cam00 (test view, identity colour) is reported the same way, from its LR, as a diagnostic.

  camres_probe.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--frames 0 50 100 150 200 250 299] [--sigma 6]
"""
from __future__ import annotations

import argparse
import json
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


def affine_of(payload):
    a = payload.get('affine')
    if not a or 'delta' not in a:
        return None
    d = a['delta'].float()
    return dict(names=list(a['names']), d=d - d.mean(0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=[0, 50, 100, 150, 200, 250, 299])
    p.add_argument('--sigma', type=float, default=6.0)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    result = dict(kind='camres_probe', args=vars(a), sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models={})
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        aff = affine_of(payload)
        del payload
        manifest = m['manifest']
        root = Path(manifest['_root'])
        cams = list(manifest['splits']['train']) + ['cam00']
        per = {}
        for cam in cams:
            obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == cam}
            rs, renders, lrs = [], [], []
            for f in a.frames:
                o = obs[f]
                with torch.no_grad():
                    camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                    st = native.native_effective_state(m['g'], float(o['time']), torch)
                    img = native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float()
                    if aff is not None and cam in aff['names']:
                        d = aff['d'][aff['names'].index(cam)].to(img.device)
                        img = torch.einsum('ij,jhw->ihw', torch.eye(3, device=img.device) + d[:, :3], img) + d[:, 3, None, None]
                    lr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['lr_path']))).permute(2, 0, 1).float().to(img.device)
                    fy = img.shape[-2] // lr.shape[-2]
                    down = F.avg_pool2d(img[None], fy)[0].clamp(0, 1)
                rs.append(down - lr)
                renders.append(down)
                lrs.append(lr)
            r = torch.stack(rs)                      # T, 3, h, w
            tot = float(r.pow(2).mean())
            pm = r.mean(0)
            lf = blur(pm, a.sigma)
            x = torch.stack(renders).permute(1, 0, 2, 3).reshape(3, -1)
            y = torch.stack(lrs).permute(1, 0, 2, 3).reshape(3, -1)
            X = torch.cat([x, torch.ones(1, x.shape[1], device=x.device)], 0)        # 4 x N
            sol = torch.linalg.lstsq(X.T.double(), y.T.double()).solution.float()    # 4 x 3
            refit = (X.T @ sol).T - y
            frac_lf = float(lf.pow(2).mean()) / max(tot, 1e-20)
            per[cam] = dict(rms=tot ** 0.5, closure_psnr=-10 * float(np.log10(max(tot, 1e-20))),
                            persist_frac=float(pm.pow(2).mean()) / max(tot, 1e-20), persist_lf_frac=frac_lf,
                            lf_gain_db=-10 * float(np.log10(max(1 - frac_lf, 1e-12))),
                            affine_refit_frac=1 - float(refit.pow(2).mean()) / max(tot, 1e-20),
                            persist_lf_rms_per_channel=[float(v) for v in lf.pow(2).mean((1, 2)).sqrt()])
        train = [c for c in cams if c != 'cam00']
        med = lambda k: float(np.median([per[c][k] for c in train]))
        result['models'][label] = dict(per_camera=per, info=m['info'], train_median={k: med(k) for k in (
            'rms', 'closure_psnr', 'persist_frac', 'persist_lf_frac', 'lf_gain_db', 'affine_refit_frac')},
            train_max_lf_gain_db=max(per[c]['lf_gain_db'] for c in train))
        print(json.dumps({label: result['models'][label]['train_median'], 'cam00': {k: per['cam00'][k] for k in ('persist_frac', 'persist_lf_frac', 'lf_gain_db')}}), flush=True)
        del m
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
