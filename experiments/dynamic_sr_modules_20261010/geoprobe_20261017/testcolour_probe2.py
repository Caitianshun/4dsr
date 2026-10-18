"""Test-view colour from the training cameras' colour affines, v2 (Claude, session 4dsr-e9, 2026-10-17). Diagnostic; no training.
New file (v1 testcolour_probe.py is byte-frozen on the A100; its HR least squares failed in cusolver for 27M rows): the affine
fits use 4x4 normal equations in float64, and a split oracle is added. cullretrain / ens_eval_view are imported, never modified.

Background: every training camera has a learned 3x4 colour affine (mean over training cameras fixed to the identity); the test
view cam00 is rendered with the identity. camres_probe.py found that a global colour affine fitted on cam00's own LR would
remove 11-26% of cam00's LR residual, and the Cook07 A100 controls differ by up to 0.3 dB in the low band only. Question: can
cam00's colour be predicted from the training cameras alone (camera positions and their affines), as a legal test-view
rendering choice? For each checkpoint and the cam00 frames in --frames, cam00 HR PSNR (clamped render vs HR) is reported for:
  identity              the registered evaluation;
  knn{k}                mean of the centred affines of the k nearest training cameras (camera centres), k in --ks;
  idw                   inverse-square-distance weighted mean of all training affines;
  oracle_lr             least-squares 3x4 affine fitted on cam00's own LR (box-reduced render vs LR; test-LR diagnostic);
  oracle_hr             least-squares 3x4 affine fitted on cam00's HR (upper bound, diagnostic);
  split_hr              fitted on cam00's HR of the even-indexed frames of --frames, scored on the odd-indexed ones (psnr_odd):
                        a colour-gauge-aligned PSNR in the spirit of the NeRF-W test-time appearance protocol (diagnostic metric,
                        removes the test camera's global colour gauge, never a method).
Also: the correlation of each training camera's affine with the mean of its 2 nearest neighbours (leave-one-out), i.e. how
smooth the colour affines are over the rig.

  testcolour_probe2.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--frames ...] [--ks 1 2 3 4]
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


def apply_affine(d, img):
    """cullretrain.CameraAffine.apply with a centred delta d (3 x 4)"""
    return torch.einsum('ij,jhw->ihw', torch.eye(3, device=img.device) + d[:, :3], img) + d[:, 3, None, None]


def fit_affine(x, y):
    """least-squares centred delta d with (I + d[:, :3]) x + d[:, 3] ~ y; x, y: 3 x N (normal equations, float64)"""
    X = torch.cat([x, torch.ones(1, x.shape[1], device=x.device)], 0).double()
    G = X @ X.T                                             # 4 x 4
    B = X @ y.double().T                                    # 4 x 3
    sol = torch.linalg.solve(G + 1e-9 * torch.eye(4, dtype=G.dtype, device=G.device), B).T.float()  # 3 x 4: y = A x + b
    d = sol.clone()
    d[:, :3] -= torch.eye(3, device=x.device)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=list(range(5, 300, 15)))
    p.add_argument('--ks', type=int, nargs='*', default=[1, 2, 3, 4])
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    result = dict(kind='testcolour_probe', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models={})
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        aff = payload.get('affine')
        del payload
        if not aff or 'delta' not in aff:
            raise SystemExit(f'{label}: no colour affine in the checkpoint')
        names = list(aff['names'])
        D = aff['delta'].float()
        D = (D - D.mean(0)).cuda()                          # centred, as applied in training
        manifest = m['manifest']
        root = Path(manifest['_root'])
        centres = {}
        for cam in names + ['cam00']:
            o = next(o for o in manifest['observations'] if o['camera_id'] == cam)
            camera = native.hr_camera(ev.camera_item(manifest, o, torch), int(o['frame_index']), manifest)
            cc = getattr(camera, 'camera_center', None)
            cc = camera.world_view_transform.inverse()[3, :3] if cc is None else cc
            centres[cam] = cc.detach().float().cpu()
        dist = {c: float((centres[c] - centres['cam00']).norm()) for c in names}
        order = sorted(names, key=lambda c: dist[c])
        cands = {'identity': torch.zeros(3, 4, device='cuda')}
        for k in a.ks:
            cands[f'knn{k}'] = torch.stack([D[names.index(c)] for c in order[:k]]).mean(0)
        w = torch.tensor([1 / max(dist[c], 1e-6) ** 2 for c in names], device='cuda')
        cands['idw'] = (w[:, None, None] * D).sum(0) / w.sum()
        # smoothness of the affines over the rig (leave-one-out 2-NN prediction)
        pred_err, var = [], []
        for c in names:
            others = sorted((float((centres[o] - centres[c]).norm()), o) for o in names if o != c)[:2]
            pr = torch.stack([D[names.index(o)] for _, o in others]).mean(0)
            pred_err.append(float((D[names.index(c)] - pr).pow(2).sum()))
            var.append(float(D[names.index(c)].pow(2).sum()))
        smooth = 1 - sum(pred_err) / max(sum(var), 1e-20)
        obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == 'cam00'}
        renders, hrs, downs, lrs = [], [], [], []
        with torch.no_grad():
            for f in a.frames:
                o = obs[f]
                camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                img = native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float()
                hr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['hr_path']))).permute(2, 0, 1).float().cuda()
                lr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['lr_path']))).permute(2, 0, 1).float().cuda()
                fy = img.shape[-2] // lr.shape[-2]
                renders.append(img)
                hrs.append(hr)
                downs.append(F.avg_pool2d(img[None], fy)[0])
                lrs.append(lr)
            cat = lambda xs: torch.stack(xs).permute(1, 0, 2, 3).reshape(3, -1)
            cands['oracle_lr'] = fit_affine(cat(downs), cat(lrs))
            cands['oracle_hr'] = fit_affine(cat(renders), cat(hrs))
            cands['split_hr'] = fit_affine(cat(renders[0::2]), cat(hrs[0::2]))
            psnr, psnr_odd = {}, {}
            for name, d in cands.items():
                vals = []
                for img, hr in zip(renders, hrs):
                    mse = float((apply_affine(d, img).clamp(0, 1) - hr).pow(2).mean())
                    vals.append(-10 * np.log10(max(mse, 1e-12)))
                psnr[name] = float(np.mean(vals))
                psnr_odd[name] = float(np.mean(vals[1::2]))
        base = psnr['identity']
        result['models'][label] = dict(info=m['info'], psnr=psnr, psnr_odd=psnr_odd,
                                       delta_db={k: v - base for k, v in psnr.items()},
                                       delta_db_odd={k: v - psnr_odd['identity'] for k, v in psnr_odd.items()},
                                       affine_smoothness_2nn=smooth, nearest=order[:4], distances={c: dist[c] for c in order},
                                       candidates={k: v.cpu().tolist() for k, v in cands.items()})
        print(json.dumps({label: dict(delta_db={k: round(v - base, 3) for k, v in psnr.items()}, identity=round(base, 3),
                                      odd_identity=round(psnr_odd['identity'], 3), odd_split_hr=round(psnr_odd['split_hr'], 3),
                                      smooth=round(smooth, 3), nearest=order[:4])}), flush=True)
        del m, renders, hrs, downs, lrs
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
