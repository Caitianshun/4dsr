"""Test-view colour by leave-one-out cross-validation over the training cameras (Claude, session 4dsr-e9, 2026-10-17).
Diagnostic for a legal, zero-training test-view colour rule. New file; cullretrain / ens_eval_view are imported, never modified.

testcolour_probe2.py found that the mean colour affine of the 4 nearest training cameras (knn4) raises cam00 PSNR by +0.06 to
+0.2 dB on Cook07, Cut07 and Discussion07 for every model, while knn1-3 are unstable; k = 4 was picked after seeing cam00, so it
is not a method. Here the rule is chosen without the test view: every training camera c is predicted from the other training
cameras only (the centred 3x4 colour affines D_j learned in training; camera centres from the manifest), and the prediction is
scored on c's own training LR (render at HR with c's calibsr warp when the checkpoint has one, predicted affine, exact 4x4 box,
vs LR) over --frames. Predictors: mean of the others (the registered gauge, leave-one-out), knn{k} for k in --ks, idw (inverse
square distance over all others). The predictor with the best mean LOO PSNR is then applied to cam00 (test HR, all --test-frames)
and compared with the identity (registered evaluation). The camera's own affine is reported as the LOO upper bound.

  testcolour_probe3.py --checkpoint LABEL=rel/ck.pt [...] --out rel/dir [--frames ...] [--test-frames ...] [--ks 1 2 3 4 5 6 8]
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


def apply_affine(d, img):
    return torch.einsum('ij,jhw->ihw', torch.eye(3, device=img.device) + d[:, :3], img) + d[:, 3, None, None]


def warp_fn(aff):
    """calibsr_v2 CalibAffine.warp_image with the checkpoint's warp (identity without one)"""
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


def centre_of(manifest, cam):
    o = next(o for o in manifest['observations'] if o['camera_id'] == cam)
    camera = native.hr_camera(ev.camera_item(manifest, o, torch), int(o['frame_index']), manifest)
    cc = getattr(camera, 'camera_center', None)
    cc = camera.world_view_transform.inverse()[3, :3] if cc is None else cc
    return cc.detach().float().cpu()


def predictors(D, names, centres, target, ks):
    """predicted centred affine for camera `target` from the cameras in `names` other than target"""
    others = [c for c in names if c != target]
    dist = {c: float((centres[c] - centres[target]).norm()) for c in others}
    order = sorted(others, key=lambda c: dist[c])
    P = {'mean_others': torch.stack([D[c] for c in others]).mean(0)}
    for k in ks:
        P[f'knn{k}'] = torch.stack([D[c] for c in order[:k]]).mean(0)
    w = torch.tensor([1 / max(dist[c], 1e-6) ** 2 for c in others], device=D[others[0]].device)
    P['idw'] = (w[:, None, None] * torch.stack([D[c] for c in others])).sum(0) / w.sum()
    return P


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=[10, 70, 130, 190, 250])
    p.add_argument('--test-frames', type=int, nargs='*', default=list(range(0, 300, 10)))
    p.add_argument('--ks', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6, 8])
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy = ens_eval.legacy_helpers()
    started = time.time()
    result = dict(kind='testcolour_probe3', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, models={})
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        aff = payload.get('affine')
        del payload
        names = list(aff['names'])
        Dall = aff['delta'].float()
        Dall = (Dall - Dall.mean(0)).cuda()
        D = {c: Dall[i] for i, c in enumerate(names)}
        warp = warp_fn(aff)
        manifest = m['manifest']
        root = Path(manifest['_root'])
        centres = {c: centre_of(manifest, c) for c in names + ['cam00']}
        loo = {}
        with torch.no_grad():
            for c in names:
                obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == c}
                xs, ys = [], []
                for f in a.frames:
                    o = obs[f]
                    camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                    st = native.native_effective_state(m['g'], float(o['time']), torch)
                    img = warp(native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float(), c)
                    lr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['lr_path']))).permute(2, 0, 1).float().cuda()
                    fy = img.shape[-2] // lr.shape[-2]
                    xs.append(img)
                    ys.append(lr)
                P = predictors(D, names, centres, c, a.ks)
                P['own'] = D[c]
                row = {}
                for name, d in P.items():
                    mse = float(np.mean([float((F.avg_pool2d(apply_affine(d, x)[None], fy)[0].clamp(0, 1) - y).pow(2).mean()) for x, y in zip(xs, ys)]))
                    row[name] = -10 * math.log10(max(mse, 1e-12))
                loo[c] = row
            keys = [k for k in loo[names[0]] if k != 'own']
            loo_mean = {k: float(np.mean([loo[c][k] for c in names])) for k in keys + ['own']}
            best = max(keys, key=lambda k: loo_mean[k])
            # test view
            obs = {int(o['frame_index']): o for o in manifest['observations'] if o['camera_id'] == 'cam00'}
            Pt = predictors(D, names + ['cam00'], centres, 'cam00', a.ks)
            Pt['identity'] = torch.zeros(3, 4, device='cuda')
            test = {k: [] for k in Pt}
            for f in a.test_frames:
                o = obs[f]
                camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                img = native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].float()
                hr = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(root / o['hr_path']))).permute(2, 0, 1).float().cuda()
                for k, d in Pt.items():
                    test[k].append(-10 * math.log10(max(float((apply_affine(d, img).clamp(0, 1) - hr).pow(2).mean()), 1e-12)))
            tpsnr = {k: float(np.mean(v)) for k, v in test.items()}
        rule = 'mean_others' if best == 'mean_others' else best
        test_key = 'identity' if rule == 'mean_others' else rule
        result['models'][label] = dict(info=m['info'], loo_mean_psnr=loo_mean, loo_per_camera=loo, loo_best=best,
                                       test_psnr=tpsnr, test_delta_db={k: v - tpsnr['identity'] for k, v in tpsnr.items()},
                                       chosen_rule_test_delta_db=tpsnr[test_key] - tpsnr['identity'])
        print(json.dumps({label: dict(loo_best=best, loo={k: round(v - loo_mean['mean_others'], 3) for k, v in loo_mean.items()},
                                      test={k: round(v - tpsnr['identity'], 3) for k, v in tpsnr.items()},
                                      chosen=round(tpsnr[test_key] - tpsnr['identity'], 3))}), flush=True)
        del m
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
