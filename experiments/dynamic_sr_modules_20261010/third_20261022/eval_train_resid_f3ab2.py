"""[eval_train_resid_f3ab2.py, Claude 4dsr-f3 2026-10-22 21:06 PDT: copy of perframe_20261021/eval_train_resid.py whose only change is
the camera model of checkpoints WITHOUT a warp field (the tm rows of the user-requested stacked ablation): the checkpoint's own colour
affine (cullretrain.CameraAffine, the same apply() used in its training) instead of an error. Warp-field checkpoints are unchanged.]
Training-view evaluation AGAINST HR of a parent and its per-frame residual arms (Claude 4dsr-25, 2026-10-21; user request ~20:15
PDT: the P / P+C gains measured against HR on training images and on test images, not against the LR). Training-camera HR is used
for scoring only; it never entered any fit.

Views: every training camera of the parent's manifest at frames --frames (default 0, 10, ..., 290). Models: the culled parent and each
--resid LABEL=FITDIR (perframe_fit.py outputs; FITDIR/run.json must match the evaluated checkpoint and its post-hoc cull). For every
view and model:
  R      native render of the deformed state at frame f (+ the frame's residuals via perframe_fit.apply_residual), clamped;
  R_obs  the checkpoint's camera model applied at HR (warp field, then colour affine: the fitter's closure path before the area
         operator), clamped -- the render as that training camera would record it;
  metrics of R_obs against the camera's HR: PSNR, SSIM (frozen legacy helper, 5 px border, as eval_full.py), LPIPS (alex 0.1),
  dynamic / static PSNR (eval_full.dyn_mask: |HR - background| > 0.06 max over RGB, 5 px dilation; background = per-camera median of
  the evaluated HR frames), high-band PSNR (eval_full.band_mse); raw PSNR of R (no camera model) against HR; LR closure PSNR (4x4 area
  of R_obs against the LR, observed space).
Summary per model: mean over views of per-view PSNR / SSIM / LPIPS / raw PSNR / LR closure; pooled dynamic / static PSNR; per-camera
means. TF32 off.
Usage: eval_train_resid.py --checkpoint CK --resid P=FITDIR [--resid PC=FITDIR] --out DIR [--frames 0 10 ...] [--threads 8]"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABS = HERE.parent / 'absorb_20261021'
GEO = HERE.parent / 'geoprobe_20261017'
for _p in (ABS, GEO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import absorb_bound2 as ab  # noqa: E402  (cullretrain runtime, camera model, model loader, area operator)
import perframe_fit as pf  # noqa: E402  (apply_residual / load_residual: the fitter's own code)
import eval_full  # noqa: E402  (ens_eval legacy metric helpers, band_mse)
import numpy as np  # noqa: E402
import torch  # noqa: E402

cr, native, ev = ab.cr, ab.native, ab.ev
ROOT = ab.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--resid', action='append', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=list(range(0, 300, 10)))
    p.add_argument('--cameras', nargs='*', default=None)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError(f'completed evaluation preserved: {out}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    started = time.time()
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    plan, raw_aff = payload['plan'], payload.get('affine')
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    root = Path(manifest['_root'])
    train = list(manifest['splits']['train'])
    cams = [c for c in train if a.cameras is None or c in a.cameras]
    if raw_aff is not None and 'wfield' not in raw_aff:   # f3ab2 (4dsr-f3, 10-22 21:06): colour-affine-only checkpoints (tm arms)
        model = cr.CameraAffine(list(raw_aff['names']), device=dev)
        with torch.no_grad():
            model.delta.copy_(raw_aff['delta'].to(device=dev, dtype=torch.float32))
        model.delta.requires_grad_(False)
    else:
        model = ab.camera_model(raw_aff, dev)
    g = ab.load_model(ck, plan, dev)
    h, opt, pipe, sh = cr.configuration(plan)
    cull_record = cr.cull(g, cr.Culler(manifest, device=dev), None, 'posthoc', optimizer_backed=False)
    xyz_sha = hashlib.sha256(g._xyz.detach().float().cpu().numpy().tobytes()).hexdigest()
    ck_sha = cr.entry(ck)['sha256']
    resid = {}
    for spec in a.resid:
        lab, d = spec.split('=', 1)
        run = json.loads((ROOT / d / 'run.json').read_text())
        if run.get('checkpoint_sha256') != ck_sha or int(run.get('points_after_cull', -1)) != int(g._xyz.shape[0]) \
                or run.get('post_cull_xyz_sha256') != xyz_sha:
            raise SystemExit(f'{lab}: residuals do not belong to this checkpoint / cull')
        resid[lab] = ROOT / d
    labels = ['parent'] + list(resid)
    lr_w, lr_h = manifest['resolutions']['lr']
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    index = {c: i for i, c in enumerate(train)}
    background = torch.ones(3, device=dev) if cr.white_background(plan) else torch.zeros(3, device=dev)
    legacy = eval_full.ens_eval.legacy_helpers()
    # as eval_full.py: pins torch.hub to the runtime copy's torch_hub, where the local AlexNet weights live (10-22: without it the local
    # service, which has no network, tried to download them; remote hosts had downloaded the same file, sha256 7be5be79...)
    runtime = eval_full.rt.check_runtime()
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    read = lambda path: legacy.read_rgb(root / path)                 # HWC float [0, 1], as eval_full.py
    med = {}
    for c in cams:
        med[c] = np.median(np.stack([read(obs[(c, f)]['hr_path']) for f in a.frames]), axis=0)
    interior = None
    pool = cf.ThreadPoolExecutor(max_workers=a.threads)
    rows = []
    agg = {lab: dict(dyn=0.0, sta=0.0) for lab in labels}
    n_dyn = n_sta = 0

    def cpu_metrics(pred, hr, mask):
        ssim = legacy.ssim_map_rgb(pred, hr)
        e = ((pred.astype(np.float64) - hr) ** 2).sum(axis=2)
        hb = eval_full.band_mse(pred, hr)
        return dict(psnr=legacy.metric_psnr(float(e.mean() / 3)), ssim=legacy.masked_mean(ssim, mask), high_psnr=-10 * math.log10(max(hb[2], 1e-12)))

    for f in a.frames:
        t0 = time.time()
        with torch.no_grad():
            S = native.native_effective_state(g, float(obs[(train[0], f)]['time']), torch)
            states = {'parent': S}
            for lab, d in resid.items():
                states[lab] = pf.apply_residual(S, pf.load_residual(d / f'f{f:04d}.npz', dev))
        for c in cams:
            o = obs[(c, f)]
            hr = read(o['hr_path'])
            lr = torch.from_numpy(np.ascontiguousarray(read(o['lr_path']))).permute(2, 0, 1).to(dev)
            if interior is None:
                interior = np.zeros(hr.shape[:2], bool)
                interior[5:-5, 5:-5] = True
            dm = eval_full.dyn_mask(hr, med[c])
            nd = int(dm.sum())
            n_dyn += 3 * nd
            n_sta += 3 * (dm.size - nd)
            cam = native.hr_camera(ev.camera_item(manifest, o, torch), index[c], manifest)
            preds, raws, clos = {}, {}, {}
            with torch.no_grad():
                for lab in labels:
                    R = native.native_rgb(cam, g, states[lab], pipe, background)['render']
                    Robs = model.apply(R, c)
                    raws[lab] = legacy.image_array(R.clamp(0, 1).detach())
                    preds[lab] = legacy.image_array(Robs.clamp(0, 1).detach())
                    clos[lab] = float(-10 * torch.log10(((ab.area(Robs, (lr_h, lr_w)) - lr) ** 2).mean().clamp_min(1e-12)))
            fut = {lab: pool.submit(cpu_metrics, preds[lab], hr, interior) for lab in labels}
            ty = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1)[None].cuda() * 2 - 1
            for lab in labels:
                with torch.inference_mode():
                    tx = torch.from_numpy(np.ascontiguousarray(preds[lab])).permute(2, 0, 1)[None].cuda() * 2 - 1
                    lp = float(metric(tx, ty).item())
                e = ((preds[lab].astype(np.float64) - hr) ** 2).sum(axis=2)
                agg[lab]['dyn'] += float(e[dm].sum())
                agg[lab]['sta'] += float(e[~dm].sum())
                q = fut[lab].result()
                raw_psnr = legacy.metric_psnr(float(((raws[lab].astype(np.float64) - hr) ** 2).mean()))
                rows.append(dict(frame=f, camera=c, name=lab, psnr=q['psnr'], ssim=q['ssim'], lpips=lp, high_psnr=q['high_psnr'],
                                 raw_psnr=raw_psnr, lr_closure_psnr=clos[lab], dyn_fraction=nd / dm.size))
        print(json.dumps(dict(frame=f, seconds=round(time.time() - t0, 1), elapsed=round(time.time() - started, 1))), flush=True)
    pool.shutdown()
    by = {lab: [r for r in rows if r['name'] == lab] for lab in labels}
    mean = lambda lab, k: float(np.mean([r[k] for r in by[lab]]))
    ps = lambda se, n: -10 * math.log10(max(se / max(n, 1), 1e-12))
    summary = {lab: dict(views=len(by[lab]), psnr=mean(lab, 'psnr'), ssim=mean(lab, 'ssim'), lpips=mean(lab, 'lpips'),
                         high_psnr=mean(lab, 'high_psnr'), raw_psnr=mean(lab, 'raw_psnr'), lr_closure_psnr=mean(lab, 'lr_closure_psnr'),
                         dyn_psnr=ps(agg[lab]['dyn'], n_dyn), static_psnr=ps(agg[lab]['sta'], n_sta),
                         per_camera_psnr={c: float(np.mean([r['psnr'] for r in by[lab] if r['camera'] == c])) for c in cams})
               for lab in labels}
    for lab in labels:
        if lab != 'parent':
            summary[lab]['delta_vs_parent'] = {k: summary[lab][k] - summary['parent'][k]
                                               for k in ('psnr', 'ssim', 'lpips', 'high_psnr', 'raw_psnr', 'lr_closure_psnr', 'dyn_psnr', 'static_psnr')}
            summary[lab]['cameras_improved'] = sum(summary[lab]['per_camera_psnr'][c] > summary['parent']['per_camera_psnr'][c] for c in cams)
    result = dict(doc=__doc__, kind='eval_train_resid', checkpoint=cr.entry(ck), manifest=cr.entry(manifest_path), cull_record=cull_record,
                  post_cull_xyz_sha256=xyz_sha, resid={k: str(v.relative_to(ROOT)) for k, v in resid.items()}, cameras=cams, frames=a.frames,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(), runtime=runtime, dyn_share=n_dyn / max(n_dyn + n_sta, 1),
                  summary=summary, seconds=time.time() - started)
    with (out / 'per_view.jsonl').open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({lab: {k: round(v, 4) for k, v in s.items() if isinstance(v, float)} for lab, s in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
