"""Fixed-checkpoint diagnostic of the monocular depth prior (Claude, session 4dsr-e9, 2026-10-18). Inference only, no training.
New file; modkit5.py (byte-frozen, hash pinned below) supplies the prior network and the training-time loss functions; cullretrain /
ens_eval_view / the pinned 4DGaussians runtime are imported, never modified.

Codex feedback 10-18 (verified): the rasteriser returns the unnormalised depth sum D = sum_i w_i z_i (forward.cu l.359, 377) and
modkit5 used 1 / D as inverse depth, which is only right where the coverage A = sum_i w_i is close to 1; the loss was local Pearson
only. This probe measures, on training views of the given checkpoints (models trained without and with the depth prior):
  A        coverage per pixel, A = 1 - mean_rgb(render on white background - render on black background);
  cur      the training-time quantity: modkit5.render_inverse_depth(D) against the prior on the teacher image, per 32 x 32 patch
           (1 - Pearson; modkit5's prior-std validity rule), split into patches whose minimum A (on the prior grid) is >= --a-min
           and the others; its global Pearson;
  norm     the coverage-normalised inverse mean depth A / D (pixels with A < --a-min excluded), the same statistics;
  prior    when the view's HR exists locally: the prior on the HR image vs the prior on the teacher image (global and patch Pearson),
           a check of the teacher-borne prior's quality (reads training-view HR, diagnostic only).
Writes summary.json in --out with per-view rows and per-model means.

  depth_diag.py --checkpoint LABEL=rel/ck.pt [...] --teacher-dir rel/sr_dir --out rel/dir [--views cam01:30 ...] [--a-min 0.9]
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
for _p in (CULL, VAR, HERE, HERE.parent):
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
PINNED = {'modkit5.py': '6d7108e73adf085a'}


def load_modkit5():
    """modkit5's helper functions without importing its training chain (modkit4 needs the srfix runtime at import)"""
    import types
    src = (HERE / 'modkit5.py').read_text()
    if rt.sha(HERE / 'modkit5.py')[:16] != PINNED['modkit5.py']:
        raise SystemExit('modkit5.py changed')
    ns = dict(__name__='modkit5_helpers', json=json, sys=sys, time=time, types=types, Path=Path)
    ns['modkit4'] = types.SimpleNamespace(modkit2=types.SimpleNamespace(modkit=types.SimpleNamespace(
        calibsr_v2=types.SimpleNamespace(sha=lambda p: rt.sha(p)))))
    ns['THIRD'] = ROOT / 'output/dynamic_sr_modules_20261010/geoprobe_20261017/third_party'
    ns['DA_DIR'] = ns['THIRD'] / 'Depth-Anything-V2'
    ns['DA_WEIGHT'] = ns['THIRD'] / 'depth_anything_v2_vits.pth'
    ns['DA_SHA'] = '715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
    ns['MIN_STD'] = 0.05
    exec(src[src.index('def load_prior('):src.index('def install(')], ns)
    return types.SimpleNamespace(**{k: ns[k] for k in ('load_prior', 'prior_inverse_depth', 'render_inverse_depth', 'patch_pearson_loss')})


def per_patch(render, prior, amin, patch, min_std=0.05):
    """per patch: 1 - Pearson (both maps scaled by their image-wide std, as modkit5), prior-std validity, minimum coverage"""
    h, w = render.shape
    h2, w2 = h // patch * patch, w // patch * patch

    def blocks(x):
        return x[:h2, :w2].reshape(h2 // patch, patch, w2 // patch, patch).permute(0, 2, 1, 3).reshape(-1, patch * patch)
    a = blocks(render / (render.std() + 1e-8))
    b = blocks(prior / (prior.std() + 1e-8))
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    sa, sb = a.pow(2).mean(1).add(1e-6).sqrt(), b.pow(2).mean(1).sqrt()
    corr = (a * b).mean(1) / (sa * sb + 1e-6)
    return 1 - corr, sb >= min_std, blocks(amin).min(1).values


def gcorr(x, y, mask=None):
    x, y = x.flatten(), y.flatten()
    if mask is not None:
        x, y = x[mask.flatten()], y[mask.flatten()]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


def coverage(black, white):
    """renders of the same view on black and white backgrounds (3 x H x W): render = C + (1 - A) bg, so A = 1 - (white - black)"""
    return (1 - (white - black).mean(0)).clamp(0, 1)


def view_stats(mk5, A, D, pt, ph, a_min, patch):
    """A, D: coverage and unnormalised depth sum at render resolution; pt / ph: prior on the teacher / HR image (ph may be None)"""
    grid = pt.shape[-2:]
    inv_cur = mk5.render_inverse_depth(D, grid)
    Ag = F.interpolate(A[None, None], size=grid, mode='area')[0, 0]
    Dg = F.interpolate(D[None, None], size=grid, mode='area')[0, 0]
    mask = Ag > a_min
    zbar = Dg / Ag.clamp_min(1e-3)
    inv_norm = 1.0 / zbar.clamp_min(0.05 * float(zbar[mask].median()) if bool(mask.any()) else 1e-3)
    lc, vc, amin_c = per_patch(inv_cur, pt, Ag, patch)
    ln, _, _ = per_patch(torch.where(mask, inv_norm, inv_norm[mask].mean()), pt, Ag, patch)
    hi = vc & (amin_c >= a_min)
    lo = vc & (amin_c < a_min)
    return dict(A_lt_099=float((Ag < 0.99).float().mean()), A_lt_09=float((Ag < 0.9).float().mean()), A_lt_05=float((Ag < 0.5).float().mean()),
                loss_cur=float(lc[vc].mean()), loss_cur_hiA=float(lc[hi].mean()) if hi.any() else None,
                loss_cur_loA=float(lc[lo].mean()) if lo.any() else None, loA_patch_share=float(lo.float().sum() / vc.float().sum()),
                loA_loss_share=float(lc[lo].sum() / lc[vc].sum()) if lo.any() else 0.0,
                loss_norm_hiA=float(ln[hi].mean()) if hi.any() else None,
                gcorr_cur=gcorr(inv_cur, pt), gcorr_norm=gcorr(inv_norm, pt, mask),
                prior_teacher_vs_hr_gcorr=None if ph is None else gcorr(pt, ph),
                prior_teacher_vs_hr_patchloss=None if ph is None else float(per_patch(pt, ph, Ag, patch)[0][vc].mean()),
                gcorr_norm_vs_hrprior=None if ph is None else gcorr(inv_norm, ph, mask))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--teacher-dir', required=True, help='directory with <camera>/<frame:04d>.png teacher images')
    p.add_argument('--out', required=True)
    p.add_argument('--views', nargs='*', default=None, help='camera:frame; default 6 training cameras x frames 30, 150, 270')
    p.add_argument('--a-min', type=float, default=0.9)
    p.add_argument('--patch', type=int, default=32)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed probe preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mk5 = load_modkit5()
    legacy = ens_eval.legacy_helpers()
    net = mk5.load_prior('cuda')
    started = time.time()
    result = dict(kind='depth_diag', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), HERE / 'modkit5.py', Path(cr.__file__))}, models={})
    prior_cache = {}
    for spec in a.checkpoint:
        label, path = spec.split('=', 1)
        m = ens_eval.load_model(label, path, True)
        manifest = m['manifest']
        root = Path(manifest['_root'])
        train = list(manifest['splits']['train'])
        views = [v.split(':') for v in a.views] if a.views else [(c, f) for c in train[::max(1, len(train) // 6)][:6] for f in (30, 150, 270)]
        obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
        rows = []
        for cam, f in views:
            f = int(f)
            o = obs[(cam, f)]
            with torch.no_grad():
                camera = native.hr_camera(ev.camera_item(manifest, o, torch), f, manifest)
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                black = torch.zeros(3, device='cuda')
                white = torch.ones(3, device='cuda')
                pb = native.native_rgb(camera, m['g'], st, m['pipe'], black)
                pw = native.native_rgb(camera, m['g'], st, m['pipe'], white)
                A = coverage(pb['render'], pw['render'])
                D = pb['depth'].reshape(A.shape).float()
                key = (cam, f)
                if key not in prior_cache:
                    timg = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(ROOT / a.teacher_dir / cam / f'{f:04d}.png'))).permute(2, 0, 1).float().cuda()
                    pt = mk5.prior_inverse_depth(net, timg)
                    hrp = root / o['hr_path']
                    ph = None
                    if hrp.exists():
                        himg = torch.from_numpy(np.ascontiguousarray(legacy.read_rgb(hrp))).permute(2, 0, 1).float().cuda()
                        ph = mk5.prior_inverse_depth(net, himg)
                    prior_cache[key] = (pt, ph)
                pt, ph = prior_cache[key]
                row = dict(view=f'{cam}:{f}', **view_stats(mk5, A, D, pt, ph, a.a_min, a.patch))
                rows.append(row)
        keys = [k for k in rows[0] if k != 'view']
        mean = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
        result['models'][label] = dict(info=m['info'], views=rows, mean=mean)
        print(json.dumps({label: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mean.items()}, 's': round(time.time() - started)}), flush=True)
        del m
        torch.cuda.empty_cache()
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
