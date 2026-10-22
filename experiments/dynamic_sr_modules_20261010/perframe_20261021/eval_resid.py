"""cam00 evaluation of a checkpoint plus per-frame residuals, with a temporal-difference (flicker) read-out (Claude 4dsr-25, 2026-10-21;
LR-anchored per-frame dynamic refinement, joint design with 4dsr-f3, whose absorb_20261021 fitters define the residuals).
geoprobe_20261017/eval_full.py runs unchanged (same metrics, post-hoc training-camera cull, TF32 off); only
full_refine_registered_v2.native_effective_state and the frozen image_array helper are wrapped.

Residuals: for a model registered with --resid LABEL=DIR, the deformed state at the time of frame f receives DIR/f{f:04d}.npz on the
post-cull Gaussians, in the order of absorb_bound2.build():
  xyz[idx] += dxyz;  sh[idx, 0, :] += dsh0;  scales[idx] *= exp(dlogscale);  opacity[idx] = sigmoid(logit(opacity[idx]) + dlogit)
npz arrays: idx (int, K; 'free' accepted) is required; dxyz (K, 3, world units), dsh0 (K, 3; 'dc' accepted), dlogscale (K, 3),
dlogit (K, 1) are optional. DIR/run.json (or meta.json) must name the evaluated checkpoint's sha256 (key checkpoint_sha256, or
parent_sha256, or parent.sha256 / checkpoint.sha256) and the post-cull point count (points_after_cull, or cull.points_after); a
mismatch stops the run. A frame without a file is rendered unchanged and listed (missing_frames); --require-all makes it an error.

Temporal read-out (every model, also those without residuals; frames must be consecutive): for each pair (f - 1, f)
  d_R = R(f) - R(f - 1), d_H = HR(f) - HR(f - 1);  tdiff_psnr = -10 log10(pooled mean (d_R - d_H)^2)  (whole image) and the same
  inside dyn(f) | dyn(f - 1) (eval_full.dyn_mask, background = median of every 5th cam00 HR frame, as eval_full); plus the render's own
  temporal energy ratio mean|d_R| / mean|d_H| in that region (1 = as much frame-to-frame change as HR; > 1 = flicker, < 1 = smoothing).
Written to <out>/temporal.json; residual bookkeeping to <out>/resid.json.
  eval_resid.py [--resid LABEL=DIR ...] [--require-all] <eval_full.py arguments>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEO = HERE.parent / 'geoprobe_20261017'
if str(GEO) not in sys.path:
    sys.path.insert(0, str(GEO))

import eval_full  # noqa: E402  (sets up the runtime paths, torch, cullretrain, ens_eval, native)
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = eval_full.ROOT
MODELS = {}    # id(GaussianModel) -> dict(label, times, root, test, resid=None | dict)
CTX = {}
ALIASES = dict(idx=('idx', 'free'), dxyz=('dxyz',), dsh0=('dsh0', 'dc'), dlogscale=('dlogscale',), dlogit=('dlogit',))


def first(z, names):
    for n in names:
        if n in z:
            return z[n]
    return None


def apply(st, z):
    """absorb_bound2.build() with stored arrays; returns a new state dict (the input is not modified)."""
    st = dict(st)
    dev = st['xyz'].device
    idx = first(z, ALIASES['idx'])
    if idx is None:
        raise KeyError('residual file without idx')
    idx = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(dev)
    arr = lambda k: (None if first(z, ALIASES[k]) is None else
                     torch.from_numpy(np.asarray(first(z, ALIASES[k]), dtype=np.float32)).to(dev))
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= st['xyz'].shape[0]):
        raise IndexError('residual index outside the post-cull model')
    if arr('dxyz') is not None:
        st['xyz'] = st['xyz'].index_add(0, idx, arr('dxyz').to(st['xyz'].dtype))
    if arr('dsh0') is not None:
        sh = st['sh'].clone()
        sh[idx, 0, :] = sh[idx, 0, :] + arr('dsh0').to(sh.dtype)
        st['sh'] = sh
    if arr('dlogscale') is not None:
        st['scales'] = st['scales'].index_put((idx,), st['scales'][idx] * torch.exp(arr('dlogscale')))
    if arr('dlogit') is not None:
        o = st['opacity'][idx].clamp(1e-6, 1 - 1e-6)
        st['opacity'] = st['opacity'].index_put((idx,), torch.sigmoid(torch.log(o / (1 - o)) + arr('dlogit')))
    return st


def meta_value(meta, paths):
    for p in paths:
        v = meta
        for k in p.split('.'):
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            return v
    return None


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--resid', action='append', default=[])
    p.add_argument('--require-all', action='store_true')
    a, rest = p.parse_known_args()
    resid = dict(s.split('=', 1) for s in a.resid)
    out = ROOT / rest[rest.index('--out') + 1]
    orig_load, orig_state, orig_helpers = (eval_full.ens_eval.load_model, eval_full.native.native_effective_state,
                                           eval_full.ens_eval.legacy_helpers)
    temporal = {}

    def load_model(label, path, cull):
        m = orig_load(label, path, cull)
        times = {}
        for o in m['manifest']['observations']:
            t, f = float(o['time']), int(o['frame_index'])
            if times.setdefault(t, f) != f:
                raise SystemExit(f'{label}: time {t} maps to frames {times[t]} and {f}')
        rec = dict(label=label, times=times, root=Path(m['manifest']['_root']), resid=None)
        if label in resid:
            d = ROOT / resid[label]
            mp = d / 'run.json' if (d / 'run.json').exists() else d / 'meta.json'
            meta = json.loads(mp.read_text())
            sha = meta_value(meta, ('checkpoint_sha256', 'parent_sha256', 'parent.sha256', 'checkpoint.sha256'))
            pts = meta_value(meta, ('points_after_cull', 'cull.points_after'))
            if sha != m['info']['checkpoint']['sha256']:
                raise SystemExit(f'{label}: residuals for checkpoint {sha}, evaluated {m["info"]["checkpoint"]["sha256"]}')
            if pts is None or int(pts) != m['info']['points_evaluated']:
                raise SystemExit(f'{label}: residuals for {pts} points, model has {m["info"]["points_evaluated"]} after the cull')
            rec['resid'] = dict(dir=d, meta_file=mp.name, applied=[], missing=[], k=[])
            m['info']['residuals'] = dict(dir=resid[label], meta_file=mp.name)
        MODELS[id(m['g'])] = rec
        return m

    def native_effective_state(g, t, torch_):
        st = orig_state(g, t, torch_)
        rec = MODELS.get(id(g))
        if rec is None:
            return st
        f = rec['times'][float(t)]
        CTX['cur'] = (rec['label'], f)
        r = rec['resid']
        if r is None:
            return st
        path = r['dir'] / f'f{f:04d}.npz'
        if not path.exists():
            if a.require_all:
                raise SystemExit(f'{rec["label"]}: no residuals for frame {f}')
            r['missing'].append(f)
            return st
        with np.load(path) as z:
            z = {k: z[k] for k in z.files}
        r['applied'].append(f)
        r['k'].append(int(np.asarray(first(z, ALIASES['idx'])).size))
        return apply(st, z)

    def legacy_helpers():
        lg = orig_helpers()
        image_array, read_rgb = lg.image_array, lg.read_rgb
        bgs = {}

        def capture(x):
            arr = image_array(x)
            cur = CTX.pop('cur', None)
            if cur is None:
                return arr
            label, f = cur
            hr = read_rgb(VIEW[label]['root'] / VIEW[label]['obs'][f]['hr_path'])
            if label not in bgs:
                bgs[label] = np.median(np.stack([read_rgb(VIEW[label]['root'] / VIEW[label]['obs'][k]['hr_path'])
                                                 for k in range(0, len(VIEW[label]['obs']), 5)]), axis=0)
            dm = eval_full.dyn_mask(hr, bgs[label])
            s = temporal.setdefault(label, dict(prev=None, se=0.0, n=0, se_d=0.0, n_d=0, er=0.0, eh=0.0, er_d=0.0, eh_d=0.0, pairs=0))
            cur_t = (f, arr.astype(np.float32), hr.astype(np.float32), dm)
            if s['prev'] is not None and s['prev'][0] == f - 1:
                _, r0, h0, m0 = s['prev']
                dr, dh = cur_t[1] - r0, cur_t[2] - h0
                e = ((dr - dh) ** 2).sum(axis=2)
                m = dm | m0
                s['se'] += float(e.sum()); s['n'] += e.size * 3
                s['se_d'] += float(e[m].sum()); s['n_d'] += int(m.sum()) * 3
                s['er'] += float(np.abs(dr).sum()); s['eh'] += float(np.abs(dh).sum())
                s['er_d'] += float(np.abs(dr)[m].sum()); s['eh_d'] += float(np.abs(dh)[m].sum())
                s['pairs'] += 1
            s['prev'] = cur_t
            return arr

        lg.image_array = capture
        return lg

    VIEW = {}
    orig_view = eval_full.ens_eval.view_observations

    def view_observations(manifest, view, label):
        obs = orig_view(manifest, view, label)
        VIEW[label] = dict(root=Path(manifest['_root']), obs={int(o['frame_index']): o for o in obs})
        return obs

    eval_full.ens_eval.load_model = load_model
    eval_full.ens_eval.view_observations = view_observations
    eval_full.ens_eval.legacy_helpers = legacy_helpers
    eval_full.native.native_effective_state = native_effective_state
    sys.argv = [str(GEO / 'eval_full.py')] + rest
    eval_full.main()
    ps = lambda se, n: -10 * float(np.log10(max(se / max(n, 1), 1e-12)))
    tsum = {k: dict(pairs=s['pairs'], tdiff_psnr=ps(s['se'], s['n']), tdiff_dyn_psnr=ps(s['se_d'], s['n_d']),
                    temporal_energy_ratio=s['er'] / max(s['eh'], 1e-12), temporal_energy_ratio_dyn=s['er_d'] / max(s['eh_d'], 1e-12))
            for k, s in temporal.items()}
    (out / 'temporal.json').write_text(json.dumps(dict(doc=__doc__, summary=tsum), indent=1))
    report = {r['label']: dict(dir=str(r['resid']['dir'].relative_to(ROOT)), meta_file=r['resid']['meta_file'],
                               applied_frames=len(r['resid']['applied']), missing_frames=r['resid']['missing'],
                               k_mean=float(np.mean(r['resid']['k'])) if r['resid']['k'] else 0.0)
              for r in MODELS.values() if r['resid'] is not None}
    (out / 'resid.json').write_text(json.dumps(dict(wrapper=str(Path(__file__).relative_to(ROOT)), report=report), indent=1))
    print(json.dumps(dict(temporal={k: (v['pairs'], round(v['tdiff_psnr'], 3), round(v['tdiff_dyn_psnr'], 3),
                                        round(v['temporal_energy_ratio_dyn'], 3)) for k, v in tsum.items()},
                          resid={k: (v['applied_frames'], len(v['missing_frames'])) for k, v in report.items()})), flush=True)


if __name__ == '__main__':
    main()
