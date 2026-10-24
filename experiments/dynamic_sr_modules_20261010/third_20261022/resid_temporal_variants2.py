"""Zero-training premise test for a temporally coherent per-frame refinement (Claude 4dsr-f3, 2026-10-22; third-module study
with 4dsr-46). Reads one perframe_fit residual directory (fNNNN.npz: idx, dxyz, dsh0) and writes temporally filtered variants in
the same format, so eval_resid.py scores them against the parent and the unfiltered fit in one pass.

Per Gaussian u the fit only exists at its selected frames T_u (M[t,u] = 1). Variants (values are written at the original selected
entries only; idx sets are unchanged):
  SMs  masked Gaussian smoothing along time, sigma = s frames: conv(D*M) / conv(M)  (normalised convolution, reflect padding)
  SMCs / SMXs  as SMs, but only the colour residual (dsh0) / only the position residual (dxyz) is smoothed; the other is kept
  KFk  keyframes every k frames: local estimate at keyframe times (normalised convolution, sigma 0.75), linear interpolation between
       keyframes. Storage = |U| x n_key x 6 values instead of sum_t |idx_t| x 6.
Values are written as float16 (eval_resid casts to float32). run.json copies the source run.json (checkpoint / cull identity) and
adds the variant block. No training, no GPU, no cam00.
Usage: resid_temporal_variants.py --fit FITDIR --out-root DIR [--variants SM1,SM2,KF4,KF10] [--chunk 20000]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def gconv(x, sigma, axis=0):
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * x.ndim
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode='reflect')
    out = np.zeros_like(x, dtype=np.float64)
    n = x.shape[axis]
    for j, w in enumerate(k):
        out += w * np.take(xp, range(j, j + n), axis=axis)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fit', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--variants', default='SM1,SM2,KF4,KF10')
    p.add_argument('--chunk', type=int, default=20000)
    a = p.parse_args()
    t0 = time.time()
    fit = ROOT / a.fit
    frames = sorted(int(q.stem[1:]) for q in fit.glob('f[0-9][0-9][0-9][0-9].npz'))
    if frames != list(range(len(frames))) or not frames:
        raise SystemExit(f'{fit}: frames must be 0..N-1, got {len(frames)}')
    T = len(frames)
    src = []
    for f in frames:
        with np.load(fit / f'f{f:04d}.npz') as z:
            idx = np.asarray(z['idx'] if 'idx' in z.files else z['free'], dtype=np.int64)
            dxyz = np.asarray(z['dxyz'], dtype=np.float32) if 'dxyz' in z.files else np.zeros((idx.size, 3), np.float32)
            dsh0 = np.asarray(z['dsh0'] if 'dsh0' in z.files else z['dc'], dtype=np.float32) if ('dsh0' in z.files or 'dc' in z.files) \
                else np.zeros((idx.size, 3), np.float32)
            extra = {k: z[k] for k in z.files if k not in ('idx', 'free', 'dxyz', 'dsh0', 'dc')}
        src.append((idx, dxyz, dsh0, extra))
    U = np.unique(np.concatenate([s[0] for s in src]))
    pos = {f: np.searchsorted(U, src[f][0]) for f in frames}
    variants = [v for v in a.variants.split(',') if v]
    outs = {v: [[np.zeros((src[f][0].size, 3), np.float32), np.zeros((src[f][0].size, 3), np.float32)] for f in frames]
            for v in variants}
    for c0 in range(0, U.size, a.chunk):
        c1 = min(U.size, c0 + a.chunk)
        n = c1 - c0
        D = np.zeros((T, n, 6), np.float64)
        M = np.zeros((T, n), np.float64)
        sel = {}
        for f in frames:
            pu = pos[f]
            m = (pu >= c0) & (pu < c1)
            loc = pu[m] - c0
            sel[f] = (np.nonzero(m)[0], loc)
            D[f, loc, :3] = src[f][1][m]
            D[f, loc, 3:] = src[f][2][m]
            M[f, loc] = 1.0
        for v in variants:
            if v.startswith('SMC') or v.startswith('SMX'):     # smooth one attribute only (C = colour dsh0, X = position dxyz)
                s = float(v[3:])
                num = gconv(D * M[:, :, None], s)
                den = gconv(M, s)[:, :, None]
                V = np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0.0)
                keep = slice(0, 3) if v.startswith('SMC') else slice(3, 6)
                V[:, :, keep] = D[:, :, keep]
            elif v.startswith('SM'):
                s = float(v[2:])
                num = gconv(D * M[:, :, None], s)
                den = gconv(M, s)[:, :, None]
                V = np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0.0)
            elif v.startswith('KF'):
                k = int(v[2:])
                kt = list(range(0, T, k))
                if kt[-1] != T - 1:
                    kt.append(T - 1)
                kt = np.asarray(kt)
                num = gconv(D * M[:, :, None], 0.75)
                den = gconv(M, 0.75)[:, :, None]
                L = np.where(den > 1e-3, num / np.maximum(den, 1e-12), 0.0)
                K = L[kt]
                j = np.clip(np.searchsorted(kt, np.arange(T), side='right') - 1, 0, len(kt) - 2)
                w = ((np.arange(T) - kt[j]) / (kt[j + 1] - kt[j]))[:, None, None]
                V = (1 - w) * K[j] + w * K[j + 1]
            else:
                raise SystemExit(f'unknown variant {v}')
            for f in frames:
                rows, loc = sel[f]
                outs[v][f][0][rows] = V[f, loc, :3]
                outs[v][f][1][rows] = V[f, loc, 3:]
    meta = json.loads((fit / 'run.json').read_text())
    stats = {}
    for v in variants:
        d = ROOT / a.out_root / v
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'complete.json').exists():
            raise FileExistsError(f'completed variant preserved: {d}')
        dx = dc = nx = nc = 0.0
        for f in frames:
            idx, sx, sc, extra = src[f]
            vx, vc = outs[v][f]
            dx += float(((vx - sx) ** 2).sum()); nx += float((sx ** 2).sum())
            dc += float(((vc - sc) ** 2).sum()); nc += float((sc ** 2).sum())
            tmp = d / f'f{f:04d}.tmp.npz'
            np.savez(tmp, idx=idx.astype(np.int32), dxyz=vx.astype(np.float16), dsh0=vc.astype(np.float16), **extra)
            tmp.replace(d / f'f{f:04d}.npz')
        stored = (U.size * len(range(0, T, int(v[2:]))) * 6) if v.startswith('KF') else sum(src[f][0].size for f in frames) * 6
        stats[v] = dict(rel_change_dxyz=(dx / max(nx, 1e-30)) ** 0.5, rel_change_dsh0=(dc / max(nc, 1e-30)) ** 0.5,
                        stored_values=int(stored), source_values=int(sum(src[f][0].size for f in frames) * 6))
        rec = dict(meta)
        rec['variant'] = dict(name=v, source_fit=a.fit, doc=__doc__, stats=stats[v], union_gaussians=int(U.size))
        (d / 'run.json').write_text(json.dumps(rec, indent=1))
        (d / 'complete.json').write_text(json.dumps(dict(frames=T, seconds=time.time() - t0), indent=1))
    summary = dict(fit=a.fit, union=int(U.size), frames=T, mean_selected=float(np.mean([s[0].size for s in src])), stats=stats,
                   seconds=time.time() - t0)
    (ROOT / a.out_root / 'variants_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
