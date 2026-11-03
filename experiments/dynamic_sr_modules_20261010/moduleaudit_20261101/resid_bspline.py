"""Temporal B-spline coding of per-frame P+C residuals (Claude 4dsr-4e, module audit 2026-11-01, candidate U1 step 1: post-hoc
projection, no training, no GPU, no cam00). Reads one perframe_fit residual directory (fNNNN.npz: idx, dxyz, dsh0) and writes
variants in the same format, so perframe_20261021/eval_resid.py (cam00) and third_20261022/loco_eval.py (held-out cam10) score them.

Per Gaussian u of the union U of all per-frame dynamic sets, the fit exists only at its selected frames (M[t,u] = 1). Basis: uniform
cubic B-splines on frames 0..T-1 with knot spacing D (B: T x m, m ~ T/D + 3; every frame touches 4 basis functions). Coefficients
per Gaussian solve the weighted ridge problem  sum_t w_t (B_t c - d_t)^2 + eps |c|^2,  w_t = 1 on selected frames and w0 elsewhere
(absent frames pull the spline softly towards 0), exactly, by sparse normal equations and a batched solve. Variants (kind + D):
  BPd  w0 = 0; written at the selected entries only (idx sets unchanged). Stored: |U| x m x 6 coefficients + T x |U| bitmap.
  BEd  w0 = --absent-weight (default 0.05); written at EVERY frame for every u in U (membership-free; idx = U). Stored: |U| x m x 6.
  BUd  w0 = 1 (absent frames count as residual 0, i.e. plain projection); written at every frame for every u in U.
  F16  precision control: the source residuals unchanged except float16 storage (same idx sets); isolates the fp16 effect (D = 1).
Values are written as float16 (eval_resid casts to float32); run.json copies the source run.json (checkpoint / cull identity) and adds
a variant block with stored-value counts and relative changes. No training, no GPU, no cam00, no HR.
usage: resid_bspline.py --fit FITDIR --out-root DIR [--variants BP4,BP8,BE4,BE8,BU4] [--chunk 8000] [--absent-weight 0.05]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def bspline_basis(n: int, spacing: float) -> np.ndarray:
    knots = np.arange(-3 * spacing, (n - 1) + 4 * spacing, spacing)
    t = np.arange(n, dtype=np.float64)
    cols = []
    for j in range(len(knots) - 4):
        u = (t - knots[j]) / spacing
        b = np.zeros(n)
        r = (u >= 0) & (u < 1); b[r] = u[r] ** 3 / 6
        r = (u >= 1) & (u < 2); v = u[r] - 1; b[r] = (-3 * v ** 3 + 3 * v ** 2 + 3 * v + 1) / 6
        r = (u >= 2) & (u < 3); v = u[r] - 2; b[r] = (3 * v ** 3 - 6 * v ** 2 + 4) / 6
        r = (u >= 3) & (u < 4); v = u[r] - 3; b[r] = (1 - v) ** 3 / 6
        if b.sum() > 1e-9:
            cols.append(b)
    return np.stack(cols, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fit', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--variants', default='BP4,BP8,BE4,BE8,BU4')
    p.add_argument('--chunk', type=int, default=8000)
    p.add_argument('--absent-weight', type=float, default=0.05)
    p.add_argument('--eps', type=float, default=1e-4)
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
            dxyz = np.asarray(z['dxyz'], dtype=np.float32)
            dsh0 = np.asarray(z['dsh0'] if 'dsh0' in z.files else z['dc'], dtype=np.float32)
            if set(z.files) - {'idx', 'free', 'dxyz', 'dsh0', 'dc', 'frame', 'px_world'}:
                raise SystemExit(f'{fit}: unsupported residual arrays {sorted(z.files)} (only P and C are coded)')
        src.append((idx, dxyz, dsh0))
    U = np.unique(np.concatenate([s[0] for s in src]))
    pos = [np.searchsorted(U, src[f][0]) for f in frames]
    variants = [v for v in a.variants.split(',') if v]
    kinds = {v: (('F1', 1) if v == 'F16' else (v[:2], int(v[2:]))) for v in variants}
    for v, (k, d) in kinds.items():
        if not (k == 'F1' or (k in ('BP', 'BE', 'BU') and d >= 2)):
            raise SystemExit(f'unknown variant {v}')
    bases = {d: bspline_basis(T, d) for d in {d for k, d in kinds.values() if k != 'F1'}}
    sparse = {}
    for d, B in bases.items():
        cols = np.zeros((T, 4), np.int64)
        vals = np.zeros((T, 4))
        for t in range(T):
            nz = np.nonzero(B[t] > 1e-12)[0]
            if len(nz) > 4:
                raise SystemExit('basis row with more than 4 nonzeros')
            j0 = min(int(nz[0]), B.shape[1] - 4)
            cols[t] = np.arange(j0, j0 + 4)
            vals[t] = B[t, j0:j0 + 4]
        sparse[d] = (cols, vals, B.shape[1])
    w0 = {'BP': 0.0, 'BE': a.absent_weight, 'BU': 1.0}

    def solve(Dm, M, d, kind):
        cols, vals, m = sparse[d]
        n = Dm.shape[1]
        W = M[:, :, 0] + w0[kind] * (1 - M[:, :, 0])
        G = np.zeros((n, m, m))
        R = np.zeros((n, m, 6))
        for t in range(T):
            c, v = cols[t], vals[t]
            G[:, c[:, None], c[None, :]] += W[t][:, None, None] * np.outer(v, v)[None]
            R[:, c, :] += W[t][:, None, None] * v[None, :, None] * Dm[t][:, None, :]
        G[:, np.arange(m), np.arange(m)] += a.eps
        C = np.linalg.solve(G, R)
        V = np.zeros_like(Dm)
        for t in range(T):
            V[t] = np.einsum('i,nik->nk', vals[t], C[:, cols[t], :])
        return V
    # outputs: BP keeps the per-frame idx sets; BE / BU write every u in U at every frame
    out_sel = {v: [[np.zeros((src[f][0].size, 3), np.float32), np.zeros((src[f][0].size, 3), np.float32)] for f in frames]
               for v in variants if kinds[v][0] == 'BP'}
    out_all = {v: np.zeros((T, U.size, 6), np.float16) for v in variants if kinds[v][0] in ('BE', 'BU')}
    for c0 in range(0, U.size, a.chunk):
        c1 = min(U.size, c0 + a.chunk)
        n = c1 - c0
        Dm = np.zeros((T, n, 6))
        M = np.zeros((T, n, 1))
        sel = []
        for f in frames:
            pu = pos[f]
            m = (pu >= c0) & (pu < c1)
            loc = pu[m] - c0
            sel.append((np.nonzero(m)[0], loc))
            Dm[f, loc, :3] = src[f][1][m]
            Dm[f, loc, 3:] = src[f][2][m]
            M[f, loc, 0] = 1.0
        for v in variants:
            kind, d = kinds[v]
            if kind == 'F1':
                continue
            V = solve(Dm, M, d, kind)
            if kind == 'BP':
                for f in frames:
                    rows, loc = sel[f]
                    out_sel[v][f][0][rows] = V[f, loc, :3]
                    out_sel[v][f][1][rows] = V[f, loc, 3:]
            else:
                out_all[v][:, c0:c1] = V.astype(np.float16)
    meta = json.loads((fit / 'run.json').read_text())
    src_vals = int(sum(s[0].size for s in src) * 6)
    stats = {}
    for v in variants:
        kind, d = kinds[v]
        od = ROOT / a.out_root / v
        od.mkdir(parents=True, exist_ok=True)
        if (od / 'complete.json').exists():
            raise FileExistsError(f'completed variant preserved: {od}')
        ex = nx = ec = nc = 0.0
        for f in frames:
            idx, sx, sc = src[f]
            if kind == 'F1':
                vx, vc = sx.astype(np.float16).astype(np.float32), sc.astype(np.float16).astype(np.float32)
                widx, ox, oc = idx, vx, vc
            elif kind == 'BP':
                vx, vc = out_sel[v][f]
                widx = idx
                ox, oc = vx, vc
            else:
                full = out_all[v][f].astype(np.float32)
                widx = U
                ox, oc = full[:, :3], full[:, 3:]
                vx, vc = ox[pos[f]], oc[pos[f]]
            ex += float(((vx - sx) ** 2).sum()); nx += float((sx ** 2).sum())
            ec += float(((vc - sc) ** 2).sum()); nc += float((sc ** 2).sum())
            tmp = od / f'f{f:04d}.tmp.npz'
            np.savez(tmp, idx=widx.astype(np.int32), dxyz=ox.astype(np.float16), dsh0=oc.astype(np.float16))
            tmp.replace(od / f'f{f:04d}.npz')
        m = bases[d].shape[1] if kind != 'F1' else T
        stored = int(U.size * m * 6) if kind != 'F1' else src_vals
        stats[v] = dict(kind=kind, knot_spacing=d, n_basis=int(m), stored_values=stored, source_values=src_vals,
                        stored_ratio=stored / src_vals, membership_bitmap_bytes=int(T * U.size / 8) if kind == 'BP' else 0,
                        stored_fp16_mib=((stored * 2 + (T * U.size / 8 if kind == 'BP' else 0) + U.size * 4) if kind != 'F1'
                                         else (src_vals * 2 + sum(s[0].size for s in src) * 4)) / 2 ** 20,
                        source_fp32_mib=(src_vals * 4 + sum(s[0].size for s in src) * 4) / 2 ** 20,
                        rel_change_dxyz_on_selected=(ex / max(nx, 1e-30)) ** 0.5,
                        rel_change_dsh0_on_selected=(ec / max(nc, 1e-30)) ** 0.5)
        rec = dict(meta)
        rec['variant'] = dict(name=v, source_fit=a.fit, doc=__doc__, stats=stats[v], union_gaussians=int(U.size),
                              absent_weight=w0.get(kind), eps=a.eps)
        (od / 'run.json').write_text(json.dumps(rec, indent=1))
        (od / 'complete.json').write_text(json.dumps(dict(frames=T, seconds=time.time() - t0), indent=1))
    summary = dict(fit=a.fit, union=int(U.size), frames=T, mean_selected=float(np.mean([s[0].size for s in src])), stats=stats,
                   seconds=time.time() - t0)
    (ROOT / a.out_root / 'variants_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
