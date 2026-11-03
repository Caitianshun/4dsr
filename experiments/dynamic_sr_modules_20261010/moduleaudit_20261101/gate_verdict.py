"""Mechanical verdict of the zero-training free-set gate screening (Claude 4dsr-4e, 2026-11-01; tmp/claude_moduleaudit_20261101/
prereg_M2gate.md + addendum A). Reads the four gate.json of pc_gate_diag.py; d = cam10 whole-image LR PSNR change against the parent,
dM = the same inside cam10's dynamic mask, k = PSNR change inside the fitting cameras' masks (closure). A gate G is 'promising' iff
  c1  loss cells (Cook07 x4, Coffee07 x2):  d(G) >= 0.5 d(full)
  c2  all four cells:                       k(G) >= 0.8 k(full)
  c3  gain cells (Flame07 x2, VRH07 x2):    d(G) >= 0.8 d(full)  and  dM(G) >= dM(full) - 0.05
  c4  ~50 % gates only, both loss cells:    d(G) - d(rand_drop50) >= 0.1
Addendum A sensitivity (reported, never decisive): c2 with 0.6 instead of 0.8, and c2 as k(full) - k(G) <= 0.05 dB.
  gate_verdict.py --gate-root DIR --out OUT.json"""
import argparse
import json
from pathlib import Path

LOSS = ('Cook07_x4', 'Coffee07_x2')
GAIN = ('Flame07_x2', 'VRHeadset07_x2')
GATES = ('agr_drop25', 'agr_drop50', 'agr_drop75', 'agrP_drop50', 'agrC_drop50', 'rmask_drop50', 'rho_near', 'rho_dropfar',
         'defo_moving')
HALF = ('agr_drop50', 'agrP_drop50', 'agrC_drop50', 'rmask_drop50')

ap = argparse.ArgumentParser()
ap.add_argument('--gate-root', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[3]
cells = {c: json.loads((ROOT / a.gate_root / f'{c}_cam10' / 'gate.json').read_text())['summary'] for c in LOSS + GAIN}


def d(c, arm):
    return cells[c][arm]['cam_d_psnr']['all']


def dm(c, arm):
    return cells[c][arm]['cam_d_psnr']['M']


def k(c, arm):
    return cells[c][arm]['closure_d_psnr']['dyn']


rows = {}
for g in GATES:
    r = dict(c1={c: d(c, g) >= 0.5 * d(c, 'full') for c in LOSS},
             c2={c: k(c, g) >= 0.8 * k(c, 'full') for c in LOSS + GAIN},
             c3={c: bool(d(c, g) >= 0.8 * d(c, 'full') and dm(c, g) >= dm(c, 'full') - 0.05) for c in GAIN},
             c4={c: d(c, g) - d(c, 'rand_drop50') >= 0.1 for c in LOSS} if g in HALF else None,
             c2_sens60={c: k(c, g) >= 0.6 * k(c, 'full') for c in LOSS + GAIN},
             c2_sensabs={c: k(c, 'full') - k(c, g) <= 0.05 for c in LOSS + GAIN},
             numbers={c: dict(d=d(c, g), d_full=d(c, 'full'), dM=dm(c, g), dM_full=dm(c, 'full'), k=k(c, g), k_full=k(c, 'full'),
                              k_ratio=k(c, g) / k(c, 'full'), d_minus_rand=d(c, g) - d(c, 'rand_drop50'),
                              kept_count=cells[c][g]['kept_count'], kept_mu2=cells[c][g]['kept_mu2'],
                              kept_c2=cells[c][g]['kept_c2']) for c in LOSS + GAIN})
    main = [all(r['c1'].values()), all(r['c2'].values()), all(r['c3'].values())] + ([all(r['c4'].values())] if r['c4'] else [])
    r['promising'] = all(main)
    r['promising_if_c2_sens60'] = all([all(r['c1'].values()), all(r['c2_sens60'].values()), all(r['c3'].values())]
                                      + ([all(r['c4'].values())] if r['c4'] else []))
    r['failed'] = [f'{name}:{c}' for name in ('c1', 'c2', 'c3', 'c4') if r[name] for c, ok in r[name].items() if not ok]
    rows[g] = r
rec = dict(rule=__doc__, gate_root=a.gate_root, gates=rows, any_promising=any(r['promising'] for r in rows.values()))
out = ROOT / a.out
if out.exists():
    raise SystemExit(f'verdict already written: {a.out}')
out.write_text(json.dumps(rec, indent=1))
for g, r in rows.items():
    print(f"{g:14s} promising={r['promising']!s:5s} (sens60 {r['promising_if_c2_sens60']!s:5s}) failed={r['failed']}")
print('any_promising', rec['any_promising'])
