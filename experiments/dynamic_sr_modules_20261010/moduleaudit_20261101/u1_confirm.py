"""U1 cam00 confirmation, read ONCE after the held-out selection (Claude 4dsr-4e, 2026-11-01; prereg_U1.md). Requires the selection
file of u1_select.py; refuses to run without it. Compares, in the same eval_resid pass, the selected variant with the cell's v3:
  PSNR >= v3 - 0.02,  LPIPS <= v3 + 0.001,  tdiff_psnr >= v3 - 0.05,  static_psnr >= v3 - 0.01   (cam00, 300 frames).
All other variants' cam00 values are recorded as development data only. Writes OUT with the read time.
  u1_confirm.py --selection SEL.json --eval EVAL_DIR --v3-label PC|SMX1 --out OUT.json"""
import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--selection', required=True)
ap.add_argument('--eval', required=True)
ap.add_argument('--v3-label', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[3]
sel = json.loads((ROOT / a.selection).read_text())
read_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
summ = json.loads((ROOT / a.eval / 'summary.json').read_text())['summary']
temp = json.loads((ROOT / a.eval / 'temporal.json').read_text())['summary']
KEYS = ('psnr', 'ssim', 'lpips', 'dyn_psnr', 'static_psnr', 'low_psnr', 'mid_psnr', 'high_psnr')
table = {k: dict({m: v.get(m) for m in KEYS}, tdiff_psnr=temp.get(k, {}).get('tdiff_psnr'),
                 tdiff_dyn_psnr=temp.get(k, {}).get('tdiff_dyn_psnr'),
                 temporal_energy_ratio_dyn=temp.get(k, {}).get('temporal_energy_ratio_dyn')) for k, v in summ.items()}
v3 = table[a.v3_label]
deltas = {k: {m: (r[m] - v3[m]) if (r.get(m) is not None and v3.get(m) is not None) else None
              for m in ('psnr', 'ssim', 'lpips', 'dyn_psnr', 'static_psnr', 'high_psnr', 'tdiff_psnr')} for k, r in table.items()}
choice = sel['choice']
verdict = None
if choice != 'keep_v3':
    d = deltas[choice]
    checks = dict(psnr=d['psnr'] >= -0.02, lpips=d['lpips'] <= 0.001, tdiff=d['tdiff_psnr'] >= -0.05, static=d['static_psnr'] >= -0.01)
    verdict = dict(checks=checks, passed=all(checks.values()))
rec = dict(rule=__doc__, selection=a.selection, selection_decided_at=sel['decided_at'], cam00_read_at=read_at, choice=choice,
           v3_label=a.v3_label, verdict=verdict, deltas_vs_v3=deltas, table=table, eval=a.eval)
out = ROOT / a.out
if out.exists():
    raise SystemExit(f'confirmation already written: {a.out}')
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(dict(choice=choice, verdict=verdict, deltas_selected=deltas.get(choice))), flush=True)
