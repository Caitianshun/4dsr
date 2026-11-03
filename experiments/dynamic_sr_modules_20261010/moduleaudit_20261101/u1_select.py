"""U1 selection from the held-out camera only (Claude 4dsr-4e, 2026-11-01; tmp/claude_moduleaudit_20261101/prereg_U1.md + addendum A).
Reads ONLY the LOCO read-out (loco.json of loco_eval.py at cam10, LR space) and the variants' stored bytes (variants_summary.json of
the main-table projections; storage is not an image metric). Never opens a cam00 evaluation.
Rule: reference = the cell's v3 choice (v2 or SMX1); a variant V in {BP4, BE4, BE8} is admissible if
  psnr(V) - psnr(ref) >= -0.01 dB  and  tdiff_psnr(V) - tdiff_psnr(ref) >= -0.05 dB  (whole image, LR space, 300 frames);
among admissible variants choose the smallest stored fp16 bytes; if another admissible variant is within 5 % of it and is BP, prefer
BP (support set unchanged). No admissible variant -> 'keep_v3' (U1 fails for the cell). Writes OUT with the decision and a timestamp.
  u1_select.py --loco LOCO_DIR --ref v2|SMX1 --main-variants VARIANTS_SUMMARY_JSON --out OUT.json"""
import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--loco', required=True)
ap.add_argument('--ref', required=True, choices=('v2', 'SMX1'))
ap.add_argument('--main-variants', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[3]
loco = json.loads((ROOT / a.loco / 'loco.json').read_text())['summary']
store = json.loads((ROOT / a.main_variants).read_text())['stats']
ref = loco[a.ref]
rows = {}
for v in ('BP4', 'BE4', 'BE8'):
    if v not in loco or v not in store:
        rows[v] = dict(missing=True)
        continue
    dp = loco[v]['psnr'] - ref['psnr']
    dt = loco[v]['tdiff_psnr'] - ref['tdiff_psnr']
    rows[v] = dict(d_psnr=dp, d_tdiff=dt, d_dyn=loco[v]['dyn_psnr'] - ref['dyn_psnr'], admissible=bool(dp >= -0.01 and dt >= -0.05),
                   stored_fp16_mib=store[v]['stored_fp16_mib'])
adm = [v for v, r in rows.items() if r.get('admissible')]
choice = 'keep_v3'
if adm:
    best = min(adm, key=lambda v: rows[v]['stored_fp16_mib'])
    near_bp = [v for v in adm if v.startswith('BP') and rows[v]['stored_fp16_mib'] <= 1.05 * rows[best]['stored_fp16_mib']]
    choice = near_bp[0] if near_bp and not best.startswith('BP') else best
rec = dict(rule=__doc__, loco=a.loco, ref=a.ref, ref_values=ref, rows=rows, choice=choice,
           decided_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), cam00_read=False)
(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
if (ROOT / a.out).exists():
    raise SystemExit(f'selection already written: {a.out}')
(ROOT / a.out).write_text(json.dumps(rec, indent=1))
print(json.dumps(dict(choice=choice, rows=rows)), flush=True)
