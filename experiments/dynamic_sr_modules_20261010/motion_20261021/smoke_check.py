"""Smoke check of a short modkit13.py run (Claude 4dsr-25, 2026-10-21): every switched-on term must have fired with finite values.
  smoke_check.py OUTDIR [--need flow app tf]"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
out = ROOT / sys.argv[1]
need = sys.argv[sys.argv.index('--need') + 1:] if '--need' in sys.argv else ['flow', 'app', 'tf']
rec = json.loads((out / 'modkit13.json').read_text())
st, terms = rec['stats'], rec['terms']
print(json.dumps(dict(stats=st, terms=terms), indent=1))
trace = (out / 'modkit13_trace.jsonl')
if trace.exists():
    for line in trace.read_text().splitlines()[:5]:
        print(line[:600])
ok = True
if 'flow' in need:
    ok &= st.get('flow_steps', 0) > 0 and terms.get('flow', {}).get('first500') is not None and math.isfinite(terms['flow']['first500'])
if 'app' in need:
    ok &= st.get('app_steps', 0) > 0
if 'tf' in need:
    ok &= st.get('tf_reads', 0) > 0
ok &= (out / 'complete.json').exists()
print('SMOKE', 'OK' if ok else 'FAILED', flush=True)
sys.exit(0 if ok else 1)
