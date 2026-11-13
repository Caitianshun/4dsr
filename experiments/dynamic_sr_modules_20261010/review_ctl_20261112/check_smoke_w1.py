"""Smoke check of a 1x1 camera displacement field run (Claude 4dsr-a3, 2026-11-12): the run completed, modkit2.json records wf_grid 1,
and every training camera has a finite field RMS.  check_smoke_w1.py RUN_DIR"""
import json
import math
import sys
from pathlib import Path

d = Path(sys.argv[1])
if not d.is_absolute():
    d = Path(__file__).resolve().parents[3] / d
assert (d / 'complete.json').exists(), 'run did not complete'
m = json.loads((d / 'modkit2.json').read_text())
assert m['switches']['wf_grid'] == 1, m['switches']
rms = m['warp_field_rms_px']
assert rms and all(math.isfinite(v) for v in rms.values()), rms
print(json.dumps(dict(ok=True, cameras=len(rms), max_rms_px=max(rms.values()))))
