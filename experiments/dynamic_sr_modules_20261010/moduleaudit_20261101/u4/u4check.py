"""Acceptance check of the U4 40-step smoke (Claude 4dsr-4e, 2026-11-01). Stdlib only; reads logs, no GPU.
Step 1 of the co-training run uses the parent checkpoint and the first schedule row, exactly as the single-student control does, and
the consistency term only enters through the separate C_term. So student A's and B's step-1 LR_L1 / SR_L1 / LR_term / SR_term must
equal those of the same-card single-student control's trajectory.jsonl line 1 (same card, same parent, same batch) up to --rtol.
Also requires: cotrain.json assertions (initial_parameters_equal, disjoint_student_storage), calls == backward_calls == stop, both
complete.json present, keys of step 1 identical to the control's. Exit 1 with a JSON report on any failure.
  u4check.py --run DIR_A --run-b DIR_B --control CONTROL_DIR --stop 40 [--rtol 1e-5] --out REPORT.json"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--run-b', required=True)
ap.add_argument('--control', required=True)
ap.add_argument('--rtol', type=float, default=1e-5)
ap.add_argument('--stop', type=int, required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[4]
A, B, C = ROOT / a.run, ROOT / a.run_b, ROOT / a.control
problems, rep = [], {}


def close(x, y):
    return abs(x - y) <= a.rtol * max(abs(x), abs(y), 1e-12)


ctl = json.loads((C / 'trajectory.jsonl').read_text().splitlines()[0])
first = json.loads((A / 'cotrain.jsonl').read_text().splitlines()[0])
rep['control_step1'] = {k: ctl[k] for k in ('cursor', 'LR_L1', 'SR_L1', 'LR_term', 'SR_term', 'keys')}
for key in ('a', 'b'):
    s = first['students'][key]
    rep[f'{key}_step1'] = {k: s[k] for k in ('cursor', 'LR_L1', 'SR_L1', 'LR_term', 'SR_term', 'keys')}
    if s['keys'] != ctl['keys']:
        problems.append(f'{key}: step-1 keys {s["keys"]} != control {ctl["keys"]}')
    for k in ('LR_L1', 'SR_L1'):
        if len(s[k]) != len(ctl[k]) or not all(close(x, y) for x, y in zip(s[k], ctl[k])):
            problems.append(f'{key}: {k} {s[k]} != control {ctl[k]}')
    for k in ('LR_term', 'SR_term'):
        if not close(s[k], ctl[k]):
            problems.append(f'{key}: {k} {s[k]} != control {ctl[k]}')
for d, name in ((A, 'a'), (B, 'b')):
    if not (d / 'complete.json').exists():
        problems.append(f'{name}: complete.json missing')
rec = json.loads((A / 'cotrain.json').read_text())
rep['cotrain_record_keys'] = sorted(rec.keys())
flat = json.dumps(rec)
for flag in ('initial_parameters_equal', 'disjoint_student_storage'):
    if f'"{flag}": true' not in flat:
        problems.append(f'cotrain.json: {flag} not true')
calls, back, stop = rec.get('calls'), rec.get('backward_calls'), a.stop
rep.update(calls=calls, backward_calls=back, stop=stop)
if not (calls == back == stop and stop):
    problems.append(f'calls/backward_calls/stop = {calls}/{back}/{stop}')
rep['problems'] = problems
rep['passed'] = not problems
Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
(ROOT / a.out).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep)[:3000], flush=True)
sys.exit(0 if not problems else 1)
