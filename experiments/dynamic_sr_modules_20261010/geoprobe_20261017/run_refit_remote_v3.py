"""Remote driver for appearance_refit_v2.py arms + group_eval_bands.py (Claude, session 4dsr-e9, 2026-10-17).
v3 of run_refit_remote.py (v1/v2 byte-frozen on the remotes): identical to v2 except that the scoring uses group_eval_bands.py
(group_eval.py plus FFT band PSNR, same directory) and --eval-tag names the evaluation directory.
v2: optional 7th arm field PLAN_FROM (checkpoint whose plan, i.e. supervision manifest, is used -- needed when the geometry
source was trained on another manifest, e.g. a clean-protocol HRref).

Runs the refit arms one after another (one training process at a time), then scores every finished refit checkpoint together
with the listed reference checkpoints (e.g. the unmodified sources) on one view with group_eval.py (same scored quantities as
`cullretrain eval`), so refits and their sources are compared on the same host and code. Arms whose complete.json exists are
skipped; a failing arm does not stop the others.

  run_refit_remote.py --baseline-sr DIR --out-root DIR --tag T --view cam00 [--ref LABEL=CK ...]
                      --arm NAME=SOURCE_CK,INIT,SUPERVISION,SH_DEGREE,STEPS,LR_DC[,PLAN_FROM] [...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GROUP_EVAL = HERE / 'group_eval_bands.py'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--ref', action='append', default=[])
    p.add_argument('--eval-tag')
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    done, failed = [], []
    for spec in a.arm:
        name, rest = spec.split('=', 1)
        fields = rest.split(',')
        source, init, sup, deg, steps, lr = fields[:6]
        plan_from = fields[6] if len(fields) > 6 else None
        label = f'{a.tag}_{name}'
        out = root / 'refit' / label
        ck = out / f'checkpoint_{int(steps):05d}.pt'
        if (out / 'complete.json').exists():
            done.append((name, ck))
            continue
        cmd = [sys.executable, '-u', str(HERE / 'appearance_refit_v2.py'), '--source', source, '--baseline-sr', a.baseline_sr,
               '--supervision', sup, '--operator', 'area', '--init', init, '--sh-degree', deg, '--steps', steps, '--lr-dc', lr,
               '--out', str(out.relative_to(ROOT)), '--arm', label] + (['--plan-from', plan_from] if plan_from else [])
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(arm=label, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        (done if r.returncode == 0 and ck.exists() else failed).append((name, ck))
    entries = [f'{n}={c.relative_to(ROOT)}' for n, c in done] + a.ref
    if entries:
        cmd = [sys.executable, '-u', str(GROUP_EVAL)]
        for e in entries:
            cmd += ['--checkpoint', e]
        cmd += ['--view', a.view, '--out', str((root / 'eval' / f'{a.eval_tag or a.tag}_{a.view}').relative_to(ROOT))]
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(group_eval_rc=r.returncode)), flush=True)
        if r.returncode:
            failed.append(('group_eval', None))
    print(json.dumps(dict(done=[n for n, _ in done], failed=[n for n, _ in failed])), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
