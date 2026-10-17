"""Remote driver for appearance_refit_v2.py arms + group_eval.py (Claude, session 4dsr-e9, 2026-10-17).

Runs the refit arms one after another (one training process at a time), then scores every finished refit checkpoint together
with the listed reference checkpoints (e.g. the unmodified sources) on one view with group_eval.py (same scored quantities as
`cullretrain eval`), so refits and their sources are compared on the same host and code. Arms whose complete.json exists are
skipped; a failing arm does not stop the others.

  run_refit_remote.py --baseline-sr DIR --out-root DIR --tag T --view cam00 [--ref LABEL=CK ...]
                      --arm NAME=SOURCE_CK,INIT,SUPERVISION,SH_DEGREE,STEPS,LR_DC [...]
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
GROUP_EVAL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009/selfens_20261016/group_eval.py'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--ref', action='append', default=[])
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    done, failed = [], []
    for spec in a.arm:
        name, rest = spec.split('=', 1)
        source, init, sup, deg, steps, lr = rest.split(',')
        label = f'{a.tag}_{name}'
        out = root / 'refit' / label
        ck = out / f'checkpoint_{int(steps):05d}.pt'
        if (out / 'complete.json').exists():
            done.append((name, ck))
            continue
        cmd = [sys.executable, '-u', str(HERE / 'appearance_refit_v2.py'), '--source', source, '--baseline-sr', a.baseline_sr,
               '--supervision', sup, '--operator', 'area', '--init', init, '--sh-degree', deg, '--steps', steps, '--lr-dc', lr,
               '--out', str(out.relative_to(ROOT)), '--arm', label]
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(arm=label, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        (done if r.returncode == 0 and ck.exists() else failed).append((name, ck))
    entries = [f'{n}={c.relative_to(ROOT)}' for n, c in done] + a.ref
    if entries:
        cmd = [sys.executable, '-u', str(GROUP_EVAL)]
        for e in entries:
            cmd += ['--checkpoint', e]
        cmd += ['--view', a.view, '--out', str((root / 'eval' / f'{a.tag}_{a.view}').relative_to(ROOT))]
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(group_eval_rc=r.returncode)), flush=True)
        if r.returncode:
            failed.append(('group_eval', None))
    print(json.dumps(dict(done=[n for n, _ in done], failed=[n for n, _ in failed])), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
