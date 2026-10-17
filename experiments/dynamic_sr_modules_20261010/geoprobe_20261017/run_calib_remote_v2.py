"""Remote driver for calibsr_v2.py arms + group_eval.py (Claude, session 4dsr-e9, 2026-10-17). v2 of run_calib_remote.py:
arm spec NAME=CALIB[:WARP_LR[:SH_CAP[:ROUTE]]] (SH_CAP "-" = none).

Runs the SR arms one after another (one training process at a time), then scores every finished checkpoint on one view with
experiments/dynamic_sr_cull_retrain_20261009/selfens_20261016/group_eval.py (same scored quantities as `cullretrain eval`:
all 300 frames, post-hoc culling, TF32 off). Arms whose complete.json exists are skipped; an arm that fails does not stop the
others. All arms: area operator, lambda_SR 0 (pure multi-view; zero teachers served), D-SSIM 0.2, --prune-gate none,
--affine, srfix --mode fix -- the 4dsr-7c gate recipe of areaL0 -- plus the --calib switch.

  run_calib_remote.py --parent CK --baseline-sr DIR --teacher-zeros H W --out-root DIR --tag T --view cam00
                      --arm NAME=CALIB[:WARP_LR[:SH_CAP[:ROUTE]]] [...]
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
    p.add_argument('--parent', required=True)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--teacher-zeros', type=int, nargs=2, required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    done, failed = [], []
    for spec in a.arm:
        name, rest = spec.split('=', 1)
        parts = rest.split(':')
        calib, lr = parts[0], (parts[1] if len(parts) > 1 else '')
        cap = parts[2] if len(parts) > 2 and parts[2] != '-' else None
        route = parts[3] if len(parts) > 3 else 'all'
        label = f'{a.tag}_{name}'
        out = root / 'sr' / label
        if (out / 'complete.json').exists():
            done.append((name, out))
            continue
        cmd = [sys.executable, '-u', str(HERE / 'calibsr_v2.py'), '--lr-operator', 'area', '--calib', calib,
               '--warp-lr', lr or '2e-3', '--teacher-zeros', str(a.teacher_zeros[0]), str(a.teacher_zeros[1]), '--route-teacher', route]
        cmd += (['--sh-cap', cap] if cap is not None else []) + ['--',
               '--mode', 'fix', 'cullretrain', 'sr', '--parent-checkpoint', a.parent, '--baseline-sr', a.baseline_sr,
               '--arm', label, '--prune-gate', 'none', '--affine', '--lambda-sr', '0', '--dssim', '0.2',
               '--out', str(out.relative_to(ROOT))]
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(arm=label, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        (done if r.returncode == 0 else failed).append((name, out))
    ok = [(n, o) for n, o in done if (o / 'checkpoint_06000.pt').exists()]
    if ok:
        cmd = [sys.executable, '-u', str(GROUP_EVAL)]
        for n, o in ok:
            cmd += ['--checkpoint', f'{n}={(o / "checkpoint_06000.pt").relative_to(ROOT)}']
        cmd += ['--view', a.view, '--out', str((root / 'eval' / f'{a.tag}_{a.view}').relative_to(ROOT))]
        r = subprocess.run(cmd, cwd=ROOT)
        print(json.dumps(dict(group_eval_rc=r.returncode)), flush=True)
        if r.returncode:
            failed.append(('group_eval', None))
    print(json.dumps(dict(done=[n for n, _ in done], failed=[n for n, _ in failed])), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
