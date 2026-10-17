"""Sequential arm driver for appearance_refit.py (Claude 4dsr-e9, 2026-10-17). One training process at a time (PRO6000
rule); skips arms whose eval summary exists; does not start an arm whose estimated end passes the deadline (unix s).

  run_arms.py ARMS.json [DEADLINE_UNIX]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    arms = json.loads(Path(sys.argv[1]).read_text())
    deadline = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    est, failed = 420.0, []
    for arm in arms:
        if (ROOT / arm['eval_out'] / 'summary.json').exists():
            print(json.dumps(dict(arm=arm['arm'], skipped='eval exists')), flush=True)
            continue
        if time.time() + est > deadline:
            print(json.dumps(dict(stopped_before=arm['arm'], estimate_s=est)), flush=True)
            break
        t = time.time()
        r = subprocess.run([sys.executable, '-u', str(HERE / 'appearance_refit.py')] + arm['argv'], cwd=ROOT)
        dur = time.time() - t
        est = max(dur, 0.5 * est + 0.5 * dur)
        print(json.dumps(dict(arm=arm['arm'], rc=r.returncode, seconds=round(dur, 1))), flush=True)
        if r.returncode:
            failed.append(arm['arm'])
    print(json.dumps(dict(done=True, failed=failed)), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
