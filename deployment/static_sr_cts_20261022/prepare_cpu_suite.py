"""Register all six scenes, then run CPU camera-order planning with terminal receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--prepared', default='output/static_sr_wu_20261022/cts_prepared')
    parser.add_argument('--plans', default='output/static_sr_wu_20261022/cts_cpu_prior_plans')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    record = dict(status='running', stages=[], GPU_calls=0)
    try:
        for name, extra in [('register', []), ('plan', ['--workers', '4'])]:
            command = [sys.executable, str(ROOT/'experiments/static_sr_wu_20261022/prepare_suite.py'),
                name, '--prepared', str(ROOT/args.prepared), '--plans', str(ROOT/args.plans), *extra]
            begin = time.monotonic()
            with (out/(name+'.log')).open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record['stages'].append(dict(name=name, returncode=result.returncode, seconds=time.monotonic()-begin))
            if result.returncode:
                raise RuntimeError(name+' failed; inspect its log')
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (out/'status.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__': main()
