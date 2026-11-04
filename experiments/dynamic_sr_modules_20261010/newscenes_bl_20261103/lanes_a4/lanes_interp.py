"""Lane runner with a per-line interpreter (Claude 4dsr-a4, 2026-11-03), for the CTS CEM lanes of make_lanes_a4.py.

Same line format as static_sr_v2_20261026/sv2_lanes.py, plus an optional '@<absolute interpreter>' prefix:
  @/abs/python <script> <args>   run <script> (path relative to the project root) with that interpreter (python -u)
  <script> <args>                run it with this runner's interpreter
The lanes given on the command line run concurrently, one thread each. A lane stops at its first failing line (the
'set -e' of the .sh it was derived from); the other lanes continue. Output lines are prefixed with the lane letter.
Exit code 1 if any line failed.
  lanes_interp.py LANE_A.txt [LANE_B.txt ...]"""
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
lock = threading.Lock()
failed = []


def say(lane, text):
    with lock:
        print(f'[{lane}] {text}', flush=True)


def run_lane(lane, listfile):
    for line in Path(listfile).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = shlex.split(line)
        interp = sys.executable
        if parts[0].startswith('@'):
            interp, parts = parts[0][1:], parts[1:]
        say(lane, '>> ' + line)
        t0 = time.time()
        p = subprocess.Popen([interp, '-u', str(ROOT / parts[0])] + parts[1:], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for out in p.stdout:
            say(lane, out.rstrip('\n'))
        rc = p.wait()
        say(lane, f'rc={rc} after {time.time() - t0:.0f}s: {line}')
        if rc != 0:
            with lock:
                failed.append((lane, line))
            say(lane, 'lane stops at its first failure')
            return


if len(sys.argv) < 2:
    sys.exit(__doc__)
threads = [threading.Thread(target=run_lane, args=(chr(65 + i), f)) for i, f in enumerate(sys.argv[1:])]
for t in threads:
    t.start()
for t in threads:
    t.join()
if failed:
    print('failed:', failed, flush=True)
    sys.exit(1)
