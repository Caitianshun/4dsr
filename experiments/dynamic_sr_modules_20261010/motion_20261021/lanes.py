"""Parallel lanes inside ONE training-service job (Claude 4dsr-25, 2026-10-21; after 4dsr-5d's par_abort.py). Each lane file is run
line by line:
  WAIT <path>            block until the file exists (path relative to the project root); stops the lane if ABORT appears
  TOUCH <path>           write a marker file
  CRITICAL <script> ...  run a project script; on failure write the ABORT file (all lanes stop at their next line)
  <script> ...           run a project script (python -u, same interpreter); a failure is logged and the lane continues
Output lines are prefixed with the lane name. Exit code 1 if anything failed. 3090: at most 2 lanes (2 training processes).
  lanes.py --abort FILE LANE_A.txt [LANE_B.txt]"""
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
        if ABORT.exists():
            say(lane, 'ABORT present, lane stops')
            return
        if line.startswith('WAIT '):
            target = ROOT / line[5:].strip()
            say(lane, f'waiting for {target.relative_to(ROOT)}')
            while not target.exists():
                if ABORT.exists():
                    say(lane, 'ABORT present, lane stops')
                    return
                time.sleep(15)
            continue
        if line.startswith('TOUCH '):
            target = ROOT / line[6:].strip()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
            continue
        critical = line.startswith('CRITICAL ')
        parts = shlex.split(line[9:] if critical else line)
        say(lane, '>> ' + line)
        t0 = time.time()
        p = subprocess.Popen([sys.executable, '-u', str(ROOT / parts[0])] + parts[1:], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for out in p.stdout:
            say(lane, out.rstrip('\n'))
        rc = p.wait()
        say(lane, f'rc={rc} after {time.time() - t0:.0f}s: {line}')
        if rc != 0:
            failed.append((lane, line))
            if critical:
                ABORT.parent.mkdir(parents=True, exist_ok=True)
                ABORT.write_text(f'lane {lane} failed: {line}\n')
                return


if len(sys.argv) < 4 or sys.argv[1] != '--abort':
    sys.exit(__doc__)
ABORT = ROOT / sys.argv[2]
threads = [threading.Thread(target=run_lane, args=(chr(65 + i), f)) for i, f in enumerate(sys.argv[3:])]
for t in threads:
    t.start()
for t in threads:
    t.join()
if failed:
    print('failed:', failed, flush=True)
    sys.exit(1)
