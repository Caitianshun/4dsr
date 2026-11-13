"""Queue worker for the 11-12 fill-in runs (Claude, 2026-11-12). Claims task files from QUEUE/pending (sorted by name, so a
numeric prefix sets the priority) by an atomic rename into QUEUE/claimed, runs their lines, and moves them to QUEUE/done or
QUEUE/failed. Two workers in one job give two concurrent processes on one card (3090 only, at most two; user 11-13: the PRO6000
runs one training process at a time). Line syntax (paths relative to the project root, same interpreter):
  CRITICAL <script> ...   a failure ends this task (moved to failed); the worker continues with the next task
  <script> ...            a failure is logged and the task continues
--seed SRC: on first start the task files of SRC (read-only for the training account) are copied into QUEUE/pending, under a lock
directory so that only one worker seeds. Exits when QUEUE/pending is empty or QUEUE/STOP exists.
  queue_worker.py --seed SRC --queue DIR --worker NAME"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument('--seed', required=True)
p.add_argument('--queue', required=True)
p.add_argument('--worker', required=True)
a = p.parse_args()
Q = ROOT / a.queue
Q.mkdir(parents=True, exist_ok=True)
try:
    (Q / 'seed.lock').mkdir()
    for sub in ('pending', 'claimed', 'done', 'failed'):
        (Q / sub).mkdir(exist_ok=True)
    for f in sorted((ROOT / a.seed).glob('*.txt')):
        (Q / 'pending' / f.name).write_text(f.read_text())
    (Q / 'seeded').write_text(a.worker + '\n')
except FileExistsError:
    for _ in range(120):
        if (Q / 'seeded').exists():
            break
        time.sleep(1)


def say(msg):
    print(f'[{a.worker}] {msg}', flush=True)


while not (Q / 'STOP').exists():
    task = None
    for f in sorted((Q / 'pending').glob('*.txt')):
        target = Q / 'claimed' / f'{f.stem}__{a.worker}.txt'
        try:
            os.rename(f, target)
        except OSError:
            continue
        task = target
        break
    if task is None:
        say('queue empty, worker exits')
        break
    say(f'claimed {task.name}')
    ok = True
    for line in task.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        critical = line.startswith('CRITICAL ')
        parts = shlex.split(line[9:] if critical else line)
        t = time.time()
        r = subprocess.run([sys.executable, '-u', str(ROOT / parts[0]), *parts[1:]], cwd=ROOT)
        say(f'rc={r.returncode} after {time.time() - t:.0f}s: {line[:300]}')
        if r.returncode and critical:
            ok = False
            break
    final = Q / ('done' if ok else 'failed') / task.name
    os.rename(task, final)
    say(f'{"done" if ok else "FAILED"} {task.name}')
