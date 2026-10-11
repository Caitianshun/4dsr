"""Run a dependency graph of cullretrain.py tasks as concurrent subprocesses on the job's single GPU.

tasks.json: [{"name": str, "argv": [script arguments], "script": optional, default cullretrain.py, "deps": [names], "done": "path relative to the
project whose existence means the task already completed (it is then skipped, never re-run)"}].
Several pipelines (one per GPU job) may share one task list: a task is claimed by atomically creating
<claims>/<name>; a task claimed by another pipeline is awaited through its done file or <name>.failed.
A failed task marks its dependents 'skipped_dependency_failed'; independent tasks continue.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import rt

HERE = Path(__file__).resolve().parent
TERMINAL = {'succeeded', 'already_complete', 'failed', 'skipped_dependency_failed', 'done_elsewhere', 'failed_elsewhere'}
GOOD = {'succeeded', 'already_complete', 'done_elsewhere'}
BAD = {'failed', 'skipped_dependency_failed', 'failed_elsewhere'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--claims', required=True)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    out, claims = Path(a.out), Path(a.claims)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(a.tasks).read_text())
    by_name = {t['name']: t for t in tasks}
    if len(by_name) != len(tasks):
        raise SystemExit('duplicate task names')
    state = {t['name']: dict(status='pending', deps=t.get('deps', [])) for t in tasks}
    done = lambda name: bool(by_name[name].get('done')) and (rt.ROOT / by_name[name]['done']).exists()
    for name in state:
        if done(name):
            state[name]['status'] = 'already_complete'
    running = {}
    started = time.time()

    def persist():
        rt.write_json(out / 'status.json', dict(updated_unix=time.time(), elapsed=time.time() - started,
                                                workers=a.workers, host_pid=os.getpid(), tasks=state))

    persist()
    while True:
        changed = False
        for name, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            running.pop(name)
            proc.log.close()
            state[name].update(status='succeeded' if code == 0 else 'failed', exit_code=code,
                               end_unix=time.time(), seconds=time.time() - state[name]['start_unix'])
            if code != 0:
                (claims / f'{name}.failed').write_text(json.dumps(dict(code=code, pid=os.getpid())))
            print(json.dumps(dict(event='task_end', name=name, code=code, seconds=round(state[name]['seconds']))), flush=True)
            changed = True
        for name, s in state.items():
            if s['status'] == 'claimed_elsewhere':
                if done(name):
                    s['status'] = 'done_elsewhere'; changed = True
                elif (claims / f'{name}.failed').exists():
                    s['status'] = 'failed_elsewhere'; changed = True
        for name, s in state.items():
            if s['status'] == 'pending' and any(state[d]['status'] in BAD for d in s['deps']):
                s['status'] = 'skipped_dependency_failed'; changed = True
        for t in tasks:
            if len(running) >= a.workers:
                break
            s = state[t['name']]
            if s['status'] != 'pending' or not all(state[d]['status'] in GOOD for d in s['deps']):
                continue
            try:
                fd = os.open(claims / t['name'], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            except FileExistsError:
                s['status'] = 'claimed_elsewhere'; changed = True
                continue
            os.write(fd, json.dumps(dict(pid=os.getpid(), out=str(out), time=time.time())).encode())
            os.close(fd)
            log = (out / 'logs' / f"{t['name']}.log").open('ab')
            argv = [sys.executable, '-u', str(HERE / t.get('script', 'cullretrain.py')), *t['argv']]
            proc = subprocess.Popen(argv, cwd=str(rt.ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4')))
            proc.log = log
            running[t['name']] = proc
            s.update(status='running', pid=proc.pid, start_unix=time.time(), argv=argv)
            print(json.dumps(dict(event='task_start', name=t['name'], pid=proc.pid)), flush=True)
            changed = True
        if changed:
            persist()
        waiting_elsewhere = any(s['status'] == 'claimed_elsewhere' for s in state.values())
        startable = any(s['status'] == 'pending' and all(state[d]['status'] in GOOD for d in s['deps']) for s in state.values())
        blocked_on_elsewhere = any(s['status'] == 'pending' and any(state[d]['status'] in ('claimed_elsewhere', 'running', 'pending') for d in s['deps'])
                                   for s in state.values())
        if not running and not startable and not (waiting_elsewhere and blocked_on_elsewhere):
            break
        time.sleep(2)
    persist()
    counts = {}
    for s in state.values():
        counts[s['status']] = counts.get(s['status'], 0) + 1
    print(json.dumps(dict(event='pipeline_end', counts=counts, elapsed=round(time.time() - started))), flush=True)
    return 0 if set(counts) <= GOOD | {'claimed_elsewhere'} else 1


if __name__ == '__main__':
    sys.exit(main())
