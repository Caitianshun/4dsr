"""Assigned-share variant of pipeline.py: run a fixed subset of a shared task graph on this job's GPU.

Same task format and claim protocol as pipeline.py, which keeps running unchanged on the other GPU.
Tasks carrying "owner" equal to --owner are this job's share: before scheduling, each one that is not
complete is pre-claimed by atomically creating <claims>/<name> with {"owner": ...}, so pipelines running
the full list see it as claimed elsewhere from the start. Tasks without "owner" are dependencies run
elsewhere and are only awaited (done file or <name>.failed). A share task whose claim already exists is
run only if the claim carries this owner and was never launched (a restart before launch); a claim of
another owner is awaited. If any own claim was launched but its task is incomplete, the pipeline refuses
to start (exit 2) without creating claims, so the earlier attempt is reviewed by hand rather than
propagated as a failure. One process per owner (exclusive lock file in the claims directory).
A failed or dependency-skipped share task writes <name>.failed so that other pipelines stop waiting;
dependency failures are propagated to a fixed point before the exit test.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import rt

HERE = Path(__file__).resolve().parent
GOOD = {'succeeded', 'already_complete', 'done_elsewhere'}
BAD = {'failed', 'skipped_dependency_failed', 'failed_elsewhere'}
AWAITED = {'claimed_elsewhere', 'external'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--claims', required=True)
    p.add_argument('--owner', required=True)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    out, claims = Path(a.out), Path(a.claims)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(a.tasks).read_text())
    by_name = {t['name']: t for t in tasks}
    if len(by_name) != len(tasks):
        raise SystemExit('duplicate task names')
    missing = {d for t in tasks for d in t.get('deps', []) if d not in by_name}
    if missing:
        raise SystemExit(f'dependencies not in the task list: {sorted(missing)}')
    own = {t['name'] for t in tasks if t.get('owner') == a.owner}
    lock = open(claims / f'.owner-{a.owner}.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'another pipeline_owned process holds owner {a.owner}')
    done = lambda name: bool(by_name[name].get('done')) and (rt.ROOT / by_name[name]['done']).exists()
    read_claim = lambda name: json.loads((claims / name).read_text() or '{}') if (claims / name).exists() else None
    stale = sorted(n for n in own if not done(n) and (read_claim(n) or {}).get('owner') == a.owner and (read_claim(n) or {}).get('launched'))
    if stale:
        print(json.dumps(dict(event='refused_stale_own_claims', names=stale)), flush=True)
        return 2
    state = {}
    for t in tasks:
        name = t['name']
        s = state[name] = dict(status='pending', deps=t.get('deps', []), share=name in own)
        if done(name):
            s['status'] = 'already_complete'
        elif name not in own:
            s['status'] = 'external'
        else:
            try:
                fd = os.open(claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            except FileExistsError:
                if (read_claim(name) or {}).get('owner') != a.owner:
                    s['status'] = 'claimed_elsewhere'
                continue
            os.write(fd, json.dumps(dict(owner=a.owner, pid=os.getpid(), out=str(out), time=time.time())).encode())
            os.close(fd)
    running = {}
    started = time.time()

    def persist():
        rt.write_json(out / 'status.json', dict(updated_unix=time.time(), elapsed=time.time() - started, owner=a.owner,
                                                workers=a.workers, host_pid=os.getpid(), tasks=state))

    def mark_failed(name, **info):
        (claims / f'{name}.failed').write_text(json.dumps(dict(owner=a.owner, pid=os.getpid(), **info)))

    persist()
    print(json.dumps(dict(event='pipeline_start', owner=a.owner, share=len(own),
                          counts={k: sum(s['status'] == k for s in state.values()) for k in {s['status'] for s in state.values()}})), flush=True)
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
                mark_failed(name, code=code)
            print(json.dumps(dict(event='task_end', name=name, code=code, seconds=round(state[name]['seconds']))), flush=True)
            changed = True
        for name, s in state.items():
            if s['status'] in AWAITED:
                if done(name):
                    s['status'] = 'done_elsewhere'; changed = True
                elif (claims / f'{name}.failed').exists():
                    s['status'] = 'failed_elsewhere'; changed = True
        propagated = True
        while propagated:
            propagated = False
            for name, s in state.items():
                if s['status'] == 'pending' and any(state[d]['status'] in BAD for d in s['deps']):
                    s['status'] = 'skipped_dependency_failed'; changed = propagated = True
                    mark_failed(name, code=None, skipped_dependency_failed=True)
        for t in tasks:
            if len(running) >= a.workers:
                break
            s = state[t['name']]
            if s['status'] != 'pending' or not all(state[d]['status'] in GOOD for d in s['deps']):
                continue
            claim = claims / t['name']
            info = json.loads(claim.read_text())
            if info.get('owner') != a.owner:
                raise RuntimeError(f"claim of {t['name']} no longer carries this owner: {info}")
            rt.write_json(claim, dict(info, launched=time.time(), launch_pid=os.getpid()))
            log = (out / 'logs' / f"{t['name']}.log").open('ab')
            argv = [sys.executable, '-u', str(HERE / 'cullretrain.py'), *t['argv']]
            proc = subprocess.Popen(argv, cwd=str(rt.ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4')))
            proc.log = log
            running[t['name']] = proc
            s.update(status='running', pid=proc.pid, start_unix=time.time(), argv=argv)
            print(json.dumps(dict(event='task_start', name=t['name'], pid=proc.pid)), flush=True)
            changed = True
        if changed:
            persist()
        startable = any(s['status'] == 'pending' and all(state[d]['status'] in GOOD for d in s['deps']) for s in state.values())
        blocked_on_elsewhere = any(s['status'] == 'pending' and any(state[d]['status'] in AWAITED | {'running', 'pending'} for d in s['deps'])
                                   for s in state.values())
        if not running and not startable and not blocked_on_elsewhere:
            break
        time.sleep(2)
    persist()
    counts = {}
    for s in state.values():
        counts[s['status']] = counts.get(s['status'], 0) + 1
    print(json.dumps(dict(event='pipeline_end', counts=counts, elapsed=round(time.time() - started))), flush=True)
    share_ok = all(state[n]['status'] in GOOD for n in own)
    return 0 if share_ok else 1


if __name__ == '__main__':
    sys.exit(main())
