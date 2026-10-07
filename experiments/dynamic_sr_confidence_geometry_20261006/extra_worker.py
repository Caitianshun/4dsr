"""Exit-event queue for frozen endpoint extras under the shared GPU1 lock.

Run with system Python3 (pidfd support). No busy polling, training or remote
mutation is performed here. A later invocation incrementally covers new arms.
"""
import argparse
import fcntl
import os
from pathlib import Path
import select
import subprocess
import time
import traceback
from cg_common import ROOT, HERE, OUT, read, write, bound, sha, entry

PYTHON = '/home/cai_tianshun/Project/4dgs/.venv/bin/python'


def wait_dependencies(pids):
    fds = []
    for pid in pids:
        if not pid:
            continue
        try:
            fds.append(os.pidfd_open(pid))
        except ProcessLookupError:
            pass
    while fds:
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            os.close(fd); fds.remove(fd)


def main(args):
    wait_dependencies(args.dependencies)
    protocol = read(OUT/'protocol.json'); tasks = [('U6000', bound(protocol['parent']))]
    for task in protocol['core_task_plan']:
        label = f"r{task['repeat']}_{task['method']}"; path = OUT/'runs'/label/'complete.json'
        if path.exists():
            tasks.append((label, bound(read(path)['checkpoint'])))
    if args.include_modules:
        for directory in sorted((OUT/'runs').glob('r*')):
            if (directory/'complete.json').exists() and directory.name not in {label for label, _ in tasks}:
                receipt = read(directory/'complete.json')
                if receipt.get('updates') == 6000:
                    tasks.append((directory.name, bound(receipt['checkpoint'])))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    statuses = []
    for label, checkpoint in tasks:
        out = OUT/'evaluation'/label/'extra'
        if (out/'complete.json').exists():
            assert read(out/'complete.json')['identity']['checkpoint_sha256'] == sha(checkpoint)
            statuses.append(dict(endpoint=label, status='already_completed')); continue
        with (OUT/'evaluation_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
            busy = [r for r in processes.splitlines() if uuid in r and '/opt/todesk/' not in r]
            assert not busy, ('Evaluation GPU busy outside shared lock', busy)
            write(OUT/'extra_worker_state.json', dict(status='evaluating', endpoint=label, checkpoint=entry(checkpoint), gpu_uuid=uuid, started_unix=time.time()))
            log = OUT/f'{label}_extra.log'
            with log.open('a') as f:
                subprocess.run([PYTHON, '-u', str(HERE/'evaluate_extra.py'), '--checkpoint', str(checkpoint), '--label', label, '--out', str(out)],
                               stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
        statuses.append(dict(endpoint=label, status='completed', receipt=entry(out/'complete.json')))
    subprocess.run(['python3', str(HERE/'summarize.py')], check=True, cwd=ROOT)
    write(OUT/'extra_worker_state.json', dict(status='completed_available_endpoints', endpoints=statuses,
                                            note='Missing core or conditional endpoints remain pending in quality_summary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dependencies', nargs='*', type=int, default=[])
    parser.add_argument('--include-modules', action='store_true'); args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        write(OUT/'extra_worker_state.json', dict(status='failed', traceback=traceback.format_exc())); raise
