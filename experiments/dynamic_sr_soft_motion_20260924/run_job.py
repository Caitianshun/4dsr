#!/usr/bin/env python3
"""One persistent-systemd job: train or reuse, then immediately queue evaluations.

This is an exit-driven subprocess chain, not a training-curve monitor. The
calling systemd service owns persistence. Every child runs in the activated
Python environment with an explicit full GPU UUID. No remote dispatch occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BRANCH = {'A': 'joint', 'B': 'ordinary_split', 'C': 'bound_split', 'S': 'soft_motion'}
ALLOWED_DISPLAY = {'/opt/todesk/bin/ToDesk_Session', '/usr/libexec/gnome-remote-desktop-daemon'}
GPU_PATTERN = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def parse_apps(text, gpu):
    rows = []
    for row in csv.reader(text.splitlines()):
        if not row or row[0].strip() != gpu:
            continue
        if len(row) < 4:
            raise RuntimeError(f'Malformed nvidia-smi process record: {row}')
        rows.append({'gpu_uuid': row[0].strip(), 'pid': int(row[1].strip()),
                     'process_name': row[2].strip(), 'used_memory_mib': row[3].strip()})
    return rows


def gpu_snapshot(gpu):
    commands = {
        'gpu': ['nvidia-smi', '--id=' + gpu,
                '--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
        'apps': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                 '--format=csv,noheader,nounits'],
    }
    result = {'queried_utc': stamp(), 'requested_gpu': gpu}
    for name, command in commands.items():
        process = subprocess.run(command, text=True, capture_output=True, timeout=20)
        result[name] = {'command': command, 'returncode': process.returncode,
                        'stdout': process.stdout, 'stderr': process.stderr}
    return result


def require_available(snapshot):
    if any(snapshot[name]['returncode'] != 0 for name in ('gpu', 'apps')):
        raise RuntimeError('GPU/process query failed; refusing an unverified device')
    gpu = snapshot['requested_gpu']
    gpu_rows = list(csv.reader(snapshot['gpu']['stdout'].splitlines()))
    if len(gpu_rows) != 1 or len(gpu_rows[0]) < 3 or gpu_rows[0][2].strip() != gpu:
        raise RuntimeError('The exact requested GPU UUID did not resolve uniquely')
    blocked = [row for row in parse_apps(snapshot['apps']['stdout'], gpu)
               if row['process_name'] not in ALLOWED_DISPLAY]
    if blocked:
        raise RuntimeError(f'GPU has other compute processes; refusing concurrent work: {blocked}')


@contextmanager
def lock_file(path, blocking=True):
    # Locks are advisory across this job runner; occupancy is rechecked after
    # acquisition to reject unrelated jobs that do not use this protocol.
    with Path(path).open('a+') as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Job:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.started = time.monotonic()
        self.child = None
        self.stage = 'initializing'
        self.state = {'status': 'running', 'method': args.method, 'pid': os.getpid(),
                      'started_utc': stamp(), 'train_gpu': args.train_gpu, 'eval_gpu': args.eval_gpu,
                      'out': str(args.out), 'runner': identity(__file__),
                      'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      'persistence': 'Caller systemd service; subprocess exits trigger the next stage directly'}
        write(self.out / 'status.json', self.state)

    def update(self, stage):
        self.stage = stage
        self.state.update(stage=stage, updated_utc=stamp(), elapsed_seconds=time.monotonic() - self.started)
        write(self.out / 'status.json', self.state)

    def terminated(self, signum, frame):
        # systemd normally signals the whole service cgroup. Explicitly forward
        # too, so direct invocation cannot leave an untracked GPU child behind.
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)
        raise RuntimeError(f'Runner received signal {signum}')

    def child_run(self, label, command, gpu):
        self.update(label)
        event_path = self.out / 'events' / (label + '.json')
        tick = time.monotonic()
        event = {'status': 'checking_gpu', 'command': command, 'gpu': gpu, 'started_utc': stamp(),
                 'log': str(self.out / (label + '.log')), 'entrypoint': identity(command[1])}
        try:
            event['gpu_before'] = gpu_snapshot(gpu)
            write(event_path, event)  # Preserve the snapshot even when refused.
            require_available(event['gpu_before'])
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='4', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
            with (self.out / (label + '.log')).open('w') as log:
                self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                event.update(status='running', pid=self.child.pid)
                write(event_path, event)
                rc = self.child.wait()
            self.child = None
            event.update(returncode=rc, status='completed' if rc == 0 else 'failed')
            if rc:
                raise RuntimeError(f'{label} exited {rc}; see {event["log"]}')
        except BaseException:
            event.update(status='failed', traceback=traceback.format_exc())
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
                event['returncode'] = self.child.returncode
            self.child = None
            raise
        finally:
            event.update(elapsed_seconds=time.monotonic() - tick, finished_utc=stamp())
            write(event_path, event)

    def validate_training(self, train_dir):
        config, done = read(train_dir / 'config.json'), read(train_dir / 'complete.json')
        if done.get('status') != 'completed' or done.get('parameter_updates') != 6000 or done.get('smoke', False):
            raise RuntimeError('Training receipt must describe a completed formal 6000-step run')
        expected = {'manifest_sha256': sha(self.args.manifest), 'parent_sha256': sha(self.args.parent),
                    'branch': BRANCH[self.args.method], 'steps': 6000, 'seed': 20260923, 'sr_weight': .1}
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(f'Training identity mismatch: {key}: {config.get(key)} != {value}')
        if config.get('prior_cameras', '').split(',') != self.args.prior_cameras.split(','):
            raise RuntimeError('Training teacher cameras differ from the requested protocol')
        if config.get('prior_subdir', 'sr_swinir_x4') != 'sr_swinir_x4':
            raise RuntimeError('Expected the frozen original SwinIR cache')
        if self.args.selection and config.get('selection_sha256') != sha(self.args.selection):
            raise RuntimeError('Saved training selection differs from the requested selection')
        checkpoint = train_dir / 'checkpoint_6000.pt'
        return {'status': 'completed_training_verified', 'method': self.args.method,
                'reused_training': self.args.reuse_train is not None, 'train_dir': str(train_dir),
                'config': identity(train_dir / 'config.json'), 'complete': identity(train_dir / 'complete.json'),
                'checkpoint': identity(checkpoint), 'draw_sha256': done.get('draw_sha256'),
                'parent': identity(self.args.parent), 'manifest': identity(self.args.manifest),
                'exposure': identity(train_dir / 'exposure.json'),
                'checkpoint_1200': identity(train_dir / 'checkpoint_1200.pt') if (train_dir / 'checkpoint_1200.pt').is_file() else None,
                'verified_utc': stamp()}

    def train(self):
        args = self.args
        if args.reuse_train:
            self.update('validating_reused_training')
            train_dir = args.reuse_train
        else:
            train_dir = self.out / 'train'
            script = {'A': ROOT / 'experiments/dynamic_sr_scene_residual_20260923/train.py',
                      'B': ROOT / 'experiments/dynamic_sr_motion_bound_20260923/train.py',
                      'S': HERE / 'train.py'}[args.method]
            command = [sys.executable, str(script), '--manifest', str(args.manifest), '--checkpoint', str(args.parent),
                       '--out', str(train_dir), '--branch', BRANCH[args.method], '--steps', '6000',
                       '--milestones', '1200,6000', '--seed', '20260923', '--sr-weight', '0.1',
                       '--prior-cameras', args.prior_cameras, '--prior-subdir', 'sr_swinir_x4']
            if args.method in ('B', 'S'):
                command += ['--selection', str(args.selection)]
            if args.method == 'S':
                command += ['--reference', str(args.reference), '--calibration', str(args.calibration)]
            self.update('acquiring_training_gpu')
            # Refuse overlapping training launches. Never carry this lease
            # into the evaluation queue, including when both UUIDs are equal.
            with lock_file(f'/tmp/4dsr_soft_gpu_{args.train_gpu}.lock', blocking=False):
                self.child_run('train', command, args.train_gpu)
        receipt = self.validate_training(train_dir)
        write(self.out / 'train_receipt.json', receipt)
        return train_dir, receipt

    def evaluate(self, train_dir, train_receipt):
        args = self.args
        required = [('train_fixed', 6000), ('dev', 6000), ('test', 6000)]
        if args.eval_dev_1200 and (train_dir / 'checkpoint_1200.pt').is_file():
            required.append(('dev', 1200))
        elif args.eval_dev_1200:
            write(self.out / 'events/eval_dev_1200.json', {'status': 'skipped', 'reason': 'Optional checkpoint1200 absent'})
        queue_started = time.monotonic()
        lock = f'/tmp/4dsr_soft_eval_{args.eval_gpu}.lock'
        self.update('waiting_for_evaluation_lock')
        queued = {'status': 'waiting', 'gpu': args.eval_gpu, 'lock_path': lock, 'queued_utc': stamp()}
        write(self.out / 'events/evaluation_queue.json', queued)
        evaluations = {}
        # flock sleeps in the kernel; no timer, GPU polling, or curve monitoring.
        with lock_file(lock):
            queued.update(status='acquired', acquired_utc=stamp(), wait_seconds=time.monotonic() - queue_started)
            write(self.out / 'events/evaluation_queue.json', queued)
            with lock_file(f'/tmp/4dsr_soft_gpu_{args.eval_gpu}.lock'):
                for split, step in required:
                    label = f'eval_{split}_{step}'
                    dest = self.out / label
                    checkpoint = train_dir / f'checkpoint_{step}.pt'
                    command = [sys.executable, str(HERE / 'evaluate.py'), '--manifest', str(args.manifest),
                               '--checkpoint', str(checkpoint), '--out', str(dest), '--split', split,
                               '--method', args.method, '--prior-cameras', args.prior_cameras]
                    self.child_run(label, command, args.eval_gpu)
                    receipt = read(dest / 'complete.json')
                    if (receipt.get('status') != 'completed_evaluation' or receipt.get('parameter_updates') != 0
                            or receipt.get('split') != split or receipt.get('method') != args.method
                            or receipt.get('checkpoint_sha256') != sha(checkpoint)
                            or receipt.get('metrics_sha256') != sha(dest / 'metrics.json')
                            or receipt.get('observations') != (16 if split == 'train_fixed' else 60)):
                        raise RuntimeError(f'Incomplete/mismatched evaluation receipt: {label}')
                    evaluations[f'{split}_{step}'] = str(dest)
        queued.update(status='released', released_utc=stamp())
        write(self.out / 'events/evaluation_queue.json', queued)
        item = {'role': args.method, 'label': args.method, 'checkpoint': train_receipt['checkpoint']['path'],
                'train_dir': str(train_dir), 'evaluations': evaluations, 'job_dir': str(self.out),
                'training_reused': args.reuse_train is not None, 'train_receipt': str(self.out / 'train_receipt.json')}
        write(self.out / 'methods-entry.json', {'methods': {args.method: item}})

    def run(self):
        try:
            train_dir, receipt = self.train()
            self.evaluate(train_dir, receipt)
            self.state.update(status='completed_and_evaluated', stage='finished', finished_utc=stamp(),
                              elapsed_seconds=time.monotonic() - self.started, train_dir=str(train_dir),
                              train_receipt=str(self.out / 'train_receipt.json'),
                              methods_entry=str(self.out / 'methods-entry.json'), draw_sha256=receipt['draw_sha256'])
            write(self.out / 'status.json', self.state)
            write(self.out / 'complete.json', self.state)
        except BaseException:
            self.state.update(status='failed', stage=self.stage, failed_utc=stamp(),
                              elapsed_seconds=time.monotonic() - self.started, traceback=traceback.format_exc())
            write(self.out / 'failed.json', self.state)
            write(self.out / 'status.json', self.state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--method', choices=BRANCH, required=True)
    parser.add_argument('--train-gpu', required=True, help='Full GPU UUID; unused for reuse-only jobs')
    parser.add_argument('--eval-gpu', required=True, help='Full GPU UUID; shared flock queue serializes evaluations')
    parser.add_argument('--prior-cameras', required=True, help='Exactly four registered training cameras, comma separated')
    for name in ('selection', 'reference', 'calibration', 'reuse-train'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--eval-dev-1200', action='store_true', help='Also evaluate checkpoint1200 when present; optional')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if any(not GPU_PATTERN.fullmatch(gpu) for gpu in (args.train_gpu, args.eval_gpu)):
        parser.error('Both GPU arguments must be full GPU-UUID strings')
    cameras = [c.strip() for c in args.prior_cameras.split(',')]
    if len(cameras) != 4 or len(set(cameras)) != 4 or any(not re.fullmatch(r'cam\d{2}', c) for c in cameras):
        parser.error('Exactly four distinct camNN teacher cameras are required')
    args.prior_cameras = ','.join(cameras)
    if args.method == 'C' and not args.reuse_train:
        parser.error('C supports existing-training evaluation only; no new C training in this batch')
    if not args.reuse_train:
        if args.method in ('B', 'S') and args.selection is None:
            parser.error('New B/S training requires --selection')
        if args.method == 'S' and (args.reference is None or args.calibration is None):
            parser.error('New S training requires --reference and --calibration')
    for name in ('manifest', 'parent', 'selection', 'reference', 'calibration'):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f'Input file not found: {name}={value}')
    if args.reuse_train is not None and not args.reuse_train.is_dir():
        parser.error('--reuse-train must be an existing completed training directory')
    args.out.mkdir(parents=True, exist_ok=False)
    job = Job(args)
    signal.signal(signal.SIGTERM, job.terminated)
    signal.signal(signal.SIGINT, job.terminated)
    job.run()
    print(json.dumps({'status': 'completed_and_evaluated', 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
