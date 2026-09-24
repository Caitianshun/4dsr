#!/usr/bin/env python3
"""One registered G/L job: train/reuse, then immediately evaluate.

Caller systemd owns persistence. Shared soft/detail GPU flock leases prevent
cooperating old and new runners from overlapping; occupancy is rechecked
after acquisition. No model tensors are loaded by this orchestration layer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_soft_job_runner', ROOT / 'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
soft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soft)
write, read, sha, identity, stamp = soft.write, soft.read, soft.sha, soft.identity, soft.stamp
lock_file = soft.lock_file


class Job(soft.Job):
    def __init__(self, args):
        super().__init__(args)
        self.inputs = {name: identity(getattr(args, name)) for name in
                       ('manifest', 'parent', 'selection', 'teacher_index', 'schedule', 'roi_protocol', 'reference_config')}
        self.state.update(runner=identity(__file__), inherited_runner=identity(soft.__file__),
                          inputs=self.inputs, training_branch='ordinary_split',
                          fixed_sr_weight=.1)
        write(self.out / 'status.json', self.state)

    def validate_training(self, folder):
        args = self.args
        config, done = read(folder / 'config.json'), read(folder / 'complete.json')
        if done.get('status') != 'completed' or done.get('parameter_updates') != 6000 or done.get('smoke', False):
            raise RuntimeError('Expected completed formal 6000-update training')
        expected = {'manifest_sha256': self.inputs['manifest']['sha256'],
                    'parent_sha256': self.inputs['parent']['sha256'],
                    'selection_sha256': self.inputs['selection']['sha256'],
                    'teacher_index_sha256': self.inputs['teacher_index']['sha256'],
                    'schedule_sha256': self.inputs['schedule']['sha256'],
                    'method': args.method, 'branch': 'ordinary_split', 'steps': 6000, 'seed': 20260923}
        expected['reference_config_sha256'] = self.inputs['reference_config']['sha256']
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(f'Training identity mismatch for {key}: {config.get(key)!r} != {value!r}')
        if 'sr_weight' in config and config['sr_weight'] != (.1):
            raise RuntimeError('Saved SR weight differs from fixed method definition')
        for name, saved in self.inputs.items():
            if sha(saved['path']) != saved['sha256']:
                raise RuntimeError(f'Input changed during run: {name}')
        receipt = {'status': 'completed_training_verified', 'method': args.method,
                   'reused_training': args.reuse_train is not None, 'train_dir': str(folder),
                   'config': identity(folder / 'config.json'), 'complete': identity(folder / 'complete.json'),
                   'checkpoint': identity(folder / 'checkpoint_6000.pt'),
                   'checkpoint_1200': identity(folder / 'checkpoint_1200.pt'),
                   'draw_sha256': done.get('draw_sha256'), 'inputs': self.inputs,
                   'exposure': identity(folder / 'exposure.json'), 'verified_utc': stamp()}
        return receipt

    def train(self):
        args = self.args
        if args.reuse_train:
            self.update('validating_reused_training')
            folder = args.reuse_train
        else:
            folder = self.out / 'train'
            command = [sys.executable, str(HERE / 'train.py'), '--manifest', str(args.manifest),
                       '--checkpoint', str(args.parent), '--selection', str(args.selection),
                       '--out', str(folder), '--method', args.method, '--steps', '6000',
                       '--milestones', '1200,6000', '--seed', '20260923',
                       '--teacher-index', str(args.teacher_index), '--schedule', str(args.schedule)]
            command += ['--reference-config', str(args.reference_config)]
            self.update('acquiring_training_gpu')
            with lock_file(f'/tmp/4dsr_soft_gpu_{args.train_gpu}.lock'):
                self.child_run('train', command, args.train_gpu)
        receipt = self.validate_training(folder)
        write(self.out / 'train_receipt.json', receipt)
        return folder, receipt

    def evaluate(self, folder, train_receipt):
        args = self.args
        endpoints = [('train_fixed', 6000), ('dev', 6000), ('test', 6000)]
        if args.eval_dev_1200:
            endpoints.append(('dev', 1200))
        queued = {'status': 'waiting', 'gpu': args.eval_gpu, 'queued_utc': stamp()}
        tick = time.monotonic()
        self.update('waiting_for_evaluation_lock')
        write(self.out / 'events/evaluation_queue.json', queued)
        evaluations = {}
        with lock_file(f'/tmp/4dsr_soft_eval_{args.eval_gpu}.lock'):
            queued.update(status='acquired', acquired_utc=stamp(), wait_seconds=time.monotonic() - tick)
            write(self.out / 'events/evaluation_queue.json', queued)
            with lock_file(f'/tmp/4dsr_soft_gpu_{args.eval_gpu}.lock'):
                for split, step in endpoints:
                    label = f'eval_{split}_{step}'
                    dest, checkpoint = self.out / label, folder / f'checkpoint_{step}.pt'
                    command = [sys.executable, str(HERE / 'evaluate.py'), '--manifest', str(args.manifest),
                               '--checkpoint', str(checkpoint), '--out', str(dest), '--split', split,
                               '--method', args.method, '--original-prior-cameras', args.original_prior_cameras,
                               '--roi-protocol', str(args.roi_protocol)]
                    self.child_run(label, command, args.eval_gpu)
                    receipt = read(dest / 'complete.json')
                    expected = {'status': 'completed_evaluation', 'parameter_updates': 0, 'split': split,
                                'method': args.method, 'checkpoint_sha256': sha(checkpoint),
                                'metrics_sha256': sha(dest / 'metrics.json'),
                                'observations': 16 if split == 'train_fixed' else 60}
                    for key, value in expected.items():
                        if receipt.get(key) != value:
                            raise RuntimeError(f'Incomplete/mismatched evaluation {label}: {key}')
                    evaluations[f'{split}_{step}'] = str(dest)
        queued.update(status='released', released_utc=stamp())
        write(self.out / 'events/evaluation_queue.json', queued)
        item = {'role': args.method, 'label': args.method, 'checkpoint': train_receipt['checkpoint']['path'],
                'train_dir': str(folder), 'evaluations': evaluations, 'job_dir': str(self.out),
                'training_reused': args.reuse_train is not None, 'train_receipt': str(self.out / 'train_receipt.json')}
        write(self.out / 'methods-entry.json', {'methods': {args.method: item}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'parent', 'selection', 'teacher-index', 'schedule', 'roi-protocol', 'reference-config', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--method', choices=('G', 'L'), required=True)
    parser.add_argument('--train-gpu', required=True)
    parser.add_argument('--eval-gpu', required=True)
    parser.add_argument('--original-prior-cameras', required=True)
    parser.add_argument('--reuse-train', type=Path)
    parser.add_argument('--eval-dev-1200', action='store_true')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if any(not soft.GPU_PATTERN.fullmatch(gpu) for gpu in (args.train_gpu, args.eval_gpu)):
        parser.error('Explicit full GPU UUID required')
    cameras = [c.strip() for c in args.original_prior_cameras.split(',')]
    if len(cameras) != 4 or len(set(cameras)) != 4 or any(not soft.re.fullmatch(r'cam\d{2}', c) for c in cameras):
        parser.error('Exactly four original training-camera IDs required')
    args.original_prior_cameras = ','.join(cameras)
    for name in ('manifest', 'parent', 'selection', 'teacher_index', 'schedule', 'roi_protocol', 'reference_config'):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f'Missing input: {name}={value}')
    if args.reuse_train is not None and not args.reuse_train.is_dir():
        parser.error('Missing reusable training directory')
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        job = Job(args)
        signal.signal(signal.SIGTERM, job.terminated)
        signal.signal(signal.SIGINT, job.terminated)
        job.run()
    except BaseException:
        if not (args.out / 'failed.json').is_file():
            write(args.out / 'failed.json', {'status': 'failed_before_training', 'traceback': traceback.format_exc()})
        raise
    print(json.dumps({'status': 'completed_and_evaluated', 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
