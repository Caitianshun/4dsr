#!/usr/bin/env python3
"""Run controlled training, then immediately evaluate its successful exit.

Source activate_a100.sh first. This runner uses the current Python environment.
Logs/status live in <out>_job; evaluation lives in <out>_evaluation. All three
paths must be new. Completion triggers local evaluation, not a chat notification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, state):
    state['updated_at'] = now()
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


class ChildRunner:
    def __init__(self, env):
        self.env = env
        self.child = None
        self.launching = False
        self.interrupted_by = None

    def on_signal(self, signum, _frame):
        self.interrupted_by = signum
        # Defer interruption until Popen has returned its handle, so a signal
        # during launch cannot leave a child that cleanup cannot identify.
        if not self.launching:
            raise Interrupted(signum)

    def run(self, command, log_path, on_start):
        with log_path.open('x') as log:
            self.launching = True
            try:
                self.child = subprocess.Popen(
                    command, cwd=ROOT, env=self.env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True)
            finally:
                self.launching = False
            if self.interrupted_by is not None:
                raise Interrupted(self.interrupted_by)
            on_start(self.child.pid)
            # No polling: waitpid blocks until this exact child exits.
            return self.child.wait()

    def stop(self, signum):
        if self.child is None:
            return None

        def send_group(value):
            try:
                os.killpg(self.child.pid, value)
            except ProcessLookupError:
                pass

        # Graceful interrupt, bounded termination fallback, then forced cleanup.
        # Repeated Ctrl+C is ignored by main while this bounded cleanup runs.
        send_group(signum)
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            send_group(signal.SIGTERM)
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                send_group(signal.SIGKILL)
                self.child.wait()
        # A group member can survive after its leader exits. Remove such
        # descendants too; normal successful runs do not take this path.
        send_group(signal.SIGKILL)
        return self.child.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--teacher', choices=('none', 'sr', 'hr'), default='sr')
    parser.add_argument('--weight', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--milestones', default='1200,3000,6000')
    parser.add_argument('--prior-cameras', default='cam02,cam06,cam12,cam18')
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args(argv)
    if args.steps <= 0 or not math.isfinite(args.weight) or args.weight < 0:
        parser.error('steps must be positive; weight must be finite and nonnegative')
    try:
        milestones = [int(item) for item in args.milestones.split(',')]
    except ValueError:
        parser.error('milestones must be comma-separated positive integers')
    if not milestones or min(milestones) <= 0:
        parser.error('milestones must be comma-separated positive integers')
    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for source in (manifest, checkpoint):
        if not source.is_file():
            parser.error(f'Input file does not exist: {source}')
    # absolute(), not resolve(): even a dangling output symlink must be refused.
    out = args.out.expanduser().absolute()
    job = out.with_name(out.name + '_job')
    evaluation = out.with_name(out.name + '_evaluation')
    for destination in (out, job, evaluation):
        if os.path.lexists(destination):
            parser.error(f'Refusing to overwrite existing path: {destination}')
    job.mkdir(parents=True, exist_ok=False)
    status_path = job / 'status.json'
    train = [sys.executable, '-u', str(ROOT / 'experiments/dynamic_sr_20260919/controlled_fit.py'),
             '--manifest', str(manifest), '--checkpoint', str(checkpoint), '--out', str(out),
             '--teacher', args.teacher, '--weight', str(args.weight), '--steps', str(args.steps),
             '--milestones', args.milestones, '--prior-cameras', args.prior_cameras]
    if args.dense:
        train.append('--dense')
    evaluate = [sys.executable, '-u', str(ROOT / 'experiments/dynamic_sr_20260918/evaluate.py'),
                '--manifest', str(manifest), '--checkpoint', str(out / 'checkpoint_final.pt'),
                '--out', str(evaluation), '--no-video']
    state = dict(status='starting', stage=None, started_at=now(), runner_pid=os.getpid(),
                 project_root=str(ROOT), python=sys.executable, gpu=args.gpu,
                 training_out=str(out), evaluation_out=str(evaluation),
                 notification='Local evaluation only; no chat notification is sent or guaranteed.',
                 stages={name: dict(command=command, status='pending', returncode=None,
                                    log=str(job / (name + '.log')))
                         for name, command in [('training', train), ('evaluation', evaluate)]})
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    child_runner = ChildRunner(env)
    original_handlers = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
    exit_code = 1
    try:
        for sig in STOP_SIGNALS:
            signal.signal(sig, child_runner.on_signal)
        write_status(status_path, state)
        for name, command, artifact in [
                ('training', train, out / 'complete.json'),
                ('evaluation', evaluate, evaluation / 'metrics.json')]:
            state.update(status='running', stage=name)
            record = state['stages'][name]
            record.update(status='running', started_at=now())
            child_runner.child = None
            write_status(status_path, state)
            destination = out if name == 'training' else evaluation
            if os.path.lexists(destination):
                raise FileExistsError(f'Refusing to overwrite existing path: {destination}')

            def on_start(pid):
                record['pid'] = pid
                write_status(status_path, state)

            print(f'{name.upper()} {record["log"]}', flush=True)
            rc = child_runner.run(command, job / (name + '.log'), on_start)
            record.update(returncode=rc, exited_at=now())
            if rc != 0:
                record['status'] = 'failed'
                state.update(status='failed', reason=f'{name} exited with return code {rc}')
                exit_code = rc if rc > 0 else 128 - rc
                break
            # An exit code alone cannot establish that expected work completed.
            with artifact.open() as handle:
                json.load(handle)
            if name == 'training' and not (out / 'checkpoint_final.pt').is_file():
                raise FileNotFoundError(out / 'checkpoint_final.pt')
            record['status'] = 'complete'
            write_status(status_path, state)
        else:
            state.update(status='complete', stage=None)
            exit_code = 0
    except (Interrupted, KeyboardInterrupt) as error:
        for sig in STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        signum = error.signum if isinstance(error, Interrupted) else signal.SIGINT
        rc = child_runner.stop(signum)
        state.update(status='interrupted', signal=signal.Signals(signum).name)
        if state['stage'] is not None:
            state['stages'][state['stage']].update(status='interrupted', returncode=rc, exited_at=now())
        exit_code = 128 + signum
    except Exception as error:
        for sig in STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        rc = child_runner.stop(signal.SIGTERM)
        state.update(status='failed', reason=repr(error))
        if state['stage'] is not None:
            state['stages'][state['stage']].update(status='failed', returncode=rc, exited_at=now())
        exit_code = 1
    finally:
        # Finish recording before restoring interrupt handlers.
        for sig in STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        state.update(finished_at=now(), runner_returncode=exit_code)
        try:
            write_status(status_path, state)
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)
    print(f'{state["status"].upper()} {status_path}', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
