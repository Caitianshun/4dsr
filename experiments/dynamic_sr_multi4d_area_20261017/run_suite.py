"""Persistent CPU waiter and exit-driven area4 comparison.

Wait for the user's global Claude-idle condition before every GPU stage. A
resumed Claude session requests a complete training checkpoint and GPU release.
No message is sent to Claude. Run this standard-library program via systemd.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

import idle_gate as gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    with Path(path).open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Runner:
    def __init__(self, plan_path):
        self.plan_path = plan_path.resolve()
        self.plan = read(plan_path)
        self.out = Path(self.plan['output'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.pause = Path(self.plan['pause_file'])
        self.python = self.plan['python']
        self.gpu = self.plan['gpu_uuid']
        self.state_path = self.out / 'controller_status.json'
        self.state = read(self.state_path) if self.state_path.exists() else dict(stages={}, events=[])
        if self.state.get('plan_sha256', sha(plan_path)) != sha(plan_path):
            raise ValueError('The existing plan bytes changed; do not replay this queue')
        self.state.update(schema='multi4d_area_controller_v1', pid=os.getpid(),
                          started_utc=self.state.get('started_utc', gate.stamp()),
                          plan=str(self.plan_path), plan_sha256=sha(plan_path),
                          scientific_status='no_new_Multi4D_result_until_complete_evaluation')
        self.child = None
        self.stopping = False
        self.quiet_since = None
        self.source_hashes = self.plan['source_sha256']

    def update(self, status, **values):
        self.state.update(status=status, updated_utc=gate.stamp(), **values)
        gate.write(self.state_path, self.state)

    def verify_sources(self):
        for path, digest in self.source_hashes.items():
            if sha(path) != digest:
                raise ValueError('Frozen preparation changed: ' + path)

    def stop(self, number, _frame):
        self.stopping = True
        gate.write(self.pause, dict(reason='controller_stop', signal=number, time=gate.stamp()))

    def wait_idle(self, stage):
        last_signature = None
        while not self.stopping:
            if (self.out / 'USER_HOLD').exists():
                self.update('waiting_user_hold', waiting_for=stage)
                time.sleep(30)
                continue
            sample = gate.check(remote=False)
            signature = (sample['safe'], tuple(sorted((b['reason'], str(b.get('pid', b.get('job_id', ''))))
                                                      for b in sample['blockers'])))
            if signature != last_signature:
                gate.write(self.out / 'admission_latest.json', sample)
                self.update('waiting_all_claude_idle_and_no_claude_gpu', waiting_for=stage,
                            admission_summary=dict(live_claude_sessions=len(sample['conversations']['live_sessions']),
                                                   blockers=len(sample['blockers']), safe=sample['safe']))
                last_signature = signature
            if sample['safe']:
                self.quiet_since = self.quiet_since or time.monotonic()
                if time.monotonic() - self.quiet_since >= self.plan.get('quiet_seconds', 300):
                    full = gate.check(remote=True)
                    gate.write(self.out / 'admission_latest.json', full)
                    if full['safe']:
                        gate.write(self.out / 'admission_first_remote_sample.json', full)
                        # Both remote hosts also need separated empty readings.
                        time.sleep(self.plan.get('remote_quiet_seconds', 20))
                        if self.stopping:
                            raise InterruptedError('Stopped during remote resource confirmation')
                        full = gate.check(remote=True)
                        gate.write(self.out / 'admission_latest.json', full)
                        # Recheck locally after the potentially slower remote reads.
                        final = gate.check(remote=False)
                        if full['safe'] and final['safe']:
                            self.verify_sources()
                            self.pause.unlink(missing_ok=True)  # only this queue's own request
                            return
                    self.quiet_since = None
            else:
                self.quiet_since = None
            time.sleep(self.plan.get('check_seconds', 60))
        raise InterruptedError('Controller stopped before GPU admission')

    def environment(self):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(('ANTHROPIC_', 'OPENAI_', 'CLAUDE_CODE_')):
                env.pop(key)
        env.update(CUDA_VISIBLE_DEVICES=self.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='8.6;12.0',
                   TORCH_EXTENSIONS_DIR=str(Path(self.plan['upstream']) / '.extensions'),
                   MULTI4D_UPSTREAM=self.plan['upstream'], MAX_JOBS='4', OMP_NUM_THREADS='4',
                   MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        env['PATH'] = '/usr/local/cuda-12.8/bin:' + env.get('PATH', '')
        return env

    def execute_child(self, command, log_path, key, train):
        interrupted_evaluation = False
        with log_path.open('x') as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=self.environment(),
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            self.state['stages'][key]['attempts'][-1]['pid'] = self.child.pid
            gate.write(self.state_path, self.state)
            deadline = time.monotonic() + self.plan.get('stage_timeout_seconds', 14400)
            remote_check_due = time.monotonic() + 300
            yield_deadline = None
            try:
                while True:
                    try:
                        code = self.child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        # No model calls: only process/registry/ownership metadata.
                        remote = time.monotonic() >= remote_check_due
                        sample = gate.check(remote=remote, ignore_pids=(self.child.pid,))
                        if remote:
                            remote_check_due = time.monotonic() + 300
                        if not sample['safe'] or self.stopping:
                            self.quiet_since = None
                            gate.write(self.pause, dict(reason='Claude_became_active_or_resource_changed',
                                                        checked_utc=sample['checked_utc'], blockers=sample['blockers']))
                            self.update('yield_requested', active_stage=key)
                            yield_deadline = yield_deadline or (time.monotonic() + self.plan.get('pause_grace_seconds', 180))
                            if not train:
                                # Evaluation updates no parameters; preserve the partial
                                # attempt and start a fresh evaluation after admission.
                                self.child.terminate()
                                interrupted_evaluation = True
                        if yield_deadline and time.monotonic() >= yield_deadline:
                            raise TimeoutError(f'{key} did not yield within the checkpoint grace period')
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'{key} exceeded its recorded stage deadline')
            except BaseException:
                # Never release the lease while this queue's CUDA child survives.
                # An observation/logging failure closes admission and stops our own
                # child; it cannot orphan training outside the dispatcher.
                remaining = (max(0, yield_deadline - time.monotonic())
                             if yield_deadline else self.plan.get('pause_grace_seconds', 180))
                self.stop_owned_child(train, grace=remaining)
                raise
            finally:
                if self.child is not None and self.child.poll() is not None:
                    self.child = None
        return code, interrupted_evaluation

    def stop_owned_child(self, train, *, grace=None):
        child = self.child
        if child is None or child.poll() is not None:
            return
        if train:
            try:
                gate.write(self.pause, dict(reason='controller_error_fail_closed', time=gate.stamp()))
                child.wait(timeout=self.plan.get('pause_grace_seconds', 180) if grace is None else grace)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if child.poll() is None:
            child.terminate()  # Only the exact child created by this controller.
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        self.child = None

    def stage(self, key, argv, out, *, train=False, resume=None, smoke_eval=False):
        record = self.state['stages'].setdefault(key, dict(attempts=[], status='pending'))
        if record['status'] == 'completed':
            receipt = Path(record['output']) / ('complete.json' if train else 'summary.json')
            if not receipt.is_file() or sha(receipt) != record['receipt_sha256']:
                raise ValueError('Completed stage receipt missing or changed: ' + key)
            return Path(record['output'])
        if record['status'] in ('running', 'failed', 'outcome_unknown'):
            raise RuntimeError('Review the existing attempt instead of replaying: ' + key)
        pause_checkpoint = record.get('resume_checkpoint')
        while True:
            self.wait_idle(key)
            lease = lock('/tmp/4dsr_soft_gpu_' + self.gpu + '.lock')
            try:
                lease.__enter__()
            except BlockingIOError:
                self.quiet_since = None
                self.update('waiting_native_gpu_lease', waiting_for=key)
                time.sleep(self.plan.get('check_seconds', 60))
                continue
            # This advisory native-runner lock is acquired only after admission.
            # Root-managed Claude reservations are checked separately, never edited.
            try:
                final = gate.check(remote=False)
                if not final['safe']:
                    self.quiet_since = None
                    continue
                free = shutil.disk_usage(self.out).free
                if free < self.plan.get('minimum_free_bytes', 40 << 30):
                    raise OSError(f'Insufficient checkpoint/evaluation space: {free} bytes free')
                if pause_checkpoint and sha(pause_checkpoint) != record['resume_sha256']:
                    raise ValueError('Paused checkpoint bytes changed: ' + key)
                number = len(record['attempts']) + 1
                destination = out / f'attempt_{number:02d}'
                if destination.exists():
                    raise FileExistsError(destination)
                command = [self.python, *map(str, argv), '--out', str(destination)]
                if pause_checkpoint or resume:
                    command += ['--resume', str(pause_checkpoint or resume)]
                attempt = dict(number=number, command=command, output=str(destination), started_utc=gate.stamp())
                record.update(status='running', output=str(destination))
                record['attempts'].append(attempt)
                self.update('running', active_stage=key)
                log_path = self.out / 'logs' / f'{key}_attempt_{number:02d}.log'
                log_path.parent.mkdir(exist_ok=True)
                code, interrupted_evaluation = self.execute_child(command, log_path, key, train)
            finally:
                try:
                    self.stop_owned_child(train)
                finally:
                    lease.__exit__(None, None, None)
            attempt.update(exit_code=code, ended_utc=gate.stamp(), log=str(log_path))
            if code == 85 and train:
                checkpoint = destination / 'pause.pt'
                receipt = destination / 'pause.json'
                if not checkpoint.is_file() or not receipt.is_file():
                    raise ValueError('Pause exit without complete state')
                pause_checkpoint = checkpoint
                record.update(status='paused_for_claude', resume_checkpoint=str(checkpoint),
                              resume_sha256=sha(checkpoint))
                self.update('paused_for_claude', active_stage=None)
                continue
            if interrupted_evaluation:
                record.update(status='paused_for_claude')
                self.update('paused_for_claude', active_stage=None)
                continue
            if code:
                record.update(status='failed')
                self.update('failed', error=f'{key} exit {code}', active_stage=None)
                raise RuntimeError(f'{key} failed; preserving {destination}')
            receipt = destination / ('complete.json' if train else 'summary.json')
            value = read(receipt)
            if not train and not smoke_eval and value['status'] != 'completed':
                raise ValueError('Formal evaluation is incomplete: ' + str(receipt))
            if smoke_eval and (value['status'] != 'incomplete' or value['frames'] != 3):
                raise ValueError('Smoke evaluation must remain explicitly incomplete, with three test frames')
            if train and value['status'] not in ('smoke_completed', 'fine0_prepared', 'completed'):
                raise ValueError('Unexpected training completion: ' + str(receipt))
            record.update(status='completed', receipt_sha256=sha(receipt), output=str(destination))
            self.update('stage_completed', active_stage=None)
            return destination

    def scene(self, item):
        name, specification = item['name'], item['spec']
        out = self.out / name
        out.mkdir(exist_ok=True)
        self.stage(name + '_coarse_smoke', [HERE / 'train.py', '--spec', specification,
                   '--method', 'coarse', '--stop', '2'], out / 'coarse_smoke', train=True)
        coarse = self.stage(name + '_coarse', [HERE / 'train.py', '--spec', specification, '--method', 'coarse'],
                            out / 'coarse', train=True)
        fine0 = coarse / 'fine0.pt'
        smoke = self.stage(name + '_smoke', [HERE / 'train.py', '--spec', specification, '--method', 'areaTm', '--stop', '20'],
                           out / 'smoke', train=True, resume=fine0)
        self.stage(name + '_smoke_eval', [HERE / 'evaluate.py', '--spec', specification,
                   '--checkpoint', smoke / 'fine_20.pt', '--method', 'areaTm',
                   '--frames', '0', '150', '299', '--train-cameras', 'cam01'],
                   out / 'smoke_eval', smoke_eval=True)
        summaries = {}
        for method in ('areaL0', 'areaTm', 'areaOracle'):
            trained = self.stage(name + '_' + method,
                                 [HERE / 'train.py', '--spec', specification, '--method', method],
                                 out / method, train=True, resume=fine0)
            evaluated = self.stage(name + '_' + method + '_eval',
                                   [HERE / 'evaluate.py', '--spec', specification, '--method', method,
                                    '--checkpoint', trained / 'fine_20000.pt'],
                                   out / 'eval' / method)
            summaries[method] = evaluated / 'summary.json'
        report = out / 'assessment.json'
        command = [sys.executable, str(HERE / 'assess.py'), '--scene', item['scene'],
                   '--multi-root', str(out / 'eval'), '--out', str(report)]
        for method, path in summaries.items():
            command += ['--' + method, str(path)]
        subprocess.run(command, cwd=ROOT, check=True)
        return read(report)

    def run(self):
        self.verify_sources()
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        with lock(self.out / 'controller.lock'):
            self.update('prepared_waiting_all_claude_idle_and_no_claude_gpu_use')
            first = self.scene(self.plan['scenes'][0])
            contrast = first['cross_model']['areaTm']
            within = first['within_model']['Multi4D']['SR_gain_matched_minus_LR']
            wu_within = first['within_model']['Wu']['SR_gain_matched_minus_LR']
            temporal_ratio = None
            tm = first['Multi4D']['areaTm'].get('temporal') or {}
            l0 = first['Multi4D']['areaL0'].get('temporal') or {}
            if l0.get('frame_difference_residual_l1', 0) > 0:
                temporal_ratio = tm.get('frame_difference_residual_l1', float('inf')) / l0['frame_difference_residual_l1']
            # Confirm a meaningful advantage OR disadvantage on the second scene.
            # Selecting only successful Multi4D outcomes would bias this study.
            healthy = (first['status'] == 'completed' and contrast['status'] == 'completed'
                       and within['status'] == 'completed' and wu_within['status'] == 'completed'
                       and temporal_ratio is not None)
            sr_gain_difference = (within['delta']['psnr'] - wu_within['delta']['psnr']) if healthy else None
            decision = healthy and (abs(contrast['delta']['psnr']) >= .05
                                    or abs(contrast['delta']['lpips']) >= .003
                                    or abs(sr_gain_difference) >= .05)
            gate.write(self.out / 'expansion_decision.json', dict(expand_to_Cook07=decision,
                       rule='Complete Cut protocol; confirm Cook when absolute matched PSNR difference >=.05dB, absolute LPIPS difference >=.003, or between-model SR-gain difference >=.05dB. Both win and loss qualify.',
                       temporal_ratio=temporal_ratio, SR_gain_difference_dB=sr_gain_difference,
                       comparison=contrast, within_model=within,
                       limitation='Development rule; human visual acceptability and full-system ranking remain separately assessed'))
            if decision:
                self.scene(self.plan['scenes'][1])
            self.update('completed', active_stage=None, expanded_to_Cook07=decision,
                        result_boundary='Full protocol comparisons; no universal backbone-strength or visual-acceptance claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    runner = Runner(args.plan)
    try:
        runner.run()
    except BaseException as error:
        runner.update('failed' if not isinstance(error, InterruptedError) else 'paused',
                      error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
