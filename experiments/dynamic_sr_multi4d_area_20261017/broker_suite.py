"""Submit each already-admitted GPU stage through the existing root broker.

The host controller never uses CUDA or treats the wait CLI as the training PID.
Only the project wrapper can stop the native process group it created. Root
reservations remain authoritative until terminal status and an empty cgroup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid

import idle_gate as gate
import run_suite
import target_gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENVIRONMENT_KEYS = {
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST',
    'TORCH_EXTENSIONS_DIR', 'MULTI4D_UPSTREAM', 'MAX_JOBS', 'OMP_NUM_THREADS',
    'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MPLBACKEND', 'PYTHONUNBUFFERED',
    'PYTHONHASHSEED', 'PYTHONNOUSERSITE', 'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'PATH',
}


class BrokerOutcomeUnknown(RuntimeError):
    pass


class BrokerRunner(run_suite.Runner):
    waiting_status = 'waiting_target_gpu_idle_for_root_broker'
    prepared_status = 'prepared_waiting_target_gpu_idle_for_root_broker'

    def __init__(self, plan_path):
        super().__init__(plan_path)
        if self.plan.get('admission_policy') != 'target_gpu_broker':
            raise ValueError('broker_suite requires admission_policy=target_gpu_broker')
        for key in ('broker_stage_wrapper', 'broker_runtime_python', 'broker_cache_root',
                    'broker_request_root', 'minimum_free_vram_gib', 'broker_memory_mib'):
            if key not in self.plan:
                raise ValueError('Missing broker plan field: ' + key)
        self.python = self.plan['broker_runtime_python']
        timeout = self.plan.get('stage_timeout_seconds', 86400)
        if (isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86400
                or self.plan.get('pause_grace_seconds', 180) != 180):
            raise ValueError('Broker wrapper requires a 1..86400s stage and exactly 180s pause grace')
        budget = self.plan['broker_memory_mib']
        if isinstance(budget, bool) or not isinstance(budget, int) or not 256 <= budget <= 196608:
            raise ValueError('Broker GPU budget must be 256..196608 integer MiB')
        for key in ('broker_stage_wrapper', 'broker_runtime_python', 'broker_cache_root', 'broker_request_root'):
            path = Path(self.plan[key])
            if not path.is_absolute() or not path.resolve().is_relative_to(ROOT):
                raise ValueError('Broker runtime/request paths must stay inside this project: ' + key)
        self.request_root = Path(self.plan['broker_request_root'])
        self.request_root.mkdir(parents=True, exist_ok=True, mode=0o2770)
        info = self.request_root.stat()
        if info.st_uid != os.getuid() or info.st_gid != 1001 or info.st_mode & 0o002:
            raise ValueError('Broker request directory must be controller-owned, group 1001 and not world writable')
        os.chmod(self.request_root, 0o2770)
        self.broker_cli = self.plan.get('broker_cli', '/usr/local/bin/claude-training')
        if self.broker_cli != '/usr/local/bin/claude-training':
            raise ValueError('Only the installed authorized broker CLI is supported')
        self.active_broker_job = None
        self.broker_log = None
        self.last_broker_state = None
        self.state['admission_policy'] = 'target_gpu_broker'

    def environment(self):
        # Build a fresh allowlist; do not serialize the inherited host environment.
        prefix = str(Path(self.python).parent.parent)
        return dict(CUDA_VISIBLE_DEVICES=self.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='8.6;12.0',
                    TORCH_EXTENSIONS_DIR=self.plan['broker_cache_root'],
                    MULTI4D_UPSTREAM=self.plan.get('broker_upstream', self.plan['upstream']),
                    MAX_JOBS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0',
                    PYTHONNOUSERSITE='1', CONDA_PREFIX=prefix, LD_LIBRARY_PATH=prefix + '/lib',
                    PATH=prefix + '/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin')

    def admission(self, *, remote=False, ignore_pids=()):
        # Neither the host wait process nor its PID grants a GPU exemption.
        job = getattr(self, 'active_broker_job', None)
        return target_gate.check(gpu_uuid=self.gpu,
                                 minimum_free_vram_gib=max(float(self.plan['minimum_free_vram_gib']),
                                                          (self.plan['broker_memory_mib'] + self.plan.get('broker_headroom_mib', 2048)) / 1024),
                                 broker_job=job, expected_gpu_name=self.plan.get('target_gpu_name'))

    def wait_idle(self, stage):
        separation = max(20., float(self.plan.get('target_quiet_seconds', 20)))
        if not math.isfinite(separation):
            raise ValueError('Target idle sample separation must be finite')
        last_signature = None
        while not self.stopping:
            if (self.out / 'USER_HOLD').exists():
                self.update('waiting_user_hold', waiting_for=stage)
                time.sleep(30)
                continue
            first = self.admission()
            signature = (first['safe'], tuple(sorted(item['reason'] for item in first['blockers'])))
            if signature != last_signature:
                gate.write(self.out / 'admission_latest.json', first)
                self.update(self.waiting_status, waiting_for=stage,
                            admission_summary=dict(gpu_uuid=self.gpu, safe=first['safe'],
                                                   blockers=len(first['blockers']), policy='target_gpu_broker'))
                last_signature = signature
            if first['safe']:
                started = time.monotonic()
                gate.write(self.out / 'admission_first_target_sample.json', first)
                time.sleep(separation)  # No native/common GPU lease is held here.
                if self.stopping:
                    raise InterruptedError('Stopped during target GPU resource confirmation')
                if (self.out / 'USER_HOLD').exists():
                    continue
                second = self.admission()
                second['idle_read_separation_seconds'] = time.monotonic() - started
                gate.write(self.out / 'admission_latest.json', second)
                if second['safe'] and second['idle_read_separation_seconds'] >= 20:
                    self.verify_sources()
                    self.pause.unlink(missing_ok=True)
                    return
            time.sleep(self.plan.get('check_seconds', 60))
        raise InterruptedError('Controller stopped before target GPU admission')

    def broker_call(self, operation, *args, timeout=35):
        process = subprocess.run([self.broker_cli, operation, *map(str, args)], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=timeout)
        if process.returncode:
            raise BrokerOutcomeUnknown(f'Broker {operation} CLI exit {process.returncode}: {process.stderr[-500:]}')
        try:
            value = json.loads(process.stdout)
        except ValueError as error:
            raise BrokerOutcomeUnknown('Unparseable broker response; do not resubmit') from error
        if not isinstance(value, dict):
            raise BrokerOutcomeUnknown('Broker response must be an object')
        return value

    def record_broker_state(self, value):
        job = self.active_broker_job
        if value.get('state') == 'submitted' and 'job_id' not in value:
            frozen = target_gate.frozen_manifest_proof(job, self.gpu)
            if (value.get('unit') != 'claude-local-' + job['job_id'] + '.service'
                    or value.get('manifest_sha256') != frozen['manifest_sha256']):
                raise BrokerOutcomeUnknown('Pre-running submission has no matching root job identity')
            value = dict(value, job_id=job['job_id'], identity_source='verified_root_submission')
        if value.get('job_id') != job['job_id']:
            raise BrokerOutcomeUnknown('Broker returned another job identity')
        gate.write(job['status_copy'], value)
        attempt = self.state['stages'][job['stage_key']]['attempts'][-1]
        attempt['broker_status'] = value
        attempt['broker_status_path'] = str(target_gate.common.ROOT_JOBS / job['job_id'] / 'run/status.json')
        # These are root worker/native launcher identities, never wait CLI PIDs.
        attempt['root_runner'] = value.get('runner')
        attempt['root_child'] = value.get('child')
        if (gate.ROOT_JOBS / job['job_id'] / 'manifest.json').exists():
            self.frozen_job(job)
        signature = (value.get('state'), value.get('stage'), value.get('command_started'), value.get('exit_code'))
        if self.broker_log is not None and signature != self.last_broker_state:
            self.broker_log.write(json.dumps(dict(kind='root_broker_status', value=value)) + '\n')
            self.broker_log.flush()
            self.last_broker_state = signature
        gate.write(self.state_path, self.state)

    def record_root_logs(self, job, directory):
        attempt = self.state['stages'][job['stage_key']]['attempts'][-1]
        logs = {}
        for name in ('stdout.log', 'stderr.log'):
            source = directory / 'run' / name
            if source.exists():
                destination = self.request_root / (job['job_id'] + '.' + name)
                shutil.copyfile(source, destination)
                logs[name] = dict(root_path=str(source), local_copy=str(destination),
                                  sha256=run_suite.sha(destination), bytes=destination.stat().st_size)
            else:
                logs[name] = dict(root_path=str(source), status='absent')
        attempt['native_root_logs'] = logs
        wrapper = Path(job['wrapper_receipt'])
        if wrapper.exists():
            value = run_suite.read(wrapper)
            request = run_suite.read(job['request_file'])
            if value.get('job_id') != job['job_id'] or value.get('command') != request['command']:
                raise BrokerOutcomeUnknown('Project wrapper receipt belongs to another native stage')
            attempt['wrapper_receipt'] = dict(path=str(wrapper), sha256=run_suite.sha(wrapper),
                                             native_pid=value.get('native_pid'),
                                             native_start_ticks=value.get('native_start_ticks'),
                                             status=value.get('status'), exit_code=value.get('exit_code'))

    def frozen_job(self, job):
        frozen = target_gate.frozen_manifest_proof(job, self.gpu)
        directory, manifest_path = Path(frozen['directory']), Path(frozen['manifest_path'])
        digest = frozen['manifest_sha256']
        copied = self.request_root / (job['job_id'] + '.root-manifest.json')
        if copied.exists() and run_suite.sha(copied) != digest:
            raise BrokerOutcomeUnknown('Frozen root manifest changed after first observation')
        if not copied.exists():
            # Save the exact root-normalized bytes, not a reserialized hash.
            copied.write_bytes(manifest_path.read_bytes())
            os.chmod(copied, 0o440)
        attempt = self.state['stages'][job['stage_key']]['attempts'][-1]
        attempt.update(root_manifest_sha256=digest, root_manifest_copy=str(copied))
        return directory

    def terminal_ready(self, job, value):
        if value.get('state') not in ('ended', 'completed', 'failed'):
            return False
        directory = self.frozen_job(job)
        path = directory / 'run/status.json'
        actual = json.loads(path.read_text())
        if actual != value or actual.get('job_id') != job['job_id']:
            raise BrokerOutcomeUnknown('CLI and root terminal status disagree')
        reservation = target_gate._root_json(gate.RESERVATIONS / (job['job_id'] + '.json'))
        if reservation.get('gpu_uuids') != [self.gpu]:
            raise BrokerOutcomeUnknown('Terminal reservation belongs to another GPU')
        finished, proof = gate.reservation_finished(reservation)
        if not finished:
            return False
        cgroup = '/claudevpn.slice/claude-local-' + job['job_id'] + '.service'
        if reservation.get('expected_cgroup') != cgroup:
            raise BrokerOutcomeUnknown('Terminal root cgroup differs from the submitted job')
        try:
            populated = dict(line.split() for line in (Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'cgroup.events').read_text().splitlines()).get('populated')
        except FileNotFoundError:
            populated = '0'  # Original runner exit was proved independently.
        if populated != '0':
            return False
        if not isinstance(actual.get('exit_code'), int) or isinstance(actual.get('exit_code'), bool):
            raise BrokerOutcomeUnknown('Terminal broker job has no numeric exit code')
        gate.write(job['terminal_copy'], dict(status=actual, original_runner_proof=proof,
                                              cgroup=cgroup, cgroup_populated=populated))
        self.record_root_logs(job, directory)
        if actual['exit_code'] in (0, 85):
            wrapper = run_suite.read(job['wrapper_receipt'])
            wrapper_code = wrapper.get('exit_code')
            command_code = actual.get('command_exit_code')
            if (not isinstance(wrapper_code, int) or isinstance(wrapper_code, bool)
                    or not isinstance(command_code, int) or isinstance(command_code, bool)
                    or wrapper.get('status') != 'ended'
                    or wrapper_code != actual['exit_code']
                    or command_code != actual['exit_code']):
                raise BrokerOutcomeUnknown('Successful/cooperative root exit does not match the frozen wrapper receipt')
        return True

    def request_yield(self, reason, *, abort=False):
        job = self.active_broker_job
        if not job:
            return
        gate.write(self.pause, dict(reason=reason, job_id=job['job_id'], time=gate.stamp()))
        if abort:
            gate.write(job['abort_file'], dict(reason=reason, job_id=job['job_id'], time=gate.stamp()))

    def stop_owned_child(self, train, *, grace=None):
        # There is no owned host CUDA subprocess to signal. Keep the root job
        # reserved and write only this job's wrapper cancellation request.
        job = getattr(self, 'active_broker_job', None)
        if not job:
            return
        if job.get('cleanup_outcome_unknown'):
            raise BrokerOutcomeUnknown('Previously unverified root cleanup remains unknown; no second wait/replay')
        self.request_yield('controller_stop_or_error', abort=True)
        deadline = time.monotonic() + self.plan.get('broker_cleanup_timeout_seconds', 240)
        while time.monotonic() < deadline:
            value = self.broker_call('wait', job['job_id'], '--timeout', '5', timeout=10)
            self.record_broker_state(value)
            if self.terminal_ready(job, value):
                self.active_broker_job = None
                return
            time.sleep(2)
        job['cleanup_outcome_unknown'] = True
        raise BrokerOutcomeUnknown('Own root job did not prove terminal + empty cgroup; reservation remains authoritative, no replay')

    def execute_child(self, command, log_path, key, train):
        jid = 'codex-m4d-' + uuid.uuid4().hex
        domain, errors = gate.domain_context()
        if errors or not domain.get('host_view_verified'):
            raise BrokerOutcomeUnknown('Cannot submit outside a verified host namespace')
        abort_file = self.request_root / (jid + '.abort.json')
        request_path = self.request_root / (jid + '.request.json')
        wrapper_receipt = self.request_root / (jid + '.wrapper.json')
        environment = self.environment()
        if set(environment) - ENVIRONMENT_KEYS:
            raise ValueError('Unexpected host environment entry in broker request')
        request = dict(schema='multi4d_broker_stage_v1', job_id=jid, command=command, cwd=str(ROOT),
                       environment=environment, pause_file=str(self.pause), abort_file=str(abort_file),
                       wrapper_receipt=str(wrapper_receipt), train=bool(train),
                       source_sha256=dict(self.source_hashes),
                       stage_timeout_seconds=int(self.plan.get('stage_timeout_seconds', 86400)),
                       pause_grace_seconds=int(self.plan.get('pause_grace_seconds', 180)))
        if request_path.exists():
            raise FileExistsError(request_path)
        gate.write(request_path, request)
        os.chmod(request_path, 0o440)
        request_sha = run_suite.sha(request_path)
        manifest = dict(version=1, job_id=jid, project_id='4dsr', role='local', operation='run',
                        env_id='sr4d', cwd=str(ROOT), gpu_uuids=[self.gpu],
                        gpu_memory_mib=int(self.plan['broker_memory_mib']),
                        timeout_seconds=int(self.plan.get('stage_timeout_seconds', 86400)) + 300,
                        argv=['/usr/bin/python3', self.plan['broker_stage_wrapper'], '--request',
                              str(request_path), '--request-sha256', request_sha])
        manifest_path = self.request_root / (jid + '.manifest.json')
        gate.write(manifest_path, manifest)
        os.chmod(manifest_path, 0o440)
        job = dict(job_id=jid, request_manifest=manifest, host_boot_id=domain['boot_id'],
                   host_pid_domain=domain['pid_domain'], stage_key=key, abort_file=str(abort_file),
                   request_file=str(request_path), request_sha256=request_sha,
                   wrapper_receipt=str(wrapper_receipt),
                   status_copy=str(self.request_root / (jid + '.status.json')),
                   terminal_copy=str(self.request_root / (jid + '.terminal.json')))
        job['expected_broker_identity'] = dict(project_root=str(ROOT), runner_uid=995,
            env_root=self.plan.get('broker_registered_env_root', '/home/cai_tianshun/miniconda3/envs/sr4d'),
            gpu_admission=dict(mode='training_free', memory_budget_mib=manifest['gpu_memory_mib'],
                               headroom_mib=self.plan.get('broker_headroom_mib', 2048),
                               sample_interval_seconds=self.plan.get('broker_sample_interval_seconds', 2)))
        attempt = self.state['stages'][key]['attempts'][-1]
        attempt.update(broker_job_id=jid, broker_manifest=str(manifest_path),
                       broker_manifest_sha256=run_suite.sha(manifest_path),
                       wrapper_request=str(request_path), wrapper_request_sha256=request_sha)
        # Record the unique submission identity BEFORE the non-idempotent CLI.
        self.active_broker_job = job
        gate.write(self.state_path, self.state)
        started = time.monotonic()
        deadline = started + request['stage_timeout_seconds'] + 240
        yield_deadline = None
        interrupted_evaluation = False
        with log_path.open('x') as stream:
            self.broker_log = stream
            try:
                submitted = self.broker_call('submit', manifest_path)
                if submitted.get('job_id') != jid:
                    raise BrokerOutcomeUnknown('Broker submit response changed the unique job identity')
                attempt['broker_submission'] = submitted
                gate.write(self.state_path, self.state)
                while True:
                    value = self.broker_call('wait', jid, '--timeout', '15', timeout=20)
                    self.record_broker_state(value)
                    state = value.get('state')
                    if state in ('outcome_unknown', 'rejected'):
                        raise BrokerOutcomeUnknown('Root job is ' + state + '; no automatic replay')
                    if self.terminal_ready(job, value):
                        code = value['exit_code']
                        self.active_broker_job = None
                        if code == 75 and value.get('command_started') is False:
                            # The root admission refused before native work began.
                            attempt['retry_reason'] = 'root_admission_refused_before_command'
                            return code, True
                        if code == 85 and not train:
                            return code, True
                        return code, interrupted_evaluation
                    if state == 'running' and value.get('command_started'):
                        sample = self.admission()
                        gate.write(self.out / 'admission_running_latest.json', sample)
                        if not sample['safe'] or self.stopping:
                            self.request_yield('target_resource_changed_or_controller_stop')
                            self.update('yield_requested', active_stage=key)
                            yield_deadline = yield_deadline or (time.monotonic() + request['pause_grace_seconds'] + 40)
                            interrupted_evaluation = not train
                    elif self.stopping:
                        self.request_yield('controller_stop_before_native_command', abort=True)
                    if yield_deadline and time.monotonic() >= yield_deadline:
                        raise TimeoutError('Project wrapper did not finish its owned native-child cleanup')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Root broker stage exceeded its frozen deadline')
            except BaseException:
                self.request_yield('controller_exception_fail_closed', abort=True)
                self.stop_owned_child(train, grace=0)
                raise
            finally:
                self.broker_log = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    runner = BrokerRunner(args.plan)
    try:
        runner.run()
    except BaseException as error:
        runner.update('paused' if isinstance(error, InterruptedError) else 'failed',
                      error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
