"""Run one project-contained native stage inside the existing GPU broker.

The host controls admission; the broker owns the reservation and confinement.
This worker supervises only its own native child and never touches another job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENV_KEYS = {
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_HOME',
    'TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR', 'MULTI4D_UPSTREAM',
    'MAX_JOBS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MPLBACKEND', 'PYTHONUNBUFFERED', 'PYTHONHASHSEED', 'PYTHONNOUSERSITE',
    'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'PATH',
}
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'


def contained(value):
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Stage path escapes the registered project: ' + str(path))
    return path


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate(request):
    if request.get('schema') != 'multi4d_broker_stage_v1':
        raise ValueError('Unrecognized stage request')
    jid = request.get('job_id', '')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', jid):
        raise ValueError('Invalid broker job identity')
    if os.getuid() != pwd.getpwnam('claude-training').pw_uid:
        raise ValueError('This entry must run as the broker worker, never in the host environment')
    expected_cgroup = '0::/claudevpn.slice/claude-local-' + jid + '.service'
    if Path('/proc/self/cgroup').read_text().strip() != expected_cgroup:
        raise ValueError('Broker job cgroup identity mismatch')
    command = request['command']
    if not isinstance(command, list) or len(command) < 2 or not all(isinstance(x, str) for x in command):
        raise ValueError('Invalid native command')
    python = contained(command[0])
    if python.name not in ('python', 'python3') or python.parent.name != 'bin':
        raise ValueError('Expected the project-contained native Python')
    if contained(command[1]).resolve() not in {(HERE / 'train.py').resolve(), (HERE / 'evaluate.py').resolve()}:
        raise ValueError('Only the reviewed native train/evaluation entry is allowed')
    if contained(request['cwd']).resolve() != ROOT:
        raise ValueError('Unexpected working directory')
    for name in ('pause_file', 'abort_file', 'wrapper_receipt'):
        contained(request[name])
    if not isinstance(request['train'], bool):
        raise ValueError('Missing training/evaluation distinction')
    if not 1 <= request['stage_timeout_seconds'] <= 86400 or request['pause_grace_seconds'] != 180:
        raise ValueError('Unreviewed stage timeout or checkpoint grace')
    environment = request['environment']
    if not isinstance(environment, dict) or set(environment) - ENV_KEYS:
        raise ValueError('Unexpected environment fields; never import host credentials')
    if not all(isinstance(v, str) and '\x00' not in v for v in environment.values()):
        raise ValueError('Invalid environment value')
    if environment.get('CUDA_VISIBLE_DEVICES') != PRO6000 or os.environ.get('CUDA_VISIBLE_DEVICES') != PRO6000:
        raise ValueError('The frozen native stage must use only the broker-admitted PRO6000')
    prefix = python.parent.parent
    expected_path = str(prefix / 'bin') + ':/usr/local/cuda-12.8/bin:/usr/bin:/bin'
    if environment.get('CONDA_PREFIX') != str(prefix) or environment.get('LD_LIBRARY_PATH') != str(prefix / 'lib'):
        raise ValueError('Native runtime environment identity mismatch')
    if environment.get('PATH') != expected_path:
        raise ValueError('Unexpected native executable search path')
    contained(environment['MULTI4D_UPSTREAM'])
    contained(environment['TORCH_EXTENSIONS_DIR'])
    return command


def verify_sources(request):
    sources = request.get('source_sha256')
    if not isinstance(sources, dict) or not sources or str(Path(__file__).resolve()) not in sources:
        raise ValueError('The host must bind this wrapper and scientific sources to the request')
    for name, digest in sources.items():
        path = contained(name)
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid frozen source digest')
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                value.update(block)
        if value.hexdigest() != digest:
            raise ValueError('Frozen source changed before/after native execution: ' + name)
    return len(sources)


def execute(request):
    command = validate(request)
    source_count = verify_sources(request)
    os.umask(0o007)
    environment = dict(os.environ)
    environment.update(request['environment'])
    # The broker worker has a different UID from the source owner. Authorize
    # only this already-hashed project snapshot for Git's ownership check.
    environment.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory',
                       GIT_CONFIG_VALUE_0=environment['MULTI4D_UPSTREAM'])
    # Only the preverified AlexNet checkpoint is placed here by the host.
    # The sandbox has no network and must never download weights implicitly.
    environment['TORCH_HOME'] = str(Path(environment['TORCH_EXTENSIONS_DIR']).parent / 'torch')
    pause, abort = Path(request['pause_file']), Path(request['abort_file'])
    receipt = Path(request['wrapper_receipt'])
    child = None
    received_signals = []
    old_handlers = {}
    started = time.monotonic()
    pause_started = None
    termination_reason = None
    result = dict(schema='multi4d_broker_wrapper_receipt_v1', job_id=request['job_id'],
                  wrapper_pid=os.getpid(), command=command, started_unix=time.time(),
                  train=request['train'], source_count=source_count, status='starting')

    def requested(number, _frame):
        received_signals.append(number)

    def stop_child():
        nonlocal termination_reason
        if child is None:
            return
        # This process group is created by this exact Popen, never discovered by
        # a name match or selected from another task's PID.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        # Retire surviving workers even if the original native leader exited.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[number] = signal.signal(number, requested)
    try:
        child = subprocess.Popen(command, cwd=request['cwd'], env=environment,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        result.update(status='running', native_pid=child.pid)
        try:
            result['native_start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        except FileNotFoundError:
            result['native_already_exited'] = child.poll() is not None
        atomic(receipt, result)
        while True:
            try:
                code = child.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            signalled = bool(received_signals)
            if (signalled or abort.exists()) and not pause.exists():
                atomic(pause, dict(reason='own_broker_wrapper_signal_or_abort', signals=received_signals))
            requested_pause = pause.exists() or abort.exists() or signalled
            if requested_pause:
                pause_started = now if pause_started is None else pause_started
                if not request['train']:
                    termination_reason = 'evaluation_interrupted_for_resource_release'
                    stop_child()
                    code = 85
                    break
                if now - pause_started >= request['pause_grace_seconds']:
                    termination_reason = 'native_checkpoint_grace_exhausted'
                    stop_child()
                    code = 86
                    break
            if now - started >= request['stage_timeout_seconds']:
                termination_reason = 'native_stage_deadline_exhausted'
                stop_child()
                code = 87
                break
        result.update(status='ended', exit_code=code, termination_reason=termination_reason,
                      ended_unix=time.time(), elapsed_s=time.monotonic() - started,
                      received_signals=received_signals)
        result['sources_unchanged_after_execution'] = verify_sources(request) == source_count
        # Native train exit 85 means its complete checkpoint is already saved.
        atomic(receipt, result)
        return code if code >= 0 else 128 - code
    except BaseException as error:
        termination_reason = 'wrapper_error'
        # A supervision/logging error still grants training its remaining
        # checkpoint grace. An earlier request never starts a second 180s.
        if request['train'] and child is not None and child.poll() is None:
            pause_started = time.monotonic() if pause_started is None else pause_started
            try:
                if not pause.exists():
                    atomic(pause, dict(reason='own_wrapper_error', error=repr(error)))
            except OSError as pause_error:
                result['pause_write_error'] = repr(pause_error)
            remaining = max(0, request['pause_grace_seconds'] - (time.monotonic() - pause_started))
            try:
                result['native_exit_during_error_grace'] = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                result['error_grace_exhausted'] = True
        stop_child()
        result.update(status='failed', error=repr(error), ended_unix=time.time(),
                      elapsed_s=time.monotonic() - started)
        atomic(receipt, result)
        raise
    finally:
        if child is not None:
            stop_child()
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    raw = contained(args.request).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError('Stage request changed after the host submitted it')
    return execute(json.loads(raw))


if __name__ == '__main__':
    sys.exit(main())
