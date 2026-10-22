"""Execute audited static-SR stages serially inside the admitted CTS snapshot.

The existing broker owns GPU admission and persistence. A training return code
of zero immediately starts its registered rendering/evaluation command.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def diagnostics(record, out, error=None, tail_limit=65536):
    """Keep bounded failure text, with the full log identity left auditable."""
    stages = record.get('stages', [])
    result = dict(schema='static_sr_cts_diagnostics_v1', job_id=record['job_id'],
                  status='failed' if error is not None else 'completed',
                  stages=[{key: stage[key] for key in ('name', 'returncode', 'seconds') if key in stage}
                          for stage in stages], exception=error)
    if error is None or not stages:
        return result
    stage = stages[-1]
    path = out / (stage['name'] + '.log')
    result['failed_stage'] = dict(name=stage['name'], log=dict(path=str(path)))
    log = result['failed_stage']['log']
    try:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                value.update(chunk)
            stream.seek(max(0, before.st_size - tail_limit))
            tail = stream.read(tail_limit)
            after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Stage log changed while diagnostics were collected')
        log.update(bytes=before.st_size, sha256=value.hexdigest(), tail_bytes=len(tail),
                   tail_truncated=before.st_size > tail_limit, tail_text=tail.decode('utf-8', errors='replace'))
    except (OSError, ValueError) as log_error:
        log['read_error'] = type(log_error).__name__ + ': ' + str(log_error)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    source = Path(args.job).resolve()
    job = json.loads(source.read_text())
    context_path = Path(os.environ['CLAUDE_REMOTE_JOB_DIR']) / 'execution_context.json'
    context = json.loads(context_path.read_text())
    assert context['project_id'] == '4dsr'
    assert context['selected_gpu_uuids'] == [job['gpu_uuid']]
    assert job['training_only_inputs'] and job['no_proposed_modules']
    for entry in job.get('required_files', []):
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen identity mismatch: ' + entry['path'])
    if job.get('input_receipt'):
        receipt = json.loads((ROOT / job['input_receipt']).read_text())
        assert receipt['status'] == 'verified'
        assert receipt['inventory_sha256'] == job['input_inventory_sha256']
    out = ROOT / job['out']
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, out / 'job.json')
    began = time.monotonic()
    record = dict(schema='static_sr_cts_execution_v1', job_id=job['job_id'], method=job['method'], scene=job['scene'],
                  original_LR=True, no_proposed_modules=True, host=os.uname().nodename,
                  job_sha256=sha(source), gpu_uuid=job['gpu_uuid'], broker_context=context,
                  started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    locks = []
    failure = None
    try:
        if job.get('shared_project_root'):
            shared_root = Path(job['shared_project_root'])
            if shared_root != Path('/home/cts/Project/4DSR') or not ROOT.is_relative_to(shared_root):
                raise ValueError('Unexpected CTS snapshot location')
            shared = shared_root / 'deployment/cts_20260921/project_access.lock'
            for path, mode in [(shared, fcntl.LOCK_SH), (shared.with_name('gpu0.lock'), fcntl.LOCK_EX)]:
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                locks.append(fd)
            if (shared_root / 'deployment/cts_20260921/sync_pending.json').exists():
                raise ValueError('Existing CTS shared deployment has an unverified synchronization')
        environment = os.environ.copy()
        environment.update({key: str(value).replace('{ROOT}', str(ROOT)) for key,value in job.get('env', {}).items()})
        environment.setdefault('CUDA_HOME', str(Path(job['shared_project_root']) / 'tools/cuda-12.8-minimal'))
        environment.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
        environment.setdefault('MPLBACKEND', 'Agg')
        environment.setdefault('HF_HUB_OFFLINE', '1')
        environment.setdefault('TRANSFORMERS_OFFLINE', '1')
        environment.setdefault('FOURDSR_UPSTREAM', str(ROOT / 'third_party/4dgs_843d5ac_runtime'))
        environment.setdefault('PYTHONFAULTHANDLER', '1')
        environment['PATH'] = environment['CUDA_HOME'] + '/bin:' + environment.get('PATH','/usr/bin:/bin')
        environment['LD_LIBRARY_PATH'] = ':'.join([str(Path(job['shared_project_root'])/'tools/runtime-libs'), environment['CUDA_HOME']+'/lib64', environment.get('LD_LIBRARY_PATH','')])
        for command in job['commands']:
            argv = [str(value).replace('{ROOT}', str(ROOT)) for value in command['argv']]
            cwd = ROOT / command.get('cwd', '.')
            assert cwd.resolve().is_relative_to(ROOT)
            stage = dict(name=command['name'], argv=argv, cwd=str(cwd), started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(stage)
            write(out/'status.json', dict(record, status='running', elapsed_seconds=time.monotonic()-began))
            before = time.monotonic()
            with (out/(command['name']+'.log')).open('x') as stream:
                process = subprocess.run(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
            stage.update(returncode=process.returncode, seconds=time.monotonic()-before)
            write(out/'status.json', dict(record, status='stage_completed' if process.returncode==0 else 'failed', elapsed_seconds=time.monotonic()-began))
            if process.returncode:
                raise RuntimeError('Stage failed: ' + command['name'] + ' rc=' + str(process.returncode))
        result = dict(record, status='completed', elapsed_seconds=time.monotonic()-began,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'status.json', result)
        write(out/'complete.json', result)
    except BaseException:
        failure = traceback.format_exc()
        write(out/'status.json', dict(record, status='failed', elapsed_seconds=time.monotonic()-began, error=failure))
        raise
    finally:
        try:
            write(out/'diagnostics.json', diagnostics(record, out, failure))
        finally:
            for fd in locks:
                os.close(fd)


if __name__ == '__main__':
    main()
