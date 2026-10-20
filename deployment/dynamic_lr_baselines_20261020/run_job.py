"""Execute frozen author stages serially inside an admitted remote GPU job.

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
    assert job['original_LR'] and job['no_proposed_modules']
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
    record = dict(schema='dynamic_lr_remote_execution_v1', job_id=job['job_id'], method=job['method'], scene=job['scene'],
                  original_LR=True, no_proposed_modules=True, host=os.uname().nodename,
                  job_sha256=sha(source), gpu_uuid=job['gpu_uuid'], broker_context=context,
                  started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    locks = []
    try:
        if (ROOT / 'activate_cts.sh').exists():
            shared = ROOT / 'deployment/cts_20260921/project_access.lock'
            for path, mode in [(shared, fcntl.LOCK_SH), (shared.with_name('gpu0.lock'), fcntl.LOCK_EX)]:
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                locks.append(fd)
            if (ROOT / 'deployment/cts_20260921/sync_pending.json').exists():
                raise ValueError('Existing CTS shared deployment has an unverified synchronization')
        environment = os.environ.copy()
        environment.update({key: str(value).replace('{ROOT}', str(ROOT)) for key,value in job.get('env', {}).items()})
        environment.setdefault('CUDA_HOME', str(ROOT / 'tools/cuda-12.8-minimal'))
        environment.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
        environment.setdefault('MPLBACKEND', 'Agg')
        environment.setdefault('PYTHONFAULTHANDLER', '1')
        environment['PATH'] = environment['CUDA_HOME'] + '/bin:' + environment.get('PATH','/usr/bin:/bin')
        environment['LD_LIBRARY_PATH'] = ':'.join([str(ROOT/'tools/runtime-libs'), environment['CUDA_HOME']+'/lib64', environment.get('LD_LIBRARY_PATH','')])
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
        write(out/'status.json', dict(record, status='failed', elapsed_seconds=time.monotonic()-began, error=traceback.format_exc()))
        raise
    finally:
        for fd in locks:
            os.close(fd)


if __name__ == '__main__':
    main()
