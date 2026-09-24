#!/usr/bin/env python3
"""Sequential U/W/F batch and event-driven remote dispatch/verified return.

run --spec JSON --out NEW_DIR executes one scene's exact U,W,F sequence.
The spec contains scene, common runner arguments (underscore keys), methods
{U:{},W:{},F:{calibration:...}}, nonempty file_identities and optional
required_receipts. It must use absolute paths valid on the execution host.

dispatch runs this module in a persistent remote systemd unit, waits for its
exit event, then returns all outputs and verifies a streaming hash manifest.
Run dispatch itself under local systemd. Source/data deployment and frozen
teacher preparation must already be finished; this tool does not sync inputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback

from run_job import HERE, ROOT, identity, read, sha, stamp, write, soft


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_spec(spec, calibration_pending=False):
    common = spec['common']
    check(set(spec['methods']) == {'U', 'W', 'F'}, 'Exactly U/W/F; no additional branch')
    required = {'manifest', 'parent', 'selection', 'teacher_index', 'schedule', 'roi_protocol',
                'original_prior_cameras', 'train_gpu', 'eval_gpu'}
    check(required <= set(common), 'Incomplete common runner arguments')
    allowed = required | {'eval_dev_1200'}
    check(set(common) <= allowed, 'Unexpected common runner argument')
    check(read(common['manifest'])['scene'] == spec['scene'], 'Scene/manifest mismatch')
    for method in ('U', 'W', 'F'):
        options = spec['methods'][method]
        check(set(options) <= {'calibration'}, 'Per-method overrides may only provide calibration')
        if method == 'F':
            check(options.get('calibration'), 'F needs its fixed calibration')
    rows = spec.get('file_identities', [])
    check(rows, 'Refusing batch without a frozen deployment identity list')
    indexed = {}
    for row in rows:
        path = Path(row['path'])
        check(path.is_absolute() and path.is_file(), f'Missing absolute input: {path}')
        check(sha(path) == row['sha256'], f'Input SHA mismatch: {path}')
        if 'bytes' in row:
            check(path.stat().st_size == row['bytes'], f'Input size mismatch: {path}')
        indexed[str(path.resolve())] = row['sha256']
    for key in ('manifest', 'parent', 'selection', 'teacher_index', 'schedule', 'roi_protocol'):
        check(str(Path(common[key]).resolve()) in indexed, f'Unregistered input identity: {key}')
    if not calibration_pending:
        check(str(Path(spec['methods']['F']['calibration']).resolve()) in indexed, 'Unregistered calibration identity')
    for receipt in spec.get('required_receipts', []):
        actual = read(receipt['path'])
        check(actual.get('status') == receipt['status'], f'Prerequisite is not ready: {receipt["path"]}')
        if receipt.get('sha256'):
            check(sha(receipt['path']) == receipt['sha256'], f'Prerequisite changed: {receipt["path"]}')
    return common


def runner_command(common, method, extra, out):
    args = dict(common, **extra, method=method, out=str(out))
    command = [sys.executable, str(HERE / 'queued_job.py')]
    for key, value in args.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    return command


def return_manifest(out, status):
    rows = []
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.name == 'return_manifest.json' or '.transport' in path.parts:
            continue
        record = identity(path)
        rows.append({'path': str(path.relative_to(out)), 'sha256': record['sha256'],
                     'bytes': record['bytes'], 'symlink': path.is_symlink()})
    write(out / 'return_manifest.json', {'status': status, 'root': str(out), 'host': socket.gethostname(),
                                       'files': rows, 'finished_utc': stamp()})


def run(args):
    out = args.out.resolve()
    check(not out.exists(), f'Output already exists: {out}')
    spec = read(args.spec)
    # A partial or invalid deployment cannot create a formally running batch.
    common = validate_spec(spec, calibration_pending=args.calibrate)
    out.mkdir(parents=True)
    tick = time.monotonic()
    state = {'status': 'running_sequential_batch', 'scene': spec['scene'], 'order': ['U', 'W', 'F'],
             'spec': identity(args.spec), 'source': identity(__file__), 'host': socket.gethostname(),
             'pid': os.getpid(), 'python': sys.executable, 'started_utc': stamp(), 'jobs': {}}
    write(out / 'batch_spec.json', spec)
    write(out / 'status.json', state)
    try:
        if args.calibrate:
            calibration = Path(spec['methods']['F']['calibration']).resolve()
            check(calibration == out / 'calibration/calibration.json',
                  'New calibration must be inside this new batch output')
            command = [sys.executable, str(HERE / 'calibrate.py'), '--manifest', common['manifest'],
                       '--checkpoint', common['parent'], '--selection', common['selection'],
                       '--teacher-index', common['teacher_index'], '--schedule', common['schedule'],
                       '--out', str(calibration.parent)]
            start = time.monotonic()
            event = {'status': 'waiting_for_gpu', 'command': command, 'started_utc': stamp()}
            write(out / 'calibration_event.json', event)
            with soft.lock_file(f'/tmp/4dsr_soft_gpu_{common["train_gpu"]}.lock'):
                event['gpu_before'] = soft.gpu_snapshot(common['train_gpu'])
                write(out / 'calibration_event.json', event)
                soft.require_available(event['gpu_before'])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=common['train_gpu'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           OMP_NUM_THREADS='4', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
                with (out / 'calibration.log').open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event.update(status='completed' if result.returncode == 0 else 'failed', returncode=result.returncode,
                         elapsed_seconds=time.monotonic() - start, finished_utc=stamp())
            write(out / 'calibration_event.json', event)
            check(result.returncode == 0, 'Calibration failed; no formal branch launched')
            receipt = read(calibration.parent / 'complete.json')
            check(receipt.get('status') == 'completed_calibration' and
                  receipt['calibration_sha256'] == sha(calibration), 'Calibration receipt mismatch')
            check(read(calibration)['status'] == 'calibrated', 'Degenerate calibration; no branch launch')
            spec['file_identities'].append(identity(calibration))
            spec.setdefault('required_receipts', []).append({'path': str(calibration.parent / 'complete.json'),
                'status': 'completed_calibration', 'sha256': sha(calibration.parent / 'complete.json')})
            validate_spec(spec)
            write(out / 'batch_spec.json', spec)
        for method in state['order']:
            dest = out / method
            command = runner_command(common, method, spec['methods'][method], dest)
            event = {'status': 'running', 'command': command, 'started_utc': stamp()}
            start = time.monotonic()
            write(out / f'{method}_event.json', event)
            with (out / f'{method}_service.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            event.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - start,
                         finished_utc=stamp(), status='completed' if result.returncode == 0 else 'failed')
            write(out / f'{method}_event.json', event)
            check(result.returncode == 0, f'{method} failed; remaining methods not launched')
            receipt = read(dest / 'complete.json')
            check(receipt.get('status') == 'completed_and_evaluated', f'{method} completion mismatch')
            state['jobs'][method] = {'out': str(dest), 'complete': identity(dest / 'complete.json'),
                                     'methods_entry': identity(dest / 'methods-entry.json')}
            write(out / 'status.json', state)
        # This gate concerns the registered draw stream, never quality scores.
        draws = [read(out / m / 'train_receipt.json')['draw_sha256'] for m in state['order']]
        check(draws[0] and len(set(draws)) == 1, 'U/W/F realized draw streams differ or are absent')
        state.update(status='completed_sequential_batch', draw_sha256=draws[0],
                     elapsed_seconds=time.monotonic() - tick, finished_utc=stamp())
        write(out / 'complete.json', state)
        write(out / 'status.json', state)
        return_manifest(out, state['status'])
    except BaseException:
        state.update(status='failed_sequential_batch', traceback=traceback.format_exc(),
                     elapsed_seconds=time.monotonic() - tick, finished_utc=stamp())
        write(out / 'failed.json', state)
        write(out / 'status.json', state)
        return_manifest(out, state['status'])
        raise


def receive(host, remote_out, out, record_dir):
    check(not any(p.name != '.transport' for p in out.iterdir()), 'Refusing to overwrite existing returned files')
    command = ['rsync', '-a', '--partial', '--timeout=90', '-e',
               'ssh -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=3',
               host + ':' + remote_out.rstrip('/') + '/', str(out) + '/']
    with (record_dir / 'rsync.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    manifest = read(out / 'return_manifest.json')
    rows = []
    for row in manifest['files']:
        relative = Path(row['path'])
        check(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe returned path')
        path = out / relative
        check(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], f'Returned identity mismatch: {path}')
        rows.append({'recorded_path': manifest['root'] + '/' + str(relative), 'local_path': str(path),
                     'sha256': row['sha256'], 'bytes': row['bytes']})
    write(record_dir / 'returned_file_index.json', {'status': 'all_returned_hashes_match', 'host': host, 'files': rows})
    check(manifest['status'] == 'completed_sequential_batch', 'Remote failure preserved on return')
    check(read(out / 'complete.json')['status'] == 'completed_sequential_batch', 'Missing successful batch receipt')


def dispatch(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = out / '.transport'
    records.mkdir()
    started = time.monotonic()
    state = {'status': 'starting_remote_persistent_batch', 'host': args.host, 'unit': args.unit,
             'remote_root': args.remote_root, 'remote_out': args.remote_out, 'local_out': str(out),
             'started_utc': stamp(), 'inputs_predeployed': True}
    write(records / 'status.json', state)
    try:
        module = args.remote_root.rstrip('/') + '/experiments/dynamic_sr_detail_supervision_20260924/remote_batch.py'
        # Only selected host aliases; all shell values are quoted literally.
        run_args = [module, 'run', '--spec', args.remote_spec, '--out', args.remote_out]
        if args.calibrate:
            run_args.append('--calibrate')
        shell = 'source ' + shlex.quote(args.activation) + '\nexec python ' + shlex.join(run_args)
        remote = ['systemd-run', '--user', '--wait', '--collect', '--unit=' + args.unit,
                  '--property=WorkingDirectory=' + args.remote_root,
                  '--property=StandardOutput=journal', '--property=StandardError=journal',
                  '/bin/bash', '-lc', shell]
        command = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
                   '-o', 'ServerAliveCountMax=3', args.host, shlex.join(remote)]
        with (records / 'remote_wait.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
        state.update(status='remote_service_returned', returncode=result.returncode)
        write(records / 'status.json', state)
        if result.returncode == 255:
            raise RuntimeError('SSH transport failed; remote service may persist. No blind relaunch or assumed completion.')
        receive(args.host, args.remote_out, out, records)
        check(result.returncode == 0, 'Remote systemd command failed; returned records preserved')
        state.update(status='completed_remote_batch_and_verified_return', elapsed_seconds=time.monotonic() - started,
                     finished_utc=stamp())
        write(records / 'complete.json', state)
        write(records / 'status.json', state)
    except BaseException:
        state.update(status='failed_dispatch_or_return', traceback=traceback.format_exc(),
                     elapsed_seconds=time.monotonic() - started, finished_utc=stamp())
        write(records / 'failed.json', state)
        write(records / 'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    local = subs.add_parser('run')
    local.add_argument('--spec', type=Path, required=True)
    local.add_argument('--out', type=Path, required=True)
    local.add_argument('--calibrate', action='store_true', help='First calibrate on the same training GPU; no branch if it fails')
    remote = subs.add_parser('dispatch')
    remote.add_argument('--host', choices=('a100-train', 'cts'), required=True)
    for key in ('remote-root', 'remote-spec', 'remote-out', 'activation', 'unit'):
        remote.add_argument('--' + key, required=True)
    remote.add_argument('--out', type=Path, required=True)
    remote.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        run(args)
    else:
        dispatch(args)


if __name__ == '__main__':
    main()
