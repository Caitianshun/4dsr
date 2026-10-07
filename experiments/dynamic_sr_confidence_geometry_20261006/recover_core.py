"""Bounded, identity-checked recovery of the A100 repeat-2 core endpoints.

This local coordinator reuses completed B2perm and the frozen worker APIs.  It
never changes a training schedule/loss, and retries only SSH transport failures.
Run CPU acceptance first; --execute is deliberately explicit.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import traceback

from cg_common import ROOT, HERE, OUT, read, write, sha, entry
import worker_async as worker

HOST = 'a100-train'
GPU = '1'
REMOTE_OUT = worker.REMOTE + '/output/' + HERE.name
RECOVERY_UNIT = '4dsr-cg-a100-core-recovery-r2-20261006'
REGISTRATION = OUT / 'core_recovery_registration.json'
STATE = OUT / 'core_recovery_state.json'
INCIDENTS = OUT / 'recovery_incidents'
PLACEHOLDER_NAMES = {'retrieval_manifest.json', 'retrieval_journal.log'}
BACKOFF = (5, 15, 30)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def archive_retrieval_only(directory, incidents):
    """The same CPU-only function executes locally and in the remote helper."""
    directory, incidents = Path(directory), Path(incidents)
    if not directory.exists():
        return {'status': 'absent_no_archive', 'path': str(directory)}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Not an ordinary run directory')
    children = list(directory.iterdir())
    if not children or any(p.is_symlink() or not p.is_file() for p in children):
        raise ValueError('Only registered failed retrieval files may be archived')
    if {p.name for p in children} != PLACEHOLDER_NAMES:
        raise ValueError('Non-retrieval run contents preserved; refusing archive')
    manifest = json.loads((directory / 'retrieval_manifest.json').read_text())
    expected_unit = '4dsr-cg-' + directory.name.lower().replace('_', '-') + '-20261006'
    if manifest.get('task') != directory.name or manifest.get('unit') != expected_unit:
        raise ValueError('Placeholder task/unit ownership mismatch')
    items = manifest.get('files')
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError('Unexpected placeholder retrieval inventory')
    item = items[0]
    journal = directory / 'retrieval_journal.log'
    if (item.get('path') != 'runs/' + directory.name + '/retrieval_journal.log'
            or item.get('sha256') != digest_file(journal)
            or item.get('bytes') != journal.stat().st_size):
        raise ValueError('Placeholder journal inventory mismatch')
    files = {p.name: digest_file(p) for p in sorted(children)}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    destination = incidents / (directory.name + '_retrieval_only_' + fingerprint[:16])
    incidents.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError('Archive destination already exists; source preserved')
    directory.rename(destination)
    return {'status': 'archived_retrieval_only', 'original': str(directory),
            'archive': str(destination), 'files': files, 'fingerprint': fingerprint,
            'training_or_checkpoint_files_present': False}


def is_ssh_transport_failure(error):
    if isinstance(error, subprocess.CalledProcessError):
        command = error.cmd
        first = Path(command[0]).name if isinstance(command, (list, tuple)) else ''
        return error.returncode == 255 and first in ('ssh', 'rsync')
    if isinstance(error, RuntimeError) and isinstance(error.args[0] if error.args else None, str):
        value = str(error).lower()
        return any(x in value for x in (
            'ssh: connect to host', 'connection reset by peer', 'connection closed by',
            'connection timed out', 'broken pipe', 'ssh_dispatch_run_fatal'))
    return False


def bounded_transport(operation, reconnect, report, sleep=time.sleep):
    """Retries use stable identities, not new jobs; normal waiting stays pidfd."""
    for attempt in range(len(BACKOFF) + 1):
        try:
            return operation()
        except Exception as error:
            transport = is_ssh_transport_failure(error)
            report(attempt, error, transport)
            if not transport or attempt == len(BACKOFF):
                raise
            sleep(BACKOFF[attempt])
            # Recovery probes receipts/unit state only after a connection error.
            # An error here is bounded by the next operation attempt as well.
            try:
                reconnect()
            except Exception as probe_error:
                report(attempt, probe_error, is_ssh_transport_failure(probe_error))
                if not is_ssh_transport_failure(probe_error):
                    raise


def ssh_cpu(code, *args, stdin=None):
    command = ['ssh', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
               HOST, shlex.join(['python3', '-c', code, *map(str, args)])]
    cp = subprocess.run(command, input=stdin, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    if cp.returncode:
        raise subprocess.CalledProcessError(cp.returncode, command, cp.stdout, cp.stderr)
    return json.loads(cp.stdout)


def remote_endpoint(method):
    code = '''import pathlib,json,hashlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);out=root/'output/dynamic_sr_confidence_geometry_20261006'
method=sys.argv[2];run=out/'runs'/('r2_'+method)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
unit='4dsr-cg-r2-'+method.lower()+'-20261006'
p=subprocess.run(['systemctl','--user','show',unit,'--no-pager',*[f'--property={k}' for k in ('LoadState','ActiveState','SubState','MainPID','Result','ExecMainStatus')]],text=True,capture_output=True)
if p.returncode and 'LoadState=not-found' not in p.stdout:raise RuntimeError(p.stderr or p.stdout)
state=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)
value={'method':method,'unit':unit,'unit_state':state,'run_exists':run.exists(),'files':[x.name for x in run.iterdir()] if run.exists() else []}
for name in ('config.json','complete.json'):
 path=run/name
 if path.exists():value[name]={'data':json.loads(path.read_text()),'sha256':digest(path)}
if 'complete.json' in value:
 receipt=value['complete.json']['data'];path=pathlib.Path(receipt['checkpoint']['path'])
 path=path if path.is_absolute() else root/path
 if path.resolve()!=(run/'checkpoint_12000.pt').resolve():raise ValueError('Remote endpoint path mismatch')
 value['checkpoint_actual_sha256']=digest(path)
print(json.dumps(value))
'''
    return ssh_cpu(code, worker.REMOTE, method)


def validate_remote_completed(value, registration, method):
    receipt = value.get('complete.json', {}).get('data', {})
    config = value.get('config.json', {}).get('data', {})
    if (receipt.get('status') != 'completed_training' or receipt.get('updates') != 6000
            or receipt.get('method') != method or str(receipt.get('repeat')) != '2'
            or config.get('method') != method or str(config.get('repeat')) != '2'
            or config.get('protocol_sha256') != registration['protocol']['sha256']
            or config.get('schedule_sha256') != registration['schedule']['sha256']
            or config.get('parent') != registration['parent']
            or receipt.get('checkpoint', {}).get('sha256') != value.get('checkpoint_actual_sha256')):
        raise ValueError('Remote completed endpoint identity mismatch: ' + method)
    if value['unit_state'].get('ActiveState') == 'failed':
        raise RuntimeError('Existing training unit failed; preserving evidence')
    expected_source = dict(registration['training_source_identity'])
    if method in ('R', 'RG'):
        for name in ('degradation_operator.py', 'confidence_cache.py'):
            rel = 'experiments/' + HERE.name + '/' + name
            expected_source[rel] = registration['critical_files'][rel]
    if method in ('G', 'RG'):
        rel = 'experiments/' + HERE.name + '/depth_prior.py'
        expected_source[rel] = registration['critical_files'][rel]
    if config.get('source_identity') != expected_source or str(config.get('physical_gpu')) != GPU:
        raise ValueError('Completed A100 training source/GPU identity mismatch')
    if method == 'B2perm' and (
            receipt['checkpoint']['sha256'] != registration['completed_B2_checkpoint']['sha256']
            or value['complete.json']['sha256'] != registration['completed_B2_receipt']['sha256']):
        raise ValueError('B2 differs from the already retrieved fixed endpoint')
    return receipt


def verify_local_retrieval(task):
    receipt = read(OUT / (task + '_retrieval_complete.json'))
    if (receipt.get('status') != 'all_remote_files_verified' or receipt.get('host') != HOST
            or not receipt.get('training_complete')):
        raise ValueError('Local full retrieval receipt is incomplete')
    manifest_path = worker.bound(receipt['manifest'])
    manifest = read(manifest_path)
    expected_unit = '4dsr-cg-' + task.lower().replace('_', '-') + '-20261006'
    if manifest.get('task') != task or manifest.get('unit') != expected_unit or receipt.get('unit') != expected_unit:
        raise ValueError('Local retrieval task mismatch')
    for item in manifest['files']:
        path = (OUT / item['path']).resolve()
        if not path.is_relative_to(OUT.resolve()) or digest_file(path) != item['sha256']:
            raise ValueError('Local full retrieval hash mismatch')
    return worker.verify_training(task, task.split('_', 1)[1], '2')


def prepare_b2(remote_proof, existing_complete, transfer, verify, dispatch):
    """No B2 training function is accepted by this recovery path."""
    if not existing_complete():
        transfer()
    checkpoint = verify()
    if digest_file(checkpoint) != remote_proof['checkpoint']['sha256']:
        raise ValueError('Local B2 endpoint differs from strictly checked remote endpoint')
    return checkpoint, dispatch(checkpoint)


def verify_remote_inputs(registration):
    raw = (OUT / 'deployment_files.json').read_text()
    code = '''import pathlib,hashlib,json,sys
root=pathlib.Path(sys.argv[1]);payload=sys.stdin.read();inventory=json.loads(payload)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for item in inventory['files']:
 rel=pathlib.Path(item['path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe deployment path')
 if digest(root/rel)!=item['sha256']:raise ValueError('Deployment identity changed: '+str(rel))
critical=json.loads(sys.argv[2]);source=json.loads(sys.argv[3])
for rel,want in critical.items():
 if digest(root/rel)!=want:raise ValueError('Critical frozen identity changed: '+rel)
for rel,want in source.items():
 path=root/'vendor/4dgs'/rel[9:] if rel.startswith('upstream/') else root/rel
 if digest(path)!=want:raise ValueError('Actual training import changed: '+rel)
print(json.dumps({'status':'passed_frozen_recovery_inventory','file_count':len(inventory['files']),'inventory_sha256':hashlib.sha256(payload.encode()).hexdigest(),'critical_sha256':critical,'source_identity':source,'GPU_calls':0}))
'''
    result = ssh_cpu(code, worker.REMOTE, json.dumps(registration['critical_files']),
                     json.dumps(registration['training_source_identity']), stdin=raw)
    if result['inventory_sha256'] != registration['deployment_inventory']['sha256']:
        raise ValueError('Recovery deployment inventory changed')
    result.update(registration=entry(REGISTRATION), source=entry(Path(__file__)))
    write(OUT / 'remote_recovery_verification.json', result)
    return result


def preflight_remaining(registration):
    for method in ('R', 'G', 'RG'):
        proof = remote_endpoint(method)
        state = proof['unit_state']
        if state.get('ActiveState') == 'failed':
            raise RuntimeError('Existing training unit genuinely failed: ' + method)
        if 'complete.json' in proof:
            validate_remote_completed(proof, registration, method)
            if not (OUT / 'runs' / ('r2_' + method) / 'complete.json').exists():
                if int(state.get('MainPID', '0')):
                    worker.wait_unit(proof['unit'], HOST)
                worker.retrieve_remote(HOST, 'r2_' + method, OUT / ('r2_' + method + '_recovery_retrieval.log'))
            worker.verify_training('r2_' + method, method, '2')
            continue
        if not proof['run_exists']:
            # An SSH loss can occur after remote rename but before its reply.
            # Finish archiving the still-local retrieval placeholder on retry.
            local = OUT / 'runs' / ('r2_' + method)
            if local.exists():
                if method != 'R':
                    raise ValueError('Unexpected local run without remote ownership')
                archived = archive_retrieval_only(local, INCIDENTS)
                write(INCIDENTS / ('R_local_after_remote_archive_' + str(time.time_ns()) + '.json'), archived)
            continue
        if 'config.json' in proof:
            config = proof['config.json']['data']
            dispatch = read(OUT / ('dispatch_r2_' + method + '.json'))
            if (not int(state.get('MainPID', '0')) or state.get('LoadState') != 'loaded'
                    or config.get('method') != method or str(config.get('repeat')) != '2'
                    or config.get('protocol_sha256') != registration['protocol']['sha256']
                    or dispatch.get('host') != HOST or str(dispatch.get('gpu')) != GPU
                    or dispatch.get('unit') != proof['unit']):
                raise ValueError('Nonempty run is not an owned running unit: ' + method)
            # A legitimate existing unit is attached by frozen worker.main.
            continue
        if method != 'R' or state.get('LoadState') != 'not-found':
            raise ValueError('Unknown nonempty run preserved: ' + method)
        code = ('from pathlib import Path\nimport hashlib,json,sys\n'
                + 'PLACEHOLDER_NAMES=' + repr(PLACEHOLDER_NAMES) + '\n'
                + inspect.getsource(digest_file) + '\n'
                + inspect.getsource(archive_retrieval_only) + '\n'
                + 'print(json.dumps(archive_retrieval_only(sys.argv[1],sys.argv[2])))\n')
        remote_archive = ssh_cpu(code, REMOTE_OUT + '/runs/r2_R', REMOTE_OUT + '/recovery_incidents')
        local_archive = archive_retrieval_only(OUT / 'runs/r2_R', INCIDENTS)
        write(INCIDENTS / ('R_placeholder_archive_' + str(time.time_ns()) + '.json'),
              dict(remote=remote_archive, local=local_archive, registration=entry(REGISTRATION)))


def registration():
    protocol = read(OUT / 'protocol.json')
    inventory = read(OUT / 'deployment_files.json')
    by_path = {e['path']: e['sha256'] for e in inventory['files']}
    critical = {str(p.relative_to(ROOT)): sha(p) for p in [
        OUT / 'protocol.json', OUT / 'calibration.json', OUT / 'schedule_2.json',
        HERE / 'train.py', HERE / 'degradation_operator.py', HERE / 'depth_prior.py',
        HERE / 'confidence_cache.py']}
    for rel, digest in critical.items():
        if by_path.get(rel) != digest:
            raise ValueError('Current critical bytes differ from deployed inventory: ' + rel)
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in [
        Path(__file__), HERE / 'worker_async.py', HERE / 'cg_common.py', HERE / 'evaluate_extra.py',
        HERE / 'summarize.py', ROOT / 'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py']}
    b2_checkpoint = verify_local_retrieval('r2_B2perm')
    return dict(status='registered_CPU_checked_recovery_only', host=HOST, gpu=GPU,
                recovery_unit=RECOVERY_UNIT, methods_to_train=['R', 'G', 'RG'],
                never_train_B2perm=True, protocol=entry(OUT / 'protocol.json'),
                schedule=entry(OUT / 'schedule_2.json'), parent=protocol['parent'],
                calibration=entry(OUT / 'calibration.json'), deployment_inventory=entry(OUT / 'deployment_files.json'),
                completed_B2_checkpoint=entry(b2_checkpoint),
                completed_B2_receipt=entry(OUT / 'runs/r2_B2perm/complete.json'),
                completed_B2_retrieval=entry(OUT / 'r2_B2perm_retrieval_complete.json'),
                original_remote_verification=entry(OUT / 'remote_verification.json'),
                training_source_identity=read(OUT / 'remote_verification.json')['source_identity'],
                critical_files=critical, dependencies=dependencies, source=entry(Path(__file__)),
                SSH_only_retry_backoff_seconds=list(BACKOFF), scene_training_updates_during_registration=0)


def execute():
    reg = read(REGISTRATION)
    for rel, digest in reg['dependencies'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen recovery/evaluation source changed: ' + rel)
    for key in ('protocol', 'schedule', 'calibration', 'deployment_inventory', 'original_remote_verification'):
        worker.bound(reg[key])
    INCIDENTS.mkdir(exist_ok=True)
    old_worker = OUT / 'worker_a100-train_r2.json'
    if old_worker.exists():
        snapshot = INCIDENTS / ('worker_state_before_recovery_' + sha(old_worker)[:16] + '.json')
        if not snapshot.exists():
            snapshot.write_bytes(old_worker.read_bytes())
    def report(attempt, error, transport):
        detail = dict(status='SSH_transport_interruption' if transport else 'stopped_real_failure',
                      attempt=attempt, error_type=type(error).__name__, error=str(error),
                      stdout=getattr(error, 'stdout', None), stderr=getattr(error, 'stderr', None),
                      traceback=traceback.format_exc(), source=entry(Path(__file__)),
                      worker_source=entry(HERE / 'worker_async.py'), training_units_preserved=True)
        write(INCIDENTS / ('recovery_exception_' + str(time.time_ns()) + '.json'), detail)
        write(STATE, detail)
    verification = bounded_transport(lambda: verify_remote_inputs(reg), lambda: None, report)
    proof = bounded_transport(lambda: remote_endpoint('B2perm'), lambda: None, report)
    remote_b2 = validate_remote_completed(proof, reg, 'B2perm')
    if int(proof['unit_state'].get('MainPID', '0')):
        bounded_transport(lambda: worker.wait_unit(proof['unit'], HOST), lambda: remote_endpoint('B2perm'), report)
    def already_retrieved():
        if (OUT / 'r2_B2perm_retrieval_complete.json').exists():
            verify_local_retrieval('r2_B2perm')
            return True
        return False
    cp, b2_unit = bounded_transport(lambda: prepare_b2(remote_b2, already_retrieved,
        lambda: worker.retrieve_remote(HOST, 'r2_B2perm', OUT / 'r2_B2perm_recovery_retrieval.log'),
        lambda: verify_local_retrieval('r2_B2perm'),
        lambda checkpoint: worker.dispatch_evaluation('r2_B2perm', checkpoint)),
        lambda: remote_endpoint('B2perm'), report)
    args = argparse.Namespace(host=HOST, gpu=GPU, repeat='2', methods=['R', 'G', 'RG'],
                              training_dependency_pid=0, evaluation_dependency_pid=0)
    def remaining():
        preflight_remaining(reg)
        write(STATE, dict(status='running_remaining_owned_stable_units',
                         B2perm_checkpoint=entry(cp), B2perm_evaluation_unit=b2_unit,
                         remote_verification=entry(OUT / 'remote_recovery_verification.json'),
                         recovery_unit=RECOVERY_UNIT, source=entry(Path(__file__))))
        worker.main(args)
    bounded_transport(remaining, lambda: preflight_remaining(reg), report)
    if b2_unit:
        worker.wait_unit(b2_unit)
    for method in ('C1', 'Jperm', 'B2perm', 'R', 'G', 'RG'):
        task = 'r2_' + method
        checkpoint = worker.verify_training(task, method, '2')
        worker.primary_complete(task, checkpoint, True)
        worker.extra_complete(task, checkpoint, True)
    worker.summarize()
    write(STATE, dict(status='batch_completed',
                     methods=['C1', 'Jperm', 'B2perm', 'R', 'G', 'RG'], B2perm_retrained=False,
                     new_training_budget=18000, source=entry(Path(__file__)),
                     verification=entry(OUT / 'remote_recovery_verification.json')))


def cpu_checks():
    checks = {}
    with tempfile.TemporaryDirectory(prefix='cg_recovery_CPU_') as temporary:
        base = Path(temporary)
        def placeholder(path):
            path.mkdir(); journal = path / 'retrieval_journal.log'; journal.write_text('-- No entries --\n')
            write(path / 'retrieval_manifest.json', dict(task=path.name,
                unit='4dsr-cg-' + path.name.lower().replace('_', '-') + '-20261006',
                files=[dict(path='runs/' + path.name + '/retrieval_journal.log',
                            sha256=digest_file(journal), bytes=journal.stat().st_size)]))
        run = base / 'r2_R'; placeholder(run)
        receipt = archive_retrieval_only(run, base / 'incidents')
        assert not run.exists() and Path(receipt['archive']).exists()
        assert archive_retrieval_only(run, base / 'incidents')['status'] == 'absent_no_archive'
        checks['only_retrieval_files_archived_and_repeat_idempotent'] = True
        run.mkdir(); (run / 'config.json').write_text('{}')
        try:
            archive_retrieval_only(run, base / 'incidents')
        except ValueError:
            pass
        else:
            raise AssertionError('Nonempty training directory was moved')
        assert (run / 'config.json').exists(); checks['nonempty_R_never_moved'] = True
        checkpoint = base / 'b2.pt'; checkpoint.write_bytes(b'completed fixed B2 checkpoint')
        calls = {'transfer': 0, 'dispatch': 0, 'B2_training': 0}
        def transfer(): calls['transfer'] += 1
        def dispatch(cp): calls['dispatch'] += 1; return 'stable_existing_evaluation_unit'
        for _ in range(2):
            prepare_b2({'checkpoint': {'sha256': digest_file(checkpoint)}}, lambda: True,
                       transfer, lambda: checkpoint, dispatch)
        assert calls['transfer'] == calls['B2_training'] == 0
        checks['completed_B2_reused_without_transfer_or_training'] = True
        checks['B2_evaluation_uses_existing_stable_dispatch_API'] = True
        launches = []
        worker.ensure_unit('CPU_existing_unit', lambda: launches.append(1),
                           state_fn=lambda: dict(LoadState='loaded', ActiveState='active', MainPID='1'))
        assert not launches
        checks['existing_running_stable_unit_attached_without_launch'] = True
        attempts = []; reconnects = []; delays = []
        def interrupted_wait():
            attempts.append('attach_same_unit')
            if len(attempts) == 1:
                raise subprocess.CalledProcessError(255, ['ssh', HOST, 'pidfd_wait_existing_unit'])
            return 'completed_after_reattach'
        assert bounded_transport(interrupted_wait, lambda: reconnects.append('same_unit_and_receipt'),
                                 lambda *args: None, delays.append) == 'completed_after_reattach'
        assert len(attempts) == 2 and len(reconnects) == 1 and delays == [5]
        checks['SSH255_reattaches_same_identity_without_redispatch'] = True
        completed = []
        for error in [RuntimeError('Remote GPU occupied'), RuntimeError('Unit did not finish successfully'),
                      ValueError('Inventory mismatch')]:
            try:
                bounded_transport(lambda e=error: (_ for _ in ()).throw(e), lambda: completed.append(False),
                                  lambda *args: None, lambda value: completed.append(False))
            except type(error):
                pass
            else:
                raise AssertionError('Real failure accepted')
        assert not completed
        checks['failed_worker_unknown_GPU_and_hash_errors_stop_not_complete'] = True
        failures = []
        def always_disconnected():
            failures.append(1); raise subprocess.CalledProcessError(255, ['ssh', HOST])
        try:
            bounded_transport(always_disconnected, lambda: None, lambda *args: None, lambda _: None)
        except subprocess.CalledProcessError:
            pass
        assert len(failures) == 4; checks['transport_retry_limit_is_four_attempts'] = True
        # Exercise the actual execute control flow with only CPU file fixtures.
        # A worker exception must leave a failed state, never batch_completed.
        fixture = base / 'coordinator'; fixture.mkdir()
        fixture_registration = fixture / 'registration.json'
        write(fixture_registration, dict(dependencies={}, **{
            key: {} for key in ('protocol', 'schedule', 'calibration', 'deployment_inventory',
                               'original_remote_verification')}))
        write(fixture / 'r2_B2perm_retrieval_complete.json', {})
        write(fixture / 'remote_recovery_verification.json', {'status': 'CPU_fixture'})
        replacements = dict(OUT=fixture, INCIDENTS=fixture / 'incidents', STATE=fixture / 'state.json',
            REGISTRATION=fixture_registration, verify_remote_inputs=lambda _: None,
            entry=lambda path: {'path': str(path), 'sha256': digest_file(path)},
            remote_endpoint=lambda _: {'unit_state': {'MainPID': '0'}},
            validate_remote_completed=lambda *args: {'checkpoint': {'sha256': digest_file(checkpoint)}},
            verify_local_retrieval=lambda _: checkpoint, preflight_remaining=lambda _: None)
        saved = {key: globals()[key] for key in replacements}
        saved_worker = {name: getattr(worker, name) for name in ('bound', 'dispatch_evaluation', 'main')}
        try:
            globals().update(replacements)
            worker.bound = lambda _: fixture
            worker.dispatch_evaluation = lambda *args: None
            worker.main = lambda _: (_ for _ in ()).throw(RuntimeError('CPU fixture actual worker failure'))
            try:
                execute()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed worker was accepted by actual coordinator')
            assert read(fixture / 'state.json')['status'] == 'stopped_real_failure'
            assert list((fixture / 'incidents').glob('recovery_exception_*.json'))
            checks['actual_coordinator_failed_worker_never_marks_batch_complete'] = True
        finally:
            globals().update(saved)
            for name, value in saved_worker.items():
                setattr(worker, name, value)
    return dict(status='passed_CPU_recovery_acceptance', checks=checks, GPU_calls=0,
                SSH_calls=0, services_started=0, scene_training_updates=0, source=entry(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-self-test', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.cpu_self_test:
        result = cpu_checks()
        write(OUT / 'core_recovery_CPU_checks.json', result)
        write(REGISTRATION, registration())
        print(json.dumps(result))
        return
    with (OUT / 'core_recovery_r2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute()
        except BaseException:
            write(STATE, dict(status='failed_recovery_preserved', traceback=traceback.format_exc(),
                             source=entry(Path(__file__)), training_units_preserved=True))
            raise


if __name__ == '__main__':
    main()
