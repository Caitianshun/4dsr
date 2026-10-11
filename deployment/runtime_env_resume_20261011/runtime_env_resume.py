"""Resume Desk after the verified sync and failed runtime import, with CTS library paths."""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'deployment/resume_verified_sync_20261011'))
import resume_sync as previous

parent = previous.parent
PREVIOUS_ID = 'deskgames_cube_sync_resume_20261011t0800z'
PREVIOUS_REL = 'output/dynamic_sr_highres_20261011/orchestration/' + PREVIOUS_ID
LD_LIBRARY_PATH = '/home/cts/Project/4DSR/tools/runtime-libs:/home/cts/Project/4DSR/tools/cuda-12.8-minimal/lib64'
SCHEMA = 'highres_verified_sync_runtime_environment_resume_v1'
FILES = ('runtime_env_resume.py', 'README.md')
SCIENCE_KEYS = ('schema', 'dataset', 'condition', 'host', 'remote_root', 'remote_python',
                'seed', 'rung', 'local_python', 'workers', 'cpu_threads', 'reserve_bytes',
                'remote_future_gb', 'gpus', 'expansion_policy')


def bound_checkpoint():
    out = parent.safe_local_member(PREVIOUS_REL)
    spec = parent.read(out / 'spec.json')
    original = previous.validate_recovery_spec(spec)
    inventory = parent.read(out / 'sync_inputs.json')
    if inventory != original['inventory']:
        raise ValueError('Successful sync inventory differs from the original prepared inputs')
    accepted = parent.read(out / 'events/006_sync_verified.json')
    if accepted.get('status') != 'verify_accepted' or accepted.get('missing') != [] or accepted.get('reused_count') != len(inventory['files']) or len(inventory['files']) != 13335:
        raise ValueError('The complete original13335-file SHA verification is required')
    failures = sorted(out.glob('failure_*.json'))
    if len(failures) != 1:
        raise ValueError('One preserved startup failure required')
    failure = parent.read(failures[0])
    if failure.get('status') != 'explicit_recovery_failed_preserved' or failure.get('remote_started') is not False or failure.get('smoke_started') is not True or 'Startup persistent service/SSH returned 1' not in failure.get('error', ''):
        raise ValueError('Only the pre-formal-training startup failure may resume')
    events = sorted((out / 'events').glob('*.json'))
    if len(events) != 15 or any('remote_dispatch_attempted' in p.name for p in events):
        raise ValueError('Previous run does not end at the audited failed startup')
    smoke = parent.OUT / 'smoke' / f'dgH_cube_seed{parent.SEED}_{PREVIOUS_ID}'
    log = smoke / 'logs/coarse50.log'
    if 'ImportError: libBLT.2.5.so.8.6: cannot open shared object file' not in log.read_text() or 'from tkinter import W' not in log.read_text():
        raise ValueError('Unexpected startup failure; do not retry')
    returned = parent.read(smoke / 'events/coarse50_returned.json')
    if returned.get('exit_code') != 1 or returned.get('log') != parent.entry(log) or (smoke / 'complete.json').exists() or list((smoke / 'coarse50').glob('*.pt')):
        raise ValueError('Failed import must have no completed diagnostic or trained checkpoint')
    refs = [out / 'spec.json', out / 'sync_inputs.json', out / 'resume_binding.json',
            failures[0], *events, log, smoke / 'events/coarse50_returned.json',
            parent.OUT / 'pipelines' / (PREVIOUS_ID + '.startup.service.log')]
    pins = {str(p.relative_to(PROJECT)): parent.sha(p) for p in refs}
    return dict(spec=spec, inventory=inventory, table=original['table'], pins=pins,
                failure=failures[0], sync_verified=out / 'events/006_sync_verified.json')


def validate_spec(spec):
    parent.validate_spec(spec)
    bound = bound_checkpoint()
    old = bound['spec']
    recipe = spec.get('runtime_environment_resume', {})
    if spec['host'] != 'cts' or spec['condition'] != 'dgH_cube' or spec['seed'] != 20261007 or any(spec.get(k) != old.get(k) for k in SCIENCE_KEYS):
        raise ValueError('Only unchanged CTS Desk half seed07 settings are supported')
    if spec['run_id'] in (PREVIOUS_ID, previous.OLD_ID) or spec.get('supervisor') is not None or recipe.get('schema') != SCHEMA or recipe.get('previous_out') != PREVIOUS_REL or recipe.get('LD_LIBRARY_PATH') != LD_LIBRARY_PATH:
        raise ValueError('Fresh explicit runtime-environment recovery required')
    for key, path in [('sync_verified', bound['sync_verified']), ('inventory', parent.safe_local_member(PREVIOUS_REL) / 'sync_inputs.json'), ('table', bound['table']), ('failure', bound['failure'])]:
        if recipe.get(key) != parent.entry(path):
            raise ValueError('Recovery provenance differs: ' + key)
    audit = recipe.get('operator_runtime_audit', {})
    if not audit.get('path') or audit != parent.entry(parent.safe_local_member(audit['path'])) or not isinstance(parent.read(parent.safe_local_member(audit['path'])), dict):
        raise ValueError('Pin the operator actual CTS runtime-library/import audit')
    runtime = parent.read(parent.safe_local_member(audit['path']))
    if runtime.get('status') != 'actual_backbone_and_extensions_import_passed' or runtime.get('host') != 'cts' or runtime.get('LD_LIBRARY_PATH') != LD_LIBRARY_PATH or runtime.get('gpu', {}).get('uuid') != spec['gpus'][0]['uuid'] or runtime.get('no_training') is not True or runtime.get('lib', {}).get('sha256') != 'c5072c0c111d974e3d69503ca2275b29a49186c6fabe62395a83791eb372495a':
        raise ValueError('Actual CTS runtime-library/import acceptance differs')
    expected = dict(old['freeze']['inputs'], **bound['pins'])
    expected[audit['path']] = audit['sha256']
    expected.update({str((DEPLOY / n).relative_to(PROJECT)): parent.sha(DEPLOY / n) for n in FILES})
    if spec['freeze'] != dict(old['freeze'], inputs=expected):
        raise ValueError('Exact original science freeze plus recovery provenance required')
    return bound


class RuntimeEnvRecovery(previous.SyncRecovery):
    def sync(self, inventory):
        raise RuntimeError('Synchronization already completed; this adapter never transfers inputs')

    def rewrite_ssh(self, command):
        """Only rewrite the exact inherited SSH boundary, never local/rsync commands."""
        if not isinstance(command, (list, tuple)) or list(command[:len(self.ssh)]) != self.ssh:
            return command
        if len(command) != len(self.ssh) + 1 or not isinstance(command[-1], str):
            raise ValueError('Unexpected inherited SSH invocation')
        remote = shlex.split(command[-1])
        if remote and remote[0] == 'systemd-run':
            unit = next((v for v in remote if v.startswith('--unit=')), '')
            allowed = {'--unit=highres-startup-' + self.spec['run_id'], '--unit=highres-headroom-' + self.spec['run_id']}
            if unit not in allowed or self.spec['remote_python'] not in remote or any(v.startswith('--setenv=LD_LIBRARY_PATH=') for v in remote):
                raise ValueError('Only this fresh startup/runner service may receive the environment')
            remote.insert(1, '--setenv=LD_LIBRARY_PATH=' + LD_LIBRARY_PATH)
        elif remote and remote[0] == self.spec['remote_python']:
            remote = ['env', 'LD_LIBRARY_PATH=' + LD_LIBRARY_PATH, *remote]
        else:
            raise ValueError('Unexpected SSH command outside the inherited Python/service interface')
        return [*self.ssh, shlex.join(remote)]

    @contextlib.contextmanager
    def runtime_environment(self):
        # This supervisor is one synchronous process. Restore the shared module
        # function on exit; do not change SSH config or the user-manager environment.
        original = subprocess.run
        def run(command, *args, **kwargs):
            rewritten = self.rewrite_ssh(command)
            if rewritten != command:
                self.record('CTS_runtime_environment', LD_LIBRARY_PATH=LD_LIBRARY_PATH,
                            argv=rewritten, environment_scope='one SSH Python call or one fresh user service')
            return original(rewritten, *args, **kwargs)
        subprocess.run = run
        try:
            yield
        finally:
            subprocess.run = original

    def run(self):
        bound = validate_spec(self.spec)
        if self.out.exists():
            raise FileExistsError('Fresh recovery output required; preserve old failures')
        parent.mkdir(self.out); parent.mkdir(self.out / 'logs'); parent.mkdir(self.out / 'events')
        parent.write(self.out / 'spec.json', self.spec)
        parent.write(self.out / 'resume_binding.json', dict(status='completed_sync_runtime_environment_only_resume',
                     sync_verified=parent.entry(bound['sync_verified']), table=parent.entry(bound['table']),
                     CPU_repeated=False, Z5_repeated=False, synchronization_repeated=False, task_table_rewritten=False,
                     LD_LIBRARY_PATH=LD_LIBRARY_PATH))
        lock = parent.mkdir(parent.OUT / 'orchestration/claims') / 'dgH_cube_seed07.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.frozen()
                with self.runtime_environment():
                    self.stage = 'completed_sync_acceptance_reused'
                    self.record('completed_sync_acceptance_reused_no_transfer',
                                completion=parent.entry(bound['sync_verified']),
                                files=len(bound['inventory']['files']),
                                input_identity='Original complete SHA acceptance; no repeated all-input scan')
                    self.startup(); self.runner(bound['table']); self.pull(); self.summarize()
            except BaseException:
                parent.write(self.out / ('failure_' + str(time.time_ns()) + '.json'), dict(
                    status='runtime_environment_recovery_failed_preserved', stage=self.stage,
                    error=traceback.format_exc(), remote_started=self.remote_started,
                    smoke_started=self.smoke_started, automatic_retry=False, old_outputs_preserved=True))
                with self.runtime_environment():
                    if self.smoke_started and not (self.out / 'startup_acceptance.json').exists():
                        try:
                            self.pull_paths([self.smoke_rel, 'output/dynamic_sr_highres_20261011/pipelines/' + self.spec['run_id'] + '.startup.service.log'], 'startup_failure_return')
                        except BaseException:
                            parent.write(self.out / ('startup_return_failure_' + str(time.time_ns()) + '.json'), dict(status='failed_return_remote_preserved', error=traceback.format_exc()))
                    if self.remote_started and not self.formal_returned:
                        try:
                            self.pull()
                        except BaseException:
                            parent.write(self.out / ('formal_return_failure_' + str(time.time_ns()) + '.json'), dict(status='failed_return_remote_preserved', error=traceback.format_exc()))
                raise


def make_spec(run_id, target, runtime_audit):
    target = parent.local(target)
    if target.exists():
        raise FileExistsError('Preserve existing spec; use a fresh path')
    bound = bound_checkpoint()
    spec = copy.deepcopy(bound['spec'])
    spec.update(run_id=run_id, out=str((parent.OUT / 'orchestration' / run_id).relative_to(PROJECT)),
                supervisor=None, created_utc=parent.now())
    audit = parent.entry(parent.local(runtime_audit))
    spec['runtime_environment_resume'] = dict(schema=SCHEMA, previous_out=PREVIOUS_REL,
        LD_LIBRARY_PATH=LD_LIBRARY_PATH, operator_runtime_audit=audit,
        sync_verified=parent.entry(bound['sync_verified']), inventory=parent.entry(parent.safe_local_member(PREVIOUS_REL) / 'sync_inputs.json'),
        table=parent.entry(bound['table']), failure=parent.entry(bound['failure']))
    inputs = dict(bound['spec']['freeze']['inputs'], **bound['pins'])
    inputs[audit['path']] = audit['sha256']
    inputs.update({str((DEPLOY / n).relative_to(PROJECT)): parent.sha(DEPLOY / n) for n in FILES})
    spec['freeze'] = dict(bound['spec']['freeze'], inputs=inputs)
    validate_spec(spec)
    if parent.local(spec['out']).exists():
        raise FileExistsError('Fresh run output required')
    parent.write(target, spec)
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--make-spec', action='store_true')
    ap.add_argument('--run-id'); ap.add_argument('--spec-out', type=Path)
    ap.add_argument('--runtime-audit', type=Path)
    args = ap.parse_args()
    if args.make_spec:
        if args.spec or not args.run_id or not args.spec_out or not args.runtime_audit:
            ap.error('--make-spec requires --run-id/--spec-out/--runtime-audit only')
        spec = make_spec(args.run_id, args.spec_out, args.runtime_audit)
        print(json.dumps(dict(status='fresh_runtime_environment_resume_spec_not_launched', run_id=spec['run_id']))); return
    if not args.spec:
        ap.error('--spec required')
    spec = parent.read(parent.local(args.spec)); bound = validate_spec(spec)
    if args.validate_only:
        print(json.dumps(dict(status='local_interface_acceptance_no_SSH_or_GPU', run_id=spec['run_id'], files=len(bound['inventory']['files']), table=parent.entry(bound['table'])))); return
    RuntimeEnvRecovery(spec).run()


if __name__ == '__main__':
    main()
