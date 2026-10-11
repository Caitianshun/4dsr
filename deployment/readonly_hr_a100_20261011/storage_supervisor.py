"""Independent Long360 storage hook; original scientific supervisor stays frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'experiments/dynamic_sr_highres_20261011'))
import local_orchestrator as parent
import remote_storage as storage

DEPLOY_FILES = ('storage_supervisor.py', 'remote_storage.py', 'selfcheck_storage.py', 'README.md')
STORAGE_SCHEMA = 'highres_owned_rootless_readonly_HR_storage_v1'
PINNED_HELPER = '''
import hashlib,pathlib,runpy,sys
p=pathlib.Path(sys.argv[1]);expected=sys.argv[2];root=pathlib.Path(sys.argv[3])
if not p.is_absolute() or not p.resolve().is_relative_to(root):raise ValueError('Helper outside workspace')
for q in [p,*p.parents]:
 if q.is_symlink():raise ValueError('Helper symlink rejected')
 if q==root:break
if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen remote storage helper changed')
sys.argv=[str(p),*sys.argv[4:]]
runpy.run_path(str(p),run_name='__main__')
'''


def external_rsync_argv(ssh, host, source, target, listing):
    if host != 'a100-train':
        raise ValueError('External HR transfer must target A100')
    return ['rsync', '-rlt', '--no-owner', '--no-group', '--no-perms', '--ignore-existing', '--protect-args',
            '--files-from=' + str(listing), '-e', shlex.join(ssh[:-1]), str(source).rstrip('/') + '/', host + ':' + str(target).rstrip('/') + '/']


def validate_storage_spec(spec):
    parent.validate_spec(spec)
    if spec['dataset'] != 'long360' or spec['condition'] != 'l3H_long360' or spec['host'] != 'a100-train':
        raise ValueError('Readonly external storage only applies to fixed Long360 A100 condition')
    cfg = spec.get('storage_recipe', {})
    expected = dict(schema=STORAGE_SCHEMA, external_parent=str(storage.EXTERNAL), canonical_prefix=storage.PREFIX,
                    binary=str(storage.BINARY), binary_sha256=storage.BINARY_SHA,
                    package=str(storage.PACKAGE), package_sha256=storage.PACKAGE_SHA,
                    external_reserve_bytes=storage.RESERVE)
    route = cfg.get('ssh_jump')
    if route not in (None, 'cts') or dict(cfg, ssh_jump=route) != dict(expected, ssh_jump=route):
        raise ValueError('Storage deployment parameters differ from closed authorized binding')
    inputs = spec['freeze']['inputs']
    for name in DEPLOY_FILES:
        p = DEPLOY / name
        rel = str(p.relative_to(PROJECT))
        if inputs.get(rel) != parent.sha(p):
            raise ValueError('Storage wrapper/helper source is absent from freeze or changed: ' + name)
    return expected


class StorageSupervisor(parent.Supervisor):
    def __init__(self, spec):
        super().__init__(spec)
        if spec.get('storage_recipe', {}).get('ssh_jump') == 'cts':
            self.ssh = [*self.ssh[:-1], '-J', 'cts', self.ssh[-1]]
        self.storage_attempted = False

    def frozen(self):
        super().frozen()
        validate_storage_spec(dict(self.spec, freeze=self.freeze))

    def storage_call(self, mode, files):
        self.frozen()
        remote_script = self.spec['remote_root'] + '/deployment/readonly_hr_a100_20261011/remote_storage.py'
        expected = self.freeze['inputs'][str((DEPLOY / 'remote_storage.py').relative_to(PROJECT))]
        argv = [self.spec['remote_python'], '-B', '-c', PINNED_HELPER, remote_script, expected, self.spec['remote_root'], '--mode', mode, '--run-id', self.spec['run_id']]
        result = subprocess.run([*self.ssh, shlex.join(argv)], input=json.dumps(dict(files=files)), text=True, capture_output=True)
        self.record('storage_' + mode, exit_code=result.returncode, stdout=result.stdout[-16000:], stderr=result.stderr[-8000:])
        if result.returncode:
            raise RuntimeError('Readonly HR storage ' + mode + ' failed; preserved, no retry: ' + result.stderr[-2000:])
        value = json.loads(result.stdout)
        self.frozen()
        return value

    def sync(self, inventory):
        self.stage = 'readonly_HR_external_stage'
        self.frozen()
        files = {k: v for k, v in inventory['files'].items() if k.startswith(storage.PREFIX)}
        storage.validate_inventory(files)
        # Deploy source-only helper with the unchanged additive conflict policy.
        sources = {}
        for name in DEPLOY_FILES:
            p = DEPLOY / name
            sources[str(p.relative_to(PROJECT))] = dict(bytes=p.stat().st_size, sha256=parent.sha(p))
        source_inventory = dict(files=sources, future_bytes=0)
        pre = self.remote('preflight', source_inventory)
        self.record('storage_helper_preflight', **pre)
        listing = self.out / 'storage_helper_missing.txt'
        listing.write_text(''.join(k + '\n' for k in pre['missing']))
        if pre['missing']:
            cmd = ['rsync', '-a', '--no-owner', '--no-group', '--relative', '--ignore-existing', '--files-from=' + str(listing),
                   '-e', shlex.join(self.ssh[:-1]), str(PROJECT) + '/', self.spec['host'] + ':' + self.spec['remote_root'] + '/']
            with (self.out / 'logs/storage_helper_sync.log').open('xb') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Source-only storage helper sync failed; no retry')
        self.remote('verify', source_inventory)
        self.storage_attempted = True
        stage = self.storage_call('prepare', files)
        listing = self.out / 'external_HR_members.txt'
        listing.write_text(''.join(k[len(storage.PREFIX):] + '\n' for k in files))
        # -rlt deliberately avoids permissions/ownership/hardlinks on exFAT.
        cmd = external_rsync_argv(self.ssh, self.spec['host'], PROJECT / storage.PREFIX, stage['source'], listing)
        with (self.out / 'logs/external_HR_sync.log').open('xb') as log:
            started = time.monotonic()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        self.record('external_HR_transfer_returned', exit_code=result.returncode, seconds=time.monotonic()-started, files=len(files), bytes=sum(v['bytes'] for v in files.values()))
        if result.returncode:
            raise RuntimeError('External HR additive transfer failed; preserve stage, no retry')
        self.storage_call('verify', files)
        mounted = self.storage_call('mount', files)
        mounted['deployment_route'] = dict(host=self.spec['host'], ssh_jump=self.spec['storage_recipe'].get('ssh_jump'), ssh_argv=self.ssh)
        parent.write(self.out / 'storage_mapping.json', mounted)
        # The original65GB future+60GB root guard now sees every HR file reused.
        self.stage = 'canonical_HR_reuse_verified'
        super().sync(inventory)
        self.storage_call('mount-check', files)

    def runner(self, table):
        files = json.loads((self.out / 'sync_inputs.json').read_text())['files']
        hr = {k: v for k, v in files.items() if k.startswith(storage.PREFIX)}
        self.storage_call('mount-check', hr)
        return super().runner(table)

    def run(self):
        validate_storage_spec(self.spec)
        try:
            return super().run()
        finally:
            # Keep the owned readonly input mount alive for audited rerenders.
            # Return its diagnostics on success/failure without stopping it.
            if self.storage_attempted:
                rel = 'output/dynamic_sr_highres_20261011/storage/' + self.spec['run_id']
                try:
                    self.pull_paths([rel], 'readonly_storage_return')
                except BaseException as error:
                    self.record('readonly_storage_return_failed_remote_preserved', error=str(error))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    spec = parent.read(parent.local(args.spec))
    validate_storage_spec(spec)
    if args.validate_only:
        print(json.dumps(dict(status='storage_wrapper_validated_CPU_only_no_SSH_or_GPU', run_id=spec['run_id'], storage_recipe=spec['storage_recipe'])))
        return
    StorageSupervisor(spec).run()


if __name__ == '__main__':
    main()
