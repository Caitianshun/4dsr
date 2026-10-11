"""Reuse the accepted final Desk inventory and preserve the late live-copy snapshots."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'deployment/return_completed_dag_20261011'))
import return_completed_dag as base
parent = base.parent
PREVIOUS = parent.OUT/'orchestration/deskgames_terminal_complete_return_20261011t1808z'
SCHEMA = 'reuse_accepted_terminal_Desk_inventory_return_v1'
FILES = ('return_terminal_inventory.py','README.md')
DIAGNOSIS = PROJECT/'tmp/codex_highres_validation_20261010/resource_audit/Desk_active_snapshot_growth_diagnosis_20261011T1818Z.json'
DIAGNOSIS_SHA = '19a0c990215bbbe77a44f2ea88a00796dcbfcee659d9d26421e35690d6fca3b5'
PINS = {
    'return_config.json':'ae18e921f90b03ddcb7b43d8797faa5b1523e9ffee8895c258a24b1002d13b68',
    'terminal_cpu_receipt.json':'e2844c8485e77222ef899fa4730dcfd3d18765baa1919655c4d5ed3c6db9f844',
    'return_failure_1791742388363359343.json':'b8e5fb5463c36c5ac0dbb2f3bfe030892093dccb4ca941a390eb2b45f8f0b518',
    'events/001_terminal_CPU_collection_returned.json':'aabf774fbb227d6124977aafb548bd31edf6e192ef9d23e7737881ab5d3f3374',
}
EXPECTED = {
    base.MUTABLE[0]:dict(bytes=2644722,sha256='0b86a6885085ccda3e9736acbdf39ac1273c99790f828d10c9457153edb2669e'),
    base.MUTABLE[1]:dict(bytes=16,sha256='393c803af48a5ac8903c4af2efc4fa60e8efb27a652f1a7690f161edff70b9cc'),
    base.MUTABLE[2]:dict(bytes=27094,sha256='0d53e928780f46a2f8a3ff6894f2c11bd9fe525ef6029feea99164f0ecfb3421'),
    base.MUTABLE[3]:dict(bytes=0,sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
}


def sources():
    refs = {str((PREVIOUS/n).relative_to(PROJECT)):digest for n,digest in PINS.items()}
    refs[str(DIAGNOSIS.relative_to(PROJECT))] = DIAGNOSIS_SHA
    for rel,digest in refs.items():
        if parent.sha(parent.safe_local_member(rel)) != digest:raise ValueError('Accepted earlier return/failure provenance changed: '+rel)
    old_config = parent.read(PREVIOUS/'return_config.json');base.binding(old_config)
    terminal = parent.read(PREVIOUS/'terminal_cpu_receipt.json')
    event = parent.read(PREVIOUS/'events/001_terminal_CPU_collection_returned.json')
    failure = parent.read(PREVIOUS/'return_failure_1791742388363359343.json')
    if terminal.get('status') != 'completed_original_DAG_terminal_inventory' or terminal.get('GPU_launches') != 0 or terminal.get('complete') != old_config['terminal_refs']['complete'] or terminal.get('identity') != old_config['terminal_refs']['identity'] or terminal.get('original_InvocationID') != base.INVOCATION or event.get('exit_code') != 0 or event.get('worker_sha256') != base.TERMINAL_SHA:
        raise ValueError('Previously accepted stable terminal inventory differs')
    if failure.get('status') != 'completed_DAG_return_failed_preserved' or 'Differing local snapshot does not match the preserved17:13 inventory: '+base.MUTABLE[0] not in failure.get('error','') or (PREVIOUS/'local_snapshot_archive.json').exists() or (PREVIOUS/'local_snapshots').exists():
        raise ValueError('Only the pre-archive live-copy snapshot mismatch is supported')
    for name in FILES:refs[str((DEPLOY/name).relative_to(PROJECT))] = parent.sha(DEPLOY/name)
    return old_config, terminal, refs


def snapshot_binding():
    original = parent.read(base.OLD/'remote_result_inventory.json')['files']; result = {}
    for rel in base.MUTABLE:
        q = parent.safe_local_member(rel)
        if q.is_symlink() or not q.is_file():raise ValueError('Fixed historical snapshot missing/type changed: '+rel)
        before = base.stat(q); actual = dict(bytes=before['bytes'],sha256=parent.sha(q))
        if before != base.stat(q) or actual != EXPECTED[rel]:raise ValueError('Historical snapshot changed: '+rel)
        result[rel] = dict(ref=actual,stat=before,original_inventory=original[rel])
    diagnosis = parent.read(DIAGNOSIS); copied = result[base.MUTABLE[0]]
    if diagnosis['actual_local'] != dict(copied['ref'],mtime_ns=copied['stat']['mtime_ns'],ctime_ns=copied['stat']['ctime_ns']) or diagnosis['old_inventory'] != copied['original_inventory'] or diagnosis['old_prefix_SHA_verified'] is not True or diagnosis['appended_iterations'] != list(range(11744,11766)) or diagnosis['newline_complete'] is not True:raise ValueError('Fixed snapshot diagnosis differs')
    copied['live_copy_evidence'] = dict(diagnosis=parent.entry(DIAGNOSIS),old_prefix_SHA_verified=True,old_prefix_rows=11743,actual_rows=11765,appended_rows=22,appended_bytes=4979,meaning='Exact old prefix plus22 complete later rows; original collection ended with SSH/rsync failure')
    return result


def binding(config):
    old,terminal,refs = sources()
    if config.get('schema') != SCHEMA or not re.fullmatch('[a-z0-9_-]{1,64}',config['return_id']) or config['return_id'] in (base.RUN,old['return_id']) or config.get('base_config') != parent.entry(PREVIOUS/'return_config.json') or config.get('terminal_receipt') != parent.entry(PREVIOUS/'terminal_cpu_receipt.json') or config.get('diagnosis') != parent.entry(DIAGNOSIS) or config.get('snapshots') != snapshot_binding():raise ValueError('Fresh exact snapshot/inventory recovery binding required')
    inputs = dict(old['freeze']['inputs']);inputs.update(refs)
    if config['freeze'] != dict(old['freeze'],inputs=inputs):raise ValueError('Original scientific and recovery-source freeze differs')
    parent.check_freeze(config['freeze'])
    return old,terminal


class InventoryReturn(base.CompletedReturn):
    def __init__(self,config,path):
        old,self.accepted_terminal = binding(config)
        super().__init__(old,PREVIOUS/'return_config.json')
        self.recovery_config = config;self.out = parent.OUT/'orchestration'/config['return_id']
        self.freeze = dict(config['freeze'],inputs=dict(config['freeze']['inputs'],**{str(path.relative_to(PROJECT)):parent.sha(path)}))

    def terminal_inventory(self):
        # No remote collection: the earlier actual successful CPU inventory is pinned.
        self.stable_inventory = self.accepted_terminal['inventory']
        parent.write(self.out/'recovery_config.json',self.recovery_config)
        parent.write(self.out/'terminal_cpu_receipt.json',self.accepted_terminal)
        self.record('reuse_accepted_terminal_inventory',receipt=self.recovery_config['terminal_receipt'],inventory_files=len(self.stable_inventory['files']),new_remote_inventory_collections=0,new_GPU_launches=0)

    def archive_snapshots(self):
        archived = []
        for rel, expected in self.recovery_config['snapshots'].items():
            q = parent.safe_local_member(rel)
            if q.is_symlink() or base.stat(q) != expected['stat'] or dict(bytes=q.stat().st_size,sha256=parent.sha(q)) != expected['ref']:raise ValueError('Fixed actual local snapshot changed before archive: '+rel)
            stable = self.stable_inventory['files'][rel]
            if expected['ref'] == stable:continue
            dest = self.out/'local_snapshots'/rel
            if dest.exists() or dest.is_symlink():raise FileExistsError('Preserve private snapshot archive')
            parent.mkdir(dest.parent);q.rename(dest)
            if parent.sha(dest) != expected['ref']['sha256']:raise ValueError('Snapshot bytes changed during archival')
            archived.append(dict(original_path=rel,local_snapshot=expected,stable_remote=stable,archive=parent.entry(dest)))
        parent.write(self.out/'local_snapshot_archive.json',dict(scope=list(base.MUTABLE),archived=archived,historical_inventory_is_live_snapshot=True,unchanged_files_preserved=True,assets_deleted=False))


def make_config(return_id):
    old,_,refs = sources();inputs=dict(old['freeze']['inputs']);inputs.update(refs)
    config = dict(schema=SCHEMA,return_id=return_id,base_config=parent.entry(PREVIOUS/'return_config.json'),terminal_receipt=parent.entry(PREVIOUS/'terminal_cpu_receipt.json'),diagnosis=parent.entry(DIAGNOSIS),snapshots=snapshot_binding(),freeze=dict(old['freeze'],inputs=inputs))
    binding(config);return config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--make-config',action='store_true');p.add_argument('--return-id');p.add_argument('--config-out',type=Path);a=p.parse_args()
    if a.make_config:
        if a.config or not all((a.return_id,a.config_out)):p.error('--make-config requires return-id/config-out')
        target=parent.local(a.config_out)
        if target.exists() or (parent.OUT/'orchestration'/a.return_id).exists():raise FileExistsError('Fresh config/output required')
        config=make_config(a.return_id);parent.write(target,config)
        import json
        print(json.dumps(dict(status='accepted_terminal_inventory_reuse_config_created_no_SSH_GPU',config=parent.entry(target))));return
    if not a.config:p.error('--config required')
    path=parent.local(a.config);config=parent.read(path);binding(config)
    if a.validate_only:print('Actual accepted terminal/snapshot bindings passed; no SSH/GPU');return
    InventoryReturn(config,path).run()


if __name__=='__main__':main()
