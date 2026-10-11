"""One explicit Desk sync-failure recovery. Never remake prepared inputs or tasks."""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'experiments/dynamic_sr_highres_20261011'))
import local_orchestrator as parent
import persistent_runner as runner

OLD_ID = 'deskgames_cube_seed07_20261011t060613z'
OLD_REL = 'output/dynamic_sr_highres_20261011/orchestration/' + OLD_ID
CONDITION = 'dgH_cube'
DEPLOY_FILES = ('resume_sync.py', 'selfcheck_resume.py', 'README.md')
RECIPE_SCHEMA = 'highres_explicit_failed_sync_recovery_v1'


def fail_if(value, message):
    if value:
        raise ValueError(message)


def read_member(rel):
    return parent.read(parent.safe_local_member(rel))


def evidence(old_out):
    paths = [old_out / n for n in ('spec.json', 'sync_inputs.json', 'sync_missing.txt', 'screening_freeze.json', 'Z5_gate.json', 'logs/sync.log')]
    paths += sorted((old_out / 'events').glob('*.json')) + sorted(old_out.glob('failure_*.json'))
    return {str(p.relative_to(parent.ROOT)): parent.sha(p) for p in paths}


def validate_inventory(inventory, spec, full_hash=False):
    fail_if(inventory.get('schema') != 'highres_additive_stream_sync_v1' or inventory.get('host') != spec['host'] or inventory.get('destination') != spec['remote_root'] or inventory.get('future_bytes') != 65 * 10**9,
            'Original synchronization identity or65GB future budget differs')
    fail_if(not inventory.get('files'), 'Empty original synchronization inventory')
    started = time.monotonic()
    for rel, ref in inventory['files'].items():
        p = parent.safe_local_member(rel)
        fail_if(type(ref.get('bytes')) is not int or ref['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')), 'Malformed file size/SHA256')
        fail_if(not p.is_file() or p.stat().st_size != ref['bytes'], 'Original local input missing/size differs: ' + rel)
        if full_hash:
            fail_if(parent.sha(p) != ref['sha256'], 'Original local input SHA differs: ' + rel)
    return dict(files=len(inventory['files']), bytes=sum(v['bytes'] for v in inventory['files'].values()), full_SHA_checked=full_hash, seconds=time.monotonic()-started)


def validate_checkpoint(old_out, full_hash=False):
    old_out = Path(old_out)
    fail_if(str(old_out.relative_to(parent.ROOT)) != OLD_REL, 'Only the explicitly audited Desk failed run can be recovered')
    spec, inv = parent.read(old_out / 'spec.json'), parent.read(old_out / 'sync_inputs.json')
    parent.validate_spec(spec)
    fail_if(spec['dataset'] != 'deskgames_cube' or spec['condition'] != CONDITION or spec['run_id'] != OLD_ID, 'Wrong failed scene/run')
    failures = sorted(old_out.glob('failure_*.json'))
    fail_if(len(failures) != 1, 'Exactly one immutable original failure required')
    failure = parent.read(failures[0])
    fail_if(failure.get('status') != 'failed_preserved_no_retry' or failure.get('stage') != 'verified_additive_sync' or failure.get('remote_started') is not False,
            'Recovery only applies before remote dispatch at failed additive sync')
    fail_if('Additive synchronization failed, no retry' not in failure.get('error', ''), 'Unexpected failure mechanism')
    log = (old_out / 'logs/sync.log').read_text()
    fail_if('Broken pipe' not in log or 'code 10' not in log or 'not responding' not in log, 'Original network-failure log required')
    paths = sorted((old_out / 'events').glob('*.json'))
    names = ['wait_binding','supervisor_exited','started','returned','input_freeze','started','returned','input_freeze','started','returned','input_freeze','started','returned','started','returned','input_freeze','started','returned','input_freeze','input_freeze','sync_inventory','remote_preflight','sync_preflight_accepted']
    fail_if([p.name for p in paths] != [f'{i:03d}_{n}.json' for i,n in enumerate(names,1)], 'Original23events must be exact, complete and pre-dispatch')
    events = [parent.read(p) for p in paths]
    stages = [('prepare_all_video_half',3,4),('initialize_seed07',6,7),('expand_privileged_HR',9,10),('register_LR',12,13),('plan_teacher',14,15),('make_screening_tasks',17,18)]
    for stage, begin, end in stages:
        fail_if(events[begin-1]['stage'] != stage or events[end-1]['stage'] != stage or events[end-1].get('exit_code') != 0, 'Original completed CPU stage invalid: '+stage)
    freeze = inv['freeze']
    fail_if(events[19]['frozen_inputs'] != freeze['inputs'], 'Use event020 final dynamic freeze, not initial screening freeze')
    parent.check_freeze(freeze)
    initial = parent.read(old_out / 'screening_freeze.json')
    fail_if(initial['namespace_python'] != freeze['namespace_python'] or any(freeze['inputs'].get(k) != v for k,v in initial['inputs'].items()), 'Final freeze lost screening identity')
    accepted, gate = parent.freeze_screening(dict(spec, freeze=freeze))
    fail_if(accepted is None or accepted != freeze or gate['status'] != 'passed_Z5', 'Original full Z1/Z5/teacher recipe acceptance required')
    fail_if(parent.read(old_out/'Z5_gate.json')['status'] != 'passed_Z5', 'Original run did not pass Z5')
    metrics = validate_inventory(inv, spec, full_hash)
    pre = events[22]
    fail_if(events[21].get('exit_code') != 0 or pre.get('status') != 'preflight_accepted', 'Original preflight not accepted')
    missing = pre['missing']
    fail_if(len(missing) != len(set(missing)) or not set(missing) <= set(inv['files']) or pre['reused_count'] + len(missing) != len(inv['files']), 'Incomplete original preflight partition')
    fail_if((old_out/'sync_missing.txt').read_text().splitlines() != missing, 'Original transfer list differs from accepted preflight')
    fail_if(pre['disk']['reserve_bytes'] != 60*10**9 or pre['disk']['needed_bytes'] != 65*10**9 + sum(inv['files'][k]['bytes'] for k in missing), 'Original disk guard differs')
    table = parent.OUT/'pipelines'/f'tasks_{CONDITION}_seed{parent.SEED}_{OLD_ID}.json'
    fail_if(freeze['inputs'].get(str(table.relative_to(parent.ROOT))) != parent.sha(table), 'Original task table not pinned by final freeze')
    runner.validate_table(parent.read(table))
    prepared = parent.read(parent.DATA/'prepared'/CONDITION/'manifest.json')
    complete = parent.read(parent.DATA/'prepared'/CONDITION/'complete.json')
    fail_if(prepared.get('full_time_decode_verified') is not True or complete.get('GPU_calls') != 0 or len(prepared['decode_audit']) != 21 or any(e['actual_decoded_count'] != 6000 or e['packet_count'] != 6000 or e['selected_frame_count'] != 300 for e in prepared['decode_audit']), 'Completed21-video CPU preparation receipt differs')
    hr = parent.read(parent.DATA/'hr_full'/CONDITION/'hr_full_storage.json')
    fail_if(hr.get('LR_roundtrip_all_frames_SHA_equal') is not True or set(hr['cameras']) != set(prepared['splits']['train']) or any(c['frames'] != 300 for c in hr['cameras'].values()), 'Original full HR CPU acceptance differs')
    return dict(spec=spec, inventory=inv, table=table, failure=failures[0], evidence=evidence(old_out), local_input_check=metrics,
                local_dispatch_proof='Original23events end before startup(); failed sync raises before both GPU dispatch methods.')


def absent_gpu_paths(run_id=OLD_ID):
    base='output/dynamic_sr_highres_20261011/'
    return [base+'smoke/'+f'{CONDITION}_seed{parent.SEED}_{run_id}', base+'pipelines/'+f'{CONDITION}_seed{parent.SEED}_{run_id}',
            base+'pipelines/'+run_id+'.service.log',base+'pipelines/'+run_id+'.startup.service.log',
            base+'runs/'+f'{CONDITION}_seed{parent.SEED}',base+'registry/full_SR_refinement/'+f'{CONDITION}/seed_{parent.SEED}',
            base+'registry/schedules/'+f'{CONDITION}/seed_{parent.SEED}',base+'registry/teachers/'+CONDITION+'/teacher_cost.json',
            base+'registry/teachers/'+CONDITION+f'/teacher_index_seed{parent.SEED}.json','data/dynamic_sr/highres_20261011/prepared/'+CONDITION+'/sr_swinir_x4']


NO_LAUNCH_AUDIT = r'''
import json,pathlib,subprocess,sys,time
b=json.load(sys.stdin);root=pathlib.Path(b['root']);units={};absent=[]
for unit in b['units']:
 p=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState','--property=ActiveState','--property=MainPID','--property=InvocationID'],capture_output=True,text=True)
 state=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
 if state.get('LoadState')!='not-found' or state.get('MainPID','0')!='0':raise ValueError('Existing old GPU unit protected: '+unit)
 units[unit]=dict(returncode=p.returncode,values=state,stderr=p.stderr)
for rel in b['paths']:
 path=pathlib.PurePosixPath(rel)
 if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe absence member')
 q=root/rel
 for p in [q,*q.parents]:
  if p==root:break
  if p.is_symlink():raise ValueError('Symlink/unknown old result protected')
 if q.exists():raise ValueError('Existing GPU artifact protected: '+rel)
 absent.append(rel)
claim=root/'output/dynamic_sr_highres_20261011/claims'/ (b['uuid']+'.lock')
if claim.exists() or claim.is_symlink():raise ValueError('Existing shared GPU claim protected')
print(json.dumps(dict(status='scoped_old_GPU_dispatch_absence_verified',units=units,absent_paths=absent,claim_absent=str(claim),captured_unix=time.time(),training=False)))
'''


def validate_recovery_spec(spec):
    parent.validate_spec(spec)
    recipe = spec.get('sync_recovery', {})
    fail_if(recipe.get('schema') != RECIPE_SCHEMA or recipe.get('old_out') != OLD_REL or spec.get('supervisor') is not None or spec['run_id'] == OLD_ID, 'Fresh explicit Desk recovery identity required')
    old_out = parent.safe_local_member(OLD_REL)
    checkpoint = validate_checkpoint(old_out)
    old = checkpoint['spec']
    science_keys=('schema','dataset','condition','host','remote_root','remote_python','seed','rung','local_python','workers','cpu_threads','reserve_bytes','remote_future_gb','gpus','expansion_policy')
    fail_if(any(spec.get(k) != old.get(k) for k in science_keys), 'Recovery changes original scientific/hardware settings')
    fail_if(recipe.get('inventory') != parent.entry(old_out/'sync_inputs.json') or recipe.get('table') != parent.entry(checkpoint['table']) or recipe.get('failure') != parent.entry(checkpoint['failure']), 'Recovery provenance refs differ')
    expected = dict(checkpoint['inventory']['freeze']['inputs'], **checkpoint['evidence'])
    audit=recipe.get('operator_remote_audit',{})
    fail_if(not audit.get('path'), 'The operator remote audit must be preserved and frozen')
    audit_path=parent.safe_local_member(audit['path'])
    fail_if(audit != parent.entry(audit_path) or not isinstance(parent.read(audit_path),dict), 'Operator audit identity differs')
    expected[audit['path']]=audit['sha256']
    for n in DEPLOY_FILES:
        p=DEPLOY/n; expected[str(p.relative_to(PROJECT))]=parent.sha(p)
    fail_if(spec['freeze'] != dict(checkpoint['inventory']['freeze'],inputs=expected), 'Recovery must retain exact old science freeze plus immutable recovery evidence/sources')
    return checkpoint


class SyncRecovery(parent.Supervisor):
    def cpu(self, name, script, argv, estimated_gb=0):
        # The inherited final summarize() uses cpu() for read-only analysis.
        # Only that exact call is allowed; preparation/registration/table
        # generation cannot be reached through this adapter.
        target, result = self.out/'headroom_spec.json', self.out/'headroom.json'
        expected = ['--spec', str(target), '--out', str(result)]
        if self.stage != 'headroom_analysis' or name != 'headroom' or script != 'headroom.py' or estimated_gb != 0 or list(map(str, argv)) != expected:
            raise RuntimeError('Recovery permits only the fixed final headroom analysis; CPU preparation, registration and task-table rebuilding are forbidden')
        hspec = parent.read(target)
        budgets = dict(LRparent=17000, SRrecon=17000, HRref=17000, lam0=23000, base=23000, oracleHR=23000)
        runs = parent.OUT/'runs'/f'{CONDITION}_seed{parent.SEED}'
        arms = {a: str(runs/('eval_LRparent17k' if a == 'LRparent' else 'eval_'+a)) for a in budgets}
        if hspec.get('condition') != CONDITION or hspec.get('seed') != parent.SEED or hspec.get('arms') != arms or hspec.get('protocol', {}).get('budgets') != budgets:
            raise ValueError('Final headroom input does not preserve the six original endpoints/budgets')
        return super().cpu(name, script, argv, estimated_gb=0)

    def no_launch_audit(self, label):
        value=dict(root=self.spec['remote_root'], units=['highres-startup-'+OLD_ID+'.service','highres-headroom-'+OLD_ID+'.service'],paths=absent_gpu_paths(),uuid=self.spec['gpus'][0]['uuid'])
        argv=[self.spec['remote_python'],'-B','-c',NO_LAUNCH_AUDIT]
        result=subprocess.run([*self.ssh,shlex.join(argv)],input=json.dumps(value),text=True,capture_output=True)
        self.record(label,exit_code=result.returncode,stderr=result.stderr[-8000:])
        if result.returncode:
            raise RuntimeError('Old remote GPU dispatch state is not safely absent; preserve/review, no re-dispatch')
        receipt=json.loads(result.stdout)
        parent.write(self.out/(label+'.json'),receipt)
        return receipt

    def run(self):
        bound=validate_recovery_spec(self.spec)
        if self.out.exists():
            raise FileExistsError('Fresh recovery output required; no automatic retry')
        parent.mkdir(self.out);parent.mkdir(self.out/'logs');parent.mkdir(self.out/'events')
        parent.write(self.out/'spec.json',self.spec)
        parent.write(self.out/'resume_binding.json',dict(status='explicit_sync_only_recovery',old_inventory=parent.entry(parent.safe_local_member(OLD_REL)/'sync_inputs.json'),table=parent.entry(bound['table']),local_dispatch_proof=bound['local_dispatch_proof'],CPU_repeated=False,task_table_rewritten=False))
        lock=parent.mkdir(parent.OUT/'orchestration/claims')/(CONDITION+'_seed07.lock')
        with lock.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                self.frozen()
                self.no_launch_audit('before_sync_no_old_dispatch')
                # Keep the old inventory byte-for-byte; new output/log names
                # are recovery metadata only. Original sync rehashes reused
                # members, rejects conflicts and copies missing files only.
                self.sync(bound['inventory'])
                self.no_launch_audit('before_startup_no_old_dispatch')
                self.startup();self.runner(bound['table']);self.pull();self.summarize()
            except BaseException:
                parent.write(self.out/('failure_'+str(time.time_ns())+'.json'),dict(status='explicit_recovery_failed_preserved',stage=self.stage,error=traceback.format_exc(),remote_started=self.remote_started,smoke_started=self.smoke_started,automatic_retry=False,old_outputs_preserved=True))
                if self.smoke_started and not (self.out/'startup_acceptance.json').exists():
                    try:self.pull_paths([self.smoke_rel,'output/dynamic_sr_highres_20261011/pipelines/'+self.spec['run_id']+'.startup.service.log'],'startup_failure_return')
                    except BaseException:
                        parent.write(self.out/('startup_return_failure_'+str(time.time_ns())+'.json'),dict(status='failed_return_remote_preserved',error=traceback.format_exc()))
                if self.remote_started and not self.formal_returned:
                    try:self.pull()
                    except BaseException:
                        parent.write(self.out/('formal_return_failure_'+str(time.time_ns())+'.json'),dict(status='failed_return_remote_preserved',error=traceback.format_exc()))
                raise


def make_spec(run_id, target, remote_audit):
    fail_if(parent.local(target).exists(), 'Preserve existing recovery spec; use a fresh file')
    bound=validate_checkpoint(parent.safe_local_member(OLD_REL),full_hash=True)
    spec=copy.deepcopy(bound['spec'])
    spec.update(run_id=run_id,out=str((parent.OUT/'orchestration'/run_id).relative_to(parent.ROOT)),supervisor=None,created_utc=parent.now())
    audit=parent.entry(parent.local(remote_audit))
    spec['sync_recovery']=dict(schema=RECIPE_SCHEMA,old_out=OLD_REL,inventory=parent.entry(parent.safe_local_member(OLD_REL)/'sync_inputs.json'),table=parent.entry(bound['table']),failure=parent.entry(bound['failure']),operator_remote_audit=audit)
    inputs=dict(bound['inventory']['freeze']['inputs'],**bound['evidence'])
    inputs.update({str((DEPLOY/n).relative_to(PROJECT)):parent.sha(DEPLOY/n) for n in DEPLOY_FILES})
    inputs[audit['path']]=audit['sha256']
    spec['freeze']=dict(bound['inventory']['freeze'],inputs=inputs)
    validate_recovery_spec(spec)
    parent.write(parent.local(target),spec)
    return spec,bound['local_input_check']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--make-spec',action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--spec-out',type=Path)
    p.add_argument('--remote-audit',type=Path)
    args=p.parse_args()
    if args.make_spec:
        if args.spec or not args.run_id or not args.spec_out or not args.remote_audit:p.error('--make-spec requires --run-id/--spec-out/--remote-audit only')
        spec,checked=make_spec(args.run_id,args.spec_out,args.remote_audit)
        print(json.dumps(dict(status='fresh_sync_recovery_spec_not_launched',run_id=spec['run_id'],local_input_check=checked)));return
    if not args.spec:p.error('--spec required')
    spec=parent.read(parent.local(args.spec));bound=validate_recovery_spec(spec)
    if args.validate_only:
        print(json.dumps(dict(status='CPU_recovery_preflight_passed_no_SSH_or_GPU',run_id=spec['run_id'],local_input_check=bound['local_input_check'])));return
    SyncRecovery(spec).run()


if __name__=='__main__':
    main()
