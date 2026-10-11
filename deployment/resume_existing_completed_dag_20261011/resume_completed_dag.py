"""Continue the terminal Desk DAG using its original completed task artifacts."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'deployment/complete_startup_eval_20261011'))
import complete_startup_eval as original

parent = original.parent
OLD_ID = 'deskgames_cube_eval3_resume_20261011t1010z'
OLD_OUT = parent.OUT / 'orchestration' / OLD_ID
UNIT = 'highres-headroom-' + OLD_ID + '.service'
INVOCATION = 'ec3f38a6d7094b7383fc9b6c1ee0b8dd'
SCHEMA = 'resume_original_completed_Desk_DAG_v1'
FILES = ('resume_completed_dag.py', 'README.md')
COMPLETED = ('teacher_cost', 'teacher', 'register_sr', 'coarse', 'fine')

# One short CPU preflight checks terminal/task metadata. The original runner
# checks checkpoint weights and teacher indexes once when it reuses each task.
PRECHECK = r'''
import json,pathlib,subprocess,sys
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);here=root/'experiments/dynamic_sr_highres_20261011'
sys.path.insert(0,str(here))
import persistent_runner as runner
from common import OUT,entry,local,read
text=subprocess.check_output(['systemctl','--user','show',p['unit'],'--no-pager','--property=InvocationID,MainPID,ActiveState'],text=True)
unit=dict(row.split('=',1) for row in text.splitlines() if '=' in row)
if unit['InvocationID']!=p['InvocationID'] or int(unit['MainPID'])!=0 or unit['ActiveState'] not in ('failed','inactive'):raise ValueError('Original exact service must remain terminal')
tablepath=local(p['table']['path'])
if entry(tablepath)!=p['table']:raise ValueError('Original task table changed')
table=read(tablepath);runner.validate_table(table)
identity=read(local(p['old_pipeline'])/'identity.json')
runner.check_frozen_identity(identity['sources'],identity['python_inventory'])
accepted={};metadata={}
for task in table['tasks']:
 if task['name'] in p['completed']:
  accepted[task['name']]=entry(local(task['done']))
  if accepted[task['name']]!=p['completed'][task['name']]:raise ValueError('Original completed artifact changed')
  value=read(local(task['done']));metadata[task['name']]={k:value[k] for k in ('status','checkpoint','state','cursor') if k in value}
if set(accepted)!=set(p['completed']):raise ValueError('Original five completed tasks missing')
failed=next(t for t in table['tasks'] if t['name'].endswith('_eval_LRparent17k'))
if local(failed['done']).parent.exists():raise FileExistsError('Preserve partial eval output; this continuation does not clean or overwrite it')
if (OUT/'claims'/f"{p['uuid']}.lock").exists():raise FileExistsError('Existing GPU UUID claim protected')
print(json.dumps(dict(status='original_terminal_and_five_completion_metadata_accepted',unit=unit,completed=accepted,metadata=metadata,deep_completion_check='Original runner checks weights/indexes once before already_complete',eval_LRparent17k_output_absent=True,GPU_launches=0)))
'''
PRECHECK_SHA = hashlib.sha256(PRECHECK.encode()).hexdigest()


def existing_binding(terminal_path, resource_path, exit_path):
    old = parent.read(OLD_OUT / 'spec.json')
    original.validate_spec(old)
    terminal = parent.read(terminal_path)
    if terminal.get('schema') != 'Desk_existing_DAG_terminal_classification_v1' or terminal.get('unit') != UNIT or terminal.get('InvocationID') != INVOCATION or terminal.get('MainPID') != 0 or terminal.get('ActiveState') != 'failed' or terminal.get('process_alive') is not False:
        raise ValueError('Bind the actual original terminal Desk invocation')
    prefix = 'dgH_cube_seed20261007_'
    completed = {prefix + name: terminal['completed_tasks'][prefix + name]['artifact'] for name in COMPLETED}
    if terminal.get('task_counts') != dict(completed=5, failed=1, pending=22) or set(terminal['completed_tasks']) != set(completed) or set(terminal['failed_tasks']) != {prefix + 'eval_LRparent17k'} or terminal.get('classification') != 'engineering_resource_guard_rejection_before_LRparent17k_evaluation':
        raise ValueError('Only this pre-evaluation resource failure is supported')
    exited = parent.read(exit_path)
    if exited.get('status') != 'observed_original_PID_exit' or exited.get('DAG_status') != 'failed' or exited.get('original_unit') != UNIT or exited.get('InvocationID') != INVOCATION or int(exited['unit_after']['MainPID']) != 0 or exited.get('GPU_launches') != 0:
        raise ValueError('Actual original exit receipt required')
    resource = parent.read(resource_path)
    if resource.get('schema') != 'CTS_GPU0_TwoSample_Readonly_Actual_v1' or resource.get('target') != 'cts' or resource.get('ssh_jump') != 'a100-train' or resource.get('result', {}).get('returncode') != 0:
        raise ValueError('Verified CTS route and actual idle receipt required')
    observations = resource['observations']
    host = next(x['payload'] for x in observations if x['kind'] == 'host')
    if host.get('hostname') != 'cts' or host.get('uid') != 1000 or host['pid836374'].get('proc_exists') is not False:
        raise ValueError('Original ToDesk process must be confirmed absent')
    samples = [x['payload'] for x in observations if x['kind'] in ('gpu_1', 'gpu_2')]
    if len(samples) != 2:raise ValueError('Both actual GPU readings required')
    for s in samples:
        gpu, apps = s['gpu'], s['compute_apps']
        values = [v.strip() for v in gpu['stdout'].strip().split(',')]
        if gpu['exit_code'] != 0 or apps['exit_code'] != 0 or apps['stdout'].strip() or len(values) != 6 or values[0] != '0' or '3090' not in values[1] or values[2] != old['gpus'][0]['uuid'] or float(values[3]) > 1024 or float(values[5]) > 10:
            raise ValueError('Actual CTS GPU0 resource acceptance differs')
    table_ref = old['complete_startup_eval']['table']
    table = parent.safe_local_member(table_ref['path'])
    if parent.entry(table) != table_ref:raise ValueError('Original table changed')
    startup = OLD_OUT / 'startup_acceptance.json'
    if parent.read(startup).get('status') != 'passed_runtime_gradient_and_eval3_only':raise ValueError('Original startup acceptance required')
    refs = [parent.entry(p) for p in (OLD_OUT / 'spec.json', startup, terminal_path, resource_path, exit_path)]
    return dict(old=old, table=table, completed=completed, refs=refs)


def validate_spec(spec):
    recipe = spec.get('completed_DAG_resume', {})
    if recipe.get('schema') != SCHEMA or recipe.get('ssh_jump') != 'a100-train' or recipe.get('worker_sha256') != original.WORKER_SHA or recipe.get('precheck_sha256') != PRECHECK_SHA:
        raise ValueError('Explicit original-worker continuation with verified CTS route required')
    paths = [parent.safe_local_member(recipe[k]['path']) for k in ('terminal_classification', 'resource_receipt', 'native_exit_receipt')]
    bound = existing_binding(*paths)
    old = bound['old']
    if any(spec.get(k) != old.get(k) for k in original.runtime.SCIENCE_KEYS) or any(spec.get(k) != old.get(k) for k in ('sync_recovery', 'runtime_environment_resume', 'complete_startup_eval')):
        raise ValueError('Original scientific settings, sources, and 69k total update budget must remain unchanged')
    if spec['run_id'] == OLD_ID or spec.get('supervisor') is not None or recipe.get('completed') != bound['completed'] or recipe.get('table') != parent.entry(bound['table']):
        raise ValueError('Fresh run and exact original completed task/table identities required')
    for key, path in zip(('terminal_classification', 'resource_receipt', 'native_exit_receipt'), paths):
        if recipe[key] != parent.entry(path):raise ValueError('Actual receipt identity differs')
    inputs = dict(old['freeze']['inputs'])
    inputs.update({r['path']: r['sha256'] for r in bound['refs']})
    inputs.update({str((DEPLOY / name).relative_to(PROJECT)): parent.sha(DEPLOY / name) for name in FILES})
    if spec['freeze'] != dict(old['freeze'], inputs=inputs):raise ValueError('Exact original freeze plus continuation sources/evidence required')
    parent.validate_spec(spec)
    return bound


class CompletedDAGRecovery(original.CompleteStartupEval):
    def __init__(self, spec):
        super().__init__(spec)
        self.ssh = [*self.ssh[:-1], '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', '-J', 'a100-train', self.ssh[-1]]

    def startup(self):
        raise RuntimeError('Original startup already accepted; never repeat it')

    def continuation_preflight(self, bound):
        recipe = self.spec['completed_DAG_resume']
        payload = dict(root=self.spec['remote_root'], unit=UNIT, InvocationID=INVOCATION,
                       table=recipe['table'], completed=bound['completed'], uuid=self.spec['gpus'][0]['uuid'],
                       old_pipeline=f'output/dynamic_sr_highres_20261011/pipelines/dgH_cube_seed20261007_{OLD_ID}')
        command = [self.spec['remote_python'], '-B', '-c', PRECHECK, json.dumps(payload)]
        result = subprocess.run([*self.ssh, shlex.join(command)], text=True, capture_output=True)
        self.record('original_completions_preflight', exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, precheck_sha256=PRECHECK_SHA)
        if result.returncode:raise RuntimeError('Original completion/terminal preflight failed; no DAG dispatched')

    def run(self):
        bound = validate_spec(self.spec)
        if self.out.exists():raise FileExistsError('Fresh continuation output required')
        parent.mkdir(self.out); parent.mkdir(self.out / 'events'); parent.mkdir(self.out / 'logs')
        parent.write(self.out / 'spec.json', self.spec)
        with (self.out / 'startup_acceptance.json').open('xb') as dst, (OLD_OUT / 'startup_acceptance.json').open('rb') as src:shutil.copyfileobj(src, dst)
        lock = parent.mkdir(parent.OUT / 'orchestration/claims') / 'dgH_cube_seed07_DAG_resume.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.frozen()
                self.record('existing_DAG_continuation', completed=bound['completed'], table=parent.entry(bound['table']), startup_acceptance=parent.entry(self.out / 'startup_acceptance.json'), source=parent.entry(__file__), worker_sha256=original.WORKER_SHA, synchronization_repeated=False, CPU_preparation_repeated=False, startup_repeated=False, GPU_guard_changed=False, graphics_allowance=[], total_training_updates=69000)
                with self.runtime_environment():
                    self.stage = 'original_completed_task_acceptance'
                    self.continuation_preflight(bound)
                    self.runner(bound['table'])
                    self.pull()
                    self.summarize()
            except BaseException:
                parent.write(self.out / f'failure_{time.time_ns()}.json', dict(status='completed_DAG_continuation_failed_preserved', stage=self.stage, error=traceback.format_exc(), remote_started=self.remote_started, automatic_retry=False, original_completed_artifacts_preserved=True))
                if self.remote_started and not self.formal_returned:
                    try:
                        with self.runtime_environment():self.pull()
                    except BaseException:
                        parent.write(self.out / f'formal_return_failure_{time.time_ns()}.json', dict(status='failed_return_remote_preserved', error=traceback.format_exc()))
                raise


def make_spec(run_id, target, terminal_path, resource_path, exit_path):
    target = parent.local(target)
    if target.exists():raise FileExistsError('Preserve old specs; use a fresh spec path')
    bound = existing_binding(terminal_path, resource_path, exit_path)
    spec = copy.deepcopy(bound['old'])
    spec.update(run_id=run_id, out=str((parent.OUT / 'orchestration' / run_id).relative_to(PROJECT)), supervisor=None, created_utc=parent.now())
    spec['completed_DAG_resume'] = dict(schema=SCHEMA, ssh_jump='a100-train', completed=bound['completed'], table=parent.entry(bound['table']), worker_sha256=original.WORKER_SHA, precheck_sha256=PRECHECK_SHA, terminal_classification=parent.entry(terminal_path), resource_receipt=parent.entry(resource_path), native_exit_receipt=parent.entry(exit_path))
    inputs = dict(bound['old']['freeze']['inputs'])
    inputs.update({r['path']: r['sha256'] for r in bound['refs']})
    inputs.update({str((DEPLOY / name).relative_to(PROJECT)): parent.sha(DEPLOY / name) for name in FILES})
    spec['freeze'] = dict(bound['old']['freeze'], inputs=inputs)
    validate_spec(spec)
    if parent.local(spec['out']).exists():raise FileExistsError('Fresh continuation output required')
    parent.write(target, spec)
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path); ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--make-spec', action='store_true'); ap.add_argument('--run-id'); ap.add_argument('--spec-out', type=Path)
    ap.add_argument('--terminal-classification', type=Path); ap.add_argument('--resource-receipt', type=Path); ap.add_argument('--native-exit-receipt', type=Path)
    args = ap.parse_args()
    if args.make_spec:
        if args.spec or not all((args.run_id, args.spec_out, args.terminal_classification, args.resource_receipt, args.native_exit_receipt)):ap.error('--make-spec requires run-id/spec-out and the three actual receipts')
        spec = make_spec(args.run_id, args.spec_out, *[parent.local(p) for p in (args.terminal_classification, args.resource_receipt, args.native_exit_receipt)])
        print(json.dumps(dict(status='fresh_completed_DAG_continuation_spec_not_launched', run_id=spec['run_id']))); return
    if not args.spec:ap.error('--spec required')
    spec = parent.read(parent.local(args.spec)); validate_spec(spec)
    if args.validate_only:print(json.dumps(dict(status='local_original_DAG_binding_checked_no_SSH_GPU', completed_tasks=5))); return
    CompletedDAGRecovery(spec).run()


if __name__ == '__main__':main()
