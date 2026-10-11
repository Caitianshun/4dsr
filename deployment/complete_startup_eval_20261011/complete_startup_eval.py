"""Complete only the missing Desk startup evaluation using its existing50-step checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'deployment/runtime_env_resume_20261011'))
import runtime_env_resume as runtime

parent = runtime.parent
VALIDATE_RUNTIME_SPEC = runtime.validate_spec
OLD_ID = 'deskgames_cube_runtime_resume_20261011t0950z'
OLD_REL = 'output/dynamic_sr_highres_20261011/orchestration/' + OLD_ID
SMOKE_REL = 'output/dynamic_sr_highres_20261011/smoke/dgH_cube_seed20261007_' + OLD_ID
CHECKPOINT_SHA = '0fc0729727f8d20f109561f407e0dd3a9e830b3b166a36a70a648423ad3d9edf'
SCHEMA = 'highres_complete_existing50step_eval3_v1'
FILES = ('complete_startup_eval.py', 'README.md')

# Sent as an owned python-c process. No installation or frozen-source change.
WORKER = r'''
import csv,json,math,os,pathlib,socket,subprocess,sys,time,traceback
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);here=root/'experiments/dynamic_sr_highres_20261011'
sys.path.insert(0,str(here));os.chdir(root)
from common import OUT,entry,local,mkdir,read,write,disk_guard
import persistent_runner as runner
original_idle=runner.check_gpu_idle
def settled_idle(gpu,allowed):
 time.sleep(2)
 return original_idle(gpu,allowed)
runner.check_gpu_idle=settled_idle
if p['mode']=='runner':
 sys.argv=[str(here/'persistent_runner.py'),*p['argv']]
 runner.main()
else:
 out=local(p['out']);cp=local(p['checkpoint']['path']);coarse=read(local(p['coarse_completion']['path']));override=read(local(p['smoke_override']['path']));identity=read(local(p['coarse_identity']['path']))
 for ref in [p['checkpoint'],p['coarse_completion'],p['smoke_override'],p['coarse_identity']]:
  if entry(local(ref['path']))!=ref:raise ValueError('Existing50step artifact SHA differs')
 if p['condition']!='dgH_cube' or p['seed']!=20261007 or coarse.get('status')!='completed' or coarse['state']['coarse_iteration']!=50 or coarse['state']['fine_iteration']!=0 or coarse['checkpoint']!=p['checkpoint'] or override.get('not_eligible_for_formal_parent') is not True:raise ValueError('Existing50step diagnostic identity differs')
 runner.check_frozen_identity(identity['source_identity'],p['python_inventory'])
 if out.exists() or not out.is_relative_to(OUT/'smoke'):raise FileExistsError('Fresh eval-only startup directory required')
 mkdir(out);mkdir(out/'logs');mkdir(out/'events')
 gpu=runner.resolve_gpu(p['uuid'])
 if int(gpu['index'])!=0 or '3090' not in gpu['name'] or gpu['uuid']!=p['uuid']:raise ValueError('CTS GPU allocation differs')
 record=dict(status='running_existing50step_eval3_only',diagnostic_only=True,eligible_for_formal_parent=False,host=socket.gethostname(),PID=os.getpid(),gpu=gpu,condition=p['condition'],seed=p['seed'],source_identity=identity['source_identity'],adaptation_source=p['adaptation_source'],worker_sha256=p['worker_sha256'],coarse_completion=p['coarse_completion'],coarse_identity=p['coarse_identity'],checkpoint=p['checkpoint'],smoke_override=p['smoke_override'],checkpoint_retrained=False,checkpoint_copied=False,settling_seconds=2,guard='One original idle check after buffer; original thresholds/foreign-process rejection unchanged',started_unix=time.time())
 write(out/'identity.json',record)
 try:
  with runner.gpu_claims([gpu],OUT/'claims',[]):
   runner.check_frozen_identity(identity['source_identity'],p['python_inventory']);guard=disk_guard(root,3)
   evaluation=out/'eval3'
   argv=[sys.executable,'-u','-B',str(here/'entry.py'),'eval','--checkpoint',str(cp.relative_to(root)),'--cull','--frames','0','150','299','--out',str(evaluation.relative_to(root)),'--label','dgH_cube_diagnostic50steps']
   with (out/'logs/eval3.log').open('xb') as log:
    proc=subprocess.Popen(argv,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(out/'events/eval3_started.json',dict(pid=proc.pid,argv=argv,disk_guard=guard,started_unix=time.time()))
    begin=time.monotonic();code=proc.wait()
   write(out/'events/eval3_returned.json',dict(exit_code=code,seconds=time.monotonic()-begin,log=entry(out/'logs/eval3.log')))
   if code:raise RuntimeError('Existing50step eval3 exited '+str(code)+'; no formal task may follow')
   runner.check_frozen_identity(identity['source_identity'],p['python_inventory']);disk_guard(root,0)
   summary=read(evaluation/'summary.json')
   with (evaluation/'metrics_per_observation.csv').open() as f:rows=list(csv.DictReader(f))
   test=[r for r in rows if r['scope']=='test']
   if summary.get('status')!='completed' or summary.get('frames')!=3 or summary.get('checkpoint')!=p['checkpoint'] or summary.get('posthoc_cull') is not True or sorted((r['camera'],int(r['frame'])) for r in test)!=[('cam00',0),('cam00',150),('cam00',299)]:raise ValueError('Three-frame cam00 culled evaluation incomplete')
   if not all(math.isfinite(float(r[k])) for r in test for k in ['psnr','ssim','lpips']):raise ValueError('Nonfinite startup metrics')
   final=dict(record,status='completed_diagnostic50steps_and_eval3',diagnostic50steps=True,evaluation=entry(evaluation/'summary.json'),metrics_csv=entry(evaluation/'metrics_per_observation.csv'),seconds=time.time()-record['started_unix'],formal_author_schedule_unchanged=[3000,14000],meaning='Existing50step runtime-gradient acceptance plus new3-frame evaluation; no quality/convergence claim')
   write(out/'complete.json',final);print(json.dumps(final))
 except BaseException:
  write(out/('failure_'+str(time.time_ns())+'.json'),dict(record,status='failed_eval3_existing50step_preserved_no_formal_training',error=traceback.format_exc(),automatic_retry=False));raise
'''
WORKER_SHA = hashlib.sha256(WORKER.encode()).hexdigest()


def bound_checkpoint():
    out = parent.safe_local_member(OLD_REL)
    spec = parent.read(out / 'spec.json')
    original = VALIDATE_RUNTIME_SPEC(spec)
    smoke = parent.safe_local_member(SMOKE_REL)
    cp = smoke / 'coarse50/checkpoint_fine_00000.pt'
    complete = smoke / 'coarse50/complete.json'
    override = smoke / 'coarse50/smoke_override.json'
    identity = smoke / 'identity.json'
    done, marked, ident = parent.read(complete), parent.read(override), parent.read(identity)
    if parent.sha(cp) != CHECKPOINT_SHA or done.get('status') != 'completed' or done.get('checkpoint') != parent.entry(cp) or done['state']['coarse_iteration'] != 50 or done['state']['fine_iteration'] != 0 or marked.get('not_eligible_for_formal_parent') is not True or ident.get('condition') != 'dgH_cube' or ident.get('seed') != parent.SEED:
        raise ValueError('The accepted existing50step checkpoint/diagnostic marker differs')
    returned = smoke / 'events/coarse50_returned.json'
    if parent.read(returned).get('exit_code') != 0 or (smoke / 'complete.json').exists() or (smoke / 'events/eval3_started.json').exists():
        raise ValueError('Only missing eval3 after successful50step may resume')
    failures = sorted(out.glob('failure_*.json'))
    if len(failures) != 1 or parent.read(failures[0]).get('remote_started') is not False:
        raise ValueError('Previous formal training must not have started')
    service = parent.OUT / 'pipelines' / (OLD_ID + '.startup.service.log')
    if "child('eval3'" not in service.read_text() or 'Allocated GPU is not idle' not in service.read_text():
        raise ValueError('Expected eval3-before-launch idle-check failure required')
    events = sorted((out / 'events').glob('*.json'))
    if any('remote_dispatch_attempted' in p.name for p in events):
        raise ValueError('Previous formal dispatch protected')
    runtime.previous.runner.check_frozen_identity(ident['source_identity'], sorted(Path(p).name for p in spec['freeze']['namespace_python']))
    refs = [out / 'spec.json', out / 'resume_binding.json', failures[0], *events,
            cp, complete, override, identity, returned, service]
    pins = {str(p.relative_to(PROJECT)): parent.sha(p) for p in refs}
    return dict(original, spec=spec, pins=pins, checkpoint=cp, coarse_completion=complete,
                smoke_override=override, coarse_identity=identity, failed_startup=failures[0])


def validate_spec(spec):
    parent.validate_spec(spec)
    bound = bound_checkpoint()
    old = bound['spec']
    recipe = spec.get('complete_startup_eval', {})
    if any(spec.get(k) != old.get(k) for k in runtime.SCIENCE_KEYS) or spec['host'] != 'cts' or spec['condition'] != 'dgH_cube' or spec.get('supervisor') is not None or spec['run_id'] in [OLD_ID, runtime.PREVIOUS_ID, runtime.previous.OLD_ID]:
        raise ValueError('Fresh unchanged CTS Desk half seed07 recovery required')
    if recipe.get('schema') != SCHEMA or recipe.get('old_out') != OLD_REL or recipe.get('worker_sha256') != WORKER_SHA or recipe.get('settling_seconds') != 2:
        raise ValueError('Transparent eval-only worker/settling identity required')
    for key in ['checkpoint', 'coarse_completion', 'smoke_override', 'coarse_identity', 'failed_startup', 'table', 'sync_verified']:
        if recipe.get(key) != parent.entry(bound[key]):
            raise ValueError('Recovery provenance differs: ' + key)
    inputs = dict(old['freeze']['inputs'], **bound['pins'])
    inputs.update({str((DEPLOY / n).relative_to(PROJECT)): parent.sha(DEPLOY / n) for n in FILES})
    if spec['freeze'] != dict(old['freeze'], inputs=inputs):
        raise ValueError('Original scientific freeze plus exact existing50step/new source evidence required')
    return bound


class CompleteStartupEval(runtime.RuntimeEnvRecovery):
    def payload(self, mode, **extra):
        return dict(mode=mode, root=self.spec['remote_root'], condition='dgH_cube', seed=parent.SEED,
                    uuid=self.spec['gpus'][0]['uuid'], worker_sha256=WORKER_SHA,
                    adaptation_source=parent.entry(__file__), **extra)

    def rewrite_ssh(self, command):
        if isinstance(command, (list, tuple)) and list(command[:len(self.ssh)]) == self.ssh and len(command) == len(self.ssh) + 1:
            remote = shlex.split(command[-1])
            target = self.spec['remote_root'] + '/experiments/dynamic_sr_highres_20261011/persistent_runner.py'
            if target in remote:
                index = remote.index(target)
                payload = self.payload('runner', argv=remote[index + 1:])
                remote = [*remote[:index], '-c', WORKER, json.dumps(payload)]
                command = [*self.ssh, shlex.join(remote)]
        return super().rewrite_ssh(command)

    def startup(self):
        self.stage = 'GPU_existing50step_eval3_startup'
        self.frozen()
        recipe = self.spec['complete_startup_eval']
        self.smoke_rel = f'output/dynamic_sr_highres_20261011/smoke/dgH_cube_seed{parent.SEED}_{self.spec["run_id"]}'
        payload = self.payload('eval', out=self.smoke_rel,
            python_inventory=sorted(Path(p).name for p in self.freeze['namespace_python']),
            **{k: recipe[k] for k in ['checkpoint', 'coarse_completion', 'smoke_override', 'coarse_identity']})
        resources = self.remote('resources', dict(gpus=self.spec['gpus'], host='cts', future_bytes=65 * 10**9))
        self.record('startup_resources_verified', **resources)
        unit = 'highres-startup-' + self.spec['run_id']
        logfile_rel = 'output/dynamic_sr_highres_20261011/pipelines/' + self.spec['run_id'] + '.startup.service.log'
        cmd = ['systemd-run', '--user', '--wait', '--unit=' + unit,
               '--property=WorkingDirectory=' + self.spec['remote_root'],
               '--property=StandardOutput=append:' + self.spec['remote_root'] + '/' + logfile_rel,
               '--property=StandardError=append:' + self.spec['remote_root'] + '/' + logfile_rel,
               '--setenv=PYTHONDONTWRITEBYTECODE=1', self.spec['remote_python'], '-u', '-B', '-c', WORKER, json.dumps(payload)]
        self.smoke_started = True
        self.record('startup_eval3_only_dispatch_attempted', argv=cmd, checkpoint=recipe['checkpoint'],
                    unit=unit + '.service', completion_trigger='systemd-run --wait native D-Bus exit event')
        result = subprocess.run([*self.ssh, shlex.join(cmd)], capture_output=True, text=True)
        self.record('startup_exit_or_SSH_returned', exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        self.pull_paths([self.smoke_rel, logfile_rel], 'startup')
        if result.returncode:
            raise RuntimeError('Existing50step eval3 service exited/SSH returned ' + str(result.returncode) + '; no formal task started')
        smoke = parent.local(self.smoke_rel); complete = parent.read(smoke / 'complete.json')
        if complete.get('status') != 'completed_diagnostic50steps_and_eval3' or complete.get('eligible_for_formal_parent') is not False or complete.get('checkpoint') != recipe['checkpoint'] or complete.get('smoke_override') != recipe['smoke_override'] or complete.get('worker_sha256') != WORKER_SHA or complete.get('checkpoint_retrained') is not False:
            raise ValueError('New eval3 completion does not bind the original diagnostic checkpoint')
        for key in ['evaluation', 'metrics_csv']:
            if complete[key] != parent.entry(parent.local(complete[key]['path'])):
                raise ValueError('Returned eval3 artifact SHA differs')
        parent.write(self.out / 'startup_acceptance.json', dict(status='passed_runtime_gradient_and_eval3_only',
            complete=parent.entry(smoke / 'complete.json'), checkpoint=recipe['checkpoint'],
            coarse_completion=recipe['coarse_completion'], diagnostic_only=True, formal_parent_eligible=False,
            checkpoint_retrained=False, adaptation_source=parent.entry(__file__), worker_sha256=WORKER_SHA))
        self.frozen()

    def run(self):
        # Reuse its condition lock, environment scope and failure-return control
        # flow; only the spec validator/startup/owned worker differ locally.
        original = runtime.validate_spec
        runtime.validate_spec = validate_spec
        try:
            return super().run()
        finally:
            runtime.validate_spec = original


def make_spec(run_id, target):
    target = parent.local(target)
    if target.exists():
        raise FileExistsError('Preserve existing spec; use a fresh path')
    bound = bound_checkpoint(); spec = copy.deepcopy(bound['spec'])
    spec.update(run_id=run_id, out=str((parent.OUT / 'orchestration' / run_id).relative_to(PROJECT)),
                supervisor=None, created_utc=parent.now())
    spec['complete_startup_eval'] = dict(schema=SCHEMA, old_out=OLD_REL, worker_sha256=WORKER_SHA,
        settling_seconds=2, **{k: parent.entry(bound[k]) for k in ['checkpoint', 'coarse_completion', 'smoke_override', 'coarse_identity', 'failed_startup', 'table', 'sync_verified']})
    inputs = dict(bound['spec']['freeze']['inputs'], **bound['pins'])
    inputs.update({str((DEPLOY / n).relative_to(PROJECT)): parent.sha(DEPLOY / n) for n in FILES})
    spec['freeze'] = dict(bound['spec']['freeze'], inputs=inputs)
    validate_spec(spec)
    if parent.local(spec['out']).exists():
        raise FileExistsError('Fresh run output required')
    parent.write(target, spec); return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path); ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--make-spec', action='store_true'); ap.add_argument('--run-id'); ap.add_argument('--spec-out', type=Path)
    args = ap.parse_args()
    if args.make_spec:
        if args.spec or not args.run_id or not args.spec_out:ap.error('--make-spec requires --run-id/--spec-out only')
        spec = make_spec(args.run_id, args.spec_out)
        print(json.dumps(dict(status='fresh_eval3_only_spec_not_launched', run_id=spec['run_id'], worker_sha256=WORKER_SHA))); return
    if not args.spec:ap.error('--spec required')
    spec = parent.read(parent.local(args.spec)); bound = validate_spec(spec)
    if args.validate_only:
        print(json.dumps(dict(status='local_existing50step_interface_checked_no_SSH_GPU', checkpoint=parent.entry(bound['checkpoint']), table=parent.entry(bound['table']), worker_sha256=WORKER_SHA))); return
    CompleteStartupEval(spec).run()


if __name__ == '__main__':
    main()
