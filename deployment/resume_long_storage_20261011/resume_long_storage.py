"""Resume Long's failed external-HR copy in its existing stage, then the original two-GPU DAG."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import contextlib
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'deployment/readonly_hr_a100_20261011'))
import storage_supervisor as original
parent, storage = original.parent, original.storage
OLD_ID = 'long360_seed07_storage_20261011t061138z'
OLD_REL = 'output/dynamic_sr_highres_20261011/orchestration/' + OLD_ID
SCHEMA = 'highres_resume_existing_Long_HR_stage_v1'
FILES = ('resume_long_storage.py', 'README.md')
SCIENCE_KEYS = ('schema','dataset','condition','host','remote_root','remote_python','seed','rung',
                'local_python','workers','cpu_threads','reserve_bytes','remote_future_gb','gpus','expansion_policy')

PARTIAL = r'''
import hashlib,importlib.util,json,pathlib,sys,time
helper=pathlib.Path(sys.argv[1]);expected=sys.argv[2];root=pathlib.Path(sys.argv[3]);run_id=sys.argv[4]
if not helper.is_absolute() or not helper.resolve().is_relative_to(root):raise ValueError('Helper outside workspace')
for q in [helper,*helper.parents]:
 if q.is_symlink():raise ValueError('Helper symlink rejected')
 if q==root:break
if hashlib.sha256(helper.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen storage helper changed')
s=importlib.util.spec_from_file_location('owned_remote_storage',helper);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
full=json.load(sys.stdin)['files'];files=m.validate_inventory(full);source,target,out,drive,identity=m.context(run_id)
for q in [source,out,target]:m.owned(q)
if m.mountinfo(target) or any(target.iterdir()) or (out/'mount_started.json').exists():raise FileExistsError('Existing mapping/attempt protected; partial-copy recovery only')
unit='highres-hr-'+run_id+'.service';state=m.capture(['systemctl','--user','show',unit,'--property=LoadState','--value'])
if state['stdout'].strip() not in ['not-found','']:raise FileExistsError('Existing mount service protected')
seen=set();reused=[];start=time.monotonic()
for q in source.rglob('*'):
 m.no_symlinks(q)
 if q.is_file():
  rel=str(q.relative_to(source))
  if rel not in files:raise ValueError('Unexpected external member preserved: '+rel)
  v=files[rel]
  if q.stat().st_size!=v['bytes'] or m.digest(q)!=v['sha256']:raise ValueError('Different external member preserved: '+rel)
  seen.add(rel);reused.append(rel)
 elif not q.is_dir():raise ValueError('Nonregular external member protected')
missing=sorted(set(files)-seen);needed=sum(files[r]['bytes'] for r in missing)
print(json.dumps(dict(status='existing_owned_HR_stage_partial_SHA_accepted',storage_run_id=run_id,source=str(source),target=str(target),unit=unit,reused_count=len(reused),missing=missing,missing_bytes=needed,existing_hash_seconds=time.monotonic()-start,disk=m.guard(m.DRIVE,needed),root_disk=m.guard(m.ROOT,0),package_binary=identity,parent_drive=drive,training=False)))
'''
WORKER = r'''
import argparse,json,pathlib,sys,time
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);sys.path.insert(0,str(root/'experiments/dynamic_sr_highres_20261011'))
import persistent_runner as runner
idle=runner.check_gpu_idle
def settled(gpu,allowed):
 time.sleep(2)
 names=list(allowed)
 if int(gpu['index'])==0 and gpu['uuid']==p['gpu0']['uuid'] and 'gnome-remote-desktop-daemon' not in names:names.append('gnome-remote-desktop-daemon')
 return idle(gpu,names)
runner.check_gpu_idle=settled
print(json.dumps(dict(kind='owned_A100_service_adapter',source=p['source'],worker_sha256=p['worker_sha256'],startup_allocation=p.get('startup_allocation'),settling_seconds=2,original_thresholds_unchanged=True,GPU0_graphics_allowance='gnome-remote-desktop-daemon<=original600MiB')),flush=True)
sys.argv=[p['target'],*p['argv']]
if pathlib.Path(p['target']).name=='gpu_startup.py':
 import gpu_startup
 original_validate=gpu_startup.validate
 def gpu0_prerequisites(a):
  if a.condition!='l3H_long360' or a.gpu!=0 or a.uuid!=p['gpu0']['uuid']:raise ValueError('Explicit authorized Long GPU0 startup binding required')
  original_args=argparse.Namespace(**vars(a));original_args.gpu=1
  return original_validate(original_args)
 gpu_startup.validate=gpu0_prerequisites
 # Only CPU prerequisite validation sees the original default index1.
 # Original execute receives the real index0/UUID and checks actual hardware.
 gpu_startup.main()
else:runner.main()
'''
EARLY_WORKER = r'''
import hashlib,json,os,pathlib,socket,sys,time,traceback
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);sys.path.insert(0,str(root/'experiments/dynamic_sr_highres_20261011'))
import persistent_runner as r
out=r.local(p['out'])
if not out.is_relative_to(r.OUT/'pipelines') or out.exists():raise FileExistsError('Fresh owned early-LR output required')
r.mkdir(out);r.mkdir(out/'logs');r.mkdir(out/'events')
try:
 tablepath=r.local(p['table']['path'])
 if r.entry(tablepath)!=p['table']:raise ValueError('Original complete28-task table changed')
 table=r.read(tablepath);r.validate_table(table)
 if table['condition']!='l3H_long360' or table['seed']!=20261007:raise ValueError('Only original Long seed07 LR tasks')
 names=['l3H_long360_seed20261007_coarse','l3H_long360_seed20261007_fine']
 tasks=[next(t for t in table['tasks'] if t['name']==n) for n in names]
 if tasks[0]['deps']!=[] or tasks[1]['deps']!=[names[0]] or any(t['script']!='entry.py' or t['argv'][0]!='lr' for t in tasks):raise ValueError('Original LR dependency chain differs')
 for key in ('startup_complete','startup_identity'):
  if r.entry(r.local(p[key]['path']))!=p[key]:raise ValueError('Accepted startup artifact changed')
 complete=r.read(r.local(p['startup_complete']['path']));identity=r.read(r.local(p['startup_identity']['path']))
 if complete['status']!='completed_diagnostic50steps_and_eval3' or complete['eligible_for_formal_parent'] is not False or identity['condition']!='l3H_long360' or identity['seed']!=20261007 or identity['gpu']['uuid']!=p['gpu0']['uuid']:raise ValueError('Actual GPU0 startup acceptance required')
 sources=identity['source_identity'];namespace=sorted(pathlib.Path(q).name for q in p['namespace_python'])
 actual={q['path']:q['sha256'] for q in sources}
 if any(actual.get(q)!=sha for q,sha in p['namespace_python'].items()):raise ValueError('Startup sources differ from frozen scientific namespace')
 sources=[*sources,p['table'],p['startup_complete'],p['startup_identity']]
 r.check_frozen_identity(sources,namespace)
 gpu=r.resolve_gpu(p['gpu0']['uuid'])
 if int(gpu['index'])!=0 or gpu['uuid']!=p['gpu0']['uuid'] or 'A100' not in gpu['name'] or '5090' in gpu['name']:raise ValueError('Authorized A100 GPU0 binding differs')
 idle=r.check_gpu_idle
 def settled(g,names):
  time.sleep(2);return idle(g,names)
 r.check_gpu_idle=settled
 record=dict(status='running_original_LR_chain',tasks=p['table'],selected_tasks=tasks,gpu=gpu,source=p['source'],worker_sha256=p['worker_sha256'],source_identity=sources,python_inventory=namespace,host=socket.gethostname(),PID=os.getpid(),started_unix=time.time(),scientific_argv_and_outputs_unchanged=True,formal_author_schedule=[3000,14000])
 r.write(out/'identity.json',record);results={}
 with r.gpu_claims([gpu],r.OUT/'claims',['gnome-remote-desktop-daemon']):
  for task in tasks:
   if r.local(task['done']).exists():results[task['name']]=dict(status='already_complete',artifact=r.check_completion(task));continue
   if r.local(task['done']).parent.exists():raise FileExistsError('Partial original LR output preserved; no automatic restart')
   results[task['name']]=r.execute(task,gpu,out,sources,['gnome-remote-desktop-daemon'],namespace)
 # Write success after releasing the single-GPU claim, before native unit exit.
 r.write(out/'complete.json',dict(record,status='completed_original_LR_parent_17k',results=results,seconds=time.time()-record['started_unix'],GPU0_claim_released=True,automatic_retry=False))
except BaseException:
 r.write(out/f'failure_{time.time_ns()}.json',dict(status='failed_early_LR_preserved_no_formal_dispatch',error=traceback.format_exc(),automatic_retry=False));raise
'''
PARTIAL_SHA = hashlib.sha256(PARTIAL.encode()).hexdigest()
WORKER_SHA = hashlib.sha256(WORKER.encode()).hexdigest()
EARLY_WORKER_SHA = hashlib.sha256(EARLY_WORKER.encode()).hexdigest()


def native_wait(command, receipt):
    """Only this separate receipt is written by the background wait thread."""
    begin=time.monotonic()
    try:
        result=subprocess.run(command,capture_output=True,text=True)
        value=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,seconds=time.monotonic()-begin)
    except BaseException:
        value=dict(exit_code=None,error=traceback.format_exc(),seconds=time.monotonic()-begin)
    parent.write(receipt,value)
    return value


def bound_checkpoint():
    out = parent.safe_local_member(OLD_REL); spec = parent.read(out/'spec.json')
    original.validate_storage_spec(spec)
    failures = sorted(out.glob('failure_*.json'))
    if len(failures)!=1:raise ValueError('One archived original failure required; let its original parent exit first')
    failed = parent.read(failures[0])
    if failed.get('status')!='failed_preserved_no_retry' or failed.get('stage')!='readonly_HR_external_stage' or failed.get('remote_started') is not False or 'External HR additive transfer failed' not in failed.get('error',''):raise ValueError('Only failed external copy before GPU dispatch is recoverable')
    events = sorted((out/'events').glob('*.json'))
    for stage,end in [('prepare_all_video_half',4),('initialize_seed07',7),('expand_privileged_HR',10),('register_LR',13),('plan_teacher',15),('make_screening_tasks',18)]:
        e=parent.read(out/'events'/next(p.name for p in events if p.name.startswith(f'{end:03d}_')))
        if e['stage']!=stage or e.get('exit_code')!=0:raise ValueError('Completed original CPU stage differs: '+stage)
    transferred = out/'events/025_external_HR_transfer_returned.json'
    if parent.read(transferred).get('exit_code')!=10 or 'code 10' not in (out/'logs/external_HR_sync.log').read_text() or any('dispatch_attempted' in p.name for p in events):raise ValueError('Original rsync10 and absence of GPU dispatch required')
    final = dict(spec['freeze'],inputs=parent.read(out/'events/020_input_freeze.json')['frozen_inputs'])
    parent.check_freeze(final)
    table=parent.OUT/'pipelines'/f'tasks_l3H_long360_seed{parent.SEED}_{OLD_ID}.json'
    if final['inputs'].get(str(table.relative_to(PROJECT)))!=parent.sha(table):raise ValueError('Original28-task table is not frozen')
    original.parent.validate_spec(dict(spec,freeze=final))
    import persistent_runner
    persistent_runner.validate_table(parent.read(table))
    refs=[out/'spec.json',out/'screening_freeze.json',out/'Z5_gate.json',out/'external_HR_members.txt',
          failures[0],out/'logs/external_HR_sync.log',*events]
    return dict(spec=spec,final_freeze=final,table=table,failure=failures[0],
                pins={str(p.relative_to(PROJECT)):parent.sha(p) for p in refs})


def validate_spec(spec):
    original.validate_storage_spec(spec); bound=bound_checkpoint(); old=bound['spec']; r=spec.get('existing_storage_resume',{})
    if any(spec.get(k)!=old.get(k) for k in SCIENCE_KEYS) or spec['run_id']==OLD_ID or spec.get('supervisor') is not None or spec['storage_recipe'].get('ssh_jump') is not None:raise ValueError('Keep Long half/seed/twoGPU/budget settings; use fresh run and direct A100')
    if r.get('schema')!=SCHEMA or r.get('storage_run_id')!=OLD_ID or r.get('old_out')!=OLD_REL or r.get('worker_sha256')!=WORKER_SHA or r.get('partial_helper_sha256')!=PARTIAL_SHA or r.get('early_worker_sha256')!=EARLY_WORKER_SHA or r.get('early_LR_overlap') is not True or r.get('startup_gpu')!=spec['gpus'][0] or r.get('formal_gpu_count')!=2:raise ValueError('Original partial stage and owned helper/allocation identities required')
    for key,path in [('table',bound['table']),('failure',bound['failure'])]:
        if r.get(key)!=parent.entry(path):raise ValueError('Original provenance differs: '+key)
    invref=r.get('inventory',{}); audit=r.get('operator_remote_audit',{})
    for ref in [invref,audit]:
        if not ref.get('path') or ref!=parent.entry(parent.safe_local_member(ref['path'])):raise ValueError('Recovery inventory/audit SHA differs')
    inv=parent.read(parent.safe_local_member(invref['path']))
    if inv.get('schema')!='highres_additive_stream_sync_v1' or inv.get('host')!='a100-train' or inv.get('destination')!=spec['remote_root'] or inv.get('future_bytes')!=65*10**9 or inv.get('freeze')!=bound['final_freeze']:raise ValueError('Reconstructed synchronization inventory differs')
    hr={k:v for k,v in inv['files'].items() if k.startswith(storage.PREFIX)};storage.validate_inventory(hr)
    expected=dict(bound['final_freeze']['inputs'],**bound['pins'])
    expected.update({str((DEPLOY/n).relative_to(PROJECT)):parent.sha(DEPLOY/n) for n in FILES})
    expected.update({ref['path']:ref['sha256'] for ref in [invref,audit]})
    if spec['freeze']!=dict(bound['final_freeze'],inputs=expected):raise ValueError('Original final scientific freeze plus exact recovery evidence required')
    return dict(bound,inventory=inv)


class LongStorageRecovery(original.StorageSupervisor):
    def sync_non_hr(self,inventory):
        early=dict(inventory,files={k:v for k,v in inventory['files'].items() if not k.startswith(storage.PREFIX)})
        outer=self.out;self.out=parent.mkdir(outer/'early_inputs_sync')
        parent.mkdir(self.out/'logs');parent.mkdir(self.out/'events')
        try:parent.Supervisor.sync(self,early)
        finally:self.out=outer
        self.record('non_external_HR_inputs_synced',inventory=parent.entry(outer/'early_inputs_sync/sync_inputs.json'),files=len(early['files']),original_inventory=self.spec['existing_storage_resume']['inventory'])

    def start_early(self,pool,table):
        self.stage='original_LR_overlap_dispatch';self.frozen()
        acceptance=parent.read(self.out/'startup_acceptance.json')
        self.early_rel='output/dynamic_sr_highres_20261011/pipelines/'+self.spec['run_id']+'.early_LR'
        self.early_log='output/dynamic_sr_highres_20261011/pipelines/'+self.spec['run_id']+'.early_LR.service.log'
        payload=dict(root=self.spec['remote_root'],out=self.early_rel,table=parent.entry(table),
                     startup_complete=acceptance['complete'],startup_identity=parent.entry(parent.local(self.smoke_rel)/'identity.json'),
                     namespace_python=self.freeze['namespace_python'],gpu0=self.spec['gpus'][0],source=parent.entry(__file__),worker_sha256=EARLY_WORKER_SHA)
        unit='highres-early-LR-'+self.spec['run_id']
        cmd=['systemd-run','--user','--wait','--unit='+unit,'--property=WorkingDirectory='+self.spec['remote_root'],
             '--property=StandardOutput=append:'+self.spec['remote_root']+'/'+self.early_log,
             '--property=StandardError=append:'+self.spec['remote_root']+'/'+self.early_log,
             '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OMP_NUM_THREADS=4',self.spec['remote_python'],'-u','-B','-c',EARLY_WORKER,json.dumps(payload)]
        self.early_started=True
        parent.write(self.out/'early_LR_dispatch.json',dict(argv=cmd,unit=unit+'.service',payload=payload,completion_trigger='systemd-run --wait native D-Bus unit exit; no model polling',single_GPU_claim=True))
        return pool.submit(native_wait,[*self.ssh,shlex.join(cmd)],self.out/'early_LR_native_exit.json')

    def finish_early(self,future,table):
        self.stage='original_LR_overlap_exit';result=future.result()
        lr_runs='output/dynamic_sr_highres_20261011/runs/l3H_long360_seed20261007/'
        self.pull_paths([self.early_rel,self.early_log,lr_runs+'lr_coarse',lr_runs+'lr_parent'],'early_LR')
        if result['exit_code']!=0:raise RuntimeError('Early LR native service/SSH failed; artifacts preserved, no formal dispatch or retry')
        receipt=parent.local(self.early_rel)/'complete.json';done=parent.read(receipt)
        names=['l3H_long360_seed20261007_coarse','l3H_long360_seed20261007_fine']
        if done.get('status')!='completed_original_LR_parent_17k' or done.get('tasks')!=parent.entry(table) or sorted(done.get('results',{}))!=sorted(names) or done.get('GPU0_claim_released') is not True:raise ValueError('Both original LR tasks must pass completion before the full DAG')
        self.record('original_LR_overlap_completed',complete=parent.entry(receipt),native_exit=parent.entry(self.out/'early_LR_native_exit.json'),full_DAG_reuses_original_done=True)

    def result_paths(self):
        return [*super().result_paths(),self.early_rel,self.early_log]

    def remote(self,mode,value):
        if mode=='resources' and self.stage=='GPU_diagnostic50steps_startup':
            value=dict(value,gpus=[self.spec['gpus'][0]])
            self.record('GPU0_startup_operational_allocation',original_default_index=1,
                        actual_gpu=self.spec['gpus'][0],formal_gpus=self.spec['gpus'],
                        science_parameters_changed=False)
        return super().remote(mode,value)

    def cpu(self,name,script,argv,estimated_gb=0):
        if self.stage!='headroom_analysis' or name!='headroom' or script!='headroom.py' or estimated_gb!=0 or list(map(str,argv))!=['--spec',str(self.out/'headroom_spec.json'),'--out',str(self.out/'headroom.json')]:raise RuntimeError('Only original final headroom summary allowed; no repeated CPU preparation')
        return super().cpu(name,script,argv,estimated_gb=0)

    def storage_call(self,mode,files):
        if mode=='prepare':raise RuntimeError('Reuse original partial stage; never create another HR tree')
        self.frozen();remote_script=self.spec['remote_root']+'/deployment/readonly_hr_a100_20261011/remote_storage.py'
        digest=self.freeze['inputs'][str((original.DEPLOY/'remote_storage.py').relative_to(PROJECT))]
        argv=[self.spec['remote_python'],'-B','-c',original.PINNED_HELPER,remote_script,digest,self.spec['remote_root'],'--mode',mode,'--run-id',OLD_ID]
        result=subprocess.run([*self.ssh,shlex.join(argv)],input=json.dumps(dict(files=files)),text=True,capture_output=True)
        self.record('storage_'+mode,storage_run_id=OLD_ID,exit_code=result.returncode,stdout=result.stdout[-16000:],stderr=result.stderr[-8000:])
        if result.returncode:raise RuntimeError('Existing HR stage '+mode+' failed; preserved, no retry: '+result.stderr[-2000:])
        self.frozen();return json.loads(result.stdout)

    def sync(self,inventory):
        self.stage='resume_existing_external_HR';self.frozen();self.storage_attempted=True
        files={k:v for k,v in inventory['files'].items() if k.startswith(storage.PREFIX)}
        helper=self.spec['remote_root']+'/deployment/readonly_hr_a100_20261011/remote_storage.py'
        digest=self.freeze['inputs'][str((original.DEPLOY/'remote_storage.py').relative_to(PROJECT))]
        argv=[self.spec['remote_python'],'-B','-c',PARTIAL,helper,digest,self.spec['remote_root'],OLD_ID]
        result=subprocess.run([*self.ssh,shlex.join(argv)],input=json.dumps(dict(files=files)),text=True,capture_output=True)
        self.record('existing_external_HR_preflight',exit_code=result.returncode,stderr=result.stderr[-8000:],helper_sha256=PARTIAL_SHA)
        if result.returncode:raise RuntimeError('Original partial external stage rejected; preserve all contents: '+result.stderr[-2000:])
        pre=json.loads(result.stdout);parent.write(self.out/'external_HR_partial_acceptance.json',pre)
        listing=self.out/'external_HR_missing.txt';listing.write_text(''.join(k+'\n' for k in pre['missing']))
        if pre['missing']:
            cmd=original.external_rsync_argv(self.ssh,'a100-train',PROJECT/storage.PREFIX,pre['source'],listing)
            with (self.out/'logs/external_HR_sync.log').open('xb') as log:
                start=time.monotonic();p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            self.record('external_HR_missing_transfer_returned',exit_code=p.returncode,seconds=time.monotonic()-start,files=len(pre['missing']),bytes=pre['missing_bytes'])
            if p.returncode:raise RuntimeError('External HR missing-only copy failed; original/new bytes preserved, no retry')
        self.storage_call('verify',files)
        mapping=self.storage_call('mount',files);mapping['deployment_route']=dict(host='a100-train',ssh_jump=None,storage_run_id=OLD_ID,outer_run_id=self.spec['run_id'])
        parent.write(self.out/'storage_mapping.json',mapping)
        parent.Supervisor.sync(self,inventory)
        self.storage_call('mount-check',files)

    @contextlib.contextmanager
    def owned_workers(self):
        actual=subprocess.run
        def run(command,*args,**kwargs):
            if isinstance(command,(list,tuple)) and list(command[:len(self.ssh)])==self.ssh and len(command)==len(self.ssh)+1:
                remote=shlex.split(command[-1]);targets={self.spec['remote_root']+'/experiments/dynamic_sr_highres_20261011/'+n for n in ['gpu_startup.py','persistent_runner.py']}
                matches=[i for i,v in enumerate(remote) if v in targets]
                if matches:
                    i=matches[0];remote_args=remote[i+1:];allocation=None
                    if Path(remote[i]).name=='gpu_startup.py':
                        remote_args=list(remote_args);remote_args[remote_args.index('--gpu')+1]='0';remote_args[remote_args.index('--uuid')+1]=self.spec['gpus'][0]['uuid']
                        allocation=dict(original_default_index=1,actual_index=0,actual_uuid=self.spec['gpus'][0]['uuid'],reason='User authorized GPU0-first runtime acceptance; formal allocation independently checked')
                    payload=dict(root=self.spec['remote_root'],target=remote[i],argv=remote_args,source=parent.entry(__file__),worker_sha256=WORKER_SHA,gpu0=self.spec['gpus'][0],startup_allocation=allocation)
                    remote=[*remote[:i],'-c',WORKER,json.dumps(payload)];command=[*self.ssh,shlex.join(remote)]
                    self.record('owned_service_worker',argv=command,worker_sha256=WORKER_SHA,settling_seconds=2,startup_allocation=allocation)
            return actual(command,*args,**kwargs)
        subprocess.run=run
        try:yield
        finally:subprocess.run=actual

    def run(self):
        bound=validate_spec(self.spec)
        if self.out.exists():raise FileExistsError('Fresh recovery output required')
        parent.mkdir(self.out);parent.mkdir(self.out/'logs');parent.mkdir(self.out/'events');parent.write(self.out/'spec.json',self.spec)
        parent.write(self.out/'resume_binding.json',dict(status='existing_Long_HR_copy_only_resume',storage_run_id=OLD_ID,table=parent.entry(bound['table']),CPU_repeated=False,Z5_repeated=False,task_table_rewritten=False,scientific_parameters_changed=False))
        lock=parent.mkdir(parent.OUT/'orchestration/claims')/'l3H_long360_seed07.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                self.sync_non_hr(bound['inventory'])
                with self.owned_workers():self.startup()
                # The wait thread writes only its own native-exit receipt. All
                # stage/serial events and transfer operations stay on this thread.
                with cf.ThreadPoolExecutor(max_workers=1) as pool:
                    early=self.start_early(pool,bound['table']);storage_error=None;early_error=None
                    try:self.sync(bound['inventory'])
                    except BaseException as error:
                        storage_error=(error,self.stage)
                        parent.write(self.out/'overlap_storage_failure.json',dict(stage=self.stage,error=traceback.format_exc(),formal_dispatch=False))
                    try:self.finish_early(early,bound['table'])
                    except BaseException as error:early_error=error
                    if storage_error:
                        self.stage=storage_error[1];raise storage_error[0]
                    if early_error:raise early_error
                with self.owned_workers():self.runner(bound['table']);self.pull();self.summarize()
            except BaseException:
                parent.write(self.out/('failure_'+str(time.time_ns())+'.json'),dict(status='Long_storage_recovery_failed_preserved',stage=self.stage,error=traceback.format_exc(),remote_started=self.remote_started,smoke_started=self.smoke_started,early_LR_started=getattr(self,'early_started',False),automatic_retry=False,storage_run_id=OLD_ID))
                if self.smoke_started and not (self.out/'startup_acceptance.json').exists():
                    try:self.pull_paths([self.smoke_rel,'output/dynamic_sr_highres_20261011/pipelines/'+self.spec['run_id']+'.startup.service.log'],'startup_failure_return')
                    except BaseException as error:self.record('startup_return_failed_remote_preserved',error=str(error))
                if self.remote_started and not self.formal_returned:
                    try:self.pull()
                    except BaseException as error:self.record('formal_return_failed_remote_preserved',error=str(error))
                raise
            finally:
                if self.storage_attempted:
                    try:self.pull_paths(['output/dynamic_sr_highres_20261011/storage/'+OLD_ID],'readonly_storage_return')
                    except BaseException as error:self.record('readonly_storage_return_failed_remote_preserved',error=str(error))


def make_spec(run_id,target,remote_audit,inventory_path):
    target=parent.local(target);inventory_path=parent.local(inventory_path)
    if target.exists():raise FileExistsError('Preserve existing spec; use a fresh path')
    bound=bound_checkpoint();spec=copy.deepcopy(bound['spec']);spec.update(run_id=run_id,out=str((parent.OUT/'orchestration'/run_id).relative_to(PROJECT)),supervisor=None,created_utc=parent.now(),freeze=bound['final_freeze'])
    spec['storage_recipe']=dict(spec['storage_recipe'],ssh_jump=None)
    # The operator collected this exact original inventory once. Hash only its
    # small JSON reference, not the38GB prepared assets again.
    inventory=parent.read(inventory_path)
    audit=parent.entry(parent.local(remote_audit));invref=parent.entry(inventory_path)
    inputs=dict(bound['final_freeze']['inputs'],**bound['pins']);inputs.update({str((DEPLOY/n).relative_to(PROJECT)):parent.sha(DEPLOY/n) for n in FILES});inputs.update({r['path']:r['sha256'] for r in [audit,invref]})
    spec['freeze']=dict(bound['final_freeze'],inputs=inputs)
    spec['existing_storage_resume']=dict(schema=SCHEMA,old_out=OLD_REL,storage_run_id=OLD_ID,table=parent.entry(bound['table']),failure=parent.entry(bound['failure']),inventory=invref,operator_remote_audit=audit,worker_sha256=WORKER_SHA,partial_helper_sha256=PARTIAL_SHA,early_worker_sha256=EARLY_WORKER_SHA,early_LR_overlap=True,settling_seconds=2,startup_gpu=spec['gpus'][0],formal_gpu_count=2)
    validate_spec(spec)
    if parent.local(spec['out']).exists():raise FileExistsError('Fresh recovery output required')
    parent.write(target,spec);return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--make-spec',action='store_true');p.add_argument('--run-id');p.add_argument('--spec-out',type=Path);p.add_argument('--remote-audit',type=Path);p.add_argument('--inventory',type=Path);a=p.parse_args()
    if a.make_spec:
        if a.spec or not a.run_id or not a.spec_out or not a.remote_audit or not a.inventory:p.error('--make-spec requires --run-id/--spec-out/--remote-audit/--inventory only')
        s=make_spec(a.run_id,a.spec_out,a.remote_audit,a.inventory);print(json.dumps(dict(status='Long_existing_storage_recovery_spec_not_launched',run_id=s['run_id'])));return
    if not a.spec:p.error('--spec required')
    spec=parent.read(parent.local(a.spec));bound=validate_spec(spec)
    if a.validate_only:print(json.dumps(dict(status='actual_local_binding_checked_no_SSH_GPU',table=parent.entry(bound['table']),storage_run_id=OLD_ID,files=len(bound['inventory']['files']),gpus=spec['gpus'])));return
    LongStorageRecovery(spec).run()

if __name__=='__main__':main()
