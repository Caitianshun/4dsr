"""Observe the existing Desk DAG's exit, then return and summarize its outputs."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[2]
DEPLOY=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'deployment/complete_startup_eval_20261011'))
import complete_startup_eval as original
runtime,parent=original.runtime,original.parent
OLD_ID='deskgames_cube_eval3_resume_20261011t1010z'
OLD_OUT=parent.OUT/'orchestration'/OLD_ID
UNIT='highres-headroom-'+OLD_ID+'.service'
EXPECTED_INVOCATION='ec3f38a6d7094b7383fc9b6c1ee0b8dd'
FORMAL_START=PROJECT/'tmp/codex_highres_validation_20261010/resource_audit/Desk_formal_startup_actual_20261011.json'
PIPELINE='output/dynamic_sr_highres_20261011/pipelines/dgH_cube_seed20261007_'+OLD_ID
SERVICE_LOG='output/dynamic_sr_highres_20261011/pipelines/'+OLD_ID+'.service.log'
MUTABLE=(PIPELINE+'/logs/dgH_cube_seed20261007_teacher.log',PIPELINE+'/status.json',SERVICE_LOG)
FILES=('adopt_existing_dag.py','README.md')

OBSERVER=r'''
import hashlib,json,os,pathlib,select,subprocess,sys,time,traceback
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);out=root/p['out'];b=p['identity']
expected_invocation='ec3f38a6d7094b7383fc9b6c1ee0b8dd'
if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(root/'output/dynamic_sr_highres_20261011/observers'):raise FileExistsError('Fresh owned observer output required')
out.mkdir(parents=True)
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def show():
 text=subprocess.check_output(['systemctl','--user','show',b['unit'],'--no-pager','--property=InvocationID,MainPID,ActiveState,SubState,ExecMainCode,ExecMainStatus'],text=True)
 return dict(row.split('=',1) for row in text.splitlines() if '=' in row)
def proc(pid):
 d=pathlib.Path('/proc')/str(pid);raw=(d/'cmdline').read_bytes()
 return dict(start_ticks=int((d/'stat').read_text().rsplit(')',1)[1].split()[19]),uid=d.stat().st_uid,cwd=str((d/'cwd').resolve()),argv=[v.decode() for v in raw.split(bytes([0])) if v])
try:
 if b['InvocationID']!=expected_invocation:raise ValueError('Only the original accepted service invocation is allowed')
 for rel,digest in p['namespace_python'].items():
  if sha(root/rel)!=digest:raise ValueError('Original scientific source changed')
 before=show();write('before.json',dict(unit=before,identity=b,source=p['source'],observer_sha256=p['observer_sha256'],GPU_launches=0))
 if before['InvocationID']!=b['InvocationID']:raise ValueError('Different original service invocation')
 pid=int(before['MainPID'])
 if pid:
  if pid!=b['MainPID'] or not b['process_alive']:raise ValueError('Original MainPID changed')
  try:fd=os.pidfd_open(pid,0)
  except ProcessLookupError:fd=None
  if fd is not None:
   try:
    try:actual=proc(pid)
    except FileNotFoundError:actual=None
    if actual is not None:
     expected={k:b[k] for k in ['start_ticks','uid','cwd','argv']}
     if actual!=expected or actual['uid']!=os.getuid():raise ValueError('Owned original process identity changed')
     again=show()
     if again['InvocationID']!=b['InvocationID'] or int(again['MainPID']) not in (0,pid):raise ValueError('Original service changed while binding pidfd')
     write('bound_pidfd.json',dict(process=actual,unit=again,wait='select on pidfd; no model polling'))
    select.select([fd],[],[])
   finally:os.close(fd)
 after=show()
 if after['InvocationID']!=b['InvocationID'] or int(after['MainPID']) not in (0,b['MainPID']):raise ValueError('Different service invocation after native exit')
 pipeline=root/p['pipeline'];complete=pipeline/'complete.json';failures=sorted(pipeline.glob('*failure*.json'))
 status='completed' if complete.is_file() and json.loads(complete.read_text()).get('status')=='awaiting_review' else 'failed' if failures else 'unknown_no_terminal_DAG_receipt'
 receipt=dict(status='observed_original_PID_exit',DAG_status=status,original_unit=b['unit'],InvocationID=b['InvocationID'],unit_after=after,complete=dict(path=p['pipeline']+'/complete.json',sha256=sha(complete)) if complete.is_file() else None,failures=[dict(path=str(q.relative_to(root)),sha256=sha(q)) for q in failures],source=p['source'],observer_sha256=p['observer_sha256'],GPU_launches=0,finished_unix=time.time())
 write('complete.json',receipt)
except BaseException:
 write('failure.json',dict(status='observer_failed_no_dispatch',error=traceback.format_exc(),GPU_launches=0));raise
'''


def binding(config):
    spec=parent.read(OLD_OUT/'spec.json');original.validate_spec(spec)
    startup=parent.read(FORMAL_START);accepted=json.loads(startup['stdout'])
    if startup.get('host')!='cts' or startup.get('SSH_exit_code')!=0 or accepted.get('unit')!=UNIT or accepted['systemd'].get('InvocationID')!=EXPECTED_INVOCATION:raise ValueError('Original actual formal-startup receipt differs')
    if config.get('schema')!='adopt_existing_Desk_DAG_v1' or not re.fullmatch('[a-z0-9_-]{1,64}',config['observer_id']):raise ValueError('Owned fresh Desk observer config required')
    if config['original_spec']!=parent.entry(OLD_OUT/'spec.json'):raise ValueError('Original unchanged spec identity differs')
    receipt=parent.safe_local_member(config['identity_receipt']['path'])
    if parent.entry(receipt)!=config['identity_receipt']:raise ValueError('Actual identity receipt changed')
    b=parent.read(receipt)
    if b.get('schema')!='existing_owned_Desk_DAG_identity_v1' or b.get('target')!='cts' or b.get('remote_root')!=spec['remote_root'] or b.get('unit')!=UNIT or b.get('InvocationID')!=EXPECTED_INVOCATION:raise ValueError('Actual original unit identity must be confirmed before activation')
    if b.get('ssh_jump') not in (None,'a100-train','hoi-5090'):raise ValueError('Only an existing verified CTS route is allowed')
    if b['process_alive']:
        if b['MainPID']<=0 or b['start_ticks']<=0 or b['cwd']!=spec['remote_root'] or b['uid']<0:raise ValueError('Actual owned process identity incomplete')
        argv=b['argv']
        if argv[:5]!=[spec['remote_python'],'-u','-B','-c',original.WORKER] or len(argv)!=6:raise ValueError('MainPID is not the original frozen DAG worker')
        payload=json.loads(argv[5]);dispatch=parent.read(OLD_OUT/'events/015_remote_dispatch_attempted.json')['argv'];index=dispatch.index(spec['remote_root']+'/experiments/dynamic_sr_highres_20261011/persistent_runner.py')
        if payload.get('mode')!='runner' or payload.get('root')!=spec['remote_root'] or payload.get('argv')!=dispatch[index+1:] or payload.get('worker_sha256')!=original.WORKER_SHA:raise ValueError('Original task/output/worker identity differs')
    elif b['MainPID']!=0 or b.get('ActiveState') not in ('inactive','failed'):raise ValueError('Confirmed terminal original unit required')
    refs=[parent.entry(OLD_OUT/name) for name in ['spec.json','startup_acceptance.json','remote_result_inventory.json','failure_1791717557984765563.json','formal_return_failure_1791717712796026589.json']]+[parent.entry(FORMAL_START)]
    inputs=dict(spec['freeze']['inputs']);inputs.update({r['path']:r['sha256'] for r in [*refs,config['identity_receipt']]});inputs.update({str((DEPLOY/n).relative_to(PROJECT)):parent.sha(DEPLOY/n) for n in FILES})
    if config['freeze']!=dict(spec['freeze'],inputs=inputs):raise ValueError('Exact original plus observer/identity/failure freeze required')
    parent.check_freeze(config['freeze']);return spec,b


class ExistingDAGObserver(runtime.RuntimeEnvRecovery):
    def __init__(self,config,config_path):
        spec,self.identity=binding(config);super().__init__(spec);self.config=config
        self.out=parent.OUT/'orchestration'/config['observer_id'];self.pipeline_rel=PIPELINE
        self.freeze=dict(config['freeze'],inputs=dict(config['freeze']['inputs'],**{str(config_path.relative_to(PROJECT)):parent.sha(config_path)}))
        if self.identity['ssh_jump']:self.ssh=[*self.ssh[:-1],'-J',self.identity['ssh_jump'],self.ssh[-1]]

    def archive_snapshots(self):
        archived=[]
        for rel in MUTABLE:
            source=parent.safe_local_member(rel)
            if not source.exists():continue
            before=parent.entry(source);target=self.out/'local_snapshots'/rel
            if target.exists() or source.is_symlink() or not source.is_file():raise ValueError('Mutable snapshot ownership/type differs')
            parent.mkdir(target.parent);source.rename(target)
            if parent.sha(target)!=before['sha256']:raise ValueError('Archived bytes changed')
            archived.append(dict(original=before,archive=parent.entry(target)))
        parent.write(self.out/'local_snapshot_archive.json',dict(scope=list(MUTABLE),archived=archived,other_local_files_untouched=True))

    def run(self):
        if self.out.exists():raise FileExistsError('Fresh observer output required')
        parent.mkdir(self.out);parent.mkdir(self.out/'events');parent.mkdir(self.out/'logs')
        parent.write(self.out/'observer_config.json',self.config)
        with (parent.mkdir(parent.OUT/'orchestration/claims')/'dgH_cube_seed07.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                self.frozen();rel='output/dynamic_sr_highres_20261011/observers/'+self.config['observer_id']
                import hashlib
                payload=dict(root=self.spec['remote_root'],out=rel,identity=self.identity,pipeline=PIPELINE,namespace_python=self.freeze['namespace_python'],source=parent.entry(__file__),observer_sha256=hashlib.sha256(OBSERVER.encode()).hexdigest())
                cmd=['/usr/bin/python3','-u','-B','-c',OBSERVER,json.dumps(payload)];paths=[rel]
                if self.identity['process_alive']:
                    log='output/dynamic_sr_highres_20261011/pipelines/'+self.config['observer_id']+'.observer.service.log';paths.append(log)
                    cmd=['systemd-run','--user','--wait','--unit=highres-observe-'+self.config['observer_id'],'--property=WorkingDirectory='+self.spec['remote_root'],'--property=StandardOutput=append:'+self.spec['remote_root']+'/'+log,'--property=StandardError=append:'+self.spec['remote_root']+'/'+log,'--setenv=CUDA_VISIBLE_DEVICES=','--setenv=PYTHONDONTWRITEBYTECODE=1',*cmd]
                self.record('observe_existing_only',argv=cmd,new_GPU_launches=0,original_spec=self.config['original_spec'])
                # This CPU observer command stays outside runtime_environment,
                # whose frozen service whitelist permits only GPU worker names.
                result=subprocess.run([*self.ssh,shlex.join(cmd)],text=True,capture_output=True)
                self.record('observer_native_return',exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
                with self.runtime_environment():self.pull_paths(paths,'observer')
                if result.returncode:raise RuntimeError('Observer/SSH failed; original service untouched, no retry')
                terminal=parent.read(parent.local(rel)/'complete.json')
                if terminal['status']!='observed_original_PID_exit' or terminal['InvocationID']!=self.identity['InvocationID']:raise ValueError('Original native-exit receipt missing')
                self.archive_snapshots()
                with (self.out/'startup_acceptance.json').open('xb') as dest,(OLD_OUT/'startup_acceptance.json').open('rb') as src:shutil.copyfileobj(src,dest)
                with self.runtime_environment():
                    self.pull()
                    if terminal['DAG_status']=='completed':self.summarize()
                    else:parent.write(self.out/'original_terminal_failure.json',dict(status='returned_original_failure_or_unknown_no_training',terminal=terminal,new_GPU_launches=0))
            except BaseException:
                parent.write(self.out/f'observer_failure_{time.time_ns()}.json',dict(status='observer_failed_original_job_untouched',error=traceback.format_exc(),new_GPU_launches=0,automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    path=parent.local(a.config);config=parent.read(path);binding(config)
    if a.validate_only:print(json.dumps(dict(status='local_existing_DAG_binding_checked_no_SSH_GPU',original_unit=UNIT)));return
    ExistingDAGObserver(config,path).run()


if __name__=='__main__':main()
