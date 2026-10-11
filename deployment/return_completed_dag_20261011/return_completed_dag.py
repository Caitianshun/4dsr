"""Return the completed Desk DAG after its local SSH waiter failed."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'deployment/resume_existing_completed_dag_20261011'))
import resume_completed_dag as original
parent = original.parent
RUN = 'deskgames_completed_dag_resume_20261011t1243z'
OLD = parent.OUT / 'orchestration' / RUN
ACCEPTED = parent.OUT / 'orchestration/deskgames_terminal_return_20261011t1234z'
UNIT = 'highres-headroom-' + RUN + '.service'
INVOCATION = '362a7b13233349c2b94c797f7d4386f2'
PIPELINE = 'output/dynamic_sr_highres_20261011/pipelines/dgH_cube_seed20261007_' + RUN
PREFIX = 'output/dynamic_sr_highres_20261011/runs/dgH_cube_seed20261007/'
MUTABLE = (PREFIX + 'HRref/training.jsonl', PIPELINE + '/logs/dgH_cube_seed20261007_HRref.log',
           PIPELINE + '/status.json', 'output/dynamic_sr_highres_20261011/pipelines/' + RUN + '.service.log')
SCHEMA = 'return_completed_original_Desk_DAG_v1'
FILES = ('return_completed_dag.py', 'README.md')
ACTUAL_SHA = 'b5e2e845a67d4e96cb1b263a76d18ccd665eecb918ee7634ee60e2eee8ab189a'
START = PROJECT / 'tmp/codex_highres_validation_20261010/resource_audit/desk_completed_DAG_resume_actual_20261011T124813Z.json'

# CPU only: check the pinned final receipts, then collect one stable inventory.
TERMINAL = r'''
import hashlib,json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin);root=pathlib.Path(p['root']);os.chdir(root)
def safe(rel):
 q=root/rel
 if pathlib.PurePosixPath(rel).is_absolute() or '..' in pathlib.PurePosixPath(rel).parts:raise ValueError('Unsafe result path')
 for a in [q,*q.parents]:
  if a==root:break
  if a.is_symlink():raise ValueError('Symlink result protected')
 if not q.resolve().is_relative_to(root):raise ValueError('Result outside original workspace')
 return q
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bound(ref):
 q=safe(ref['path'])
 if q.stat().st_size!=ref['bytes'] or sha(q)!=ref['sha256']:raise ValueError('Terminal receipt changed: '+ref['path'])
 return json.loads(q.read_text())
if os.uname().nodename!='cts' or os.getuid()!=1000:raise ValueError('Original CTS host/UID differs')
s=subprocess.check_output(['systemctl','--user','show',p['unit'],'--no-pager','--property=LoadState,MainPID,InvocationID,ActiveState,Result,ExecMainStatus'],text=True)
unit=dict(x.split('=',1) for x in s.splitlines() if '=' in x)
if int(unit['MainPID'] or 0)!=0:raise ValueError('Original service is still active; terminal-only entry does not wait')
unloaded=unit.get('LoadState')=='not-found' and unit.get('InvocationID')=='' and unit.get('ActiveState')=='inactive'
retained=unit.get('InvocationID')==p['InvocationID'] and unit.get('ActiveState')=='inactive' and unit.get('Result')=='success' and unit.get('ExecMainStatus')=='0'
if not (unloaded or retained):raise ValueError('Original service terminal identity differs')
identity=bound(p['identity']);complete=bound(p['complete']);bound(p['table'])
if identity['pid']!=860311 or identity['host']!='cts' or identity['tasks']!={k:p['table'][k] for k in ('path','sha256')} or complete['identity']!={k:p['identity'][k] for k in ('path','sha256')} or complete['status']!='awaiting_review':raise ValueError('Original completed DAG identity differs')
for rel,digest in p['sources'].items():
 if sha(safe(rel))!=digest:raise ValueError('Frozen scientific source changed')
if sorted(str(q.relative_to(root)) for q in (root/'experiments/dynamic_sr_highres_20261011').glob('*.py'))!=sorted(p['namespace']):raise ValueError('Scientific namespace inventory differs')
files={};reused=[]
for rel in p['paths']:
 q=safe(rel)
 if not q.exists():continue
 for f in ([q] if q.is_file() else sorted(q.rglob('*'))):
  if f.is_symlink():raise ValueError('Symlink result protected')
  if not f.is_file() or '__pycache__' in f.parts:continue
  member=str(f.relative_to(root));safe(member);st=f.stat();old=p['known'].get(member)
  if old is not None:
   if st.st_size!=old['ref']['bytes'] or st.st_mtime_ns!=old['stat']['mtime_ns']:raise ValueError('Previously accepted immutable remote file changed: '+member)
   digest=old['ref']['sha256'];reused.append(member)
  else:
   digest=sha(f);after=f.stat()
   if (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Completed result changed while collecting: '+member)
  files[member]=dict(bytes=st.st_size,sha256=digest)
bound(p['identity']);bound(p['complete'])
print(json.dumps(dict(status='completed_original_DAG_terminal_inventory',unit=unit,original_InvocationID=p['InvocationID'],identity=p['identity'],complete=p['complete'],known_SHA_reused_by_size_mtime=reused,GPU_launches=0,inventory=dict(files=files,host='cts',captured_unix=time.time()))))
'''
TERMINAL_SHA = hashlib.sha256(TERMINAL.encode()).hexdigest()


def stat(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def facts(receipt):
    d = parent.read(receipt)
    if parent.sha(receipt) != ACTUAL_SHA or d.get('schema') != 'Desk_original_DAG_after_local_SSH_failure_readonly_v1' or d.get('target') != 'cts' or d.get('ssh_jump') != 'a100-train' or d['result']['returncode'] != 0:
        raise ValueError('Bind the actual successful final Desk observation')
    obs = {v['kind']: v['value'] for v in d['observations']}
    if obs['host'] != dict(hostname='cts', uid=1000) or obs['unit']['properties']['MainPID'] != '0' or obs['pipeline_status']['summary']['counts'] != dict(already_complete=5, completed=23):
        raise ValueError('Actual complete 28-task terminal evidence required')
    identity = obs['pipeline_identity']; complete = obs['terminal'][0]
    refs = {}
    for name, v in [('identity', identity), ('complete', complete), ('table', obs['source_table']['table'])]:
        refs[name] = dict(path=str(Path(v['path']).relative_to(Path(d['payload']['root']))), bytes=v['bytes'], sha256=v['sha256'])
    return d, obs, refs


def binding(config):
    old = parent.read(OLD / 'spec.json'); original.validate_spec(old)
    receipt = parent.safe_local_member(config['terminal_receipt']['path'])
    if config['terminal_receipt'] != parent.entry(receipt):raise ValueError('Actual terminal receipt changed')
    actual, obs, refs = facts(receipt)
    table = old['completed_DAG_resume']['table']
    if refs['table'] != dict(table, bytes=parent.local(table['path']).stat().st_size) or actual['payload']['unit'] != UNIT or actual['payload']['pipeline'] != PIPELINE or actual['payload']['expected_InvocationID'] != INVOCATION or actual['payload']['expected_pid'] != 860311 or actual['payload']['expected_start_ticks'] != 19169267:
        raise ValueError('Original unit/table/process binding differs')
    i = obs['pipeline_identity']['content']; c = obs['terminal'][0]['content']
    if i['pid'] != 860311 or i['host'] != 'cts' or i['tasks'] != table or i['gpus'][0]['uuid'] != old['gpus'][0]['uuid'] or c['identity'] != {k:refs['identity'][k] for k in ('path','sha256')} or c['status'] != 'awaiting_review' or set(c['tasks']) != {t['name'] for t in parent.read(parent.local(table['path']))['tasks']}:
        raise ValueError('Completed original DAG does not match the exact 28-task table')
    if any(t['status'] not in ('completed','already_complete') or t.get('result',{}).get('exit_code',0) != 0 for t in c['tasks'].values()):raise ValueError('Original DAG has an incomplete task')
    start = parent.read(START); so = {v['kind']:v['value'] for v in start['observations']}
    if so['unit']['properties']['InvocationID'] != INVOCATION or so['process']['pid'] != 860311 or so['process']['start_ticks'] != 19169267 or so['process']['cwd'] != old['remote_root'] or so['process']['uid'] != [1000]*4:raise ValueError('Actual original startup/process identity differs')
    dispatch = parent.read(OLD / 'events/008_remote_dispatch_attempted.json')
    if dispatch['unit'] != UNIT or '--out' not in dispatch['argv'] or dispatch['argv'][dispatch['argv'].index('--out')+1] != PIPELINE:raise ValueError('Original formal dispatch/output differs')
    accepted = parent.read(ACCEPTED / 'remote_result_inventory.json'); index = parent.read(ACCEPTED / 'checkpoint_index.json')
    event = parent.read(ACCEPTED / 'events/008_return_verified.json')
    if len(accepted['files']) != 43 or event['files'] != 43 or event['checkpoints'] != 2 or index['status'] != 'returned_bytes_SHA_verified' or index['inventory'] != parent.entry(ACCEPTED / 'remote_result_inventory.json'):raise ValueError('Original accepted43-file return identity differs')
    if config['schema'] != SCHEMA or not re.fullmatch('[a-z0-9_-]{1,64}',config['return_id']) or config['return_id'] == RUN or config['original_spec'] != parent.entry(OLD / 'spec.json') or config['terminal_refs'] != refs or set(config['known']) != set(accepted['files']):raise ValueError('Exact fresh terminal-return config required')
    accepted_time = (ACCEPTED / 'checkpoint_index.json').stat().st_ctime_ns
    for rel, cached in config['known'].items():
        q = parent.safe_local_member(rel)
        if q.is_symlink() or cached['ref'] != accepted['files'][rel] or stat(q) != cached['stat'] or cached['stat']['bytes'] != cached['ref']['bytes'] or max(cached['stat']['ctime_ns'],cached['stat']['mtime_ns']) > accepted_time:
            raise ValueError('Previously accepted local asset changed; do not substitute a fresh hash: ' + rel)
    paths = [OLD / n for n in ['spec.json','startup_acceptance.json','remote_result_inventory.json','failure_1791739598855407227.json','formal_return_failure_1791739613991787197.json','events/008_remote_dispatch_attempted.json','events/010_remote_exit_or_SSH_returned.json','events/012_remote_inventory.json','events/014_remote_inventory.json']]
    paths += [receipt, START, ACCEPTED/'remote_result_inventory.json', ACCEPTED/'checkpoint_index.json', ACCEPTED/'events/008_return_verified.json', *[DEPLOY/n for n in FILES]]
    inputs = dict(old['freeze']['inputs']); inputs.update({str(p.relative_to(PROJECT)):parent.sha(p) for p in paths})
    if config['freeze'] != dict(old['freeze'],inputs=inputs):raise ValueError('Original science and terminal-return provenance freeze differs')
    parent.check_freeze(config['freeze'])
    return old, refs


class CompletedReturn(original.CompletedDAGRecovery):
    def __init__(self, config, path):
        spec, self.terminal_refs = binding(config); super().__init__(spec)
        self.config = config; self.out = parent.OUT/'orchestration'/config['return_id']; self.pipeline_rel = PIPELINE
        self.freeze = dict(config['freeze'],inputs=dict(config['freeze']['inputs'],**{str(path.relative_to(PROJECT)):parent.sha(path)}))
        self.stable_inventory = None

    def runner(self, *args):raise RuntimeError('Terminal-only return never dispatches a DAG')
    def startup(self, *args):raise RuntimeError('Terminal-only return never repeats startup')

    def remote(self, mode, value):
        if mode != 'inventory' or value != dict(paths=self.result_paths()) or self.stable_inventory is None:raise RuntimeError('Only the once-collected completed result inventory is available')
        self.record('reuse_stable_terminal_inventory',capture='one terminal collection',files=len(self.stable_inventory['files']))
        return self.stable_inventory

    def terminal_inventory(self):
        payload = dict(root=self.spec['remote_root'],unit=UNIT,InvocationID=INVOCATION,**self.terminal_refs,
                       paths=self.result_paths(),known=self.config['known'],sources=self.freeze['namespace_python'],namespace=list(self.freeze['namespace_python']))
        command = ['/usr/bin/python3','-u','-B','-c',TERMINAL]
        result = subprocess.run([*self.ssh,shlex.join(command)],input=json.dumps(payload),text=True,capture_output=True)
        self.record('terminal_CPU_collection_returned',exit_code=result.returncode,stderr=result.stderr,worker_sha256=TERMINAL_SHA,new_GPU_launches=0)
        if result.returncode:raise RuntimeError('Terminal CPU collection/SSH failed; no automatic retry')
        receipt = json.loads(result.stdout)
        if receipt['status'] != 'completed_original_DAG_terminal_inventory' or receipt['GPU_launches'] != 0 or receipt['complete'] != self.terminal_refs['complete']:raise ValueError('Stable terminal collection receipt differs')
        parent.write(self.out/'terminal_cpu_receipt.json',receipt); self.stable_inventory = receipt['inventory']

    def archive_snapshots(self):
        previous = parent.read(OLD/'remote_result_inventory.json')['files']; archived = []
        for rel in MUTABLE:
            q = parent.safe_local_member(rel)
            if not q.exists():continue
            if q.is_symlink() or rel not in previous or rel not in self.stable_inventory['files']:raise ValueError('Mutable snapshot scope/source differs')
            current = dict(bytes=q.stat().st_size,sha256=parent.sha(q)); stable = self.stable_inventory['files'][rel]
            if current == stable:continue
            if current != previous[rel]:raise ValueError('Differing local snapshot does not match the preserved17:13 inventory: '+rel)
            dest = self.out/'local_snapshots'/rel
            if dest.exists() or dest.is_symlink():raise FileExistsError('Preserve existing private snapshot archive')
            parent.mkdir(dest.parent); q.rename(dest)
            archived.append(dict(original_path=rel,original_inventory=previous[rel],local_SHA=current['sha256'],stable_remote=stable,archive=parent.entry(dest)))
        parent.write(self.out/'local_snapshot_archive.json',dict(scope=list(MUTABLE),archived=archived,unchanged_files_preserved=True))

    @contextlib.contextmanager
    def accepted_sha_cache(self):
        actual = parent.sha; cache = dict(self.config['known'])
        def sha(path):
            q = Path(path); rel = str(q.relative_to(PROJECT)) if q.is_absolute() and q.is_relative_to(PROJECT) else None
            if rel in cache:
                v = cache[rel]
                if q.is_symlink() or stat(q) != v['stat']:raise ValueError('Fixed accepted/stat-cached asset changed: '+rel)
                return v['ref']['sha256']
            before = stat(q); digest = actual(q)
            if stat(q) != before:raise ValueError('File changed while hashing: '+str(q))
            if rel is not None:cache[rel] = dict(ref=dict(bytes=before['bytes'],sha256=digest),stat=before)
            return digest
        parent.sha = sha
        try:yield
        finally:parent.sha = actual

    def run(self):
        if self.out.exists():raise FileExistsError('Fresh terminal-return output required')
        parent.mkdir(self.out);parent.mkdir(self.out/'events');parent.mkdir(self.out/'logs')
        parent.write(self.out/'return_config.json',self.config)
        with (parent.mkdir(parent.OUT/'orchestration/claims')/'dgH_cube_seed07_DAG_resume.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                self.frozen();self.terminal_inventory();self.archive_snapshots()
                with (self.out/'startup_acceptance.json').open('xb') as dst,(OLD/'startup_acceptance.json').open('rb') as src:shutil.copyfileobj(src,dst)
                with self.accepted_sha_cache(),self.runtime_environment():
                    self.pull();self.summarize()
            except BaseException:
                parent.write(self.out/f'return_failure_{time.time_ns()}.json',dict(status='completed_DAG_return_failed_preserved',stage=self.stage,error=traceback.format_exc(),new_GPU_launches=0,automatic_retry=False));raise


def make_config(return_id, receipt):
    old = parent.read(OLD/'spec.json'); _,_,refs = facts(receipt)
    accepted = parent.read(ACCEPTED/'remote_result_inventory.json')
    known = {rel:dict(ref=ref,stat=stat(parent.safe_local_member(rel))) for rel,ref in accepted['files'].items()}
    paths = [OLD/n for n in ['spec.json','startup_acceptance.json','remote_result_inventory.json','failure_1791739598855407227.json','formal_return_failure_1791739613991787197.json','events/008_remote_dispatch_attempted.json','events/010_remote_exit_or_SSH_returned.json','events/012_remote_inventory.json','events/014_remote_inventory.json']]
    paths += [receipt,START,ACCEPTED/'remote_result_inventory.json',ACCEPTED/'checkpoint_index.json',ACCEPTED/'events/008_return_verified.json',*[DEPLOY/n for n in FILES]]
    inputs = dict(old['freeze']['inputs']);inputs.update({str(p.relative_to(PROJECT)):parent.sha(p) for p in paths})
    config = dict(schema=SCHEMA,return_id=return_id,original_spec=parent.entry(OLD/'spec.json'),terminal_receipt=parent.entry(receipt),terminal_refs=refs,known=known,freeze=dict(old['freeze'],inputs=inputs))
    binding(config);return config


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--make-config',action='store_true');p.add_argument('--return-id');p.add_argument('--terminal-receipt',type=Path);p.add_argument('--config-out',type=Path);a=p.parse_args()
    if a.make_config:
        if a.config or not all((a.return_id,a.terminal_receipt,a.config_out)):p.error('--make-config requires return-id/terminal-receipt/config-out')
        target=parent.local(a.config_out)
        if target.exists() or (parent.OUT/'orchestration'/a.return_id).exists():raise FileExistsError('Fresh config/output required')
        config=make_config(a.return_id,parent.local(a.terminal_receipt));parent.write(target,config);print(json.dumps(dict(status='terminal_return_config_created_no_SSH_GPU',config=parent.entry(target))));return
    if not a.config:p.error('--config required')
    path=parent.local(a.config);config=parent.read(path);binding(config)
    if a.validate_only:print(json.dumps(dict(status='actual_terminal_return_binding_checked_no_SSH_GPU',original_unit=UNIT,new_GPU_launches=0)));return
    CompletedReturn(config,path).run()


if __name__=='__main__':main()
