"""One-shot local acceptance coordinator. Plan is CPU-only; run is controller-only.

Run in a persistent service with Restart=no. No replay, transfer retry, policy
change, GPU selection, or formal training is implemented. Remote GPU admission
and artifact return remain owned by the frozen submit_remote.py entry point.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REMOTE='/home/cts/Project/4DSR/tasks/static_sr_wu_20261022'
BASE='/home/cts/Project/4DSR/.venv/bin/python'
SCHEMA='static_sr_cts_acceptance_continuation_plan_v1'
TEACHER='experiments/static_sr_wu_20261022/teacher_acceptance.py'
SUBMIT='deployment/static_sr_cts_20261022/submit_remote.py'
BOOTSTRAP='deployment/static_sr_cts_20261022/bootstrap_remote.py'
CPU_STATUS='output/static_sr_wu_20261022/cts_full_suite_cpu_v2/status.json'
BOOTSTRAP_OUT='output/static_sr_wu_20261022/runtime/cts_bootstrap'
TEMPLATE='output/static_sr_wu_20261022/splatsure_stablesr_assets/runtime_assets_v2.json'


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def resolve(path):
    path=Path(path);return path if path.is_absolute() else ROOT/path
def member(path):
    value=str(path).replace('{ROOT}/','').removeprefix(REMOTE+'/')
    p=Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p)=='.':raise ValueError('Unsafe snapshot member')
    return str(p)
def identity(path):
    path=resolve(path).absolute()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=sha(path))
def checked(item):
    path=resolve(item['path'])
    if sha(path)!=item['sha256']:raise ValueError('Frozen bytes changed: '+str(path))
    return path
def read(path):return json.loads(resolve(path).read_text())
def return_destination(path):
    source=Path(__file__).with_name('submit_remote.py')
    spec=importlib.util.spec_from_file_location('_continuation_submit_helpers',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT
    return module.checked_return_destination(path,'file')
def write(path,value,fresh=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if fresh:
        with path.open('x') as stream:stream.write(data)
    else:
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(data);temporary.replace(path)
def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def gpu_uuid(value):
    # PyTorch reports the UUID without nvidia-smi's optional GPU- prefix.
    if not isinstance(value,str) or not value:raise ValueError('Missing GPU UUID')
    return value.lower().removeprefix('gpu-')
def process_ticks(pid):
    try:text=Path(f'/proc/{pid}/stat').read_text()
    except (FileNotFoundError,ProcessLookupError):return None
    return int(text.rsplit(')',1)[1].split()[19])
def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid,0)
    # The project's Python 3.10 build omits this os wrapper. Use the same
    # kernel interface exposed by libc; never substitute process polling.
    libc=ctypes.CDLL(None,use_errno=True)
    function=libc.pidfd_open;function.argtypes=[ctypes.c_int,ctypes.c_uint];function.restype=ctypes.c_int
    result=function(pid,0)
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return result


def job_spec(path,methods):
    job=read(path);expect=[];pairs=set();inputs=[]
    if (job.get('method')!='real_teacher_GPU_acceptance_only' or not job.get('training_only_inputs')
            or not job.get('no_proposed_modules')):raise ValueError('Only the fixed teacher acceptance jobs are permitted')
    for command in job['commands']:
        argv=command['argv']
        if len(argv)<4 or argv[1]!='-u' or member(argv[2])!=TEACHER:
            raise ValueError('Coordinator cannot execute a non-teacher command')
        arguments=argv[3:]
        if len(arguments)%2 or '--plan-only' in arguments:raise ValueError('Real teacher fixture arguments required')
        flags={arguments[i]:arguments[i+1] for i in range(0,len(arguments),2)}
        if len(flags)*2!=len(arguments) or any(not k.startswith('--') for k in flags):raise ValueError('Duplicate/malformed fixture flags')
        method=flags['--method'];records=member(flags['--records']);scene=Path(records).parent.name
        if method not in methods or scene not in ('Cook07','Discussion07') or flags.get('--frame')!='0':
            raise ValueError('Use exactly the two registered domains and frame zero')
        if (method,scene) in pairs:raise ValueError('Duplicate acceptance fixture')
        pairs.add((method,scene));out=member(flags['--out'])
        if not out.startswith('output/static_sr_wu_20261022/cts_acceptance/') or out not in job['return_paths']:
            raise ValueError('Every fixture output must be returned')
        config_path='output/static_sr_wu_20261022/prepared_final/'+scene+'/splatsure.json'
        config=read(config_path);manifest=read(checked(config['manifest']))
        inputs.extend([identity(config_path),dict(config['manifest'])])
        expect.append(dict(method=method,scene=scene,records=records,path=out+'/acceptance.json',
                           hr_wh=manifest['resolutions']['hr']))
    if pairs!={(m,s) for m in methods for s in ('Cook07','Discussion07')}:
        raise ValueError('Incomplete two-domain acceptance job')
    frozen=[identity(path),*inputs]
    for item in job['required_files']:checked(item);frozen.append(dict(item))
    if TEACHER not in {x['path'] for x in frozen}:raise ValueError('Teacher wrapper must be frozen in the job')
    return dict(job=identity(path),job_id=job['job_id'],gpu_uuid=job['gpu_uuid'],out=member(job['out']),
                receipts=expect,frozen_files=frozen)


def build_plan(args):
    sequence=job_spec(args.sequence_job,['sequence_matters'])
    diffusion=job_spec(args.diffusion_job,['stablesr','threedsr'])
    if sequence['gpu_uuid']!=diffusion['gpu_uuid']:raise ValueError('Acceptance jobs must use the same CTS GPU')
    if sequence['job_id']==diffusion['job_id']:raise ValueError('Acceptance stages require distinct one-shot job identities')
    transfer_dir=member(args.transfer_dir);intent=read(transfer_dir+'/intent.json')
    if intent.get('root')!=REMOTE:raise ValueError('Transfer intent targets a different snapshot')
    inventory=identity(intent['inventory'])
    if inventory['sha256']!=intent['inventory_sha256']:raise ValueError('Transfer inventory identity changed')
    ticks=process_ticks(args.transfer_pid)
    if ticks is not None and ticks!=args.transfer_start_ticks:raise ValueError('Transfer PID was reused; refuse continuation')
    if ticks is None and not resolve(transfer_dir+'/verified.json').is_file():
        raise ValueError('Transfer PID is absent and verified receipt is missing')
    frozen=[identity(__file__),identity(SUBMIT),identity(BOOTSTRAP),identity(TEMPLATE),
            identity(transfer_dir+'/intent.json'),inventory]
    frozen+=sequence['frozen_files']+diffusion['frozen_files']
    if getattr(args,'reuse_sequence_completion',None):
        reuse=sequence_completion(args.reuse_sequence_completion,sequence)
        sequence['reuse_completion']=reuse
        frozen+=reuse['evidence_files']
    unique={x['path']:x for x in frozen}
    return dict(schema=SCHEMA,created_utc=utc(),host='cts',remote_root=REMOTE,
        cpu_status=CPU_STATUS,event_timeout_seconds=args.event_timeout_seconds,
        transfer=dict(pid=args.transfer_pid,start_ticks=args.transfer_start_ticks,boot_id=boot_id(),
            intent=identity(transfer_dir+'/intent.json'),inventory=inventory,
            verified=transfer_dir+'/verified.json',expected_files=intent['files']),
        sequence=dict(sequence,memory_mib=18000,timeout_seconds=7200),
        diffusion=dict(diffusion,memory_mib=18000,timeout_seconds=14400),
        bootstrap=dict(script=identity(BOOTSTRAP),template=identity(TEMPLATE),out=BOOTSTRAP_OUT,
            base_python=BASE,timeout_seconds=3600),
        local_python=str(Path(sys.executable).absolute()),frozen_files=list(unique.values()),
        no_replay=True,formal_training_allowed=False,GPU_calls_during_planning=0,remote_calls_during_planning=0)


def verified_transfer(plan):
    item=plan['transfer'];value=read(item['verified'])
    if (value.get('status')!='verified' or value.get('inventory_sha256')!=item['inventory']['sha256']
            or value.get('files')!=item['expected_files'] or value.get('missing')!=[]):
        raise ValueError('Transfer verification missing, failed, or bound to another inventory')
    return identity(item['verified'])


def wait_transfer(plan):
    """pidfd selects one exact process lifetime; no polling or retrying transfer."""
    item=plan['transfer'];pid=item['pid'];expected=item['start_ticks']
    if boot_id()!=item['boot_id']:raise ValueError('Machine rebooted; transfer process identity no longer valid')
    current=process_ticks(pid)
    if current is not None and current!=expected:raise ValueError('Transfer PID was reused')
    if current is None:return dict(process='already_exited',verified=verified_transfer(plan))
    try:descriptor=pidfd_open(pid)
    except ProcessLookupError:return dict(process='exited_before_pidfd_open',verified=verified_transfer(plan))
    try:
        current=process_ticks(pid)
        if current is not None and current!=expected:raise ValueError('Transfer PID was reused during pidfd acquisition')
        if not select.select([descriptor],[],[],plan['event_timeout_seconds'])[0]:
            raise TimeoutError('Transfer process exit event timed out; no replay')
        current=process_ticks(pid)
        if current is not None and current!=expected:raise ValueError('Transfer PID was reused after exit')
    finally:os.close(descriptor)
    return dict(process='pidfd_exit_observed',verified=verified_transfer(plan))


# The same implementation is executable in local CPU tests and remotely over
# stdin. Read before watch and re-read afterward close the check/watch race.
WATCH_CODE=r'''
import ctypes,errno,json,os,select,struct,sys,time
from pathlib import Path
request=json.loads(sys.stdin.readline());p=Path(request['path']);limit=time.monotonic()+request['timeout']
def terminal():
 try:value=json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
 return value if value.get('status') in ('completed','failed') else None
libc=ctypes.CDLL('libc.so.6',use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno())
watched=None;wd=None
try:
 while True:
  value=terminal()
  if value is not None:print(json.dumps(value),flush=True);break
  parent=p.parent
  while not parent.exists():parent=parent.parent
  if parent!=watched:
   if wd is not None:libc.inotify_rm_watch(fd,wd)
   wd=libc.inotify_add_watch(fd,os.fsencode(parent),0x100|0x80|0x8|0x4|0x400|0x800)
   if wd<0:
    error=ctypes.get_errno()
    if error==errno.ENOENT:continue
    raise OSError(error,os.strerror(error),str(parent))
   watched=parent
  # A deeper parent may have appeared between finding parent and installing
  # its watch. That creation event happened too early to be observed here.
  nearest=p.parent
  while not nearest.exists():nearest=nearest.parent
  if nearest!=watched:continue
  value=terminal()
  if value is not None:print(json.dumps(value),flush=True);break
  remaining=limit-time.monotonic()
  if remaining<=0:raise TimeoutError('CPU suite terminal event wait expired')
  if select.select([fd],[],[],remaining)[0]:
   data=os.read(fd,65536);offset=0
   while offset<len(data):
    event_wd,mask,cookie,length=struct.unpack_from('iIII',data,offset);offset+=16+length
    if event_wd==wd and mask&(0x8000|0x400|0x800):
     if not mask&0x8000:libc.inotify_rm_watch(fd,wd)
     watched=None;wd=None
finally:os.close(fd)
'''

VERIFY_CODE=r'''
import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);result=[]
for item in request['files']:
 path=root/item['path']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 if h.hexdigest()!=item['sha256']:raise ValueError('Remote frozen identity differs: '+str(path))
 result.append(item)
print(json.dumps(dict(status='verified',files=result)))
'''

SMALL_RECORDS_CODE=r'''
import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);base=root/request['path'];files=[]
for parent in [base,*base.parents]:
 if parent==root:break
 if parent.is_symlink():raise ValueError('Symlink bootstrap return path')
if base.exists():
 for p in sorted(base.rglob('*')):
  if p.is_symlink():raise ValueError('Symlink in bootstrap records')
  if not p.is_file():continue
  if p.suffix not in ('.json','.log') or p.stat().st_size>16*(1<<20):raise ValueError('Unexpected large bootstrap artifact')
  files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
if sum(x['bytes'] for x in files)>64*(1<<20):raise ValueError('Bootstrap records exceed 64 MiB')
print(json.dumps(dict(files=files)))
'''


def ssh_argv():
    return ['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','cts']


class Operations:
    def __init__(self,plan,out):self.plan,self.out=plan,out
    def query(self,code,request,timeout):
        result=subprocess.run(ssh_argv()+[shlex.join(['/usr/bin/python3','-c',code])],
            input=json.dumps(request)+'\n',text=True,capture_output=True,timeout=timeout)
        if result.returncode:raise RuntimeError('CPU SSH command failed: '+result.stderr[-8000:])
        return json.loads(result.stdout)
    def cpu_suite(self):
        result=self.query(WATCH_CODE,dict(path=REMOTE+'/'+self.plan['cpu_status'],timeout=self.plan['event_timeout_seconds']),
                          self.plan['event_timeout_seconds']+120)
        write(self.out/'cpu_suite_terminal.json',result,fresh=True)
        if (result.get('status')!='completed' or result.get('GPU_calls')!=0
                or [(s.get('name'),s.get('returncode')) for s in result.get('stages',[])]!=[('register',0),('plan',0)]):
            raise RuntimeError('CPU full-suite did not complete successfully')
        return dict(receipt=identity(self.out/'cpu_suite_terminal.json'))
    def job(self,name):
        item=self.plan[name]
        if item.get('reuse_completion'):
            if name!='sequence':raise ValueError('Only explicit Sequence completion reuse is permitted')
            expected=item['reuse_completion']
            for source in expected['evidence_files']:checked(source)
            observed=sequence_completion(expected['prior_launchplan']['path'],item)
            if observed!=expected:raise ValueError('Sequence reuse evidence changed after planning')
            return dict(job_id=item['job_id'],reused_completion=True,GPU_submissions=0,**observed)
        for source in item['frozen_files']:checked(source)
        self.query(VERIFY_CODE,dict(root=REMOTE,files=item['frozen_files']),300)
        directory=HERE/'dispatch'/item['job_id']
        if directory.exists():raise FileExistsError('Job already has a dispatch directory; never replay: '+str(directory))
        command=[self.plan['local_python'],str(ROOT/SUBMIT),'--host','cts','--gpu-index','0',
                 '--job',str(checked(item['job'])),'--memory-mib','18000','--timeout-seconds',str(item['timeout_seconds'])]
        with (self.out/(name+'_submit.log')).open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=item['timeout_seconds']+18000)
        if result.returncode:raise RuntimeError(name+' acceptance submission/wait/return failed; no replay')
        terminal=read(directory/'remote_terminal.json')
        if terminal.get('outcome') not in ('success','completed'):raise RuntimeError(name+' remote job did not succeed')
        complete=read(item['out']+'/complete.json')
        if complete.get('status')!='completed' or complete.get('job_id')!=item['job_id']:
            raise ValueError(name+' returned completion is missing or foreign')
        receipts=[]
        for expected in item['receipts']:
            value=read(expected['path']);validate_teacher_receipt(value,expected,item,self.plan['frozen_files'])
            receipts.append(identity(expected['path']))
        return dict(job_id=item['job_id'],terminal=identity(directory/'remote_terminal.json'),
                    completion=identity(item['out']+'/complete.json'),receipts=receipts)
    def transfer(self):return wait_transfer(self.plan)
    def bootstrap(self):
        item=self.plan['bootstrap']
        self.query(VERIFY_CODE,dict(root=REMOTE,files=[item['script'],item['template']]),300)
        command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',BASE,'-u',REMOTE+'/'+item['script']['path'],
                 'apply','--base-python',BASE,'--runtime-assets-template',REMOTE+'/'+item['template']['path'],
                 '--out',REMOTE+'/'+item['out']]
        result=None;error=None
        try:
            with (self.out/'bootstrap.log').open('x') as stream:
                result=subprocess.run(ssh_argv()+[shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,
                                      timeout=item['timeout_seconds'])
        except BaseException as exc:error=exc
        # Return even failure records immediately. Only small JSON/log files
        # are listed and copied, then checked against remote SHA256.
        listing=self.query(SMALL_RECORDS_CODE,dict(root=REMOTE,path=item['out']),120)
        write(self.out/'bootstrap_remote_files.json',listing,fresh=True)
        returned=[];errors=[]
        for source in listing['files']:
            try:
                target=return_destination(member(source['path']))
                if target.exists() and sha(target)!=source['sha256']:
                    raise FileExistsError('Preserve conflicting local bootstrap evidence: '+str(target))
                if not target.exists():
                    subprocess.run(['rsync','-a','--protect-args','--partial','cts:'+REMOTE+'/'+source['path'],str(target)],check=True,timeout=300)
                checked(source);returned.append(source)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                errors.append(dict(path=source['path'],error_type=type(exc).__name__,error=str(exc)))
        indexed={source['path']:source['sha256'] for source in listing['files']}
        if item['out']+'/receipt.json' not in indexed:
            errors.append(dict(path=item['out']+'/receipt.json',error='Current remote bootstrap receipt absent from listing'))
        write(self.out/'bootstrap_return_verified.json',dict(files=returned,transfer_errors=errors,
              local_SHA_verified=not errors),fresh=True)
        if errors:raise RuntimeError('Bootstrap return incomplete; available small records retained, no retry')
        if error is not None:raise error
        if result.returncode:raise RuntimeError('Remote CPU bootstrap failed; small evidence returned, no retry')
        receipt=read(item['out']+'/receipt.json')
        if (receipt.get('status')!='CPU_overlays_cache_imports_and_requested_registration_passed_GPU_unverified'
                or receipt.get('GPU_calls')!=0 or receipt.get('training_started') is not False
                or receipt.get('teacher_images_generated')!=0
                or set(receipt.get('checks',{}))!={'stablesr','threedsr','sequence','wu_extensions'}
                or any(v.get('returncode')!=0 for v in receipt['checks'].values())
                or receipt.get('script',{}).get('sha256')!=item['script']['sha256']):
            raise ValueError('CPU bootstrap receipt is not a complete successful import/cache acceptance')
        for source in [receipt['runtime_assets'],*(v['log'] for v in receipt['checks'].values())]:
            if indexed.get(member(source['path']))!=source['sha256']:
                raise ValueError('Bootstrap referenced evidence is not bound to this returned listing')
        return dict(receipt=identity(item['out']+'/receipt.json'),return_receipt=identity(self.out/'bootstrap_return_verified.json'))


def validate_teacher_receipt(value,expected,job,frozen):
    hashes={x['path']:x['sha256'] for x in frozen}
    if (value.get('schema')!='static_sr_wu_teacher_acceptance_fixture_v1' or value.get('status')!='fixture_passed'
            or value.get('gpu_executed') is not True or value.get('teacher_generated') is not True
            or value.get('method')!=expected['method']
            or gpu_uuid(value.get('hardware',{}).get('gpu_uuid'))!=gpu_uuid(job['gpu_uuid'])
            or value.get('wrapper',{}).get('sha256')!=hashes[TEACHER]
            or member(value.get('registered_records',{}).get('path',''))!=expected['records']):
        raise ValueError('Acceptance receipt is incomplete, foreign, or from different source/hardware')
    if expected['method']=='threedsr':
        rounds=value.get('rounds',[])
        if [r.get('round_index') for r in rounds]!=list(range(4)) or any(not r.get('outputs') for r in rounds):
            raise ValueError('3DSR fixture did not complete all four latent transitions')
        outputs=[x for r in rounds for x in r['outputs']]
    else:outputs=value.get('outputs',[])
    if not outputs or any(x['size_wh']!=expected['hr_wh'] for x in outputs):raise ValueError('Teacher original output dimensions differ')
    adapters=value.get('adapter_sources',[])
    names=['sequence_matters.py'] if expected['method']=='sequence_matters' else ['threedsr.py','splatsure_stablesr.py']
    if {member(x['path']) for x in adapters}!={'experiments/static_sr_wu_20261022/methods/'+n for n in names}:
        raise ValueError('Teacher adapter identity evidence is incomplete')
    for source in adapters:
        if hashes.get(member(source['path']))!=source['sha256']:raise ValueError('Teacher adapter source identity differs')


def sequence_completion(prior_path,item):
    """Verify one explicitly selected prior job locally; never submit or SSH.

    Paths are derived from its frozen launchplan and job, not supplied as an
    arbitrary receipt directory. The old failed run remains unchanged.
    """
    prior=read(prior_path);prior_id=identity(prior_path)
    if (prior.get('schema')!=SCHEMA or prior.get('host')!='cts' or prior.get('remote_root')!=REMOTE
            or any(prior.get('sequence',{}).get(k)!=item.get(k)
                   for k in ('job','job_id','gpu_uuid','out','receipts','frozen_files'))):
        raise ValueError('Sequence reuse must match the original frozen job and sources')
    claim_path=resolve(prior_path).with_suffix('.claimed.json');claim=read(claim_path)
    previous_status=resolve(claim['out'])/'status.json';status=read(previous_status)
    if (claim.get('launchplan')!=prior_id or status.get('launchplan')!=prior_id
            or status.get('status')!='failed' or status.get('current_stage')!='sequence'):
        raise ValueError('Sequence reuse requires the identified failed Sequence coordination run')
    evidence=[prior_id,identity(claim_path),identity(previous_status)]
    for source in item['frozen_files']:checked(source)
    job=read(checked(item['job']));directory=HERE/'dispatch'/item['job_id']
    intent=read(directory/'intent.json');terminal=read(directory/'remote_terminal.json')
    collected=read(directory/'collected.json');complete=read(item['out']+'/complete.json')
    if (intent.get('host')!='cts' or intent.get('job')!=item['job']['path']
            or intent.get('job_sha256')!=item['job']['sha256']
            or terminal.get('state')!='ended' or terminal.get('outcome')!='success'
            or terminal.get('training_exit_code')!=0 or terminal.get('job_id')!=item['job_id']
            or collected.get('status')!='collected' or collected.get('remote_terminal')!=terminal
            or collected.get('missing_paths')!=[] or collected.get('transfer_errors')!=[]
            or {x['path'] for x in collected.get('returned_paths',[])}!=set(job['return_paths'])
            or complete.get('status')!='completed' or complete.get('job_id')!=item['job_id']
            or complete.get('job_sha256')!=item['job']['sha256']
            or gpu_uuid(complete.get('gpu_uuid'))!=gpu_uuid(item['gpu_uuid'])):
        raise ValueError('Sequence completion, terminal, collection, or job identity is incomplete/foreign')
    stages=complete.get('stages',[])
    if (len(stages)!=len(job['commands']) or any(
            stage.get('name')!=command['name'] or stage.get('returncode')!=0
            or stage.get('argv')!=[x.replace('{ROOT}',REMOTE) for x in command['argv']]
            for stage,command in zip(stages,job['commands']))):
        raise ValueError('Sequence completed command identities or exit codes differ')
    evidence.extend(identity(directory/name) for name in ('intent.json','remote_terminal.json','collected.json'))
    evidence.append(identity(item['out']+'/complete.json'))
    outputs=[];receipts=[]
    def returned(source,within=None):
        path=member(source['path']);local=resolve(path)
        if within is not None and not Path(path).is_relative_to(within):
            raise ValueError('Sequence output escaped its frozen fixture directory')
        if any(p.is_symlink() for p in [local,*local.parents]):raise ValueError('Symlink in reused Sequence evidence')
        checked(dict(path=path,sha256=source['sha256']))
        evidence.append(identity(path));return identity(path)
    for expected,command in zip(item['receipts'],job['commands']):
        value=read(expected['path']);validate_teacher_receipt(value,expected,item,prior['frozen_files'])
        receipts.append(identity(expected['path']));evidence.append(receipts[-1])
        directory=Path(expected['path']).parent
        keys=[x['key'] for x in value['outputs']]
        if len(set(keys))!=len(keys) or set(keys)!=set(value.get('selected_keys',[])):
            raise ValueError('Sequence output key coverage is incomplete')
        outputs.extend(returned(x,directory) for x in value['outputs'])
        for name in ('fixture_records','teacher_receipt'):returned(value[name],directory)
        returned(value['registered_records'])
        argv=command['argv'];flags=dict(zip(argv[3::2],argv[4::2]))
        if member(value['original_sequence_plan']['path'])!=member(flags['--sequence-plan']):
            raise ValueError('Sequence original plan differs from frozen job input')
        returned(value['original_sequence_plan'])
        if {member(x['path']) for x in value['model_identities']}!={member(flags[name]) for name in ('--psrt-checkpoint','--spynet-checkpoint')}:
            raise ValueError('Sequence model identities differ from frozen job inputs')
        for source in value['model_identities']:returned(source)
    return dict(prior_launchplan=prior_id,terminal=identity(HERE/'dispatch'/item['job_id']/'remote_terminal.json'),
        completion=identity(item['out']+'/complete.json'),receipts=receipts,outputs=outputs,
        evidence_files=list({x['path']:x for x in evidence}.values()),local_output_SHA_verified=True)


def execute(plan_path,out,operations=Operations):
    out=resolve(out);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='static_sr_cts_acceptance_continuation_receipt_v1',status='starting',events=[],
                created_utc=utc(),coordinator_pid=os.getpid(),automatic_replay=False,formal_training_started=False)
    write(out/'status.json',record)
    try:
        plan=read(plan_path);record['launchplan']=identity(plan_path)
        if (plan.get('schema')!=SCHEMA or plan.get('host')!='cts' or plan.get('remote_root')!=REMOTE
                or plan.get('no_replay') is not True or plan.get('formal_training_allowed') is not False):
            raise ValueError('Unrecognized or unsafe continuation plan')
        for item in plan['frozen_files']:checked(item)
        claim=resolve(plan_path).with_suffix('.claimed.json')
        write(claim,dict(launchplan=record['launchplan'],out=str(out),pid=os.getpid(),start_ticks=process_ticks(os.getpid()),boot_id=boot_id()),fresh=True)
        ops=operations(plan,out)
        steps=[('cpu_suite',ops.cpu_suite),('sequence',lambda:ops.job('sequence')),
               ('transfer',ops.transfer),('bootstrap',ops.bootstrap),('diffusion',lambda:ops.job('diffusion'))]
        for name,action in steps:
            event=dict(stage=name,status='running',started_utc=utc());record['events'].append(event)
            record.update(status='running',current_stage=name);write(out/'status.json',record);before=time.monotonic()
            for item in plan['frozen_files']:checked(item)
            details=action();event.update(status='completed',seconds=time.monotonic()-before,details=details,ended_utc=utc())
            write(out/'status.json',record)
        record.update(status='acceptance_continuation_completed',ended_utc=utc(),formal_training_started=False)
        write(out/'complete.json',record,fresh=True);write(out/'status.json',record)
        return record
    except BaseException:
        record.update(status='failed',ended_utc=utc(),error=traceback.format_exc())
        write(out/'failure.json',record,fresh=True);write(out/'status.json',record)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--out',required=True)
    p.add_argument('--sequence-job',default='deployment/static_sr_cts_20261022/jobs/sequence_v1.json')
    p.add_argument('--diffusion-job',default='deployment/static_sr_cts_20261022/jobs/diffusion_v1.json')
    p.add_argument('--transfer-pid',type=int,required=True);p.add_argument('--transfer-start-ticks',type=int,required=True)
    p.add_argument('--transfer-dir',default='output/static_sr_wu_20261022/cts_sync/teacher_v3')
    p.add_argument('--event-timeout-seconds',type=int,default=43200)
    p.add_argument('--reuse-sequence-completion',metavar='PRIOR_LAUNCHPLAN',
                   help='Explicitly verify/reuse this failed coordinator plan\'s already completed Sequence job; never resubmit it')
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.action=='plan':
        if min(args.transfer_pid,args.transfer_start_ticks,args.event_timeout_seconds)<=0:raise ValueError('Positive process identity and timeout required')
        value=build_plan(args);write(resolve(args.out),value,fresh=True)
        print(json.dumps(dict(status='launchplan_frozen_no_execution',plan=identity(args.out))))
    else:
        result=execute(args.plan,args.out);print(json.dumps(dict(status=result['status'],out=args.out)))


if __name__=='__main__':main()
