"""Root-only exact held B0 then Bsync persistent true-exit coordinator.

Prepare/imports/preflight are CPU metadata only. Run alone creates the two
roles, verifies full actor inventories before release, and accepts exact owned
manager/callback Exit0,6000/300/299/allSHA before one Bsync successor. No retries,
method choices, global index mutations or teacher/held quality inference.
"""
from __future__ import annotations
import argparse,base64,copy,fcntl,hashlib,importlib.util,json,os,re,shlex,subprocess,sys,time,traceback
from pathlib import Path
import full_held_confirmation_contract_v2 as held
import full_held_local_RTX3090_scope_v1 as local
OUT='output/dynamic_sr_multiview_footprint_20261007'
CORE=dict(path=OUT+'/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/production_core_v3.py',sha256='590f38a55da141de8a9617c66484d7cad1dbc36ba5ae5411bb11030972f73d2d')
TRANSPORT=dict(path=OUT+'/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/runtime_adapter_v4.py',sha256='7704aa2142456d890eaf32104f0b62f1647b606f75c28c76b03a0bdc2305b580')
COMPLETION=dict(path='experiments/dynamic_sr_multiview_footprint_20261007/completion.py',sha256='e7b55bf9b504c67e4df7480be078e07b761dcca2f9b7d94fd1da2572d1fdcfeb')
COMMON=dict(path='experiments/dynamic_sr_multiview_footprint_20261007/fp_common.py',sha256='bba0a60e17f03aaa42c01ece0e8651c054a4f48cfe742e861f0f37c890292977')
def require(v,m):held.require(v,m)
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def registration_lock(scene):
 require(scene in held.HELD,'Exact held cohort required');return OUT+'/full_native_SR_root_registration_held_local_RTX3090_'+scene+'_seed20261007.lock'
def helpers(root):
 root=Path(root).resolve();require(root==held.ROOT,'Declared actual root differs')
 require(hashlib.sha256((root/CORE['path']).read_bytes()).hexdigest()==CORE['sha256'],'Original integrity source changed')
 core=load(root/CORE['path'],'_held_pair_original_core');e=core.Evidence(root)
 sys.path.insert(0,str(held.HERE));op=load(e.bound(held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py')),'_held_pair_operator')
 legacy=load(e.bound(TRANSPORT),'_held_pair_transport');e.bound(COMMON);completion=load(e.bound(COMPLETION),'_held_pair_completion')
 require('torch' not in sys.modules,'CPU coordinator imported model/tensor');return e,core,op,legacy,completion

def validate_prior_B0(p,s,core):
 require(p['status']=='accepted_actual_callback_Exit0_full6000_eval300_temporal299_allSHA' and p['task_key']==s['scene']+'/20261007/B0' and p['source']==held.entry(Path(__file__)),'True exact held B0 producer acceptance required')
 require(p['training_accepted']==dict(iterations=6000,RGB=18000,moments=0,backward=6000,Adam=6000) and p['accepted_formal_updates']==6000 and p['test_observations']==300 and p['adjacent_pairs']==299,'Previous complete B0 budget/quality/temporal differs')
 core.exact_success(p['callback_exit'],p['callback_registration_value']);core.exact_success(p['training_terminal_value'],p['training_registration_value'])
 for key in ('parent','manifest','schedule','teacher'):require(held.same(p[key],s['assets']['parent_checkpoint' if key=='parent' else key]),'Previous B0 immutable input differs: '+key)
 require(p['training_platform']==s['GPU'] and p['global_state_modified'] is False,'Previous B0 paired platform differs')
 return True

def validate_plan(q,e,source):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Root coordinator stays CPU only')
 require(q['schema']=='root_exact_held_B0_Bsync_pair_true_exit_once_v1' and q['source']==source and q['root_authorized'] is True and q['scientific_retry'] is False and q['global_state_modified'] is False and q['synthetic_contract'] is False,'Exact immutable root original held pair plan required')
 auth=local.validate_authorization(q['confirmation_authorization'],q['scene'],20261007)
 require(q['scene'] in held.HELD and q['scope']==[q['scene']+'/20261007/'+m for m in held.METHODS] and q['registration_lock']==registration_lock(q['scene']),'Only one exact held paired cohort')
 require(set(q['specs'])==set(held.METHODS),'Both exact source-bound original ready specs required')
 op=load(e.bound(held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py')),'_held_plan_operator');plans=[];specs=[]
 for method,r in q['specs'].items():
  s=e.read(r);require(s['schema']==op.SCHEMA and s['status']=='root_registered_ready_operator_v2' and s['dispatch_enabled'] is True and s['synthetic_contract'] is False and (s['scene'],s['seed'],s['method'],s['role'])==(q['scene'],20261007,method,'held_confirmation'),'Actual ready held role spec required')
  require(s['assets']['confirmation_authorization']==q['confirmation_authorization'] and s['assets']['runtime_adapter_source']==held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py') and s['GPU']==auth['cohort_training_platforms'][q['scene']]==local.GPU and s['transport']=='local' and s['workspace']==str(local.ROOT),'Frozen source/auth/platform differs')
  held.validate_quality_authorization(s['assets']['held_quality_authorization'],q['confirmation_authorization'],q['scene'],20261007,method)
  op.validate_no_repeat(s,e.read(s['assets']['root_endpoint_index']))
  p=e.read(s['assets']['plan']);require(p['missing']==[] and p['status']=='registered_ready_full_native_SR_refinement' and p['method']==method and p['planned_cost']['SR_updates']==p['planned_cost']['Adam']==6000 and p['planned_cost']['RGB_forwards']==18000 and p['planned_cost']['moment_forwards']==0,'Whole unchanged scientific plan required')
  require(p['source_files']['project'][held.NATIVE_SOURCE['path']]==held.NATIVE_SOURCE['sha256'],'Exact c278 source required')
  for role in ('manager','callback'):
   v=s['role_deployments'][role];require(v['full_inventory_verified'] is True and v['GPU_calls']==v['formal_updates']==v['image_decodes']==0,'Actual two whole role SHA closures required')
   require(v['operator_source']==s['assets']['runtime_adapter_source'] and v['role']==role,'Actual role source differs')
  from full_held_confirmation_bridge import validate_native
  validate_native(s['assets']['native_acceptance'],p['parent'],q['confirmation_authorization'],s['assets']['native_parent_terminal'])
  plans.append(p);specs.append(s)
 require(plans[0]['full_native_context']==plans[1]['full_native_context'] and specs[0]['GPU']==specs[1]['GPU'] and specs[0]['evaluation']==specs[1]['evaluation'],'Original same parent/S/teacher/paired platform required')
 require(q['run_out'].startswith(OUT+'/held_confirmation_pair_root/'+q['scene']+'_seed20261007_local_RTX3090_original_v1_20261009/') and not any(e.path(q['run_out']).is_relative_to(e.path(s['paths']['train_out'])) for s in specs),'Isolated root operational output required')
 return True

IO_HELPER=r'''
import base64,ctypes,errno,hashlib,json,os,pathlib,select,socket,struct,subprocess,sys
q=json.load(sys.stdin);mode=q['mode'];root=pathlib.Path(q['root']).resolve()
def require(v,m):
 if not v:raise ValueError(m)
def safe(v):
 p=pathlib.Path(v);require(not p.is_absolute() and p.parts and '..' not in p.parts,'Unsafe relative path')
 t=root/p;require(t.resolve().is_relative_to(root) and not any(x.is_symlink() for x in [t,*t.parents]),'Unsafe symlink/path');return t
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(v):print(json.dumps(v),flush=True)
def props(unit):
 a=subprocess.run(['systemctl','--user','show',unit,'--no-pager'],capture_output=True,text=True)
 p=dict(x.split('=',1) for x in a.stdout.splitlines() if '=' in x)
 require(a.returncode==0 or p.get('LoadState')=='not-found','systemd properties unavailable');return p
if mode=='fresh':
 for p in q['absent_paths']:require(not safe(p).exists(),'Prior task artifact blocks one-shot startup: '+p)
 for u in q['units']:
  p=props(u);require(p.get('LoadState')=='not-found' and p.get('MainPID','0')=='0' and not p.get('InvocationID'),'Existing unit blocks unique startup')
 for p in q['directories']:safe(p).mkdir(parents=True,exist_ok=True)
 emit(dict(status='fresh_CPU_units_and_paths_verified',GPU_calls=0))
elif mode=='verify':
 p=safe(q['path']);require(p.is_file() and sha(p)==q['sha256'],'Existing staged spec differs')
 emit(dict(path=q['path'],sha256=sha(p),bytes=p.stat().st_size,read_only=True))
elif mode=='publish':
 p=safe(q['path']);data=base64.b64decode(q['base64']);require(hashlib.sha256(data).hexdigest()==q['sha256'],'Payload SHA differs')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():require(p.is_file() and sha(p)==q['sha256'] and p.read_bytes()==data,'Different existing artifact protected');os.chmod(p,0o444)
 else:
  tmp=p.with_name(p.name+'.root_atomic_'+str(os.getpid()));require(not tmp.exists(),'Prior temporary protected')
  with tmp.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
  os.chmod(tmp,0o444)
  libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2
  rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
  result=rename(-100,os.fsencode(tmp),-100,os.fsencode(p),1)
  if result<0:
   code=ctypes.get_errno()
   if code==errno.EEXIST:
    require(p.is_file() and sha(p)==q['sha256'],'Concurrent different artifact protected');tmp.unlink()
   else:raise OSError(code,'renameat2(RENAME_NOREPLACE) publication failed')
  directory=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(directory)
  finally:os.close(directory)
 require(p.stat().st_mode&0o222==0 and sha(p)==q['sha256'],'Publication not frozen/exact')
 emit(dict(path=q['path'],sha256=sha(p),bytes=p.stat().st_size,mode=oct(p.stat().st_mode&0o777),atomic_noreplace=True,publication='renameat2_RENAME_NOREPLACE',parent_fsync=True))
elif mode in ('wait-json','guarded-wait-json'):
 p=safe(q['path']);require(p.parent.is_dir(),'Watch directory must preexist')
 libc=ctypes.CDLL(None,use_errno=True);libc.inotify_init1.restype=ctypes.c_int
 libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];libc.inotify_add_watch.restype=ctypes.c_int
 fd=libc.inotify_init1(os.O_CLOEXEC);require(fd>=0,'inotify init failed');pidfd=None;initial=None
 try:
  require(libc.inotify_add_watch(fd,os.fsencode(p.parent),0x8|0x80|0x400|0x800)>=0,'inotify watch failed')
  if mode=='guarded-wait-json':
   initial=props(q['unit']);pid=int(initial.get('MainPID','0'))
   require(initial.get('LoadState')=='loaded' and initial.get('Id')==q['unit'] and initial.get('ActiveState')=='active'
    and initial.get('SubState')=='running' and pid>0 and initial.get('ExecMainPID')==str(pid)
    and len(initial.get('InvocationID',''))==32 and int(initial.get('ExecMainStartTimestampMonotonic','0'))>0,'New owned service exited/unknown before registration')
   pidfd=os.pidfd_open(pid)
  def ready():
   if not p.exists():return False
   try:data=p.read_bytes();v=json.loads(data)
   except json.JSONDecodeError:return False
   if initial is not None:
    require(v['unit']==q['unit'] and v['invocation_id']==initial['InvocationID'] and int(v['pid'])==int(initial['ExecMainPID'])
     and str(v['start_monotonic'])==initial['ExecMainStartTimestampMonotonic'],'Registration differs from exact live pidfd attachment')
   emit(dict(path=q['path'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),value=v,initial_service=initial));return True
  if not ready():
   done=False
   while not done:
    if pidfd is not None:
     poll=select.poll();poll.register(fd,select.POLLIN);poll.register(pidfd,select.POLLIN)
     events=poll.poll() # True registration or process exit, no timed polling.
     if any(handle==pidfd for handle,flags in events):
      raise RuntimeError('Exact new owned service exited before root release/registration acceptance; preserved, no retry')
     require(any(handle==fd and flags&select.POLLIN for handle,flags in events),'Registration event descriptors lost')
    data=os.read(fd,65536);at=0
    while at<len(data):
     _,mask,_,length=struct.unpack_from('iIII',data,at);at+=16
     name=os.fsdecode(data[at:at+length].split(b'\0',1)[0]);at+=length
     require(not mask&(0x4000|0x400|0x800|0x8000),'Registration watch lost')
     if mask&(0x8|0x80) and name==p.name:done=ready()
 finally:
  os.close(fd)
  if pidfd is not None:os.close(pidfd)
elif mode=='owner':
 r=q['registration'];p=props(r['unit']);pid=int(r['pid']);proc=pathlib.Path('/proc')/str(pid)
 raw=(proc/'stat').read_text();start=raw[raw.rfind(')')+2:].split()[19]
 argv=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
 cg=(proc/'cgroup').read_text().strip();env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x)
 boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 require(r['host']==socket.gethostname() and r['workspace']==str(root) and r['boot_id']==boot,'Actual host/workspace/boot differs')
 require(argv==q['command']==r['command'] and start==str(r['start_ticks']) and cg=='0::'+r['owned_cgroup'],'Actual argv/start/cgroup differs')
 require(env.get('CUDA_VISIBLE_DEVICES')=='' and env.get('INVOCATION_ID')==r['invocation_id'],'Actual visibility/invocation differs')
 require((proc/'exe').resolve()==pathlib.Path(q['runtime_python']).resolve(),'Actual process executable differs')
 require(p['Id']==r['unit'] and p['LoadState']=='loaded' and p['ActiveState']=='active' and p['SubState']=='running'
  and p['MainPID']==p['ExecMainPID']==str(pid) and p['InvocationID']==r['invocation_id']
  and p['ExecMainStartTimestampMonotonic']==str(r['start_monotonic']) and p['ControlGroup']==r['owned_cgroup'],'Actual systemd identity differs')
 require(p['Restart']=='no' and p['RemainAfterExit']=='yes','Service restart/remain policy differs')
 emit(dict(status='verified_actual_CPU_service_owner',registration=r,systemd=p,proc_argv=argv,proc_start_ticks=start,proc_cgroup=cg,CUDA_VISIBLE_DEVICES='',GPU_calls=0))
else:raise ValueError('Unknown CPU IO operation')
'''

class CallbackReceiptEvidence:
    """Post-Exit0 audit only: reuse the callback's full payload SHA provenance.

    This explicit object is never used by science, actor validation or preflight.
    JSON/source bytes still receive actual SHA. Covered model/float payloads need
    both original/returned provenance and current SIZE. Uncovered payloads fail.
    """
    PAYLOAD_SUFFIXES={'.pt','.npy','.npz','.png','.jpg','.jpeg','.webp','.exr'}
    def __init__(self,parent,transfer,s):
        self.parent=parent;self.root=parent.root;self.synthetic=parent.synthetic;self.covered={};self.checked=set()
        require(transfer['status']=='completed_native_full_training_and_evaluation_all_file_SHA_return'
                and transfer['training_workspace']==s['workspace'] and transfer['evaluation_workspace']==s['evaluation']['workspace'],
                'Exact callback all-SHA provenance and workspaces required')
        require(len(transfer['entries'])==len({v['path'] for v in transfer['entries']}),'Unique callback payload identities required')
        for row in transfer['entries']:
            require(type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Typed callback SHA and bytes required')
            original=parent.path(row['path']);returned_path=s['paths']['returned_out']+'/files/'+row['path'];returned=parent.path(returned_path)
            for path in (original,returned):require(path.is_file() and path.stat().st_size==row['bytes'],'Current original/returned callback file bytes differ')
            if original.suffix.lower() in self.PAYLOAD_SUFFIXES:
                for path in (row['path'],returned_path):
                    require(path not in self.covered,'Overlapping callback source/returned identities forbidden')
                    self.covered[path]=dict(path=path,sha256=row['sha256'],bytes=row['bytes'])
    def path(self,relative):return self.parent.path(relative)
    def bound(self,item):
        path=self.path(item['path'])
        if path.suffix.lower() not in self.PAYLOAD_SUFFIXES:return self.parent.bound(item)
        require(item['path'] in self.covered,'Payload outside truecallback all-SHA provenance is refused')
        row=self.covered[item['path']];require(self.same(item,row),'Payload differs from truecallback SHA provenance')
        if 'bytes' in item:require(item['bytes']==row['bytes'],'Payload referenced bytes differ')
        require(path.is_file() and path.stat().st_size==row['bytes'],'Payload current SIZE differs')
        self.checked.add(item['path']);return path
    @staticmethod
    def same(a,b):return a['path']==b['path'] and a['sha256']==b['sha256']
    def entry(self,relative):
        p=self.path(relative)
        if p.suffix.lower() in self.PAYLOAD_SUFFIXES:
            require(relative in self.covered,'Unrecorded payload entry cannot be invented')
            row=self.covered[relative];self.bound(row);return dict(path=relative,sha256=row['sha256'])
        return self.parent.entry(relative)
    def read(self,item):
        require(self.path(item['path']).suffix.lower() not in self.PAYLOAD_SUFFIXES,'CPU audit must not decode model/image/array payloads')
        return self.parent.read(item)
    def new(self,path,value):return self.parent.new(path,value)
    def audit(self):return dict(actual_all_SHA_controller_receipt_reused=True,large_payload_SHA_rescans=0,
        original_and_returned_current_SIZE_checked=True,covered_payload_paths=len(self.covered),payload_paths_used=len(self.checked),
        scientific_SHA_mask_or_monkeypatch=False,scope='only_post_truecallback_Exit0_CPU_acceptance',small_metadata_actual_byte_SHA=True)

class Actual:
    def __init__(self,e,core,op,legacy,completion,out):
        self.e,self.core,self.op,self.legacy,self.completion,self.out=e,core,op,legacy,completion,out
    def command(self,role,argv):
        require(role in ('manager','callback'),'Only exact held roles')
        require(all(isinstance(v,str) for v in argv) and argv and argv[0] not in ('ssh','rsync'),'Only exact local CPU commands')
        return argv
    def call(self,role,argv,payload=None):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        p=subprocess.run(self.command(role,argv),input=json.dumps(payload) if payload is not None else None,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        self.e.new(self.out+'/CPU_command_'+str(time.time_ns())+'.json',dict(role=role,argv=argv,exit_code=p.returncode,stderr=p.stderr,CPU_only=True))
        require(p.returncode==0,'CPU command failed; no automatic restart: '+p.stderr)
        return p.stdout
    def io(self,s,role,mode,**kw):
        workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
        return json.loads(self.call(role,['env','CUDA_VISIBLE_DEVICES=','/usr/bin/python3','-u','-c',IO_HELPER],dict(mode=mode,root=workspace,**kw)))
    def publish(self,s,role,ref,data):
        require(hashlib.sha256(data).hexdigest()==ref['sha256'],'Publish source SHA differs')
        return self.io(s,role,'publish',path=ref['path'],sha256=ref['sha256'],base64=base64.b64encode(data).decode())
    def owner(self,s,role,ref,spec_ref):
        r=self.e.read(ref);expected=self.op.role_command(s,self.e,role,spec_ref)
        require(r['source']==held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py') and r['spec']==spec_ref and r['unit']==self.op.services(s,role)
                and r['command']==expected and re.fullmatch('[a-f0-9]{32}',r['invocation_id'])
                and r['empty_cgroup_helper_source']==s['assets']['remote_helper_source'],'Actual registered source/command differs')
        runtime=s['training'] if role=='manager' else s['evaluation']
        proof=self.io(s,role,'owner',registration=r,command=expected,runtime_python=runtime['runtime_python'])
        return self.e.new(self.out+'/'+s['method']+'/'+role+'_owner_'+str(time.time_ns())+'.json',proof)
    def start(self,s,spec_ref,previous_B0=None):
        task=s['method'];directory=self.out+'/'+task
        with self.e.path(registration_lock(s['scene'])).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            # One-time latches protect both service registration and release.
            self.e.new(directory+'/one_task_attempt.json',dict(spec=spec_ref,task_key=self.op.task_key(s),scientific_retry=False))
            for role in ('callback','manager'):
                own=s['paths']['callback_out' if role=='callback' else 'training_manager_out']
                absent=[own+'/actual_registration.json',own+'/intent.json',s['paths']['train_out'],s['paths']['evaluation_out'],s['CPU_temporal']['out']]
                release=str(Path(s['release_path'][role]).relative_to(s['workspace'] if role=='manager' else s['evaluation']['workspace']))
                self.io(s,role,'fresh',units=[self.op.services(s,role)],absent_paths=[*absent,release],directories=[own,str(Path(release).parent)])
                self.io(s,role,'verify',path=spec_ref['path'],sha256=spec_ref['sha256'])
                runtime=s['training'] if role=='manager' else s['evaluation']
                workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
                self.call(role,['env','CUDA_VISIBLE_DEVICES=',runtime['runtime_python'],'-u',str(Path(workspace)/held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py')['path']),
                    '--mode','validate','--role',role,'--root',workspace,'--spec',str(Path(workspace)/spec_ref['path']),'--spec-sha',spec_ref['sha256']])
            if s['method']=='Bsync':
                require(previous_B0 is not None,'Bsync cannot register before actual whole B0 callback acceptance')
                prior=self.e.read(previous_B0);validate_prior_B0(prior,s,self.core)
                for role in ('callback','manager'):self.publish(s,role,previous_B0,self.e.bound(previous_B0).read_bytes())
            else:require(previous_B0 is None,'B0 cannot carry a previous endpoint')
            started={}
            # Callback registers first and waits for root release; neither role
            # receives a GPU UUID until the original science wrapper does so.
            for role in ('callback','manager'):
                workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
                command=self.op.role_command(s,self.e,role,spec_ref)
                log=str(Path(workspace)/(s['paths']['callback_out' if role=='callback' else 'training_manager_out']+'/systemd.log'))
                argv=['systemd-run','--user','--unit='+self.op.services(s,role),'--property=Type=exec',
                    '--property=RemainAfterExit=yes','--property=Restart=no','--property=KillMode=control-group','--property=WorkingDirectory='+workspace,
                    '--property=StandardOutput=append:'+log,'--property=StandardError=append:'+log,'--setenv=CUDA_VISIBLE_DEVICES=',*command]
                self.call(role,argv)
                registration=s['paths']['callback_out' if role=='callback' else 'training_manager_out']+'/actual_registration.json'
                actual=self.io(s,role,'guarded-wait-json',path=registration,unit=self.op.services(s,role))
                ref=dict(path=registration,sha256=actual['sha256'])
                data=(json.dumps(actual['value'],ensure_ascii=False,indent=2)+'\n').encode()
                # Preserve exact written bytes instead of assuming formatting.
                if role=='manager':
                    raw=self.call(role,['/usr/bin/python3','-c','import pathlib,sys,base64;print(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode())',str(Path(workspace)/registration)])
                    data=base64.b64decode(raw.strip())
                    self.publish(s,'callback',ref,data)
                else:data=self.e.bound(ref).read_bytes();self.publish(s,'callback',ref,data)
                self.owner(s,role,ref,spec_ref);started[role]=ref
            for role in ('callback','manager'):
                for reference in started.values():self.publish(s,role,reference,self.e.bound(reference).read_bytes())
                self.owner(s,role,started[role],spec_ref)
            release=dict(schema='root_released_source_aware_native_full_v2',spec=spec_ref,task_key=self.op.task_key(s),
                         registrations=started,root_only_registration=True,no_scientific_asset_changes=True,
                         prior_B0_actual_acceptance=previous_B0,operator=held.entry(held.HERE/'full_held_operator_registered_local_RTX3090_v1.py'),root_startup_source=self.e.entry(str(Path(__file__).resolve().relative_to(self.e.root))))
            data=(json.dumps(release,ensure_ascii=False,indent=2)+'\n').encode();release_ref=None;release_refs={}
            for role in ('callback','manager'):
                workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
                ref=dict(path=str(Path(s['release_path'][role]).relative_to(workspace)),sha256=hashlib.sha256(data).hexdigest())
                if release_ref is not None:require(ref['sha256']==release_ref['sha256'],'Both role release bodies must have one SHA identity')
                self.publish(s,role,ref,data);release_ref=ref;release_refs[role]=ref
            return self.e.new(directory+'/startup_complete.json',dict(status='actual_both_CPU_services_bound_and_atomic_root_released',spec=spec_ref,
                registrations=started,release=release_refs['callback'],releases=release_refs,formal_updates_claimed=0,scientific_completion_claimed=False,GPU_calls=0))
    def accept(self,s,spec_ref,startup_ref):
        startup=self.e.read(startup_ref);r=self.e.read(startup['registrations']['callback'])
        completion=self.completion
        class Local(completion.Transport):
            def command(self,mode,*arguments):
                return ['env','CUDA_VISIBLE_DEVICES=','/usr/bin/python3','-u','-c',completion.REMOTE_HELPER,mode,self.workspace,completion.REL_OUT,*map(str,arguments)]
        watch=Local('local',str(self.e.root),out=self.e.path(self.out+'/'+s['method']+'/callback_event_observer'))
        terminal=None
        try:
            for value in watch.stream('wait-unit',r['unit'],r['invocation_id'],'user'):
                self.e.new(self.out+'/'+s['method']+'/callback_event_'+str(time.time_ns())+'.json',value)
                if value.get('kind')=='bound' and value.get('initial'):self.legacy.validate_bound_initial(value['initial'],r)
                if value.get('kind')=='terminal':terminal=value
        finally:watch.close()
        require(terminal is not None,'Exact callback observer ended without terminal')
        helper=self.e.bound(s['assets']['remote_helper_source'])
        terminal['owned_cgroup_exit_proof']=json.loads(self.call('callback',['env','CUDA_VISIBLE_DEVICES=','/usr/bin/python3','-u',str(helper),
            '--mode','empty-cgroup','--root',str(self.e.root),'--request',json.dumps(dict(registration=r,terminal=terminal))]))
        self.core.exact_success(terminal,r)
        bound=self.op.resolve_release(s,self.e,spec_ref,r,'callback',Path(s['release_path']['callback']),self.legacy)
        closed_ref=self.e.entry(s['paths']['callback_out']+'/complete.json');closed=self.e.read(closed_ref)
        require(closed['status']=='completed_registered_native_full_SR_exit_integrity_evaluation_and_SHA_return'
                and closed['task_key']==self.op.task_key(s) and closed['spec_digest']==self.core.digest(bound)
                and closed['selection_performed'] is False and closed['global_state_modified'] is False,'Exact callback closure identity required')
        manager_terminal=self.e.read(closed['training_exit']);manager_registration=self.e.read(bound['assets']['training_registration'])
        self.core.exact_success(manager_terminal,manager_registration)
        transfer=closed['transfer'];require(transfer==self.e.read(self.e.entry(s['paths']['returned_out']+'/manifest.json'))
            and transfer['status']=='completed_native_full_training_and_evaluation_all_file_SHA_return','All-SHA return manifest differs')
        require(len(transfer['entries'])==len({x['path'] for x in transfer['entries']}),'Duplicate returned path')
        accepted_e=CallbackReceiptEvidence(self.e,transfer,s)
        for row in transfer['entries']:
            accepted_e.bound(row)
            returned=dict(row,path=s['paths']['returned_out']+'/files/'+row['path']);accepted_e.bound(returned)
        require(closed['returned_refs']==transfer['entries'],'Closed callback returned inventory differs')
        for ref in closed['returned_refs']:accepted_e.bound(ref)
        returns=[accepted_e.read(v) for v in transfer['training_return_receipts']]
        require(len(returns)==1 and returns[0]['status']=='completed_owned_native_SR_training_all_SHA_return','Unique full training return required')
        inventory=returns[0]['inventory'];segments=[dict(path=v['path'],sha256=v['sha256']) for v in inventory['entries']
            if re.fullmatch(re.escape(s['paths']['train_out'])+r'/segment_[0-9]+_[0-9]+\.json',v['path'])]
        training=self.core.validate_training(bound,accepted_e,accepted_e.read(s['assets']['plan']),closed['training']['complete'],segments,inventory)
        require(training==closed['training'] and training['accepted']==dict(iterations=6000,RGB=18000,moments=0,backward=6000,Adam=6000),'Full6000/oneAdam training closure differs')
        require(self.core.validate_evaluation(bound,accepted_e,training['checkpoint'],closed['evaluation']['complete'])==closed['evaluation'],'Original full300 evaluator identity differs')
        protocol=accepted_e.read(s['assets']['evaluation_protocol']);diagnostics=300+4*(len(accepted_e.read(s['assets']['manifest'])['splits']['train'])-1);require(len(protocol['train_teacher_diagnostics']['keys'])==diagnostics and closed['evaluation']['cost']['RGB_forwards']==300+diagnostics,'Exact held full train diagnostics/test300 evaluator cost required')
        temporal=self.legacy.validate_temporal_completion(bound,accepted_e,accepted_e.entry(s['CPU_temporal']['out']+'/complete.json'),closed['evaluation']['complete'])
        for ref in [*training['required_refs'],*closed['evaluation']['required_refs'],*temporal['required_refs']]:
            require(any(self.core.same(ref,row) for row in transfer['entries']),'Required endpoint file missing from all-SHA return')
        return accepted_e.new(self.out+'/'+s['method']+'/accepted.json',dict(status='accepted_actual_callback_Exit0_full6000_eval300_temporal299_allSHA',
            task_key=self.op.task_key(s),spec=spec_ref,startup=startup_ref,callback_exit=terminal,complete=closed_ref,
            accepted_formal_updates=6000,test_observations=300,train_diagnostic_observations=diagnostics,adjacent_pairs=299,global_state_modified=False,
            source=held.entry(Path(__file__)),training_accepted=training['accepted'],training_terminal_value=manager_terminal,training_registration_value=manager_registration,callback_registration_value=r,
            training_platform=s['GPU'],parent=s['assets']['parent_checkpoint'],manifest=s['assets']['manifest'],schedule=s['assets']['schedule'],teacher=s['assets']['teacher'],
            training_exit=closed['training_exit'],training_registration=bound['assets']['training_registration'],
            original_all_SHA_return_manifest=self.e.entry(s['paths']['returned_out']+'/manifest.json'),small_metadata_audit=accepted_e.audit()))

def sequence(q,e,source,operations):
    out=e.path(q['run_out']);out.mkdir(parents=True,exist_ok=True)
    with (out/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            ref=e.entry(q['run_out']+'/complete.json');done=e.read(ref)
            require(done['source']==source and done['plan']==q['plan_reference'],'Completed chain identity changed')
            for ref in done['accepted_endpoints']:e.bound(ref)
            return e.entry(q['run_out']+'/complete.json')
        require(not (out/'one_sequence_attempt.json').exists(),'Prior attempt/failure/unknown blocks automatic retry')
        validate_plan(q,e,source)
        e.new(q['run_out']+'/one_sequence_attempt.json',dict(source=source,plan=q['plan_reference'],scope=q['scope'],scientific_retry=False))
        accepted=[];active='B0'
        try:
            for method in ('B0','Bsync'):
                active=method;s=e.read(q['specs'][method]);startup=operations.start(s,q['specs'][method],accepted[0] if method=='Bsync' else None)
                accepted.append(operations.accept(s,q['specs'][method],startup))
            return e.new(q['run_out']+'/complete.json',dict(status='completed_exact_held_pair_true_callback_exit_full_training_quality_temporal_allSHA',
                source=source,plan=q['plan_reference'],accepted_endpoints=accepted,accepted_formal_updates=0 if e.synthetic else 12000,
                synthetic_contract=e.synthetic,global_state_modified=False,scientific_retry=False))
        except BaseException as problem:
            e.new(q['run_out']+'/failure_'+str(time.time_ns())+'.json',dict(status='failed_exact_held_one_shot_preserved',active_task=active,
                source=source,plan=q['plan_reference'],error=repr(problem),traceback=traceback.format_exc(),automatic_retry=False,no_stop_thaw_or_global_state_write=True))
            raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('imports','prepare','preflight','run'),required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',choices=held.HELD);p.add_argument('--confirmation-authorization');p.add_argument('--B0-spec');p.add_argument('--B0-spec-sha');p.add_argument('--Bsync-spec');p.add_argument('--Bsync-spec-sha');p.add_argument('--out');p.add_argument('--plan');p.add_argument('--plan-sha');p.add_argument('--root-authorized',action='store_true');a=p.parse_args()
 e,core,op,legacy,completion=helpers(a.root);source=e.entry(str(Path(__file__).resolve().relative_to(e.root)))
 if a.mode=='imports':print(json.dumps(dict(status='passed_CPU_original_completion_and_integrity_imports',source=source,GPU_calls=0,SSH_calls=0,services_created=0)));return
 if a.mode=='prepare':
  require(a.root_authorized and all((a.scene,a.confirmation_authorization,a.B0_spec,a.B0_spec_sha,a.Bsync_spec,a.Bsync_spec_sha,a.out)),'Root actual authorization/two actual ready specs/fresh out required')
  q=dict(schema='root_exact_held_B0_Bsync_pair_true_exit_once_v1',source=source,root_authorized=True,scene=a.scene,confirmation_authorization=e.entry(a.confirmation_authorization),scope=[a.scene+'/20261007/'+m for m in held.METHODS],specs=dict(B0=dict(path=a.B0_spec,sha256=a.B0_spec_sha),Bsync=dict(path=a.Bsync_spec,sha256=a.Bsync_spec_sha)),run_out=a.out+'/actual_event_chain',registration_lock=registration_lock(a.scene),scientific_retry=False,global_state_modified=False,synthetic_contract=False)
  require(not e.path(a.out).exists(),'Fresh exact held CPU plan output required');validate_plan(q,e,source);print(json.dumps(e.new(a.out+'/plan.json',q)));return
 require(a.plan and a.plan_sha,'Relative frozen plan identity required');ref=dict(path=a.plan,sha256=a.plan_sha);q=e.read(ref);q['plan_reference']=ref
 if a.mode=='preflight':validate_plan(q,e,source);print(json.dumps(dict(status='passed_exact_held_pair_CPU_metadata_preflight',GPU_calls=0,SSH_calls=0)));return
 require(a.root_authorized,'Only root starts one persistent chain');print(json.dumps(sequence(q,e,source,Actual(e,core,op,legacy,completion,q['run_out']))))
if __name__=='__main__':main()
