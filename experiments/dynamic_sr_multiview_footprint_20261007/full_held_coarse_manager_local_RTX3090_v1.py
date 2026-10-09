"""Private owned local RTX3090 coarse manager with a declared exact display guard.

Only prefix.check_gpu is replaced in memory. Invoke the additive held-scope
producer with unchanged coarse algorithms once; call immutable76faa.verify_output
and its preserved validate_result on normal return. Actual full16 freeze and both
teachers are checked before preparing either held parent. Never
pretend this process is76faa. A root-owned model-free real resource fixture must
be accepted before a fresh prepare. Import/validate do not query GPUs.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.metadata,importlib.util,json,math,os,re,runpy,socket,subprocess,sys,time,traceback
from pathlib import Path
import full_held_confirmation_contract_v2 as held
import full_held_local_RTX3090_scope_v1 as local_scope
import xml.etree.ElementTree as ET
OUT='output/dynamic_sr_multiview_footprint_20261007';D=OUT+'/native_full_parent_coarse_overlap_20261007'
GPU=local_scope.GPU;UP=local_scope.UP;PYTHON=local_scope.PY
GNOME=None
PINS={'closure_source':(D+'/source_snapshot/root_coarse_exit_chain_v2.py','76faa033d2ab2ebb53c5d73f1138035fff7a58a7ef4e84149a44f7d42c4f7957'),'observer_source':(D+'/source_snapshot/root_coarse_exact_exit.py','7daf97ca67982a80a3149e492dd84d345ff6d8901e654acf5d59d8517f2dfdd4'),'prefix_source':('experiments/dynamic_sr_multiview_footprint_20261007/full_prefix_registered.py','92a19ed94c2bf7dab881341375c4bdc486e6f4032536e977592c7911c3ce6605'),'event_transport_source':('experiments/dynamic_sr_multiview_footprint_20261007/completion.py','e7b55bf9b504c67e4df7480be078e07b761dcca2f9b7d94fd1da2572d1fdcfeb'),'event_common_source':('experiments/dynamic_sr_multiview_footprint_20261007/fp_common.py','bba0a60e17f03aaa42c01ece0e8651c054a4f48cfe742e861f0f37c890292977')}
SCHEMA='root_exact_held_local_RTX3090_frozen_parent_coarse_resource_adapter_v1'
CONTEXT=None

def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def new(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
class Evidence:
 def __init__(self,root):self.root=Path(root);self.checked={}
 def path(self,name):
  p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Workspace evidence must be relative');p=self.root/p
  require(p.resolve()==p.absolute() and p.resolve().is_relative_to(self.root) and not any(v.is_symlink() for v in [p,*p.parents]),'Evidence path changed/symlinked');return p
 def entry(self,p):return dict(path=str(Path(p).relative_to(self.root)),sha256=sha(p))
 def bound(self,r):
  p=self.path(r['path']);require(p.is_file() and sha(p)==r['sha256'],'Bound source/spec bytes differ');self.checked[str(p)]=r['sha256'];return p
 def unchanged(self):
  for p,h in self.checked.items():require(sha(p)==h,'Bound operational/scientific source changed')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def command(s,e,mode,spec_ref):return [PYTHON,'-u',str(e.bound(s['manager_source'])),'--mode',mode,'--spec',str(e.path(spec_ref['path']))]
def validate(s,e,source):
 require(s['schema']==SCHEMA and s['scene'] in held.HELD and s['seed']==20261007 and s['runtime_root']==str(e.root)
         and e.root.resolve()==local_scope.ROOT,'Only exact isolated original held seed07 parent workspace allowed')
 authorization=local_scope.validate_authorization(s['confirmation_authorization'],s['scene'],s['seed'])
 require(authorization['cohort_training_platforms'][s['scene']]==GPU,'Root-declared exact paired physical platform required')
 lineage=authorization['coarse_operational_sources'][s['scene']]
 require(lineage==dict(producer=s['producer_source'],observer=s['observer_source'],manager=s['manager_source']),'Actual root-declared held coarse lineage differs')
 require(s['GPU']==GPU and s['display_exemption']==GNOME and s['sampling_interval_seconds']==30 and s['upstream']==UP and s['python']==PYTHON,'Exact physical GPU/display/native runtime scope differs')
 require(s['manager_source']==e.entry(source) and s['resource_adapter']==s['manager_source'],'Actual manager/resource adapter must be declared as this new source');e.bound(s['manager_source'])
 require(s['producer_source']==held.entry(held.HERE/'full_held_parent_overlap.py'),'Actual additive held producer required');e.bound(s['producer_source'])
 require(s['contract_source']==held.entry(held.HERE/'full_held_confirmation_contract_v2.py'),'Exact held strong gate contract required');e.bound(s['contract_source'])
 for key,(p,h) in PINS.items():
  if key=='producer_source':continue
  require(s[key]==dict(path=p,sha256=h),'Frozen closure/prefix/observer/event source differs');e.bound(s[key])
 require(s['unit'].startswith('4dsr-footprint-') and s['unit'].endswith('.service') and '/' not in s['unit'],'One root-owned persistent service required')
 require(s['plan']['path'].startswith(OUT+'/held_confirmation_parent_coarse_overlap/') and s['plan']['path'].endswith('/registered_plan.json'),'Fresh exact held registered plan namespace required')
 plan=read(e.bound(s['plan']));require(plan['schema']=='registered_frozen_native_parent_coarse_overlap_plan_v2' and (plan['scene'],plan['seed'])==(s['scene'],s['seed']) and plan['metadata_runtime']==dict(python='3.10.20',numpy_distribution='1.26.4') and plan['confirmation_authorization']==s['confirmation_authorization'] and plan['planned']['native_moment_forwards']==300*len(plan['data']['train_cameras']) and plan['planned']['directed_pairs']==300*len(plan['data']['train_cameras'])*(len(plan['data']['train_cameras'])-1) and plan['planned']['Adam_calls']==0,'Native frozen coarse v2 runtime/population differs')
 require(s['physical_lock']==OUT+'/locks/'+GPU+'.controller.lock' and s['physical_controller_lock']==local_scope.LOCK,'Original shared BASE physical controller lock required')
 require(s['native_inner_lock'].startswith(OUT+'/held_confirmation_parent_coarse_overlap/') and s['native_inner_lock']!=s['physical_lock'],'Separate private native inner lock required')
 require(s['mode'] in ('resource-fixture','prepare'),'No automatic scientific retry mode');output=e.path(s['adapter_output']);native=e.path(s['native_output'])
 require(output.is_relative_to(e.root/OUT/'held_confirmation_parent_coarse_overlap') and native.is_relative_to(e.root/OUT/'held_confirmation_parent_coarse_overlap') and not output.exists() and output!=native and not output.is_relative_to(native) and not native.is_relative_to(output),'Fresh separate owned outputs required')
 if s['mode']=='prepare':require(not native.exists(),'Do not redo an existing scientific output')
 auth=read(e.bound(s['authorization']));require(auth['schema']=='root_authorized_frozen_native_parent_coarse_overlap_v1' and auth['plan']==s['plan'] and auth['producer_source']==s['producer_source'] and auth['manager_source']==s['manager_source'] and auth['resource_adapter']==s['resource_adapter'] and auth['closure_source']==s['closure_source'] and auth['unit']==s['unit'] and auth['runtime_root']==str(e.root) and auth['gpu_uuid']==GPU and auth['physical_gpu_lock']==str(e.path(s['native_inner_lock'])),'Real manager/resource/closure authorization differs')
 native_argv=[str(e.bound(s['producer_source'])),'--mode','prepare','--registered-plan',str(e.bound(s['plan'])),'--upstream',UP,'--out',str(native),'--gpu-uuid',GPU,'--lock',str(e.path(s['native_inner_lock'])),'--authorization',str(e.bound(s['authorization'])),'--confirmation-authorization',str(e.bound(s['confirmation_authorization'])),'--operator-resource-resolved','--cpu-threads','2']
 require(s['producer_argv']==native_argv,'Immutable producer args differ from canonical plan/output/parent')
 if s['mode']=='prepare':
  fixture=read(e.bound(s['actual_resource_fixture']));require(fixture['status']=='passed_actual_owned_local_RTX3090_strict_no_foreign_compute_resource_fixture_zero_model' and fixture['manager_source']==s['manager_source'] and fixture['resource_adapter']==s['resource_adapter'] and fixture['closure_source']==s['closure_source'] and fixture['producer_source']==s['producer_source'] and fixture['plan']==s['plan'] and fixture['GPU']==GPU and fixture['display_exemption']==GNOME and fixture['runtime_root']==str(e.root) and fixture['source_changed'] is False and fixture['model_dispatches']==fixture['native_moment_forwards']==fixture['Adam_calls']==0,'A real matching model-free source/resource fixture is required before prepare')
  root=read(e.bound(s['root_resource_fixture_acceptance']));require(root['status']=='root_accepted_exact_owned_local_RTX3090_resource_fixture_Exit0_no_model' and root['fixture']==s['actual_resource_fixture'] and root['manager_source']==s['manager_source'],'Actual exact fixture service Exit0 acceptance required')
  reg=read(e.bound(root['registration']));terminal=read(e.bound(root['terminal']));live=reg['systemd'];end=terminal['systemd'];own=fixture['ownership']
  require(reg['source']==s['manager_source'] and live['Id']==own['unit'] and live['InvocationID']==own['invocation_id'] and int(live['ExecMainPID'])==own['pid'] and reg['process']['starttime']==own['starttime'] and reg['process']['cgroup']==own['cgroup'] and reg['process']['cmdline']==own['cmdline'],'Original resource fixture owner differs')
  require(terminal['kind']=='terminal' and terminal['unit']==live['Id'] and terminal['invocation_id']==live['InvocationID'] and terminal['exit_proved'] is True and terminal['successful'] is True and terminal['exit_kind']==1 and terminal['exit_code']==0 and end['Id']==live['Id'] and end['MainPID']=='0' and end['ExecMainPID']==live['ExecMainPID'] and end['ExecMainStartTimestampMonotonic']==live['ExecMainStartTimestampMonotonic'] and int(end['ExecMainExitTimestampMonotonic'])>0 and end['ExecMainCode']=='1' and end['ExecMainStatus']=='0' and end['Result']=='success','Actual exact model-free fixture normal Exit0 absent')
  require(s['scientific_retry_authorized'] is False,'No implicit scientific retry');no_repeat(s,e)
 return plan,auth,output,native

def no_repeat(s,e):
 a=read(e.bound(s['root_no_repeat_authorization']))
 require(a['schema']=='root_authorized_pending_exact_held_local_RTX3090_coarse_target_no_previous_dispatch_v1' and a['root_execution_authorized'] is True and a['scene']==s['scene'] and a['seed']==s['seed'] and a['workspace']==s['runtime_root'] and a['unit']==s['unit'] and a['plan']==s['plan'] and a['manager_source']==s['manager_source'] and a['GPU']==GPU and a['scientific_retry'] is False,'Fresh explicit root target/no-repeat authorization absent')
 require(type(a['unix']) in (int,float) and math.isfinite(a['unix']) and 0<=time.time()-a['unix']<=300,'Root target authorization expired; it is not a GPU reservation')
 require(all(type(a[k]) is int and a[k]==0 for k in ('accepted_same_scene_seed','existing_v2_units_same_scene_seed','producer_intents_same_scene_seed')),'Already accepted/dispatched/intent coarse target cannot be redone')
 index=read(e.bound(a['current_coarse_index']))
 require(index['schema']=='root_actual_coarse_parent_no_repeat_index_v1' and index['status']=='root_registered_actual_current_coarse_scope','Root actual coarse no-repeat snapshot schema differs')
 for name in ('accepted','existing_v2_units'):
  require(type(index[name]) is list,'Missing actual coarse accepted/dispatch scope')
  for row in index[name]:
   require(type(row) is dict and isinstance(row.get('scene'),str) and type(row.get('seed')) is int,'Malformed root coarse ledger row')
   require((row['scene'],row['seed'])!=(s['scene'],s['seed']),'Original coarse target already accepted/dispatched; no implicit scientific retry')
 require(type(a['producer_intents']) is list and all(type(row) is dict and isinstance(row.get('scene'),str) and type(row.get('seed')) is int for row in a['producer_intents']),'Root pending producer intent scope absent')
 require(not any((row['scene'],row['seed'])==(s['scene'],s['seed']) for row in a['producer_intents']),'A previous producer intent exists; explicit incident recovery required')
 return a

def parse_xml(text):
 q=ET.fromstring(text);g=q.findall('gpu');require(len(g)==1,'Ambiguous GPU XML');g=g[0]
 v=dict(uuid=g.findtext('uuid'),name=g.findtext('product_name'),driver=q.findtext('driver_version'),memory_total=g.findtext('fb_memory_usage/total'),memory_used=g.findtext('fb_memory_usage/used'),utilization=g.findtext('utilization/gpu_util'),processes=[])
 for x in g.findall('processes/process_info'):v['processes'].append(dict(pid=int(x.findtext('pid')),type=x.findtext('type'),name=x.findtext('process_name'),used_memory=x.findtext('used_memory')))
 return v

def cgroup_record(raw):
 require(isinstance(raw,str) and re.fullmatch(r'0::/(?:[A-Za-z0-9_.@:-]+/)*[A-Za-z0-9_.@:-]+\n?',raw) and raw.count('\n')<=1,'Not one exact canonical cgroup v2 record')
 result=raw[:-1] if raw.endswith('\n') else raw
 require('..' not in Path(result.split(':',2)[-1]).parts,'Cgroup path escape')
 return result

def scalar(text,unit):
 match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*'+re.escape(unit),text or '');require(match is not None,'Missing/unknown NVIDIA numeric field');return float(match[1])
def resource(v,lookup,own,idle=False):
    require(v['uuid']==GPU and v['name']==local_scope.MODEL and v['driver'],'Different physical GPU/model/driver')
    total=scalar(v['memory_total'],'MiB');used=scalar(v['memory_used'],'MiB');util=scalar(v['utilization'],'%');require(total>0 and 0<=used<=total and 0<=util<=100,'NVIDIA memory/utilization invalid')
    if idle:require(used<=1024 and util==0,'Initial RTX3090 resource gate exceeds1024MiB/0percent')
    require(len({x['pid'] for x in v['processes']})==len(v['processes']),'Duplicate GPU process');graphics=[];current=[]
    for x in v['processes']:
        require(type(x['pid']) is int and x['pid']>0 and x['type'] in ('G','C','C+G'),'Unknown NVIDIA type/PID')
        if x['type']=='G':graphics.append(x);continue
        require(x['pid']==own['pid'] and x['type']=='C','Every foreign compute process blocks local RTX3090; no display compute exception')
        p=lookup(x['pid']);require(all(p[k]==own[k] for k in ('pid','starttime','exe','exe_sha256','cgroup','cmdline')),'Owned CUDA process identity changed')
        current.append(dict(process=x,identity=p))
    return dict(v,graphics_only=graphics,exact_display_exemptions=[],own_compute=current,foreign_compute=[],no_foreign_process_control=True,idle_gate_applied=idle)
class Backend:
 def process(self,pid):
  q=Path('/proc')/str(pid)
  def start():return q.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
  began=start();raw=q.joinpath('cgroup').read_text();p=dict(pid=pid,starttime=began,exe=os.readlink(q/'exe'),exe_sha256=sha(q/'exe'),cgroup=cgroup_record(raw),raw_cgroup=raw,cmdline=[os.fsdecode(x) for x in q.joinpath('cmdline').read_bytes().split(b'\0') if x]);require(start()==began,'Process changed while reading identity');return p
 def own(self,s,expected):
  p=self.process(os.getpid());require(p['cmdline']==expected and re.fullmatch(r'0::/user\.slice/[^\n\r ]+',p['cgroup']) and '..' not in Path(p['cgroup'].split(':',2)[-1]).parts and p['cgroup'].endswith('/'+s['unit']) and Path(p['exe']).resolve()==Path(PYTHON).resolve(),'Actual root-owned manager executable/argv/cgroup differs')
  require(re.fullmatch('[a-f0-9]{32}',os.environ.get('INVOCATION_ID','')),'Actual service invocation absent');return dict(p,unit=s['unit'],invocation_id=os.environ['INVOCATION_ID'])
 def sample(self):
  r=subprocess.run(['nvidia-smi','-i',GPU,'-q','-x'],capture_output=True,text=True,check=True);return dict(parse_xml(r.stdout),at_monotonic=time.monotonic(),at_unix=time.time(),host=socket.gethostname())
 def pause(self,n):time.sleep(n)
 def invoke(self,path,argv):
  before=sys.argv;sys.argv=argv
  try:return runpy.run_path(str(path),run_name='__main__')
  finally:sys.argv=before

def check_gpu(uuid):
 global CONTEXT
 require(CONTEXT is not None and uuid==GPU and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'Resource API context/UUID changed')
 e,backend,own,out,calls=CONTEXT;e.unchanged();value=resource(backend.sample(),backend.process,own);calls.append(value);new(out/('resource_API_call_'+str(len(calls)).zfill(4)+'.json'),value)
 return ', '.join(str(value[k]) for k in ('uuid','name','memory_total','memory_used','driver'))

def execute_under_physical_lock(s,e,source,sref,mode,backend=None):
 global CONTEXT
 require(mode==s['mode'],'CLI mode/spec differs');plan,auth,out,native=validate(s,e,source);require(sys.version.split()[0]=='3.10.20' and importlib.metadata.version('numpy')=='1.26.4','Native metadata runtime differs');require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('FOURDSR_ROOT')==str(e.root) and os.environ.get('FOURDSR_UPSTREAM')==UP and Path.cwd()==e.root,'Actual CUDA/root/upstream/cwd differs')
 producer=load('_exact_G1_CPU_coarse_producer',e.bound(s['producer_source']));producer.verify_plan(plan,Path(UP));sys.path.insert(0,str(e.path(s['prefix_source']['path']).parent));import full_prefix_registered as prefix
 require(prefix.PIN=='843d5ac636c37e4b611242287754f3d4ed150144' and Path(prefix.__file__).resolve()==e.bound(s['prefix_source']).resolve() and prefix.check_gpu.__code__.co_filename==str(e.bound(s['prefix_source'])),'Original shared prefix resource API changed')
 require(not any(x in sys.modules for x in ('torch','numpy','scene','gaussian_renderer')),'Resource setup imported model/tensors');require(auth['manager_command']==command(s,e,mode,sref),'Original actual wrapper argv authorization differs')
 backend=backend or Backend();own=backend.own(s,auth['manager_command']);out.mkdir(parents=True,exist_ok=False);original=prefix.check_gpu;calls=[];began=time.monotonic();started=False
 try:
  first=resource(backend.sample(),backend.process,own,idle=True);new(out/'resource_first.json',first);backend.pause(30);second=resource(backend.sample(),backend.process,own,idle=True);require(second['at_monotonic']-first['at_monotonic']>=30 and all(first[k]==second[k] for k in ('name','driver','memory_total','host')),'Actual double read interval/physical host differs');new(out/'resource_double.json',dict(first=first,second=second,passed=True));pre=resource(backend.sample(),backend.process,own,idle=True);new(out/'resource_prelaunch.json',pre);e.unchanged()
  if mode=='resource-fixture':
   value=dict(status='passed_actual_owned_local_RTX3090_strict_no_foreign_compute_resource_fixture_zero_model',manager_source=s['manager_source'],resource_adapter=s['resource_adapter'],closure_source=s['closure_source'],producer_source=s['producer_source'],plan=s['plan'],GPU=GPU,display_exemption=GNOME,runtime_root=str(e.root),spec=sref,ownership=own,double= e.entry(out/'resource_double.json'),prelaunch=e.entry(out/'resource_prelaunch.json'),model_dispatches=0,native_moment_forwards=0,Adam_calls=0,source_changed=False,wall_seconds=time.monotonic()-began);new(out/'resource_fixture_complete.json',value);return value
  closure=load('_unchanged_coarse_closure',e.bound(s['closure_source']));CONTEXT=(e,backend,own,out,calls);prefix.check_gpu=check_gpu;require(prefix.check_gpu.__code__.co_filename==str(source),'Actual resource API co_filename is not root-bound new source')
  new(out/'single_producer_intent.json',dict(producer_source=s['producer_source'],actual_manager_source=s['manager_source'],closure_source=s['closure_source'],resource_adapter=s['resource_adapter'],producer_argv=s['producer_argv'],ownership=own,source_bytes_unchanged=True,automatic_retry=False));started=True
  namespace=backend.invoke(e.bound(s['producer_source']),s['producer_argv']);tick=time.monotonic();accepted=closure.verify_output(native,e.bound(s['plan']));require(callable(namespace.get('validate_result')),'Original frozen typed closure absent');accepted['producer_typed_observer_and_native_closure']=namespace['validate_result'](plan,native/'complete.json');accepted['cost']['postcheck_CPU_wall_seconds']=time.monotonic()-tick;accepted.update(operational_manager_source=s['manager_source'],closure_source=s['closure_source'],resource_adapter=s['resource_adapter']);e.unchanged();new(native/'exit_chain_complete.json',accepted)
  result=dict(status='completed_owned_local_RTX3090_native_coarse_normal_return_full_SHA_declared_resource_adapter',manager_source=s['manager_source'],producer_source=s['producer_source'],closure_source=s['closure_source'],resource_adapter=s['resource_adapter'],exit_chain=e.entry(native/'exit_chain_complete.json'),plan=s['plan'],spec=sref,ownership=own,resource_API_calls=len(calls),wall_seconds=time.monotonic()-began,formal_updates=0,Adam_calls=0);new(out/'manager_complete.json',result);return result
 except BaseException as exc:
  new(out/'manager_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),producer_invoked=started,wall_seconds=time.monotonic()-began,automatic_retry=False,manager_source=s['manager_source'],spec=sref,formal_updates=0,Adam_calls=0));raise
 finally:prefix.check_gpu=original;CONTEXT=None

def execute(s,e,source,sref,mode,backend=None):
 # Native producer runs in this exact owning service, so the shared flock stays
 # held across every exception and all in-process CUDA cleanup. Its own original
 # lock is a separate private inner lock; no self-deadlock or lock substitution.
 validate(s,e,source)
 physical=Path(s['physical_controller_lock'])
 require(not any(p.is_symlink() for p in (physical,*physical.parents)),'Shared physical lock cannot follow a symlink')
 physical.parent.mkdir(parents=True,exist_ok=True)
 with physical.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return execute_under_physical_lock(s,e,source,sref,mode,backend)


def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha');p.add_argument('--mode',choices=('validate','resource-fixture','prepare'),required=True);a=p.parse_args();expected=a.spec_sha if a.mode=='validate' else os.environ.get('FOURDSR_RESOURCE_SPEC_SHA');require(expected and sha(a.spec)==expected and (a.mode=='validate' or a.spec_sha is None),'Actual spec bytes changed; runtime SHA must be root-bound ENV to avoid circular authorization');s=read(a.spec);e=Evidence(s['runtime_root']);sref=e.entry(a.spec);e.bound(sref)
 if a.mode=='validate':require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only validate boundary');validate(s,e,Path(__file__).resolve());print(json.dumps(dict(status='CPU_strict_no_foreign_compute_coarse_spec_ready_no_external_operation',GPU_queries=0,model_imports=False)));return
 print(json.dumps(execute(s,e,Path(__file__).resolve(),sref,a.mode),ensure_ascii=False))
if __name__=='__main__':main()
