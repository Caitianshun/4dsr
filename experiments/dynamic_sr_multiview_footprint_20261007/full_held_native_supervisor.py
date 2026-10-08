"""Root-owned zero-update held-native supervisor, never a service dispatcher.

Root supplies actual source-bound CPU plan and starts this service once. The
CPU owner holds the base physical lock through its exact private CUDA child,
true wait/reap, integrity and full output SHA. A separate retained parent-unit
Exit0/ownedempty event is required by full_held_confirmation_bridge before any
formal SR. No teacher/HR pixels, model selection, retry or global write occurs.
"""
from pathlib import Path
import argparse, fcntl, importlib.util, json, os, re, sys, time, traceback
import full_held_confirmation_contract_v2 as held

OUT='output/dynamic_sr_multiview_footprint_20261007'
GPU='GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b'
CORE='590f38a55da141de8a9617c66484d7cad1dbc36ba5ae5411bb11030972f73d2d'
RUNTIME='7704aa2142456d890eaf32104f0b62f1647b606f75c28c76b03a0bdc2305b580'
DISPLAY='e256267d9d07397325bc757b20fecc4f5b35d772b78b79bf7d8e1b5599bf0925'

def require(ok,message):held.require(ok,message)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def native_argv(s,e):
    a=s['assets'];cmd=[s['native_python'],'-u',str(e.bound(a['native_helper_source'])),'--mode','run','--confirmation-authorization',str(e.bound(a['confirmation_authorization'])),'--upstream',s['upstream'],'--out',str(e.path(s['native_out'])),'--gpu-uuid',GPU,'--lock',str(e.path(s['native_inner_lock'])),'--operator-resource-resolved','--cpu-threads','2']
    for ref in a['native_case_requests']:cmd += ['--case',str(e.bound(ref))]
    require(bool(a['native_case_requests']),'Real held cases required');return cmd

def validate(s,e,request):
    require(s['schema']=='root_exact_held_native_zero_update_supervisor_request_v1' and s['root_execution_authorized'] is True and s['scientific_retry'] is False,'Root exact original held native request required')
    require(s['supervisor_source']==e.entry(str(Path(__file__).resolve().relative_to(e.root))),'Actual supervisor source differs')
    a=s['assets'];authorization=held.validate_authorization(a['confirmation_authorization'])
    for key,name in (('native_helper_source','full_held_native_acceptance.py'),('contract_source','full_held_confirmation_contract_v2.py'),('operational_child_guard_source','full_held_operator_registered.py')):
        require(a[key]==held.entry(held.HERE/name),'Exact held native dependency differs');e.bound(a[key])
    for key,pin in (('core_source',CORE),('transport_runtime_source',RUNTIME),('display_guard_source',DISPLAY)):
        require(a[key]['sha256']==pin,'Original operational dependency differs');e.bound(a[key])
    require(s['GPU']==GPU and s['physical_controller_lock']=='/home/ubuntu/3DGS/4dsr/'+OUT+'/locks/'+GPU+'.controller.lock','Same base A100 GPU0 physical lock required')
    require(s['workspace']==str(e.root) and s['unit'].startswith('4dsr-footprint-') and s['unit'].endswith('.service'),'Exact isolated persistent service workspace/unit required')
    require(s['native_out'].startswith(OUT+'/operator_checks/held_confirmation_native_CUDA_acceptance/runs/') and s['out'].startswith(OUT+'/held_confirmation_native_supervision/'),'Fresh native/supervisor output namespaces required')
    paths=[e.path(s[k]) for k in ('native_out','out','native_inner_lock')]
    require(not any(x==y or x.is_relative_to(y) or y.is_relative_to(x) for i,x in enumerate(paths) for y in paths[i+1:]),'Separate native inner lock and output trees required')
    helper=load('_held_native_CPU_validator',e.bound(a['native_helper_source']));plan=e.read(a['native_plan'])
    require(plan['status']=='registered_real_parent_CUDA_acceptance_ready' and not plan['missing'] and plan['confirmation_authorization']==a['confirmation_authorization'] and plan['helper_source']==a['native_helper_source'],'Actual original held native CPU plan required')
    require(plan['Adam_calls']==plan['formal_updates']==0 and plan['planned_RGB']==plan['planned_moments']==3*len(plan['cases']) and not plan['prior_receipts'],'This unique supervisor accepts only new original same-held-parent cases')
    keys={(r['scene'],r['seed']) for r in plan['cases']}
    require(keys and keys<={(scene,20261007) for scene in held.HELD} and len(keys)==len(plan['cases']),'Exact held native case population required')
    require(all(authorization['cohort_training_platforms'][scene]==GPU for scene,z in keys),'Native parent cases must use their exact paired formal training platform')
    for case in plan['cases']:
        m=e.read(case['manifest']);held.held_parent_scope(m,case['parent'],20261007)
    return helper,plan

def own(s,legacy,expected):
    argv=legacy.argv_of(os.getpid());cg=Path('/proc/self/cgroup').read_text().strip();inv=os.environ.get('INVOCATION_ID','')
    require(argv==expected and re.fullmatch('[a-f0-9]{32}',inv) and cg.startswith('0::/user.slice/') and cg.endswith('/'+s['unit']),'Exact actual CPU owner argv/invocation/cgroup required')
    return dict(unit=s['unit'],invocation_id=inv,pid=os.getpid(),start_ticks=legacy.start_ticks(os.getpid()),owned_cgroup=cg[3:],command=argv,workspace=s['workspace'],source=s['supervisor_source'])

def child(s,e,request):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'Only owned native child receives UUID')
    owner=e.read(e.entry(s['out']+'/actual_CPU_owner.json'))
    require(os.getppid()==owner['pid'] and Path('/proc/self/cgroup').read_text().strip()=='0::'+owner['owned_cgroup'] and os.environ.get('INVOCATION_ID')==owner['invocation_id'],'Native wrapper must be this exact released private child')
    helper,plan=validate(s,e,request);guard=load('_held_native_exact_display',e.bound(s['assets']['display_guard_source']))
    require(guard.GPU==GPU,'Mature exact display guard UUID differs');prefix=helper.prefix;original=prefix.check_gpu;backend=guard.Backend();identity=backend.process(os.getpid());calls=0
    def check(uuid):
        nonlocal calls
        require(uuid==GPU,'Native wrong physical UUID');v=guard.resource(backend.sample(),backend.process,identity);calls+=1
        e.new(s['out']+'/actual_native_resource_API_'+str(calls)+'.json',v)
        return ', '.join(str(v[k]) for k in ('uuid','name','memory_total','memory_used','driver'))
    prefix.check_gpu=check;old=sys.argv;sys.argv=native_argv(s,e)[2:]
    try:return helper.main()
    finally:prefix.check_gpu=original;sys.argv=old

def supervise(s,e,request):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Native owner must remain CPU-only')
    helper,plan=validate(s,e,request);legacy=load('_held_native_runtime',e.bound(s['assets']['transport_runtime_source']));op=load('_held_native_actual_child_guard',e.bound(s['assets']['operational_child_guard_source']))
    expected=[s['supervisor_python'],'-B','-u',str(e.bound(s['supervisor_source'])),'--mode','supervise','--root',str(e.root),'--request',request['path'],'--request-sha',request['sha256']]
    owner=own(s,legacy,expected);require(not e.path(s['out']).exists() and not e.path(s['native_out']).exists(),'Existing native evidence protected; no repeat/retry')
    e.new(s['out']+'/actual_CPU_owner.json',owner);op.guard_original_children(legacy,e,s['out']+'/private_child_safety')
    guard=load('_held_native_mature_display',e.bound(s['assets']['display_guard_source']));backend=guard.Backend();identity=backend.process(os.getpid());physical=Path(s['physical_controller_lock'])
    require(not any(p.is_symlink() for p in (physical,*physical.parents)),'Shared physical lock cannot follow a symlink');physical.parent.mkdir(parents=True,exist_ok=True)
    with physical.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        first=guard.resource(backend.sample(),backend.process,identity,idle=True);e.new(s['out']+'/resource_first.json',first);backend.pause(30)
        second=guard.resource(backend.sample(),backend.process,identity,idle=True);pre=guard.resource(backend.sample(),backend.process,identity,idle=True)
        require(second['at_monotonic']-first['at_monotonic']>=30,'Separated actual resource reads required');e.new(s['out']+'/actual_double_resource_and_prelaunch.json',dict(first=first,second=second,prelaunch=pre))
        command=[s['native_python'],'-B','-u',str(e.bound(s['supervisor_source'])),'--mode','child','--root',str(e.root),'--request',request['path'],'--request-sha',request['sha256']]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,FOURDSR_ROOT=str(e.root),FOURDSR_UPSTREAM=s['upstream'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        result=legacy.scientific_child(command,env,e.path(s['out']+'/native_child.log'),e,s['supervisor_source']);exitref=e.new(s['out']+'/actual_private_child_exit.json',result)
        require(result['exit_code']==0 and result['private_process_group_reaped'] is True,'Native private child failed/leaked; retain failure, no retry')
        accept=e.entry(s['native_out']+'/acceptance.json');helper.validate_receipt(e.read(accept));require(e.read(e.entry(s['native_out']+'/request_plan.json'))==plan,'Native child exact CPU plan differs')
        files=[]
        for p in sorted(e.path(s['native_out']).rglob('*')):
            if p.is_file():files.append(dict(e.entry(str(p.relative_to(e.root))),bytes=p.stat().st_size))
        value=dict(status='completed_actual_held_native_private_child_Exit0_integrity_all_output_SHA',source=s['supervisor_source'],request=request,owner=owner,native_acceptance=accept,helper_source=s['assets']['native_helper_source'],child_exit=exitref,entries=files,all_output_bytes_SHA_verified=True,parent_service_terminal_pending=True,formal_updates=0,Adam_calls=0,scientific_dispatch=False)
        e.new(s['out']+'/complete.json',value);return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','supervise','child'),required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--request',required=True);p.add_argument('--request-sha',required=True);a=p.parse_args()
    core_path=a.root/OUT/'operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/production_core_v3.py';require(held.sha(core_path)==CORE,'Original integrity core source differs before import');core=load('_held_native_original_evidence',core_path);e=core.Evidence(a.root);request=dict(path=a.request,sha256=a.request_sha);s=e.read(request)
    try:
        if a.mode=='validate':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU validate must hide CUDA');validate(s,e,request);v=dict(status='passed_CPU_exact_held_native_supervisor_registration',GPU_calls=0,model_imports=0,formal_updates=0)
        elif a.mode=='child':return child(s,e,request)
        else:v=supervise(s,e,request)
    except BaseException as error:
        if a.mode!='validate':e.new(s['out']+'/failure_'+str(time.time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),source=s['supervisor_source'],request=request,scientific_retry=False,success_claimed=False))
        raise
    print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':main()
