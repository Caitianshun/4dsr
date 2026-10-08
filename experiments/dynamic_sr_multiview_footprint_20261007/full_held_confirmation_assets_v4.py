"""CPU builders of actual held authorization, native cases, slots and roles.

No artifact is marked ready while teacher/native/coarse/schedule or actual actor
closure is missing. Machine templates supply only operational runtime bindings;
all scientific/input/source/quality references are rebuilt from actual gates.
"""
from pathlib import Path
import argparse,copy,json,os,sys,re,shlex
import full_held_confirmation_contract_v2 as held
import full_held_confirmation_bridge as bridge
import full_held_operator_registered_v2 as op

O='output/dynamic_sr_multiview_footprint_20261007'
G0='GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b'
G1='GPU-b79cd3fe-81f0-f449-30be-432e2857e517'
G1_GUARD=dict(path=O+'/operator_checks/VRHeadset07_A100_GPU1_CPU_operational_preparation_v1_20261008/source_snapshot/strict_A100_GPU1_native_resource_guard_v1.py',sha256='682dc9866270fdab3eecea9368f3f582d76d54e9c915bc436a449fe6e4725d90')
PLATFORMS=dict(coffee_martini=G1,flame_steak=G0)
DISPLAY=dict(path=O+'/native_full_parent_coarse_overlap_20261007/A100_GPU0_exact_display_manager_v1/source_snapshot/exact_display_coarse_manager.py',sha256='e256267d9d07397325bc757b20fecc4f5b35d772b78b79bf7d8e1b5599bf0925')
REMOTE=dict(path=O+'/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/remote_export_helper.py',sha256='7f2a46ecf46f32c62b8fa76c32ea4ad68644594bfc07cb21bbab61ccf47bd44b')
TRANSPORT=dict(path=O+'/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/runtime_adapter_v4.py',sha256='7704aa2142456d890eaf32104f0b62f1647b606f75c28c76b03a0bdc2305b580')
CORE=dict(path=O+'/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/production_core_v3.py',sha256='590f38a55da141de8a9617c66484d7cad1dbc36ba5ae5411bb11030972f73d2d')

def require(v,m):held.require(v,m)
def entry(p):return held.entry(p)
def write(p,v):bridge.write(p,v);return entry(p)
def bootstrap(root):
 require(Path(root).resolve()==held.ROOT,'Exact project root required');held.bound(CORE)
 core=held.load('_held_actual_asset_core',held.bound(CORE));return core.Evidence(held.ROOT),core

def authorization(a):
 require(a.root_authorized,'Root must supply both actual returned teacher acceptances')
 value=dict(schema='root_exact_held_confirmation_source_aware_authorization_v1',root_execution_authorized=True,scenes=list(held.HELD),seed=20261007,methods=list(held.METHODS),scientific_source=held.NATIVE_SOURCE,formal_updates_per_endpoint=6000,HR_training_inputs_allowed=False,test_LR_training_inputs_allowed=False,final_method_freeze=entry(a.final_method_freeze),teacher_acceptances=dict(coffee_martini=entry(a.coffee_teacher_acceptance),flame_steak=entry(a.flame_teacher_acceptance)),cohort_training_platforms=copy.deepcopy(PLATFORMS),coarse_operational_sources={s:dict(producer=entry(held.HERE/'full_held_parent_overlap.py'),observer=entry(held.OUT/'native_full_parent_coarse_overlap_20261007/source_snapshot/root_coarse_exact_exit.py'),manager=entry(held.HERE/('full_held_coarse_manager_GPU1_v2.py' if PLATFORMS[s]==G1 else 'full_held_coarse_manager.py'))) for s in held.HELD})
 held.validate_freeze(value['final_method_freeze']);held.validate_teachers(value)
 ref=write(a.out,value);held.validate_authorization(ref);return ref

def slots(a):
 ar=entry(a.confirmation_authorization);auth=held.validate_authorization(ar)
 rows=[dict(task_key=s+'/20261007/'+m,scene=s,seed=20261007,method=m,status='planned_only',actual_units_started=False,actual_formal_updates=0,actual_quality_complete=False,training_platform=auth['cohort_training_platforms'][s],budget=dict(updates=6000,RGB=18000,moments=0,Adam=6000)) for s in held.HELD for m in held.METHODS]
 return write(a.out,dict(schema='root_exact_held_confirmation_four_pending_slots_v1',confirmation_authorization=ar,source=entry(__file__),entries=rows,previous_full16_index_unchanged=True,GPU_dispatch_allowed=False,actual_services=0))

def native_case(a):
 ar=entry(a.confirmation_authorization);held.validate_authorization(ar,a.scene,20261007)
 row=next(r for r in bridge.inputs() if r['scene']==a.scene)
 return write(a.out,dict(schema='private_real_parent_native_CUDA_acceptance_case_v1',manifest=row['manifest'],parent=row['parent'],parent_complete=row['parent_complete'],confirmation_authorization=ar,scope='exact original held seed20261007 parent; diagnostic zero Adam/formal updates'))

def quality(a):
 require(a.root_authorized,'Only root authorizes post-freeze read-only held HR evaluation')
 ar=entry(a.confirmation_authorization);auth=held.validate_authorization(ar,a.scene,20261007)
 v=dict(schema='root_exact_held_HR_quality_after_method_freeze_v1',root_execution_authorized=True,confirmation_authorization=ar,final_method_freeze=auth['final_method_freeze'],scene=a.scene,seed=20261007,method=a.method,HR_may_only_be_read_by_evaluator=True,used_for_selection_or_tuning=False)
 return write(a.out,v)

def operator_template(a):
 ar=entry(a.confirmation_authorization);auth=held.validate_authorization(ar,a.scene,20261007);gpu=PLATFORMS[a.scene];require(auth['cohort_training_platforms'][a.scene]==gpu,'Exact root-frozen CoffeeG1/FlameG0 paired platform required')
 closure_ref=entry(a.materialization_complete);closure=held.read(held.bound(closure_ref));p=held.read(held.bound(closure['scientific_plan']));require((closure['scene'],closure['seed'],closure['method'])==(a.scene,20261007,a.method) and closure['confirmation_authorization']==ar,'Exact actual c278 held materialization required')
 bridge.validate_native(closure['native_acceptance'],p['parent'],ar,closure['native_parent_terminal'])
 qr=entry(a.held_quality_authorization);held.validate_quality_authorization(qr,ar,a.scene,20261007,a.method)
 template=held.read(held.bound(entry(a.machine_template)))
 require(template['transport']=='a100-train' and template['GPU_model']=='NVIDIA A100-SXM4-40GB' and template['evaluation']['transport']=='local','Existing actual compatible machine runtime bindings required')
 require(a.manager_workspace.startswith('/home/ubuntu/3DGS/4dsr/execution_workspaces/held_confirmation/'),'Isolated root-declared A100 workspace required')
 protocol_ref=entry(a.evaluation_protocol);protocol=held.read(held.bound(protocol_ref));m=held.read(held.bound(p['data']['manifest']))
 require(protocol['scene']==a.scene and protocol['seed']==20261007 and protocol['role']=='held_confirmation' and len(protocol['test_keys'])==300 and len(protocol['train_teacher_diagnostics']['keys'])==300+4*(len(m['splits']['train'])-1),'Actual full held300 +364/376 original protocol required')
 require(protocol['sources'][held.NATIVE_SOURCE['path']]==held.NATIVE_SOURCE['sha256'],'Original c278 evaluation protocol required')
 registry=entry(a.pending_slots);op.validate_no_repeat(dict(scene=a.scene,seed=20261007,method=a.method,no_automatic_scientific_retry=True),held.read(held.bound(registry)))
 namespace=O+'/held_confirmation_runtime/'+a.scene+'/seed20261007/'+a.method+'/original_v1'
 label=a.scene+'-20261007-'+a.method+'-held-original-v1';train_out=O+'/full_SR_refinement/held_confirmation/'+a.scene+'/seed20261007/'+a.method+'/original_v1'
 require(train_out.startswith(O+'/full_SR_refinement/held_confirmation/'),'Scientific plan must have the final held original train_out registered before closure')
 ev=copy.deepcopy(template['evaluation']);ev['scientific_lock']=O+'/native_evaluation_locks/'+ev['GPU']+'.'+a.scene+'-20261007-held-original-v1.inner.lock'
 training=copy.deepcopy(template['training']);training['physical_controller_lock']='/home/ubuntu/3DGS/4dsr/'+O+'/locks/'+gpu+'.controller.lock'
 s=dict(schema=op.SCHEMA,status='pending_actual_held_two_role_closures',dispatch_enabled=False,synthetic_contract=False,scene=a.scene,seed=20261007,method=a.method,role='held_confirmation',GPU=gpu,GPU_model=template['GPU_model'],host=template['host'],workspace=a.manager_workspace,transport='a100-train',transport_python='/usr/bin/python3',task_id='full_'+label,no_automatic_scientific_retry=True,production_core_source=CORE,physical_lock=O+'/locks/'+gpu+'.controller.lock',display_exemption=(copy.deepcopy(template['display_exemption']) if gpu==G0 else None),training=training,evaluation=ev,manager_unit='4dsr-footprint-'+label+('-A100GPU0.service' if gpu==G0 else '-A100GPU1.service'),callback_unit='4dsr-footprint-'+label+'-uniformRTX3090.service',CPU_temporal=dict(required=True,runtime_python=ev['runtime_python'],source=entry(held.HERE/'full_held_temporal_diagnostics.py'),out=O+'/full_temporal_diagnostics/held_confirmation/'+a.scene+'/seed20261007/'+a.method),role_deployments={})
 s['sources']={k:copy.deepcopy(template['sources'][k]) for k in ('full_refine_registered.py','full_evaluate_registered.py','full_prefix_registered.py','completion.py')};s['sources']['full_refine_registered.py']=held.NATIVE_SOURCE
 for key,name in (('held_confirmation_contract','full_held_confirmation_contract_v2.py'),('held_confirmation_bridge','full_held_confirmation_bridge.py'),('held_native_acceptance','full_held_native_acceptance.py'),('held_schedule_materializer','full_held_native_schedule_materializer.py'),('held_parent_overlap','full_held_parent_overlap.py'),('held_pair_coordinator','full_held_pair_coordinator_v2.py')):s['sources'][key]=entry(held.HERE/name)
 s['paths']=dict(train_out=train_out,training_manager_out=namespace+'/training_manager',evaluation_out=O+'/full_evaluation/held_confirmation/'+a.scene+'/seed20261007/'+a.method,callback_out=namespace+'/callback',returned_out=namespace+'/returned')
 s['release_path']=dict(manager=a.manager_workspace+'/'+namespace+'/manager_release.json',callback=ev['workspace']+'/'+namespace+'/callback_release.json')
 assets=dict(plan=closure['scientific_plan'],held_materialization_complete=closure_ref,manifest=p['data']['manifest'],selection=p['dependencies']['selection'],schedule=p['dependencies']['schedule'],teacher=p['dependencies']['teacher'],native_acceptance=closure['native_acceptance'],native_parent_terminal=closure['native_parent_terminal'],evaluation_protocol=protocol_ref,uniform_evaluation_platform=copy.deepcopy(template['assets']['uniform_evaluation_platform']),root_endpoint_index=registry,remote_helper_source=REMOTE,display_guard_source=(DISPLAY if gpu==G0 else G1_GUARD),runtime_adapter_source=entry(held.HERE/'full_held_operator_registered_v2.py'),transport_runtime_source=TRANSPORT,confirmation_authorization=ar,held_quality_authorization=qr)
 for key in ('checkpoint','complete','plan','sidecar'):assets['parent_'+key]=p['parent'][key]
 for i,r in enumerate(held.read(held.bound(p['dependencies']['selection']))['short_window_evidence']):assets['selected_short_evidence_'+str(i)]=r
 s['assets']=assets;return write(a.out,s)

def inventories(a):
 e,core=bootstrap(a.root);s=e.read(entry(a.spec));p=e.read(s['assets']['plan']);out=held.path(a.out);require(not out.exists(),'Fresh CPU inventory namespace required');out.mkdir(parents=True)
 results={}
 for role in ('manager','callback'):
  refs=op.required_role_files(s,e,p,role)
  runtime=s['training'] if role=='manager' else s['evaluation'];external={str(Path(runtime['upstream'])/name):sha for name,sha in p['source_files']['upstream'].items()}
  external.update({r['path']:r['sha256'] for r in runtime['native_extensions']})
  if role=='callback':external.update({r['path']:r['sha256'] for r in op.metric_external(e.read(s['assets']['uniform_evaluation_platform']))})
  er=write(out/(role+'_external_files.json'),[dict(path=p,sha256=h) for p,h in sorted(external.items())]);ir=write(out/(role+'_registered_inventory.json'),dict(role=role,entries=refs,all_payload_SHA_not_yet_rechecked_here=True,actual_actor_close_required=True))
  workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace'];actor_out=str((out/(role+'_actual_CPU_all_SHA.json')).relative_to(held.ROOT))
  command=['env','CUDA_VISIBLE_DEVICES=',runtime['runtime_python'],'-u',workspace+'/'+s['assets']['runtime_adapter_source']['path'],'--mode','close-role','--role',role,'--root',workspace,'--spec',workspace+'/'+entry(a.spec)['path'],'--spec-sha',entry(a.spec)['sha256'],'--out',actor_out,'--min-free-bytes',str(a.min_free_bytes),'--external-files',workspace+'/'+er['path']]
  results[role]=dict(registered_inventory=ir,external_files=er,actual_closure_expected_path=actor_out,command=command)
 return write(out/'CPU_only_actor_closure_commands.json',dict(source=entry(__file__),spec=entry(a.spec),roles=results,services_created=0,dispatch_allowed=False))

def ready(a):
 e,core=bootstrap(a.root);candidate_ref=entry(a.spec);s=e.read(candidate_ref);require(s['status']=='pending_actual_held_two_role_closures' and s['dispatch_enabled'] is False,'Original CPU candidate only')
 held.validate_authorization(s['assets']['confirmation_authorization'],s['scene'],s['seed']);s['role_deployments']={role:e.read(entry(getattr(a,role+'-closure'.replace('-','_')))) for role in ('manager','callback')}
 for role,proof in s['role_deployments'].items():
  require(proof['schema']=='actual_native_full_v2_role_CPU_all_SHA' and proof['role']==role and proof['full_inventory_verified'] is True and proof['operator_source']==s['assets']['runtime_adapter_source'],'Actual unchanged whole actor CPU SHA proof required')
  require((proof['host'],proof['workspace'])==(s['host'],s['workspace']) if role=='manager' else (proof['host'],proof['workspace'])==(s['evaluation']['host'],s['evaluation']['workspace']),'Actor actual workspace/host differs')
  required={r['path']:r['sha256'] for r in op.required_role_files(s,e,e.read(s['assets']['plan']),role)};actual={r['path']:r['sha256'] for r in proof['verified_files']};require(actual==required,'Actual whole role inventory differs')
  require(proof['GPU_calls']==proof['formal_updates']==proof['model_imports']==proof['image_decodes']==0 and proof['observed_free_bytes']>=proof['min_free_bytes']>0,'Actor actual CPU/storage boundary differs')
 s['status']='root_registered_ready_operator_v2';s['dispatch_enabled']=True
 return write(a.out,s)

def launch_artifacts(a):
 require(a.root_authorized,'Only root authorizes the exact persistent pair service command')
 e,core=bootstrap(a.root);import full_held_pair_coordinator_v2 as pair
 plan_ref=entry(a.plan);q=e.read(plan_ref);pair.validate_plan(q,e,entry(held.HERE/'full_held_pair_coordinator_v2.py'))
 require(re.fullmatch(r'4dsr-footprint-held-[A-Za-z0-9-]+\.service',a.unit),'Fresh explicit root held coordinator unit required')
 out=held.path(a.out);require(not out.exists(),'Fresh immutable once script directory required');out.mkdir(parents=True)
 source=str(held.HERE/'full_held_pair_coordinator_v2.py')
 pre=['env','CUDA_VISIBLE_DEVICES=','/usr/bin/python3','-B','-u',source,'--mode','preflight','--root',str(held.ROOT),'--plan',plan_ref['path'],'--plan-sha',plan_ref['sha256']]
 run=['/usr/bin/python3','-B','-u',source,'--mode','run','--root',str(held.ROOT),'--plan',plan_ref['path'],'--plan-sha',plan_ref['sha256'],'--root-authorized']
 intent=str(out/'once_actual_service_registration_intent.json')
 guard=[entry(held.HERE/'full_held_pair_coordinator_v2.py'),plan_ref]
 py="import hashlib,json,pathlib,subprocess;refs="+repr(guard)+";root=pathlib.Path("+repr(str(held.ROOT))+");assert all(hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in refs);p=subprocess.run(['systemctl','--user','show',"+repr(a.unit)+",'-p','LoadState','-p','MainPID','-p','InvocationID'],capture_output=True,text=True);props=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x);assert props.get('LoadState')=='not-found' and props.get('MainPID','0')=='0' and not props.get('InvocationID');f=open("+repr(intent)+",'x');json.dump(dict(source=refs[0],plan=refs[1],root_only_once=True,services_started_by_this_guard=0),f);f.close()"
 start=['systemd-run','--user','--unit='+a.unit,'--property=Type=exec','--property=RemainAfterExit=yes','--property=Restart=no','--property=KillMode=control-group','--property=WorkingDirectory='+str(held.ROOT),'--setenv=CUDA_VISIBLE_DEVICES=',*run]
 script=out/'launch_root_only_held_pair_once.sh';script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(held.ROOT))+'\n'+shlex.join(pre)+'\n'+shlex.join(['/usr/bin/python3','-B','-c',py])+'\n'+shlex.join(start)+'\n');script.chmod(0o444)
 return write(out/'root_exact_pending_once_commands.json',dict(source=entry(__file__),coordinator=entry(held.HERE/'full_held_pair_coordinator_v2.py'),plan=plan_ref,once_script=entry(script),exact_CPU_preflight_argv=pre,root_only_systemd_run_argv=start,services_created_here=0,GPU_calls=0,all_actor_actual_CPU_validation_before_release=True,Bsync_requires_actual_whole_B0_callback=True))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('authorization','pending-slots','native-case','quality-authorization','operator-template','role-inventories','ready','launch-artifacts'),required=True);p.add_argument('--root',default=str(held.ROOT));p.add_argument('--root-authorized',action='store_true');p.add_argument('--scene',choices=held.HELD);p.add_argument('--method',choices=held.METHODS);p.add_argument('--out',required=True)
 for name in ('final-method-freeze','coffee-teacher-acceptance','flame-teacher-acceptance','confirmation-authorization','materialization-complete','held-quality-authorization','machine-template','manager-workspace','evaluation-protocol','pending-slots','spec','manager-closure','callback-closure','plan','unit'):p.add_argument('--'+name)
 p.add_argument('--min-free-bytes',type=int,default=30*1024**3);a=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU builder requires empty CUDA visibility')
 actions={'authorization':authorization,'pending-slots':slots,'native-case':native_case,'quality-authorization':quality,'operator-template':operator_template,'role-inventories':inventories,'ready':ready,'launch-artifacts':launch_artifacts};print(json.dumps(actions[a.mode](a),ensure_ascii=False))
if __name__=='__main__':main()
