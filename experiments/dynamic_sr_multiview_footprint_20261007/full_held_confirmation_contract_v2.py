"""Exact post-development held scope. Metadata only; never freezes a method.

The original ten-parent registry is retained. A development view is an explicit
object subset, never a relabelled registry. No image, array or checkpoint bytes
are opened by this module. Production consumers perform their original full SHA
and native-state checks after this additional authorization gate.
"""
from pathlib import Path
import hashlib, importlib.util, json, math, os

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_multiview_footprint_20261007'
HELD=('coffee_martini','flame_steak')
DEVELOPMENT=('cook_spinach','cut_roasted_beef','meetroom_discussion','meetroom_vrheadset')
SEEDS=(20261007,20261008)
METHODS=('B0','Bsync')
GROUPS=('xyz','deformation','grid','f_dc','f_rest','opacity','scaling','rotation')
NATIVE_SOURCE=dict(path='experiments/dynamic_sr_multiview_footprint_20261007/full_refine_registered_v2.py',sha256='c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d')
HELD_INPUTS=dict(path='output/dynamic_sr_multiview_footprint_20261007/operator_checks/remaining_authorized_scope_CPU_consumer_v1_20261008/actual_CPU_consumer_v1/actual_two_unused_held_LR_parents_and_teacher_dependencies.json',sha256='69553752b8ee4b0aad8551dcd854220398bc2e729e29a6de549d78a2a10f0c17')
PIXELS={'.png','.jpg','.jpeg','.exr','.tif','.tiff','.npy','.npz','.pt','.pth'}
STEPS=6000

def require(ok,message):
    if not ok:raise ValueError(message)
def path(value):
    p=Path(value)
    if p.is_absolute():
        origin=Path(os.environ.get('FOURDSR_ORIGIN_ROOT',str(ROOT)))
        try:p=ROOT/p.relative_to(origin)
        except ValueError:pass
    else:p=ROOT/p
    require(p.resolve().is_relative_to(ROOT.resolve()) and not any(q.is_symlink() for q in (p,*p.parents)),'Escaped/symlinked held evidence')
    return p
def sha(value):
    p=path(value);require(p.suffix.lower() not in PIXELS,'Held metadata gate does not read payload bytes')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(value):
    p=path(value);return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def bound(reference):
    require(isinstance(reference,dict) and {'path','sha256'}<=set(reference),'Missing actual immutable metadata reference')
    p=path(reference['path']);require(sha(p)==reference['sha256'],'Held evidence bytes changed: '+str(p));return p
def read(value):return json.loads(path(value).read_text())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('path','sha256'))
def load(name,value):
    p=path(value);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def original_inputs():
    value=read(bound(HELD_INPUTS));require(value['schema']=='CPU_actual_held_input_dependencies_not_quality_authorization_v1','Original held input consumer differs')
    require(len(value['entries'])==2 and {r['scene'] for r in value['entries']}==set(HELD),'Exact two original held parents required')
    return value
def held_parent_scope(manifest,parent,seed):
    require(manifest['scene'] in HELD and manifest['role']=='held_confirmation' and seed==20261007,'Only the original two held seed07 parents are authorized')
    expected=next(r for r in original_inputs()['entries'] if r['scene']==manifest['scene'])
    require(same(parent['checkpoint'],expected['checkpoint']) and same(parent['complete'],expected['complete']) and same(parent['plan'],expected['parent_plan']),'Cross-parent or independent-seed substitution refused')
    for key,reference in (('complete',expected['complete']),('sidecar',expected['sidecar']),('parent_plan',expected['parent_plan']),('acceptance',expected['actual_parent_acceptance'])):bound(reference)
    cp=path(parent['checkpoint']['path']);require(cp.stat().st_size==expected['current_checkpoint_stat']['bytes'],'Accepted parent current SIZE differs')
    done=read(bound(expected['complete']));side=read(bound(expected['sidecar']));p=read(bound(expected['parent_plan']))
    require(done['status']=='completed_independent_full_author_LR_prefix' and done['scene']==p['scene']==manifest['scene'] and done['seed']==p['seed']==seed,'Incomplete/cross-domain native LR parent')
    require(all(done[k] is False for k in ('HR_used','heldout_pixels_used','teacher_used')),'Parent consumed forbidden supervision')
    require(done['state']==side['metadata']['state']==expected['actual_parent_state'] and parent['state']==expected['actual_parent_state'],'Full native17k parent state differs')
    require(side['metadata']['audit']['points']==parent['audit']['points']==expected['final_parent_points'],'Exact held parent point population differs')
    require(tuple(r['name'] for r in parent['audit']['optimizer_groups'])==GROUPS,'One native eight-group Adam is required')
    cams=manifest['splits']['train'];rows=[r for r in manifest['observations'] if r['split']=='train'];keys={(r['camera_id'],int(r['frame_index'])) for r in rows}
    require(manifest['splits']['test']==['cam00'] and not manifest['splits']['dev'] and 'cam00' not in cams and 'cam01' in cams and set(cams)==set(manifest['cameras'])-{'cam00'},'Official all-train/cam00 split required')
    require(manifest['frame_indices']==list(range(300)) and len(rows)==len(keys)==expected['legal_train_observations'] and keys=={(c,f) for c in cams for f in range(300)},'Incomplete full legal train population')
    require(manifest['resolutions']==dict(lr=[336,252],hr=[1344,1008]) and all(math.isfinite(float(r['time'])) and abs(float(r['time'])-int(r['frame_index'])/300)<1e-7 for r in rows),'Native grid/time convention differs')
    return expected
def development_parent_view(index):
    expected={(s,z) for s in DEVELOPMENT for z in SEEDS};rows=index['entries'];selected=[r for r in rows if (r['scene'],int(r['seed'])) in expected]
    require(len(rows)==10 and len(selected)==8 and {(r['scene'],int(r['seed'])) for r in selected}==expected,'Original ten-parent registry or exact eight-object subset differs')
    require(all(r['status']=='completed_independent_full_author_LR_prefix' for r in selected),'Unaccepted development parent')
    return dict(schema='explicit_original_eight_development_parent_object_view_v1',entries=selected,original_parent_registry_unchanged=True,view_only=True)
def validate_evaluation(reference,scene,seed,method):
    v=read(bound(reference));require(v['status']=='completed_full_native_test300_zero_update_evaluation' and v['test_observations']==300,'Incomplete full development evaluation')
    require(v['formal_updates']==v['Adam_calls']==v['backward_calls']==0 and v['model_RNG_unchanged'] is True,'Evaluation changed scientific state')
    identity=v['identity']['checkpoint'];p=read(bound(identity['plan']));require(identity['schema']=='registered_full_native_SR_refinement_v1' and identity['metadata']['cursor']==STEPS and (p['scene'],p['seed'],p['method'])==(scene,seed,method),'Evaluation does not bind this exact full endpoint')
    require(p['planned_cost']['SR_updates']==STEPS and p['planned_cost']['RGB_forwards']==3*STEPS and p['planned_cost']['moment_forwards']==0 and p['planned_cost']['Adam']==STEPS,'Full endpoint scientific budget differs')
    return v,p
def validate_freeze(reference):
    """All16 and paired evidence precede any held algorithm/data preparation."""
    value=read(bound(reference));require(value.get('status')=='root_frozen_single_main_configuration_before_confirmation_HR','Final root single-method freeze is pending')
    require(value.get('main_method')=='Bsync' and value.get('configuration_frozen_before_confirmation_HR') is True and value.get('confirmation_HR_used_for_selection') is False and value.get('development_results_complete') is True,'Confirmation cannot select, tune or silently change the original candidate')
    require(value.get('confirmation_scenes')==list(HELD) and value.get('root_actual_decision_reason'),'Exact held scope and root decision reason required')
    selection=read(bound(value['selection']));require(selection['status']=='registered_full_development_selection' and selection['selected_candidates']==['Bsync'] and selection['baseline_method']=='B0' and selection.get('necessary_ablation_methods',[])==[] and selection['uses_confirmation_for_selection'] is False,'Original B0/Bsync selection scope differs')
    for r in selection['short_window_evidence']:bound(r)
    cfg=read(bound(value['frozen_configuration']));require(cfg['main_method']=='Bsync' and cfg['baseline_method']=='B0' and same(cfg['scientific_source'],NATIVE_SOURCE),'Single source-aware frozen configuration differs')
    require(cfg['SR_updates']==STEPS and cfg['RGB_per_update']==3 and cfg['moment_forwards']==0 and cfg['native_Adam_calls_per_update']==1 and cfg['topology_clock']=='refinement' and cfg['SH_clock']=='fine14000+SRcursor' and cfg['SR_weights']==[0.,.05,.05] and cfg['regularizer_once'] is True,'Frozen native optimization budget/clock/loss changed')
    require(cfg.get('additional_methods',[])==[] and cfg.get('calibration_or_weight_sweep',False) is False,'No additional arms or tuning in confirmation')
    index=read(bound(value['full16_root_actual_index']));keys={f'{s}/{z}/{m}' for s in DEVELOPMENT for z in SEEDS for m in METHODS}
    require(len(index['entries'])==16 and {r['task_key'] for r in index['entries']}==keys and set(index['completed_task_keys_actual'])==keys,'All original16 independently completed quality endpoints must precede confirmation')
    require(index['full_native_verified_complete_endpoints']==index['full_native_training_complete_endpoints']==16 and index['full_native_formal_updates_accepted']==index['full_native_training_complete_updates']==96000,'Full16 quality/training budget not closed')
    pairs=read(bound(value['full8_source_grounded_pairs']));require(pairs['schema']=='existing_scalar_full_development_matrix_v2' and pairs['included_endpoints']==16 and pairs['included_pairs']==8 and pairs['method_selection_performed'] is False and pairs['held_quality_accessed'] is False,'Actual full8 paired scalar matrix required before selection')
    require({(r['scene'],int(r['seed'])) for r in pairs['paired_rows']}=={(s,z) for s in DEVELOPMENT for z in SEEDS},'Incomplete source-grounded paired development population')
    expected={f'{s}/{z}' for s in DEVELOPMENT for z in SEEDS};require(set(value['full_development_evidence'])==set(value['full_development_baseline_evidence'])==expected and set(value['full_development_chain_acceptance'])==keys,'Final method/baseline/true-chain evidence populations differ')
    for key in sorted(keys):
        scene,seed,method=key.split('/');target=value['full_development_baseline_evidence'] if method=='B0' else value['full_development_evidence'];validate_evaluation(target[scene+'/'+seed],scene,int(seed),method)
        bound(value['full_development_chain_acceptance'][key])
        row=next(r for r in index['entries'] if r['task_key']==key)
        candidates=[row[k] for k in ('completion','root_full_acceptance','actual_full_quality_acceptance','independent_full_quality_acceptance') if isinstance(row.get(k),dict)]
        require(any(same(value['full_development_chain_acceptance'][key],r) for r in candidates),'Freeze chain must reuse the actual authoritative source-bound endpoint receipt, not rename a historical schema')
    return value
def validate_teachers(authorization):
    refs=authorization['teacher_acceptances'];require(set(refs)==set(HELD),'Both actual frozen teachers are required')
    expected={r['scene']:r for r in original_inputs()['entries']}
    for scene,reference in refs.items():
        v=read(bound(reference));require(v['status']=='completed_full_teacher_CPU_acceptance' and v['teacher_images']==expected[scene]['legal_train_observations'] and v['CUDA_calls']==0 and v['HR_image_reads']==0 and v['GPU_inference_executed_by_this_program'] is False,'Actual whole original teacher acceptance remains pending')
        require(v['source']['sha256']=='c8692c441e0833c1fe54c690da1ff19efffbcd27a0ec5df78bc64228b72b1d4e','Original teacher acceptance source required');bound(v['source'])
        producer=read(bound(expected[scene]['teacher_CPU_plan']));require(same(v['producer_manifest'],producer['producer_manifest']),'Teacher acceptance must bind its original frozen producer manifest')
        teacher_index=OUT/'full_teacher_prepare'/scene/'teacher_index_seed20261007.json';ix=read(teacher_index)
        require(ix['status']=='completed_teacher_inventory' and ix['seed_consumer']==20261007 and ix['seed_independent_frozen_prior'] is True and len(ix['entries'])==expected[scene]['legal_train_observations'],'Actual complete seed07 teacher inventory missing')
        require(any(same(entry(teacher_index),r) for r in v['indexes']) and same(ix['plan'],expected[scene]['teacher_CPU_plan']),'Accepted teacher indexes or original plan differ')
    return True
def validate_authorization(reference,scene=None,seed=20261007):
    a=read(bound(reference));require(a['schema']=='root_exact_held_confirmation_source_aware_authorization_v1' and a['root_execution_authorized'] is True,'Root held authorization required')
    require(a['scenes']==list(HELD) and a['seed']==20261007 and a['methods']==list(METHODS),'Only exact two held seed07 B0/Bsync cohorts')
    require(a['scientific_source']==NATIVE_SOURCE and a['formal_updates_per_endpoint']==STEPS and a['HR_training_inputs_allowed'] is False and a['test_LR_training_inputs_allowed'] is False,'Held source/budget/information boundary differs')
    if scene is not None:require(scene in HELD and seed==20261007,'Out-of-scope confirmation task')
    validate_freeze(a['final_method_freeze']);validate_teachers(a);bound(NATIVE_SOURCE)
    require(set(a['cohort_training_platforms'])==set(HELD) and all(isinstance(v,str) and v.startswith('GPU-') for v in a['cohort_training_platforms'].values()),'Root must bind both exact paired physical training platforms')
    require(set(a['coarse_operational_sources'])==set(HELD),'Exact root-declared producer/observer/manager lineage required')
    for lineage in a['coarse_operational_sources'].values():
        require(set(lineage)=={'producer','observer','manager'},'Complete declared coarse lineage required')
        for reference in lineage.values():bound(reference)
    return a
def validate_quality_authorization(reference,confirmation_authorization,scene,seed,method):
    a=validate_authorization(confirmation_authorization,scene,seed);v=read(bound(reference))
    require(v['schema']=='root_exact_held_HR_quality_after_method_freeze_v1' and v['root_execution_authorized'] is True and v['confirmation_authorization']==confirmation_authorization and v['final_method_freeze']==a['final_method_freeze'],'Separately bound post-freeze HR evaluation authorization required')
    require((v['scene'],v['seed'],v['method'])==(scene,seed,method) and method in METHODS and v['HR_may_only_be_read_by_evaluator'] is True and v['used_for_selection_or_tuning'] is False,'Held evaluation boundary differs')
    return v


def metadata_dependencies(reference):
    """Exact small transitive gate files; never follow image/checkpoint inventories.

    Production role closure verifies their actual bytes. References embedded in
    a parent or teacher pixel inventory are deliberately not a recursive walk.
    """
    a=validate_authorization(reference);f=validate_freeze(a['final_method_freeze']);refs=[reference,HELD_INPUTS,NATIVE_SOURCE,a['final_method_freeze']]
    refs += [f[k] for k in ('selection','frozen_configuration','full16_root_actual_index','full8_source_grounded_pairs')]
    refs += read(bound(f['selection']))['short_window_evidence']
    refs += list(f['full_development_chain_acceptance'].values())
    for group in ('full_development_evidence','full_development_baseline_evidence'):
        for r in f[group].values():
            v=read(bound(r));refs += [r,v['identity']['checkpoint']['plan']]
    for r in original_inputs()['entries']:
        refs += [r[k] for k in ('complete','sidecar','parent_plan','actual_parent_acceptance','teacher_CPU_plan')]
        p=read(bound(r['parent_plan']));refs += [p['data']['manifest']]
        t=read(bound(r['teacher_CPU_plan']));refs += [t['producer_manifest']]
        v=read(bound(a['teacher_acceptances'][r['scene']]))
        refs += [a['teacher_acceptances'][r['scene']],v['source'],entry(OUT/'full_teacher_prepare'/r['scene']/'teacher_index_seed20261007.json')]
    refs += [v for d in a['coarse_operational_sources'].values() for v in d.values()]
    by={}
    for r in refs:
        bound(r);k=r['path'];require(k not in by or same(by[k],r),'Conflicting metadata path identities');by[k]={k:r[k] for k in ('path','sha256')}
    return list(by.values())
