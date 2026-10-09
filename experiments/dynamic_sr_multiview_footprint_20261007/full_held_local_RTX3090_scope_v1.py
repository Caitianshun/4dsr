"""Exact root-authorized execution-platform revision; metadata only.

The scientific method, native parents, teachers and four 6000-update slots are
unchanged. This scope is an operational local RTX3090 binding, not GPU dispatch
or native-parent acceptance. Original A100 authorization remains immutable.
"""
from pathlib import Path
import copy
import full_held_confirmation_contract_v2 as held

ROOT=held.ROOT
H='experiments/dynamic_sr_multiview_footprint_20261007'
O='output/dynamic_sr_multiview_footprint_20261007'
N=O+'/operator_checks/held_confirmation_source_aware_bridge_CPU_v1_20261008/private_native_coarse_factory_v1_20261008/local_RTX3090_exact_original_held_scope_CPU_v1_20261009'
PUB=N+'/final_local_operational_source_manifest_v1.json'
GPU='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec'
GPUS={s:GPU for s in held.HELD}
MODEL='NVIDIA GeForce RTX 3090'
PY='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
UP='/home/cai_tianshun/Project/4dgs'
LOCK=str(ROOT/O/'locks'/(GPU+'.controller.lock'))
GUARD=dict(path=O+'/operator_checks/VRHeadset08_local_RTX3090_CPU_preparation_v1_20261008/source_snapshot/strict_local_RTX3090_VR08_native_resource_guard_v1.py',sha256='8cb4c7ef0fcc00ed698ba4d5d1739301ac111e53d41e66fe3c5182e113f6c205')
ASSETS_NAME='full_held_confirmation_assets_local_RTX3090_v1.py'
OPERATOR_NAME='full_held_operator_registered_local_RTX3090_v1.py'
PAIR_NAME='full_held_pair_coordinator_local_RTX3090_v1.py'
NATIVE_NAME='full_held_native_supervisor_local_RTX3090_v1.py'
COARSE_NAME='full_held_coarse_manager_local_RTX3090_v1.py'
OLD_AUTH=dict(path=O+'/operator_checks/held_confirmation_source_aware_bridge_CPU_v1_20261008/private_native_coarse_factory_v1_20261008/root_actual_two_teachers_accepted_confirmation_authorization_v1.json',sha256='11b151b726d025512add22a5bdcbc3a59d130014808ad887ebe673c889a2c62d')
INTENT=dict(path=O+'/completion/root_actual_parallel_supervision_20261008/actual_hourly_unfinished_CPU_namespace_first_20261009_0858UTC/root_actual_authorized_local_3090_platform_preparation_intent_v35.json',sha256='7e8ce8ea0037ca775035d40fff46aec0e6e2f77c7a4f1a2ff164b2e4d39bf84c')
AUTH_PATH=N+'/root_actual_local_RTX3090_platform_revision_authorization_v1.json'

def revision_value(manager_reference):
    """Build only the authorized platform/source delta; never dispatch a job."""
    old=held.validate_authorization(OLD_AUTH)
    intent=held.read(held.bound(INTENT))
    held.require(intent['schema']=='root_authorized_unstarted_held_confirmation_local_platform_preparation_v35' and intent['preparation_root_authorized'] is True and intent['actual_GPU_dispatch_authorized_by_this_intent'] is False,'Actual root local preparation intent required')
    held.require(held.same(intent['source_previous_authorization'],OLD_AUTH) and held.same(intent['scientific_method_freeze'],old['final_method_freeze']),'Root prior authorization/freeze differs')
    held.require(intent['local_GPU']==GPU and intent['scenes']==list(held.HELD) and intent['arms']==list(held.METHODS) and intent['seed']==20261007 and intent['steps_per_arm']==6000 and intent['total_formal_budget']==24000,'Exact original local four-slot scope required')
    held.require(all(intent[k] is True for k in ('same_physical_GPU_for_both_arms_of_each_scene','one_existing_local_UUID_shared_lock_for_all_scene_native_coarse_train_and_uniform_evaluation','old_A100_authorization_and_failed_outputs_preserved','scientific_source_c278_and_method_freeze_unchanged','no_added_science_or_retraining')),'Root local scope preservation boundary absent')
    held.require(manager_reference==held.entry(held.HERE/COARSE_NAME),'Actual new local coarse source required')
    v=copy.deepcopy(old)
    v['cohort_training_platforms']=copy.deepcopy(GPUS)
    for scene in held.HELD:v['coarse_operational_sources'][scene]['manager']=manager_reference
    v['execution_platform_revision']=dict(schema='root_exact_original_held_local_RTX3090_execution_platform_revision_v1',root_execution_preparation_authorized=True,old_authorization=OLD_AUTH,root_preparation_intent=INTENT,same_scientific_configuration=True,formal_updates_added=0,GPU_dispatch_performed=False,transport='local',workspace=str(ROOT),runtime_python=PY,runtime_version='3.10.20',upstream=UP,physical_controller_lock=LOCK,guard_source=GUARD)
    return v

def validate_authorization(reference,scene=None,seed=20261007):
    """Retain every original freeze/teacher/parent gate, then check exact delta."""
    value=held.validate_authorization(reference,scene,seed)
    expected=revision_value(held.entry(held.HERE/COARSE_NAME))
    held.require(value==expected,'Local authorization changed an unapproved scientific or operational field')
    held.bound(GUARD)
    return value

def validate_runtime(root,python,upstream):
    held.require(Path(root).resolve()==ROOT and python==PY and upstream==UP,'Actual local project/native runtime binding differs')
    held.require(Path(PY).is_file() and Path(UP).is_dir(),'Previously accepted local 4dgs runtime is missing')
    return True
