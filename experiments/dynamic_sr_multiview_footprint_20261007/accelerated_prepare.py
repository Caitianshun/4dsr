"""Resume frozen support on CUDA through the unchanged preparation controller.

Install this exact file in workspace OUT/preparation/source_snapshot. The
controller retains its resource lock, double GPU check, native fixture/parent
reuse, single calibration and final diagnostics. Only the existing support
prepare CPU command and its truthful progress/log metadata are intercepted.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

EXPERIMENT='dynamic_sr_multiview_footprint_20261007'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(1048576),b''):value.update(data)
    return value.hexdigest()


def identity(path,root):
    path=Path(path);return dict(path=path.relative_to(root).as_posix(),sha256=sha(path))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def snapshot_cache(directory,root):
    """Only NPZ+JSON pairs with matching bytes qualify as completed reuse."""
    directory=Path(directory);edges=[];observations=[];incomplete=[]
    for kind,rows in (('edges',edges),('observations',observations)):
        folder=directory/kind
        names=sorted({p.stem for p in folder.glob('*.npz')} | {p.stem for p in folder.glob('*.json')})
        for name in names:
            npz=folder/(name+'.npz');sidecar=folder/(name+'.json')
            if not npz.is_file() or not sidecar.is_file():
                incomplete.extend(identity(path,root) for path in (npz,sidecar) if path.is_file());continue
            row=read(sidecar)
            if row['path']!=npz.relative_to(directory).as_posix() or sha(npz)!=row['sha256']:
                raise ValueError('Completed support cache pair changed/corrupt: '+str(npz))
            seconds=row.get('seconds')
            if seconds is not None and (not math.isfinite(float(seconds)) or float(seconds)<0):raise ValueError('Invalid cache duration')
            rows.append(dict(npz=identity(npz,root),receipt=identity(sidecar,root),seconds=seconds,
                key=[row['source'],row['target'],row['frame']] if kind=='edges' else [row['camera'],row['frame']]))
        incomplete.extend(identity(path,root) for path in sorted(folder.glob('*.tmp')) if path.is_file())
    return dict(edges=edges,observations=observations,edge_count=len(edges),observation_count=len(observations),
        completed_edge_seconds_sum=sum(float(e['seconds']) for e in edges if e['seconds'] is not None),
        edge_seconds_missing=sum(e['seconds'] is None for e in edges),incomplete_artifacts=incomplete,
        observation_generation_seconds='not independently recorded by original cache format')


def preserve_incomplete(before,root,destination):
    copied=[]
    for item in before['incomplete_artifacts']:
        source=root/item['path'];target=destination/item['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copy2(source,target)
        if sha(target)!=item['sha256']:raise ValueError('Incomplete artifact changed while preserving it')
        copied.append(dict(original=item,preserved=identity(target,root)))
    return copied


def compare_cache(before,after):
    def indexed(rows):return {row['npz']['path']:row for row in rows}
    result={}
    for kind in ('edges','observations'):
        old=indexed(before[kind]);new=indexed(after[kind])
        for path,row in old.items():
            if path not in new or new[path]!=row:raise ValueError('Previously completed CPU/GPU cache pair overwritten: '+path)
        created=[row for path,row in new.items() if path not in old]
        result[kind]=dict(reused=len(old),new=len(created),new_pairs=created,
            new_pair_seconds_sum=sum(float(row['seconds']) for row in created if row['seconds'] is not None))
    return result


def patch_controller(suite,helper_snapshot,gpu,python,interruption_receipt=None):
    """Scoped injectable adaptation; no frozen source file is written."""
    root=Path(suite.ROOT);out=Path(suite.OUT);helper_snapshot=Path(helper_snapshot)
    original_run=suite.run;original_stage=suite.stage_files;original_prepare=suite.prepare
    sources_at_import=original_stage();source_key=helper_snapshot.relative_to(root).as_posix();helper_sha=sha(helper_snapshot)
    amendment_directory=out/'preparation/device_amendments'/f'cuda_resume_{time.time_ns()}'
    holder=dict(context=None,intercepts=0)
    def stage_files():
        result=dict(original_stage())
        if sha(helper_snapshot)!=helper_sha:raise ValueError('Accelerated preparation helper bytes changed')
        result[source_key]=helper_sha
        return result
    def protected():
        names=('support/frozen/config.json','support/frozen/tau_registration.json','support/parent_moments_hr/index.json')
        return {name:identity(out/name,root) for name in names if (out/name).is_file()}
    def run(command,log,env,resource_gpu=None):
        command=list(map(str,command));support=str(suite.HERE/'support_cache.py')
        if support in command:
            if '--mode' not in command or command[command.index('--mode')+1]!='prepare':
                raise ValueError('Acceleration may not render/export parent moments')
            if '--device' not in command or command[command.index('--device')+1]!='cpu':
                raise ValueError('Only the registered CPU support command may be adapted')
            context=holder['context']
            if context is None or holder['intercepts']:raise ValueError('Unexpected duplicate/unregistered support dispatch')
            holder['intercepts']+=1;position=command.index('--device')+1;command[position]='cuda:0'
            environment=dict(env,CUDA_VISIBLE_DEVICES=gpu)
            gpu_log=out/'preparation/support_prepare_GPU_resume.log'
            tick=time.monotonic()
            context['support_command']=command;context['GPU_support_log']=str(gpu_log.relative_to(root))
            context['support_started_unix']=time.time();write(amendment_directory/'before.json',context)
            # Override the inherited literal CPU progress status immediately
            # before the GPU child, while retaining the original controller.
            suite.write(context['status_path'],dict(status='preparing_frozen_directed_support_GPU_resume',
                gpu=gpu,gpu_not_used=False,compute_device='cuda:0',CPU_closed_edges_reused=context['cache_before']['edge_count'],
                amendment=identity(amendment_directory/'before.json',root),sources=stage_files()))
            try:
                # The original run() checks the allocated GPU again and reaps
                # all private descendants before the controller lock releases.
                result=original_run(command,gpu_log,environment,gpu)
            except BaseException:
                after=snapshot_cache(out/'support/frozen',root)
                write(amendment_directory/'support_failure.json',dict(status='failed_GPU_support_resume_preserved',
                    before=identity(amendment_directory/'before.json',root),after=after,partial_changes=compare_cache(context['cache_before'],after),
                    child_wall_seconds=time.monotonic()-tick,error=traceback.format_exc(),
                    active_unreturned_support_operation_seconds=None,completed_pairs_preserved=True))
                raise
            index_path=out/'support/frozen/index.json';index=read(index_path)
            if index['status']!='completed_frozen_X_support' or index['compute_device']!='cuda:0':raise ValueError('GPU support child did not produce registered completed cache')
            after=snapshot_cache(index_path.parent,root);changes=compare_cache(context['cache_before'],after)
            if index['new_edges']!=changes['edges']['new'] or len(index['entries'])!=after['edge_count']:
                raise ValueError('Actual GPU new/reused edge counts differ from original cache index')
            if protected()!=context['protected_before']:raise ValueError('Formula registration, frozen tau or parent index changed')
            write(amendment_directory/'support_complete.json',dict(status='completed_GPU_support_resume_with_CPU_cache_reuse',
                scope='support cache device amendment only; calibration/diagnostics keep original independent cost receipts',
                before=identity(amendment_directory/'before.json',root),support_index=identity(index_path,root),changes=changes,
                current_generation_device='cuda:0',mixed_cache=index['compute_device']=='cuda:0' and changes['edges']['reused']>0,
                previous_closed_pair_device_origin=context['previous_closed_pair_device_origin'],
                prior_closed_edge_seconds_sum=context['cache_before']['completed_edge_seconds_sum'],
                new_GPU_edge_seconds_sum=changes['edges']['new_pair_seconds_sum'],
                GPU_child_wall_seconds=time.monotonic()-tick,original_index_current_invocation_seconds=index['seconds'],
                interrupted_CPU_unreturned_operation_seconds=None,CPU_interrupted_operation_cost='unknown, not estimated as zero',
                support_RGB_forwards=0,support_moment_forwards=0,parameter_updates=0,method_formulas_changed=False,
                compute_device_field_meaning='current invocation generated new edges on CUDA; existing pair seconds/bytes include reused prior device work',
                numerical_limit='No bitwise CPU-versus-CUDA equivalence claim; formulas, weights, masks, tau and thresholds source bytes remain unchanged.'))
            return result
        if str(suite.HERE/'footprint_cuda_checks.py') in command:
            raise ValueError('Already passed native fixture must be reused, not rerun')
        return original_run(command,log,env,resource_gpu)
    def prepare(a,status,env):
        # Called only inside original main(): controller lock is held and both
        # spaced GPU occupancy samples have completed.
        suite.verify_fixture()
        parent=read(out/'support/parent_moments_hr/index.json');protocol=read(out/'protocol.json')
        if parent['status']!='completed_HR_parent_moments' or len(parent['entries'])!=1140:
            raise ValueError('Complete closed 1140 HR parent cache required; acceleration never exports it')
        if parent['parent_sha256']!=protocol['parent']['sha256'] or parent['manifest_sha256']!=protocol['manifest']['sha256']:
            raise ValueError('Parent cache identity changed')
        for row in parent['entries']:
            if not (out/'support/parent_moments_hr'/row['path']).is_file():raise FileNotFoundError(row['path'])
        for name in ('config.json','tau_registration.json'):
            if not (out/'support/frozen'/name).is_file():
                raise ValueError('Acceleration requires the existing CPU frozen registration and tau: '+name)
        before=snapshot_cache(out/'support/frozen',root)
        earlier=list((out/'preparation/device_amendments').glob('*/before.json'))
        old_log=out/'preparation/support_prepare.log'
        origin='CPU_closed_before_first_device_amendment' if not earlier else 'mixed_or_prior_execution; see preceding amendment receipts'
        amendment_directory.mkdir(parents=True,exist_ok=False)
        context=dict(status='registered_support_compute_device_amendment',helper=identity(helper_snapshot,root),
            original_sources=sources_at_import,amended_sources=stage_files(),status_path=str(status),gpu=gpu,
            from_command_device='cpu',to_command_device='cuda:0',cache_before=before,protected_before=protected(),
            previous_closed_pair_device_origin=origin,prior_CPU_support_log=identity(old_log,root) if old_log.is_file() else None,
            earlier_amendments=[identity(path,root) for path in sorted(earlier)],
            explicit_root_interruption_receipt=identity(interruption_receipt,root) if interruption_receipt else None,
            CPU_interrupted_unreturned_operation_seconds=None,CPU_interrupted_operation_cost='unknown; root stop/exit receipt is authoritative',
            formula_source_unchanged=True,native_fixture_rerun=False,parent_moment_export_rerun=False,
            incomplete_artifacts_preserved=preserve_incomplete(before,root,amendment_directory/'incomplete_before'))
        holder['context']=context;write(amendment_directory/'before.json',context)
        try:
            result=original_prepare(a,status,env)
            if stage_files()!={**sources_at_import,source_key:helper_sha}:raise ValueError('Original preparation sources changed')
            write(amendment_directory/'complete.json',dict(status='completed_original_controller_preparation_after_device_amendment',
                preparation=identity(out/'preparation/complete.json',root),before=identity(amendment_directory/'before.json',root),
                support_amendment=identity(amendment_directory/'support_complete.json',root) if holder['intercepts'] else None,
                support_command_intercepts=holder['intercepts'],completed_cache_reused_without_dispatch=not bool(holder['intercepts']),
                calibration_receipt=identity(out/'calibration.json',root),parent_diagnostics_original=True,
                original_sources_unmodified=True,helper_source_key=source_key))
            return result
        except BaseException:
            write(amendment_directory/'preparation_failure.json',dict(status='failed_original_preparation_after_amendment_preserved',
                before=identity(amendment_directory/'before.json',root),error=traceback.format_exc(),
                support_command_intercepts=holder['intercepts'],cache_assets_not_deleted=True))
            raise
    suite.stage_files=stage_files;suite.run=run;suite.prepare=prepare
    def restore():suite.stage_files=original_stage;suite.run=original_run;suite.prepare=original_prepare
    return dict(restore=restore,holder=holder,amendment_directory=amendment_directory,source_key=source_key,helper_sha256=helper_sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--gpu',required=True);parser.add_argument('--python',required=True)
    parser.add_argument('--operator-resource-resolved',action='store_true');parser.add_argument('--interruption-receipt',type=Path)
    a=parser.parse_args();root=a.workspace.resolve();here=root/'experiments'/EXPERIMENT;out=root/'output'/EXPERIMENT
    snapshot=out/'preparation/source_snapshot/accelerated_prepare.py'
    if Path(__file__).resolve()!=snapshot.resolve():raise ValueError('Execute the exact registered output source_snapshot helper')
    if not a.operator_resource_resolved or not a.gpu.startswith('GPU-'):raise ValueError('Root must resolve queues and allocate the physical GPU UUID')
    if a.interruption_receipt:
        a.interruption_receipt=a.interruption_receipt.resolve();a.interruption_receipt.relative_to(root)
    sys.path.insert(0,str(here))
    spec=importlib.util.spec_from_file_location('accelerated_unchanged_suite',here/'run_suite.py')
    suite=importlib.util.module_from_spec(spec);sys.modules[spec.name]=suite;spec.loader.exec_module(suite)
    if Path(suite.ROOT).resolve()!=root or Path(suite.OUT).resolve()!=out:raise ValueError('Imported controller workspace differs')
    hooks=patch_controller(suite,snapshot,a.gpu,a.python,a.interruption_receipt)
    try:
        suite.main(SimpleNamespace(phase='prepare',gpu=a.gpu,python=a.python,operator_resource_resolved=True,
            repeat='1',task=None,evaluate_here=True))
    finally:hooks['restore']()


if __name__=='__main__':main()
