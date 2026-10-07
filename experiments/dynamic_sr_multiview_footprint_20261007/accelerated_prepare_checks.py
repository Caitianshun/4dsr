"""CPU-only controller injection and cache-provenance checks; no GPU/SSH."""
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import accelerated_prepare as ap

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'output'/HERE.name


def cuda_path_review():
    path=HERE/'support_cache.py';tree=ast.parse(path.read_text())
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    loader=next(n for n in prepare.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    assert ast.literal_eval(loader.args.defaults[0])=='cpu'
    tau=next(n for n in prepare.body if isinstance(n,ast.If) and 'tau_path.exists()'==ast.unparse(n.test))
    calls=[n for n in ast.walk(ast.Module(body=tau.orelse,type_ignores=[])) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load']
    assert len(calls)==1 and len(calls[0].args)==1 and not calls[0].keywords
    packed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pack_mask')
    assert '.detach().cpu().numpy()' in ast.unparse(packed)
    assert 'weight_lr.cpu().numpy()' in ast.unparse(prepare)
    assert "weight_lr.new_tensor([0.25, 0.5, 0.75]).float()" in ast.unparse(prepare)
    return dict(source=ap.identity(path,ROOT),tau_numpy_always_CPU=True,mask_and_weight_numpy_CPU=True,
        quantile_probabilities_same_device=True,scalar_float_conversions_synchronize=True,
        GPU_execution_performed=False,method_source_changed=False)


def fixture(directory,failing=False):
    root=directory/'workspace';here=root/'experiments'/ap.EXPERIMENT;out=root/'output'/ap.EXPERIMENT
    here.mkdir(parents=True);(here/'support_cache.py').write_text('Frozen source fixture\n')
    snapshot=out/'preparation/source_snapshot/accelerated_prepare.py';snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes((HERE/'accelerated_prepare.py').read_bytes())
    frozen=out/'support/frozen';frozen.mkdir(parents=True)
    ap.write(frozen/'config.json',dict(source='unchanged formula fixture',edge_count=2))
    ap.write(frozen/'tau_registration.json',dict(tau_z=.001,config_sha256=ap.sha(frozen/'config.json')))
    def pair(name,seconds,kind='edges'):
        path=frozen/kind/(name+'.npz');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
        row=dict(path=str(path.relative_to(frozen)),sha256=ap.sha(path),frame=0)
        if kind=='edges':row.update(source='cam02',target='cam03',seconds=seconds)
        else:row.update(camera='cam02')
        ap.write(path.with_suffix('.json'),row);return row
    old=pair('cam02_to_cam03_0000',1.25);pair('cam02_0000',None,'observations')
    orphan=frozen/'edges/cam03_to_cam04_0000.npz';orphan.write_bytes(b'unclosed CPU bytes preserved')
    parent=out/'support/parent_moments_hr';parent.mkdir(parents=True);(parent/'raw_fixture.npz').write_bytes(b'closed parent fixture')
    ap.write(parent/'index.json',dict(status='completed_HR_parent_moments',parent_sha256='parent',manifest_sha256='manifest',
        entries=[dict(path='raw_fixture.npz') for _ in range(1140)]))
    ap.write(out/'protocol.json',dict(parent={'sha256':'parent'},manifest={'sha256':'manifest'}))
    fixture_path=out/'operator_checks/footprint_cuda/passed.json';ap.write(fixture_path,dict(status='passed_native_CUDA_X_gradient_fixture'))
    ap.write(fixture_path.parent/'latest.json',dict(status='passed_native_CUDA_X_gradient_fixture',receipt=ap.identity(fixture_path,root)))
    (out/'preparation/support_prepare.log').write_text('old CPU execution --mode prepare --device cpu\n')
    calls=[];GPU_checks=[];suite=SimpleNamespace(ROOT=root,HERE=here,OUT=out,write=ap.write,read=ap.read)
    suite.entry=lambda path:ap.identity(path,root);suite.verify_fixture=lambda:fixture_path
    suite.bound=lambda value:root/value['path']
    suite.stage_files=lambda:{str((here/'support_cache.py').relative_to(root)):ap.sha(here/'support_cache.py')}
    suite.plan=lambda:None
    def occupancy(gpu):GPU_checks.append(gpu);return dict(uuid=gpu,name='CPU mock allocation')
    suite.free_sample=occupancy
    suite.check_resource=lambda gpu,status:(occupancy(gpu),occupancy(gpu))[1]
    suite.environment=lambda gpu:dict(CUDA_VISIBLE_DEVICES=gpu)
    def child(command,log,env):
        # All children must be reached inside the real original controller lock.
        lock=out/'locks/GPU-test.controller.lock'
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise AssertionError('Controller released lock before child')
        calls.append(dict(command=command,log=str(log),env=env))
        if str(here/'support_cache.py') in command:
            assert command[command.index('--device')+1]=='cuda:0' and env['CUDA_VISIBLE_DEVICES']=='GPU-test'
            assert str(log).endswith('support_prepare_GPU_resume.log')
            status=ap.read(out/'workers/prepare_r1.json')
            assert status['status']=='preparing_frozen_directed_support_GPU_resume' and not status['gpu_not_used']
            if failing:
                orphan.write_bytes(b'partially returned GPU output')
                raise RuntimeError('injected owned GPU child failure; no real GPU')
            new=pair('cam03_to_cam04_0000',.125)
            ap.write(frozen/'index.json',dict(status='completed_frozen_X_support',compute_device='cuda:0',new_edges=1,entries=[old,new],seconds=.5))
        elif str(here/'calibrate.py') in command:
            assert env['CUDA_VISIBLE_DEVICES']=='GPU-test';ap.write(out/'calibration.json',dict(status='passed',source_files={}))
        elif str(here/'diagnostics.py') in command:
            assert env['CUDA_VISIBLE_DEVICES']=='' and '--reuse-parent' in command
        else:raise AssertionError('Fixture/export or unrelated command launched')
    suite.legacy=SimpleNamespace(call=child)
    # Execute the actual frozen controller's run/prepare/main source, with only
    # OS/GPU children mocked. This covers lock ownership and skip semantics.
    source=HERE/'run_suite.py';tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('run','prepare','main')]
    scope=suite.__dict__;scope.update(dict(fcntl=fcntl))
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(source),'exec'),scope)
    return suite,snapshot,root,out,calls,GPU_checks


def exercise(directory,failing=False):
    suite,snapshot,root,out,calls,checks=fixture(directory,failing)
    original=(suite.run,suite.stage_files,suite.prepare)
    before=ap.snapshot_cache(out/'support/frozen',root)
    hooks=ap.patch_controller(suite,snapshot,'GPU-test','CPU-test-python')
    try:
        a=SimpleNamespace(phase='prepare',gpu='GPU-test',python='CPU-test-python',operator_resource_resolved=True,repeat='1')
        try:suite.main(a)
        except RuntimeError as error:
            if not failing:raise
            assert 'injected owned' in str(error)
        else:assert not failing
        amendments=hooks['amendment_directory'];registration=ap.read(amendments/'before.json')
        assert registration['cache_before']==before
        assert registration['helper']['sha256']==ap.sha(HERE/'accelerated_prepare.py')
        assert registration['incomplete_artifacts_preserved'][0]['original']['sha256']==before['incomplete_artifacts'][0]['sha256']
        assert ap.read(out/'support/frozen/edges/cam02_to_cam03_0000.json')['seconds']==1.25
        if failing:
            result=ap.read(amendments/'support_failure.json')
            assert result['active_unreturned_support_operation_seconds'] is None
            assert len(calls)==1 and result['partial_changes']['edges']['new']==0
        else:
            result=ap.read(amendments/'support_complete.json')
            assert result['changes']['edges']['reused']==1 and result['changes']['edges']['new']==1
            assert result['prior_closed_edge_seconds_sum']==1.25 and result['new_GPU_edge_seconds_sum']==.125
            assert result['mixed_cache'] and result['interrupted_CPU_unreturned_operation_seconds'] is None
            complete=ap.read(out/'preparation/complete.json')
            assert complete['sources'][hooks['source_key']]==ap.sha(snapshot)
            assert len(calls)==3 and len(checks)==4 # double startup plus support/calibration GPU prelaunch
        # Even if invoked accidentally, fixture/export paths cannot be rerun.
        try:suite.run(['CPU-test-python',str(suite.HERE/'support_cache.py'),'--mode','export'],out/'forbidden.log',{})
        except ValueError:pass
        else:raise AssertionError('Parent re-export allowed')
        try:suite.run(['CPU-test-python',str(suite.HERE/'footprint_cuda_checks.py')],out/'forbidden.log',{})
        except ValueError:pass
        else:raise AssertionError('Native fixture rerun allowed')
        frozen_bytes=snapshot.read_bytes();snapshot.write_bytes(frozen_bytes+b'changed')
        try:suite.stage_files()
        except ValueError:pass
        else:raise AssertionError('Helper source mutation accepted')
        snapshot.write_bytes(frozen_bytes)
    finally:hooks['restore']()
    assert original==(suite.run,suite.stage_files,suite.prepare)
    return dict(failure_injected=failing,original_controller_lock_held=True,commands=calls,
        GPU_inventory_calls_mocked=len(checks),controller_restored=True,receipt=ap.identity(amendments/('support_failure.json' if failing else 'support_complete.json'),ROOT))


def main():
    started=time.monotonic();directory=OUT/'operator_checks/accelerated_prepare'/str(time.time_ns());directory.mkdir(parents=True)
    for name in ('accelerated_prepare.py','accelerated_prepare_checks.py'):(directory/name).write_bytes((HERE/name).read_bytes())
    result=dict(source={name:ap.identity(HERE/name,ROOT) for name in ('accelerated_prepare.py','accelerated_prepare_checks.py')},
        GPU_started=False,SSH_used=False,parameter_updates=0,native_moment_forwards=0)
    try:
        result['CUDA_path_source_review']=cuda_path_review()
        result['successful_injected_controller']=exercise(directory/'success')
        result['failed_injected_controller']=exercise(directory/'failure',True)
        result.update(status='passed_CPU_device_amendment_controller_cache_provenance_checks',CPU_seconds=time.monotonic()-started)
    except BaseException:
        result.update(status='failed_CPU_device_amendment_check_preserved',error=traceback.format_exc(),CPU_seconds=time.monotonic()-started)
        ap.write(directory/'receipt.json',result);print(json.dumps(result));raise
    ap.write(directory/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],receipt=str(directory/'receipt.json'),CPU_seconds=result['CPU_seconds'])),flush=True)


if __name__=='__main__':main()
