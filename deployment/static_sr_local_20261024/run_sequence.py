"""Full Cook07 Sequence Matters teacher -> fresh Wu SR -> cam00 evaluation.

CPU planning only unless invoked inside the exact existing local GPU broker.
Never submits, retries, accesses remote hosts, deletes data, or changes methods.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_splatsure as shared

ROOT, EXP, RUNTIME = shared.ROOT, shared.EXP, shared.RUNTIME
CONFIG = ROOT/'output/static_sr_wu_20261022/prepared_final/Cook07/sequence_matters.json'
RECORDS = CONFIG.parent/'training_records.json'
CPU_PLAN = ROOT/'output/static_sr_wu_20261022/cpu_prior_plans/Cook07_full300/receipt.json'
UPSTREAM = ROOT/'tmp/static_sr_wu_20261022/upstream/sequence_matters'
WEIGHTS = ROOT/'weight/static_sr_wu_20261022/sequence_matters'
PSRT = WEIGHTS/'PSRT_Vimeo.pth'
SPYNET = WEIGHTS/'spynet_sintel_final-3d2a1287.pth'
MODELS = {str(PSRT): '37b7889aaf79f493f95473af05be43561f3b30d740691a2d4534ba09d653658d',
          str(SPYNET): '3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714'}
RUNS = ROOT/'output/static_sr_wu_20261024/local_sequence_runs/Cook07'
SCHEMA = 'static_sr_local_sequence_plan_v1'
GiB = shared.GiB
sha, identity, checked, read, write, now = shared.sha, shared.identity, shared.checked, shared.read, shared.write, shared.now


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_inputs():
    sys.path.insert(0, str(EXP))
    from common import training_manifest, validate_records_bundle
    from methods.sequence_matters import validate_sequence_plan, verify_upstream
    cfg = read(CONFIG)
    m, rows = training_manifest(checked(cfg['manifest']), verify_pixels=True)
    source = read(checked(cfg['source_configuration']))
    if (cfg.get('method') != 'sequence_matters' or cfg.get('degradation') != 'area4'
            or cfg['specification']['lr_steps'] != 0 or cfg['specification']['sr_steps'] != 30000
            or cfg['specification']['init_sr'] != 'fresh' or source['configuration'] != cfg['wu_configuration']
            or m['scene'] != 'cook_spinach' or cfg['scene'] != m['scene']
            or m['resolutions'] != {'hr':[1344,1008], 'lr':[336,252]}
            or m['frame_indices'] != list(range(300)) or len(rows) != 6000 or m['splits']['test'] != ['cam00']):
        raise ValueError('Only full Cook07 area4 Sequence Matters with fresh SR30000 is supported')
    records = read(RECORDS)
    validate_records_bundle(records, cfg['manifest'], m, rows)
    plan = read(CPU_PLAN)
    groups = validate_sequence_plan(plan, records, verify_upstream(UPSTREAM)/'vsr/utils_vsr.py')
    if len(groups) != 300:
        raise ValueError('CPU sequence plan must cover all 300 same-time groups')
    for path, expected in MODELS.items():
        checked(dict(path=path, sha256=expected))
    held = [o for o in m['observations'] if o['split']=='test' and o['camera_id']=='cam00']
    if len(held)!=300 or any(not (Path(m['_root'])/o['hr_path']).is_file() for o in held):
        raise ValueError('Complete cam00 evaluation inputs are absent')
    return cfg, m, records, plan


def validate_gate(kind, item, gpu_uuid):
    result = shared.validate_gate(kind, item, gpu_uuid, config=CONFIG)
    if kind == 'wiring':
        rows = [r for r in result['results'] if r['scene']=='Cook07' and r['method']=='sequence_matters']
        if len(rows)!=1 or rows[0]['status']!='passed':
            raise ValueError('Cook07 Sequence Matters wiring did not pass')
    return result


def stage_commands(run, launcher, teacher):
    run = Path(run)
    def base(script, *args):
        return [str(shared.BASE_PYTHON), '-s', str(launcher), '--mode', 'base', '--', str(EXP/script), *map(str,args)]
    common = ['--upstream',str(UPSTREAM),'--psrt-checkpoint',str(PSRT),'--spynet-checkpoint',str(SPYNET),
              '--sequence-plan',str(CPU_PLAN)]
    def node(name, argv, budget, cap=GiB):
        return dict(name=name, argv=argv, additional_bytes=budget, maximum_file_bytes=cap)
    return [
        node('teacher_acceptance', [str(teacher),str(EXP/'teacher_acceptance.py'),'--method','sequence_matters',
             '--records',str(RECORDS),'--frame','0','--out',str(run/'teacher_acceptance'),*common],
             20*shared.png_bound(1344,1008)+9*GiB//8),
        node('teacher_generate', [str(teacher),str(EXP/'methods/sequence_matters.py'),'--manifest',str(RECORDS),
             '--output',str(run/'teacher'),*common], 6000*shared.png_bound(1344,1008)+9*GiB//8),
        node('bundle', base('bundle_priors.py','--config',CONFIG,'--sr-receipt',run/'teacher/receipt.json',
             '--model',PSRT,'--model',SPYNET,'--output',run/'bundle.json'),9*GiB//8),
        node('bundle_gate',base('train_stage.py','--config',CONFIG,'--stage','sr','--prior-bundle',run/'bundle.json',
             '--check-only','--out',run/'bundle_gate'),9*GiB//8),
        node('sr_train_and_evaluate',base('train_stage.py','--config',CONFIG,'--stage','sr',
             '--prior-bundle',run/'bundle.json','--out',run/'sr'),9*GiB//2+300*shared.png_bound(1344,1008),2*GiB),
    ]


def required_paths(cfg, launcher, teacher):
    # Reuse only a read-only closure helper. No mutation of its globals or
    # running SplatSuRe entry is needed; its extra config is harmless provenance.
    paths = shared.required_paths('lr',cfg,launcher,teacher,None)
    if Path(teacher).absolute()!=RUNTIME/'sequence_python':
        raise ValueError('Use the frozen local Sequence Python shim')
    paths += [Path(__file__),CONFIG,RECORDS,CPU_PLAN,PSRT,SPYNET,Path(teacher),
              RUNTIME/'sequence_v2_acceptance.json',RUNTIME/'assets_and_plan_acceptance.json']
    m = read(checked(cfg['manifest']))
    paths.append(checked(cfg['manifest']).parent/m['initialization']['npz_path'])
    inventory=RUNTIME/'teacher_copied_files.json';paths.append(inventory)
    paths += [checked(dict(path=i['destination'],sha256=i['sha256'],bytes=i['bytes'])) for i in read(inventory)['files']]
    private=ROOT/'output/static_sr_wu_20261022/runtime/sequence_env/lib/python3.10/site-packages'
    paths += [p for p in sorted(private.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')]
    tracked=subprocess.check_output(['git','-C',str(UPSTREAM),'ls-files','-z']).decode().split('\0')
    paths += [UPSTREAM/p for p in tracked if p and (UPSTREAM/p).is_file()]
    return list(dict.fromkeys(paths))


def validate_teacher(receipt_path, records, cpu_plan, output):
    """Bind the complete producer to exact models, LR, ALS, and PNG bytes."""
    sys.path.insert(0,str(EXP))
    from methods.sequence_matters import UPSTREAM_COMMIT
    result=read(receipt_path)
    if (result.get('status')!='teacher_complete' or result.get('teacher_generated') is not True
            or result.get('manifest_sha256')!=canonical(records)
            or result.get('upstream_commit')!=UPSTREAM_COMMIT
            or result.get('source_sha256')!=sha(UPSTREAM/'vsr/utils_vsr.py')
            or result.get('weights')!=MODELS or result.get('groups')!=cpu_plan['groups']):
        raise ValueError('Teacher producer is incomplete or differs from exact inputs/models/ALS source')
    if result.get('weight_key_audit',{}).get('missing_parameters')!=[]:
        raise ValueError('PSRT model parameters were not completely loaded')
    targets=result.get('targets',{});hashes=result.get('target_sha256',{})
    keys={r['key'] for r in records['records']}
    if len(keys)!=len(records['records']) or set(targets)!=keys or set(hashes)!=keys:
        raise ValueError('Teacher has duplicate, missing, or held-out keys')
    from PIL import Image
    for key in sorted(keys):
        expected=Path(output)/(key+'.png')
        if targets[key]!=str(expected) or expected.resolve()!=expected.absolute():
            raise ValueError('Teacher target escaped the fresh formal output')
        checked(dict(path=targets[key],sha256=hashes[key]))
        with Image.open(expected) as image:
            if image.format!='PNG' or image.mode!='RGB' or image.size!=(1344,1008):
                raise ValueError('Teacher output grid or encoding differs')
    return identity(receipt_path)


def validate_fixture(run, gpu_uuid, records, cpu_plan):
    out=Path(run)/'teacher_acceptance';result=read(out/'acceptance.json')
    if (result.get('schema')!='static_sr_wu_teacher_acceptance_fixture_v1'
            or result.get('status')!='fixture_passed' or result.get('method')!='sequence_matters'
            or result.get('formal_prior_allowed') is not False or result.get('formal_parent_allowed') is not False
            or not result.get('teacher_generated') or not result.get('gpu_executed')):
        raise ValueError('A real Sequence fixture must pass before formal teacher generation')
    if (result.get('hostname')!=socket.gethostname()
            or shared.normalize_uuid(result.get('hardware',{}).get('gpu_uuid'))!=shared.normalize_uuid(gpu_uuid)):
        raise ValueError('Sequence fixture ran on a different host/GPU')
    source_ids=[result['wrapper'],result['registered_records'],result['original_sequence_plan'],
                result['fixture_records'],result['teacher_receipt'],*result['adapter_sources'],*result['model_identities']]
    for item in source_ids:checked(item)
    if (checked(result['wrapper']).resolve()!=(EXP/'teacher_acceptance.py').resolve()
            or checked(result['registered_records']).resolve()!=RECORDS.resolve()
            or checked(result['original_sequence_plan']).resolve()!=CPU_PLAN.resolve()
            or checked(result['fixture_records']).resolve()!=(out/'fixture_records.json').resolve()
            or checked(result['teacher_receipt']).resolve()!=(out/'teacher/receipt.json').resolve()
            or {str(checked(i).resolve()) for i in result['adapter_sources']}!={str((EXP/'methods/sequence_matters.py').resolve())}
            or result['registered_manifest_canonical_sha256']!=canonical(records)
            or {str(checked(i)):i['sha256'] for i in result['model_identities']}!=MODELS):
        raise ValueError('Sequence fixture source or model identities differ')
    group=cpu_plan['groups'][0]
    selected=[r for r in records['records'] if r['frame']==0]
    if result['selected_keys']!=[r['key'] for r in selected] or len(selected)!=20:
        raise ValueError('Fixture must include every frame-zero training camera')
    fixture=read(out/'fixture_records.json')
    if fixture['records']!=selected or result['fixture_manifest_canonical_sha256']!=canonical(fixture):
        raise ValueError('Fixture camera/time/LR records differ')
    calls=result.get('forward_calls',[])
    if len(calls)!=len(group['sequences']):raise ValueError('Fixture omitted an ALS forward')
    for call,sequence in zip(calls,group['sequences']):
        keys=[group['keys'][i] for i in sequence['indices']];length=len(keys)
        if (call['keys']!=keys or call['input_shape']!=[1,length,3,252,336]
                or call['output_shape']!=[1,length,3,1008,1344] or call['finite_tensors_checked']<1
                or not math.isfinite(call['elapsed_seconds']) or call['elapsed_seconds']<=0):
            raise ValueError('Fixture input order, finite output, dimensions, or timing differs')
    outputs=result.get('outputs',[])
    if len(outputs)!=20 or {x['key'] for x in outputs}!=set(group['keys']):
        raise ValueError('Fixture output coverage differs')
    for item in outputs:
        if item['size_wh']!=[1344,1008]:raise ValueError('Fixture output dimensions differ')
        if checked(item).resolve()!=(out/'teacher'/(item['key']+'.png')).resolve():
            raise ValueError('Fixture output differs from its teacher producer')
    validate_teacher(out/'teacher/receipt.json',fixture,read(out/'fixture_plan.json'),out/'teacher')
    return result


def build_plan(args):
    cfg,m,records,cpu_plan=check_inputs()
    jid=shared.job_id(args.run_id,'sequence');run=RUNS/args.run_id
    if run.exists() or any(p.is_symlink() for p in [run,*run.parents]):raise FileExistsError('Use a fresh run root')
    launcher,teacher=Path(args.base_launcher).absolute(),Path(args.teacher_python).absolute()
    gates={'wiring':identity(args.wiring_receipt),'metrics':identity(args.metric_receipt)}
    for kind,item in gates.items():validate_gate(kind,item,args.gpu_uuid)
    commands=stage_commands(run,launcher,teacher);block=os.statvfs(ROOT).f_frsize
    if block>4096:raise ValueError('PNG allocation budget requires filesystem blocks <=4096 B')
    required=[identity(p) for p in required_paths(cfg,launcher,teacher)]
    reserve=20*GiB;needed=reserve+sum(n['additional_bytes'] for n in commands);free=shutil.disk_usage(ROOT).free
    return dict(schema=SCHEMA,status='CPU_plan_ready_requires_broker' if free>=needed else 'blocked_storage',
        run_id=args.run_id,run_root=str(run),job_id=jid,host=socket.gethostname(),gpu_uuid=args.gpu_uuid,
        config=identity(CONFIG),records=identity(RECORDS),CPU_plan=identity(CPU_PLAN),gates=gates,commands=commands,
        required_files=required,execution_paths=dict(base_launcher=str(launcher),teacher_python=str(teacher)),
        storage=dict(reserve_bytes=reserve,required_free_bytes=needed,observed_free_bytes=free,allocation_unit=block),
        teacher_images=6000,teacher_groups=300,ALS_sequences=sum(len(g['sequences']) for g in cpu_plan['groups']),
        formal_iterations=30000,optimizer_updates=29999,fresh_SR=True,evaluation_frames=300,created_utc=now(),
        automatic_submission=False,automatic_replay=False,asset_deletion=False,remote_access=False)


def validate_plan(plan):
    if plan.get('schema')!=SCHEMA or plan.get('status')!='CPU_plan_ready_requires_broker':raise ValueError('Plan is not executable')
    if plan['job_id']!=shared.job_id(plan['run_id'],'sequence'):raise ValueError('Job identity changed')
    run=RUNS/plan['run_id']
    if plan['run_root']!=str(run) or any(p.is_symlink() for p in [run,*run.parents]):raise ValueError('Run escaped scope')
    paths=plan['execution_paths']
    if (plan['commands']!=stage_commands(run,paths['base_launcher'],paths['teacher_python'])
            or plan['storage']['reserve_bytes']!=20*GiB or set(plan['gates'])!={'wiring','metrics'}):
        raise ValueError('Fixed production commands, gates, or storage reserve changed')
    if plan['config']!=identity(CONFIG) or plan['records']!=identity(RECORDS) or plan['CPU_plan']!=identity(CPU_PLAN):
        raise ValueError('Protocol registration changed')
    required={str(p.absolute()) for p in required_paths(read(CONFIG),paths['base_launcher'],paths['teacher_python'])}
    if not required.issubset({i['path'] for i in plan['required_files']}):raise ValueError('Source closure is incomplete')
    return run


def run_plan(path):
    path=Path(path).absolute();plan=read(path);run=validate_plan(plan)
    broker=shared.admission(plan);shared.verify_frozen(plan)
    for kind,item in plan['gates'].items():validate_gate(kind,item,plan['gpu_uuid'])
    cfg,m,records,cpu_plan=check_inputs()
    run.mkdir(parents=True,exist_ok=False);out=run/'execution';out.mkdir()
    write(out/'claim.json',dict(plan=identity(path),pid=os.getpid(),broker=broker),new=True)
    record=dict(schema='static_sr_local_sequence_completion_v1',status='running',job_id=plan['job_id'],
        run_root=str(run),config=plan['config'],plan=identity(path),broker=broker,started_utc=now(),stages=[],artifacts=[])
    started=time.monotonic();error=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg')
    old={s:signal.signal(s,shared.signal_failure) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for index,node in enumerate(plan['commands']):
            shared.verify_frozen(plan)
            space=shared.storage_guard(plan['commands'],index,plan['storage']['reserve_bytes'])
            stage=dict(name=node['name'],argv=node['argv'],storage=space,started_utc=now());record['stages'].append(stage)
            write(out/'status.json',record);before=time.monotonic();log=out/(node['name']+'.log')
            with log.open('x') as stream:
                child=subprocess.run(node['argv'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda:shared.file_limit(node['maximum_file_bytes']))
            stage.update(returncode=child.returncode,seconds=time.monotonic()-before,log=identity(log))
            write(out/'status.json',record)
            if child.returncode:raise RuntimeError(f"Stage {node['name']} failed with code {child.returncode}; no automatic retry")
            if node['name']=='teacher_acceptance':
                fixture=validate_fixture(run,plan['gpu_uuid'],records,cpu_plan)
                stage['acceptance']=identity(run/'teacher_acceptance/acceptance.json')
                stage['hardware']=fixture['hardware'];stage['inference_seconds']=fixture['inference_seconds'];stage['memory']=fixture['memory']
            elif node['name']=='teacher_generate':
                stage['producer']=validate_teacher(run/'teacher/receipt.json',records,cpu_plan,run/'teacher')
                stage['images']=6000
        complete=read(run/'sr/complete.json');checkpoint=run/'sr/checkpoint.pt'
        if (complete.get('status')!='completed_with_evaluation' or complete.get('steps')!=30000
                or complete.get('wiring_fixture') or complete.get('stage')!='sr' or complete.get('training_keys')!=6000
                or checked(complete['checkpoint']).resolve()!=checkpoint.resolve()):
            raise ValueError('Formal fresh SR30000 and evaluation did not complete')
        record['artifacts']=[identity(run/'teacher_acceptance/acceptance.json'),identity(run/'teacher/receipt.json'),
            identity(run/'bundle.json'),identity(run/'sr/complete.json'),identity(checkpoint),
            shared.verify_metrics(run/'sr/evaluation/metrics.json',checkpoint)]
        shared.verify_frozen(plan)
        record.update(status='completed',seconds=time.monotonic()-started,finished_utc=now())
        write(out/'status.json',record);write(out/'complete.json',record,new=True)
    except BaseException:
        error=traceback.format_exc();write(out/'status.json',dict(record,status='failed',seconds=time.monotonic()-started,error=error));raise
    finally:
        for signum,handler in old.items():signal.signal(signum,handler)
        detail=dict(status='failed' if error else 'completed',error=error,stages=record['stages'],seconds=time.monotonic()-started)
        if error and record['stages']:
            log=out/(record['stages'][-1]['name']+'.log')
            if log.is_file():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read(65536)
                detail['failure_log']=dict(**identity(log),tail=tail.decode('utf-8','replace'),tail_truncated=log.stat().st_size>65536)
        write(out/'diagnostics.json',detail)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan');a.add_argument('--run-id',required=True);a.add_argument('--gpu-uuid',required=True)
    a.add_argument('--wiring-receipt',required=True);a.add_argument('--metric-receipt',required=True);a.add_argument('--out',required=True)
    a.add_argument('--base-launcher',default=RUNTIME/'launch_v2.py');a.add_argument('--teacher-python',default=RUNTIME/'sequence_python')
    a=sub.add_parser('run');a.add_argument('--plan',required=True);args=p.parse_args()
    if args.action=='run':run_plan(args.plan);return
    result=build_plan(args);output=Path(args.out)
    if output.resolve().is_relative_to(Path(result['run_root']).resolve()):raise ValueError('Keep the plan outside its fresh run root')
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result,new=True)
    print(json.dumps(dict(status=result['status'],job_id=result['job_id'],plan=identity(output),storage=result['storage'])))


if __name__=='__main__':main()
