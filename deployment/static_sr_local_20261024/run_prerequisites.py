"""Cook07 IE internal and 3DSR LR parents, each immediately evaluated.

Plan on CPU after a completed Sequence Matters run. Execute only through the
existing local PRO broker and launch_v2 base overlay. Never submits or retries.
These parents are preparation for later methods, not final SR comparisons.
"""
from __future__ import annotations
import argparse
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
import run_sequence as sequence

ROOT, EXP, RUNTIME, GiB = shared.ROOT, shared.EXP, shared.RUNTIME, shared.GiB
CONFIGS = {name: ROOT/'output/static_sr_wu_20261022/prepared_final/Cook07'/f'{name}.json'
           for name in ('iesrgs', 'threedsr')}
STAGES = {'iesrgs': 'internal', 'threedsr': 'lr'}
RUNS = ROOT/'output/static_sr_wu_20261024/local_prerequisite_runs/Cook07'
GPU = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
SCHEMA = 'static_sr_local_prerequisites_plan_v1'
sha, identity, checked, read, write, now = shared.sha, shared.identity, shared.checked, shared.read, shared.write, shared.now


def check_inputs():
    sys.path.insert(0, str(EXP))
    from train_stage import preflight
    from common import validate_records_bundle
    result = {}
    for method, path in CONFIGS.items():
        cfg = read(path); manifest, rows, _ = preflight(cfg, STAGES[method])
        if (cfg['method'] != method or cfg.get('degradation') != 'area4'
                or cfg['specification']['lr_steps'] != 30000
                or cfg.get('checkpoint') is not None or cfg.get('prior_bundle') is not None
                or cfg.get('initialization') != 'fresh_scene_local_train_LR_points'
                or manifest['scene'] != 'cook_spinach' or manifest['resolutions'] != {'hr':[1344,1008], 'lr':[336,252]}
                or manifest['frame_indices'] != list(range(300)) or len(rows) != 6000
                or manifest['splits']['test'] != ['cam00']):
            raise ValueError('Only fresh full Cook07 area4 IE internal / 3DSR LR30000 is supported')
        if method == 'iesrgs' and cfg['training']['internal_batch_views'] != 3:
            raise ValueError('IE internal same-time three-camera recipe changed')
        records = path.parent/'training_records.json'
        validate_records_bundle(read(records), cfg['manifest'], manifest, rows)
        held = [o for o in manifest['observations'] if o['split'] == 'test' and o['camera_id'] == 'cam00']
        if len(held) != 300 or any(not (Path(manifest['_root'])/o['hr_path']).is_file() for o in held):
            raise ValueError('Complete cam00 evaluation inputs are absent')
        result[method] = cfg
    if result['iesrgs']['manifest'] != result['threedsr']['manifest']:
        raise ValueError('The prerequisite parents use different input protocols')
    return result


def validate_gate(kind, item, gpu_uuid):
    result = shared.validate_gate(kind, item, gpu_uuid, config=CONFIGS['iesrgs'])
    if kind == 'wiring':
        for method in CONFIGS:
            rows = [r for r in result['results'] if r['scene'] == 'Cook07' and r['method'] == method]
            if len(rows) != 1 or rows[0]['status'] != 'passed':
                raise ValueError(f'Cook07 {method} wiring must pass')
    return result


def validate_sequence_completion(path, gpu_uuid):
    """Consume completed evidence; never wait for or launch the preceding job."""
    path = Path(path).absolute(); record = read(path)
    if record.get('schema') != 'static_sr_local_sequence_completion_v1' or record.get('status') != 'completed':
        raise ValueError('A successful full Sequence completion is required before planning')
    prior = read(checked(record['plan'])); run = sequence.validate_plan(prior)
    if (path != run/'execution/complete.json' or record.get('run_root') != str(run)
            or record.get('job_id') != prior['job_id'] or record.get('config') != identity(sequence.CONFIG)
            or prior['host'] != socket.gethostname() or prior['gpu_uuid'] != gpu_uuid):
        raise ValueError('Sequence completion does not bind the current host/GPU/full plan')
    proof = record.get('broker', {})
    if (proof.get('host') != prior['host'] or proof.get('CUDA_VISIBLE_DEVICES') != gpu_uuid
            or proof.get('cgroup') != '0::/claudevpn.slice/claude-local-'+prior['job_id']+'.service'):
        raise ValueError('Sequence broker completion differs')
    stages = record.get('stages', [])
    if ([s['name'] for s in stages] != [n['name'] for n in prior['commands']]
            or any(s.get('returncode') != 0 or not math.isfinite(s['seconds']) or s['seconds'] < 0 for s in stages)):
        raise ValueError('Sequence did not complete every production stage')
    for stage in stages: checked(stage['log'])
    shared.verify_frozen(prior)
    paths = [run/'teacher_acceptance/acceptance.json', run/'teacher/receipt.json', run/'bundle.json',
             run/'sr/complete.json', run/'sr/checkpoint.pt', run/'sr/evaluation/metrics.json']
    artifacts = record.get('artifacts', [])
    if len(artifacts) != len(paths) or {checked(x).resolve() for x in artifacts} != {p.resolve() for p in paths}:
        raise ValueError('Sequence completed artifact set differs')
    complete = read(run/'sr/complete.json')
    if (complete.get('status') != 'completed_with_evaluation' or complete.get('wiring_fixture')
            or complete.get('stage') != 'sr' or complete.get('steps') != 30000 or complete.get('training_keys') != 6000
            or checked(complete['checkpoint']).resolve() != (run/'sr/checkpoint.pt').resolve()
            or checked(complete['config']).resolve() != sequence.CONFIG.resolve()
            or complete.get('host') != prior['host']
            or shared.normalize_uuid(complete.get('gpu_uuid')) != shared.normalize_uuid(gpu_uuid)):
        raise ValueError('Sequence is not the complete formal same-host SR30000 model')
    shared.verify_metrics(run/'sr/evaluation/metrics.json', run/'sr/checkpoint.pt')
    return record


def stage_commands(run, launcher):
    run = Path(run); nodes = []
    def base(script, *args):
        return [str(shared.BASE_PYTHON), '-s', str(launcher), '--mode', 'base', '--', str(EXP/script), *map(str,args)]
    for method, config in CONFIGS.items():
        parent = run/method/STAGES[method]
        nodes += [dict(name=method+'_train', method=method, action='train',
            argv=base('train_stage.py','--config',config,'--stage',STAGES[method],'--out',parent),
            additional_bytes=9*GiB//2, maximum_file_bytes=2*GiB),
            dict(name=method+'_evaluate', method=method, action='evaluate',
            argv=base('render_eval.py','--config',config,'--checkpoint',parent/'checkpoint.pt','--out',run/method/'evaluation'),
            additional_bytes=300*shared.png_bound(1344,1008)+9*GiB//8, maximum_file_bytes=GiB)]
    return nodes


def required_paths(configs, launcher, gates, parent):
    paths = [Path(__file__), Path(sequence.__file__)]
    for method, cfg in configs.items():
        paths += shared.required_paths('lr',cfg,launcher,None,None)
        manifest_path = checked(cfg['manifest']); manifest = read(manifest_path)
        paths += [CONFIGS[method], CONFIGS[method].parent/'training_records.json',
                  manifest_path.parent/manifest['initialization']['npz_path']]
    # The actual Mip/RADE packages and receipts used by the already-passed gate.
    for method in CONFIGS:
        directory = ROOT/'output/static_sr_wu_20261022/runtime'/method
        paths += [p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.so','.json')]
    for item in gates.values(): paths.append(checked(item))
    wiring = read(checked(gates['wiring']))
    for extension in wiring['extensions'].values():
        paths += [checked(extension['wrapper']), *[checked(i) for i in extension['binaries']]]
    completion = read(checked(parent))
    paths += [checked(parent), checked(completion['plan']), *[checked(i) for i in completion['artifacts']]]
    return list(dict.fromkeys(paths))


def build_plan(args):
    if args.gpu_uuid != GPU: raise ValueError('This queue is for the registered local PRO GPU only')
    configs = check_inputs(); run = RUNS/args.run_id; jid = shared.job_id(args.run_id,'prereq')
    if run.exists() or any(p.is_symlink() for p in [run,*run.parents]): raise FileExistsError('Use a fresh run root')
    launcher = Path(args.base_launcher).absolute()
    if launcher != RUNTIME/'launch_v2.py': raise ValueError('Use the frozen base launch_v2 overlay')
    gates = {'wiring':identity(args.wiring_receipt), 'metrics':identity(args.metric_receipt)}
    for kind,item in gates.items(): validate_gate(kind,item,args.gpu_uuid)
    parent = identity(args.sequence_completion); validate_sequence_completion(checked(parent),args.gpu_uuid)
    nodes = stage_commands(run,launcher); block = os.statvfs(ROOT).f_frsize
    if block > 4096: raise ValueError('PNG allocation budget requires filesystem blocks <=4096 B')
    needed = 20*GiB + sum(n['additional_bytes'] for n in nodes); free = shutil.disk_usage(ROOT).free
    return dict(schema=SCHEMA,status='CPU_plan_ready_requires_broker' if free >= needed else 'blocked_storage',
        run_id=args.run_id,run_root=str(run),job_id=jid,host=socket.gethostname(),gpu_uuid=args.gpu_uuid,
        configs={k:identity(v) for k,v in CONFIGS.items()},sequence_completion=parent,gates=gates,
        execution_paths=dict(base_launcher=str(launcher)),commands=nodes,
        required_files=[identity(p) for p in required_paths(configs,launcher,gates,parent)],
        storage=dict(reserve_bytes=20*GiB,required_free_bytes=needed,observed_free_bytes=free,allocation_unit=block,
                     checkpoint_cap_is_failure_boundary_not_valid_truncation=True),
        created_utc=now(),iterations_per_parent=30000,optimizer_updates_per_parent=29999,
        train_observations=6000,evaluation_frames_per_parent=300,fresh_independent_parents=True,
        final_SR_results=False,teacher_generation=False,automatic_submission=False,automatic_replay=False,
        asset_deletion=False,remote_access=False)


def validate_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('status') != 'CPU_plan_ready_requires_broker': raise ValueError('Plan is not executable')
    if plan['gpu_uuid'] != GPU or plan['job_id'] != shared.job_id(plan['run_id'],'prereq'): raise ValueError('Job/GPU identity changed')
    run = RUNS/plan['run_id']; launcher = RUNTIME/'launch_v2.py'
    if plan['run_root'] != str(run) or any(p.is_symlink() for p in [run,*run.parents]): raise ValueError('Run escaped scope')
    if (plan['execution_paths'] != {'base_launcher':str(launcher)} or plan['commands'] != stage_commands(run,launcher)
            or plan['storage']['reserve_bytes'] != 20*GiB or set(plan['gates']) != {'wiring','metrics'}
            or plan['configs'] != {k:identity(v) for k,v in CONFIGS.items()}):
        raise ValueError('Fixed production recipe, configs, launchers or reserve changed')
    mandatory = {str(p.absolute()) for p in required_paths({k:read(v) for k,v in CONFIGS.items()},launcher,plan['gates'],plan['sequence_completion'])}
    if not mandatory.issubset({i['path'] for i in plan['required_files']}): raise ValueError('Source/input closure is incomplete')
    return run


def outer_argv(plan_path):
    return [str(shared.BASE_PYTHON),'-s',str(RUNTIME/'launch_v2.py'),'--mode','base','--',
            str(Path(__file__).absolute()),'run','--plan',str(Path(plan_path).absolute())]


def admission(plan, plan_path):
    proof = shared.admission(plan)
    actual = Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if os.environ.get('STATIC_SR_LOCAL_RUNTIME_MODE') != 'base' or actual != outer_argv(plan_path):
        raise ValueError('Outer process must use the exact frozen launch_v2 --mode base command')
    return dict(proof,outer_argv=actual)


def verify_parent(run, method, gpu_uuid):
    parent = Path(run)/method/STAGES[method]; result = read(parent/'complete.json')
    if (result.get('status') != 'stage_completed' or result.get('wiring_fixture') or result.get('steps') != 30000
            or result.get('stage') != STAGES[method] or result.get('training_keys') != 6000
            or result.get('prior_bundle') is not None or result.get('training_HR_read') is not False
            or result.get('heldout_pixels_read') is not False or result.get('host') != socket.gethostname()
            or shared.normalize_uuid(result.get('gpu_uuid')) != shared.normalize_uuid(gpu_uuid)
            or checked(result['config']).resolve() != CONFIGS[method].resolve()
            or checked(result['checkpoint']).resolve() != (parent/'checkpoint.pt').resolve()
            or not math.isfinite(result['seconds']) or result['seconds'] <= 0 or result['peak_bytes'] <= 0):
        raise ValueError(f'{method} is not a complete formal prerequisite parent')
    if (parent/'checkpoint.pt').stat().st_size > 2*GiB: raise ValueError('Checkpoint exceeded the registered failure boundary')
    return dict(completion=identity(parent/'complete.json'),checkpoint=identity(parent/'checkpoint.pt'),
                training_seconds=result['seconds'],peak_bytes=result['peak_bytes'],gpu_uuid=result['gpu_uuid'],gpu_name=result['gpu_name'])


def run_plan(path):
    path = Path(path).absolute(); plan = read(path); run = validate_plan(plan)
    broker = admission(plan,path); shared.verify_frozen(plan)
    for kind,item in plan['gates'].items(): validate_gate(kind,item,plan['gpu_uuid'])
    validate_sequence_completion(checked(plan['sequence_completion']),plan['gpu_uuid']); check_inputs()
    run.mkdir(parents=True,exist_ok=False); out = run/'execution'; out.mkdir()
    write(out/'claim.json',dict(plan=identity(path),pid=os.getpid(),broker=broker),new=True)
    record = dict(schema='static_sr_local_prerequisites_completion_v1',status='running',job_id=plan['job_id'],
        run_root=str(run),plan=identity(path),configs=plan['configs'],sequence_completion=plan['sequence_completion'],
        broker=broker,started_utc=now(),stages=[],artifacts=[],parents={},final_SR_results=False)
    started = time.monotonic(); error = None
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',
               IESRGS_RASTERIZER_ROOT=str(ROOT/'output/static_sr_wu_20261022/runtime/iesrgs'))
    old = {s:signal.signal(s,shared.signal_failure) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for index,node in enumerate(plan['commands']):
            shared.verify_frozen(plan)
            for item in record['artifacts']: checked(item)
            space = shared.storage_guard(plan['commands'],index,plan['storage']['reserve_bytes'])
            stage = dict(name=node['name'],argv=node['argv'],storage=space,started_utc=now()); record['stages'].append(stage)
            write(out/'status.json',record); before = time.monotonic(); log = out/(node['name']+'.log')
            with log.open('x') as stream:
                child = subprocess.run(node['argv'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,preexec_fn=lambda:shared.file_limit(node['maximum_file_bytes']))
            stage.update(returncode=child.returncode,seconds=time.monotonic()-before,log=identity(log)); write(out/'status.json',record)
            if child.returncode: raise RuntimeError(f"Stage {node['name']} failed with code {child.returncode}; no replay")
            method = node['method']
            if node['action'] == 'train':
                parent = verify_parent(run,method,plan['gpu_uuid']); record['parents'][method] = parent
                stage['cost'] = parent
                record['artifacts'] += [parent['completion'],parent['checkpoint']]
            else:
                checkpoint = run/method/STAGES[method]/'checkpoint.pt'
                metrics = shared.verify_metrics(run/method/'evaluation/metrics.json',checkpoint)
                record['parents'][method]['metrics'] = metrics; record['artifacts'].append(metrics)
        shared.verify_frozen(plan)
        for item in record['artifacts']: checked(item)
        record.update(status='completed',seconds=time.monotonic()-started,finished_utc=now())
        write(out/'status.json',record); write(out/'complete.json',record,new=True)
    except BaseException:
        error = traceback.format_exc(); write(out/'status.json',dict(record,status='failed',seconds=time.monotonic()-started,error=error)); raise
    finally:
        for signum,handler in old.items(): signal.signal(signum,handler)
        detail = dict(status='failed' if error else 'completed',error=error,stages=record['stages'],seconds=time.monotonic()-started)
        if error and record['stages']:
            log = out/(record['stages'][-1]['name']+'.log')
            if log.is_file():
                with log.open('rb') as stream: stream.seek(max(0,log.stat().st_size-65536)); tail = stream.read(65536)
                detail['failure_log'] = dict(**identity(log),tail=tail.decode('utf-8','replace'),tail_truncated=log.stat().st_size>65536)
        write(out/'diagnostics.json',detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('plan'); p.add_argument('--run-id',required=True); p.add_argument('--gpu-uuid',default=GPU)
    p.add_argument('--sequence-completion',required=True); p.add_argument('--wiring-receipt',required=True)
    p.add_argument('--metric-receipt',required=True); p.add_argument('--base-launcher',default=RUNTIME/'launch_v2.py'); p.add_argument('--out',required=True)
    p = sub.add_parser('run'); p.add_argument('--plan',required=True); args = parser.parse_args()
    if args.action == 'run': run_plan(args.plan); return
    result = build_plan(args); output = Path(args.out).absolute()
    if output.resolve().is_relative_to(Path(result['run_root']).resolve()): raise ValueError('Keep plan outside its fresh run root')
    output.parent.mkdir(parents=True,exist_ok=True); write(output,result,new=True)
    print(__import__('json').dumps(dict(status=result['status'],plan=identity(output),job_id=result['job_id'],storage=result['storage'],outer_argv=outer_argv(output))))


if __name__ == '__main__': main()
