"""CPU plans and a serial, broker-admitted Cook07 SplatSuRe pipeline.

Never submits jobs, visits a remote host, deletes assets, or initializes CUDA.
The broker supplies persistence and exclusive GPU ownership. Production child
processes retain the existing method implementation and full author schedules.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/static_sr_wu_20261022'
RUNTIME = ROOT / 'output/static_sr_wu_20261024/local_runtime'
BASE_PYTHON = Path('/home/cai_tianshun/miniconda3/envs/sr4d/bin/python')
CONFIG = ROOT / 'output/static_sr_wu_20261022/prepared_final/Cook07/splatsure.json'
RUNS = ROOT / 'output/static_sr_wu_20261024/local_scene_runs/Cook07'
GiB = 1 << 30
SCHEMA = 'static_sr_local_splatsure_plan_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path).absolute()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def checked(item):
    p = Path(item['path'])
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file() or sha(p) != item['sha256']:
        raise ValueError('Frozen file differs: ' + str(p))
    if 'bytes' in item and p.stat().st_size != item['bytes']:
        raise ValueError('Frozen file size differs: ' + str(p))
    return p


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value, *, new=False):
    path = Path(path)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if new:
        with path.open('x') as f: f.write(data)
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w') as f: f.write(data)
        os.replace(tmp, path)


def job_id(run_id, phase):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', run_id):
        raise ValueError('Run ID must be 1..120 ASCII letters/digits/underscore/hyphen')
    # The exact original identifier is hashed: a_b/a-b and A/a cannot collide
    # through slug normalization. ID always satisfies the installed broker.
    digest = hashlib.sha256((run_id + '\0' + phase).encode()).hexdigest()[:16]
    slug = re.sub('[^a-z0-9-]', '-', run_id.lower()).strip('-')[:24] or 'run'
    result = 'ss-local-' + slug + '-' + phase + '-' + digest
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', result): raise ValueError('Broker ID overflow')
    return result


def normalize_uuid(value): return str(value).lower().removeprefix('gpu-')


def validate_gate(kind, item, gpu_uuid, *, config=CONFIG):
    """Actual local receipts only; historical CTS receipts cannot grant admission."""
    receipt = read(checked(item))
    expected = {
        'wiring': ('static_sr_wu_gpu_acceptance_v1', 'passed_bounded_GPU_wiring_only'),
        'metrics': ('static_sr_wu_metric_GPU_acceptance_v1', 'passed_GPU_metrics_and_LPIPS_API_parity_only'),
        'teacher': ('static_sr_wu_teacher_acceptance_fixture_v1', 'fixture_passed'),
    }[kind]
    if (receipt.get('schema'), receipt.get('status')) != expected:
        raise ValueError(kind + ': real completed acceptance is required')
    if not receipt.get('GPU_executed', receipt.get('gpu_executed', False)):
        raise ValueError(kind + ': no real GPU execution')
    actual = receipt.get('gpu_uuid', receipt.get('GPU_uuid', receipt.get('hardware', {}).get('gpu_uuid')))
    if normalize_uuid(actual) != normalize_uuid(gpu_uuid): raise ValueError(kind + ': wrong GPU')
    if receipt.get('host', receipt.get('hostname')) != socket.gethostname():
        raise ValueError(kind + ': acceptance belongs to another host')
    sources = []
    if kind == 'wiring':
        scene = [s for s in receipt['scenes'] if s['short'] == 'Cook07' and s['resolutions']['hr'] == [1344, 1008]]
        result = [r for r in receipt['results'] if r['scene'] == 'Cook07' and r['method'] == 'splatsure']
        if len(scene) != 1 or len(result) != 1 or result[0]['status'] != 'passed':
            raise ValueError('Cook07 SplatSuRe wiring did not pass')
        if scene[0]['manifest'] != read(config)['manifest']:
            raise ValueError('Wiring uses different Cook07 inputs')
        sources += [receipt['source'], *receipt['source_snapshot'].values()]
        if not receipt.get('all_extension_modules_coexisted'):
            raise ValueError('Rasterizer coexistence is unverified')
        for value in receipt['extensions'].values(): sources += [value['wrapper'], *value['binaries']]
    elif kind == 'metrics':
        eligible = {s['short'] for s in receipt['samples'] if [4*x for x in s['original_lr_wh']] == [1344, 1008]}
        results = receipt.get('results', [])
        if (not eligible or not receipt.get('frozen_parameters_and_buffers_unchanged')
                or not any(r['short'] in eligible for r in results)
                or any(r.get('status') != 'passed' or not r['convenience_class_parity']['passed'] for r in results)):
            raise ValueError('Metric parity/dimensions did not pass')
        assets = receipt['assets']
        sources += [receipt['source'], receipt['evaluator_source'], assets['source_receipt'],
                    assets['ssim_implementation'], *assets['implementations'], *assets['model_identities']]
    else:
        if receipt.get('method') != 'stablesr' or not receipt.get('teacher_generated'):
            raise ValueError('StableSR main teacher was not executed')
        outputs = receipt.get('outputs', [])
        if not outputs or any(o['size_wh'] != [1344, 1008] for o in outputs):
            raise ValueError('StableSR dimensions did not pass')
        sources += [receipt['wrapper'], receipt['registered_records'], *receipt['adapter_sources'],
                    *receipt['model_identities'], receipt['runtime_assets_registration']]
        for output in outputs: checked(output)
    for source in sources: checked(source)
    return receipt


def png_bound(width, height):
    # RGB8 scanlines including PNG's row-filter byte. 15% exceeds generic zlib
    # deflateBound overhead; 64KiB covers PNG chunks/headers for these grids.
    # No metadata is attached by the frozen producers. Allocation unit <=4096.
    raw = (3 * width + 1) * height
    return ((math.ceil(raw * 115 / 100) + 65536 + 4095) // 4096) * 4096


def stage_commands(phase, run, base_launcher, teacher_python, runtime_assets):
    """No arbitrary commands, fixtures, checkpoints, or step overrides."""
    run = Path(run)
    def base(script, *args):
        return [str(BASE_PYTHON), '-s', str(base_launcher), '--mode', 'base', '--', str(EXP / script), *map(str, args)]
    def stable(*args): return [str(teacher_python), str(EXP / 'methods/splatsure_stablesr.py'), *map(str, args)]
    def node(name, argv, budget, cap=2*GiB):
        return dict(name=name, argv=argv, additional_bytes=budget, maximum_file_bytes=cap)
    train_png = 6000 * png_bound(1344, 1008)
    test_png = 300 * png_bound(1344, 1008)
    if phase == 'lr':
        return [node('lr_train', base('train_stage.py', '--config', CONFIG, '--stage', 'lr', '--out', run/'lr'), 9*GiB//2),
                node('lr_evaluate', base('render_eval.py', '--config', CONFIG, '--checkpoint', run/'lr/checkpoint.pt',
                     '--out', run/'lr_evaluation'), test_png + 9*GiB//8, GiB)]
    upstream = ROOT/'tmp/static_sr_wu_20261022/upstream'
    weights = ROOT/'weight/static_sr_wu_20261022/stablesr'
    return [
        node('teacher_plan', stable('plan', '--manifest', checked(read(CONFIG)['manifest']), '--out', run/'teacher',
             '--source', upstream/'stablesr', '--weight-root', weights, '--python', teacher_python,
             '--runtime-assets', runtime_assets), 9*GiB//8, GiB),
        node('teacher_generate', stable('run', '--plan', run/'teacher/plan.json'),
             train_png + 6000*png_bound(336,252) + 9*GiB//4, GiB),
        node('contribution', base('prepare_priors.py', '--config', CONFIG, '--kind', 'splatsure_weights',
             '--checkpoint', run/'lr/checkpoint.pt', '--rasterizer-source', upstream/'splatsure-rasterizer',
             '--rasterizer-binary-root', ROOT/'output/static_sr_wu_20261022/runtime/splatsure',
             '--out', run/'contribution'), 6000*((336*252*4+128+4095)//4096)*4096 + 9*GiB//8, GiB),
        node('bundle', base('bundle_priors.py', '--config', CONFIG, '--sr-receipt', run/'teacher/receipt.json',
             '--weights-receipt', run/'contribution/receipt.json', '--model', weights/'stablesr_000117.ckpt',
             '--model', weights/'vqgan_cfw_00011.ckpt', '--output', run/'bundle.json'), 9*GiB//8, GiB),
        node('bundle_gate', base('train_stage.py', '--config', CONFIG, '--stage', 'sr', '--prior-bundle', run/'bundle.json',
             '--check-only', '--out', run/'bundle_gate'), 9*GiB//8, GiB),
        # train_stage SR already evaluates immediately on successful completion.
        node('sr_train_and_evaluate', base('train_stage.py', '--config', CONFIG, '--stage', 'sr',
             '--prior-bundle', run/'bundle.json', '--out', run/'sr'), 9*GiB//2 + test_png),
    ]


def check_inputs():
    sys.path.insert(0, str(EXP))
    from train_stage import preflight
    cfg = read(CONFIG)
    m, rows, _ = preflight(cfg, 'lr')
    if (cfg['method'] != 'splatsure' or cfg.get('teacher_variant') != 'stablesr' or cfg['degradation'] != 'area4'
            or cfg['specification']['lr_steps'] != 30000 or cfg['specification']['sr_steps'] != 30000
            or cfg['specification']['init_sr'] != 'fresh' or m['scene'] != 'cook_spinach'
            or m['resolutions'] != {'hr':[1344,1008], 'lr':[336,252]}
            or m['frame_indices'] != list(range(300)) or len(rows) != 6000 or m['splits']['test'] != ['cam00']):
        raise ValueError('This runner is the full Cook07 area4 StableSR-main protocol only')
    # HR is never training supervision. Check only that later evaluation inputs
    # exist here; their pixel hashes are verified by the production evaluator.
    held = [o for o in m['observations'] if o['split'] == 'test' and o['camera_id'] == 'cam00']
    if len(held) != 300 or any(not (Path(m['_root'])/o['hr_path']).is_file() for o in held):
        raise ValueError('Complete cam00 evaluation inputs are absent')
    return cfg, m


def verify_metrics(path, expected_checkpoint):
    result = read(path)
    rows = result.get('frames', [])
    if (result.get('schema') != 'static_sr_wu_uniform_png_metrics_v1' or result.get('scene') != 'cook_spinach'
            or result.get('camera') != 'cam00' or result.get('training_view_metrics') is not False
            or len(rows) != 300 or {r['frame'] for r in rows} != set(range(300))):
        raise ValueError('Full cam00 metrics are missing or from a different protocol')
    if checked(result['checkpoint']).resolve() != Path(expected_checkpoint).resolve():
        raise ValueError('Metric checkpoint differs')
    for row in rows:
        if any(not math.isfinite(row[k]) for k in ('PSNR','SSIM','LPIPS')): raise ValueError('Nonfinite metrics')
        checked(row['render'])
    if any(not math.isfinite(result['metrics'][k]) for k in ('PSNR','SSIM','LPIPS')): raise ValueError('Nonfinite means')
    return identity(path)


def verify_lr_completion(path, run, config_id):
    record = read(path)
    if (record.get('schema') != 'static_sr_local_phase_completion_v1' or record.get('status') != 'completed'
            or record.get('phase') != 'lr' or record.get('run_root') != str(run) or record.get('config') != config_id):
        raise ValueError('A successful same-run LR phase is required')
    prior = read(checked(record['plan']))
    if prior['phase'] != 'lr' or prior['run_root'] != str(run): raise ValueError('LR plan differs')
    for item in prior['required_files']: checked(item)
    for item in record['artifacts']: checked(item)
    complete = read(Path(run)/'lr/complete.json')
    if (complete.get('status') != 'stage_completed' or complete.get('wiring_fixture')
            or complete.get('steps') != 30000 or complete.get('stage') != 'lr'
            or complete.get('training_keys') != 6000): raise ValueError('LR is not the complete formal parent')
    checked(complete['checkpoint'])
    verify_metrics(Path(run)/'lr_evaluation/metrics.json', Path(run)/'lr/checkpoint.pt')
    return record


def required_paths(phase, cfg, launcher, teacher, runtime_assets):
    """Freeze the actual shared Wu implementation, not just adaptation files."""
    result=[CONFIG,checked(cfg['manifest']),checked(cfg['source_configuration']),Path(__file__),Path(launcher),BASE_PYTHON]
    result += sorted(EXP.rglob('*.py'))
    for directory in ('experiments/dynamic_sr_cull_retrain_20261009',
                      'experiments/dynamic_sr_multiview_footprint_20261007',
                      'experiments/dynamic_sr_20260918'):
        result += sorted((ROOT/directory).glob('*.py'))
    for directory in ('third_party/4dgs_843d5ac_runtime','third_party/static_sr_eval_20261022'):
        result += [p for p in sorted((ROOT/directory).rglob('*')) if p.is_file() and p.suffix in ('.py','.so')]
    # The overlay inventories describe local copied bytes; their original source
    # paths are provenance, not execution dependencies inside broker isolation.
    inventories=[RUNTIME/'copied_files.json']
    result += [RUNTIME/'base_acceptance.json',RUNTIME/'base_v2_acceptance.json',RUNTIME/'git_owner_v2_acceptance.json',RUNTIME/'launch_v2.py']
    owner=read(RUNTIME/'git_owner_v2_acceptance.json')
    if (owner.get('status')!='different_owner_simulation_passed_no_sudo'
            or owner.get('launch_sha256')!=sha(launcher) or Path(launcher).resolve()!=(RUNTIME/'launch_v2.py').resolve()):
        raise ValueError('Launcher lacks the pinned broker Git-ownership acceptance')
    if phase=='full':
        if Path(teacher).resolve()!=(RUNTIME/'stable_python').resolve():
            raise ValueError('Use the frozen local StableSR shim')
        result += [Path(teacher),Path(runtime_assets),RUNTIME/'stablesr_acceptance.json',RUNTIME/'stablesr_v2_acceptance.json']
        inventories.append(RUNTIME/'teacher_copied_files.json')
        private=ROOT/'output/static_sr_wu_20261022/runtime/stablesr_env/lib/python3.10/site-packages'
        result += [p for p in sorted(private.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')]
    for inventory in inventories:
        result.append(inventory)
        for item in read(inventory)['files']:
            local=dict(path=item['destination'],sha256=item['sha256'],bytes=item['bytes'])
            result.append(checked(local))
    return list(dict.fromkeys(result))


def build_plan(args):
    cfg, manifest = check_inputs()
    jid = job_id(args.run_id, args.phase)
    run = RUNS/args.run_id
    if any(p.is_symlink() for p in [run,*run.parents]): raise ValueError('Symlink run root')
    if args.phase == 'lr' and run.exists(): raise FileExistsError('Fresh LR run required')
    gates = {'wiring':identity(args.wiring_receipt), 'metrics':identity(args.metric_receipt)}
    for kind,item in gates.items(): validate_gate(kind,item,args.gpu_uuid)
    lr_completion = None
    if args.phase == 'full':
        if not args.lr_completion or not args.teacher_receipt: raise ValueError('Full requires LR completion and teacher acceptance')
        verify_lr_completion(args.lr_completion, run, identity(CONFIG))
        lr_completion = identity(args.lr_completion)
        gates['teacher'] = identity(args.teacher_receipt)
        validate_gate('teacher',gates['teacher'],args.gpu_uuid)
        if any((run/n).exists() for n in ('full_phase','teacher','contribution','bundle.json','bundle_gate','sr')):
            raise FileExistsError('Full phase output already exists; no replay')
    launcher, teacher = Path(args.base_launcher).absolute(), Path(args.teacher_python).absolute()
    commands = stage_commands(args.phase, run, launcher, teacher, Path(args.runtime_assets).absolute())
    required = required_paths(args.phase,cfg,launcher,teacher,args.runtime_assets)
    block = os.statvfs(ROOT).f_frsize
    if block > 4096: raise ValueError('PNG allocation budget assumes blocks <=4096 B')
    reserve = 20*GiB
    needed = sum(n['additional_bytes'] for n in commands) + reserve
    free = shutil.disk_usage(ROOT).free
    return dict(schema=SCHEMA,status='CPU_plan_ready_requires_broker' if free >= needed else 'blocked_storage',
        phase=args.phase,job_id=jid,run_id=args.run_id,run_root=str(run),config=identity(CONFIG),
        gpu_uuid=args.gpu_uuid,host=socket.gethostname(),commands=commands,gates=gates,
        required_files=[identity(p) for p in dict.fromkeys(required)],lr_completion=lr_completion,
        execution_paths=dict(base_launcher=str(launcher),teacher_python=str(teacher),runtime_assets=str(Path(args.runtime_assets).absolute())),
        storage=dict(reserve_bytes=reserve,required_free_bytes=needed,observed_free_bytes=free,allocation_unit=block,
                     checkpoint_cap_is_failure_boundary_not_valid_truncation=True),
        created_utc=now(),formal_iterations=dict(lr=30000,sr=30000),teacher_images=6000,evaluation_frames=300,
        fresh_SR=True,automatic_submission=False,automatic_replay=False,asset_deletion=False,remote_access=False)


def admission(plan):
    if socket.gethostname() != plan['host']: raise ValueError('Wrong execution host')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['gpu_uuid']: raise ValueError('Broker GPU differs')
    group = Path('/proc/self/cgroup').read_text().strip()
    expected = '0::/claudevpn.slice/claude-local-' + plan['job_id'] + '.service'
    if group != expected: raise ValueError('Run only inside the matching local broker service')
    return dict(cgroup=group,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],host=socket.gethostname())


def validate_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('status') != 'CPU_plan_ready_requires_broker':
        raise ValueError('Plan is not executable')
    if plan['phase'] not in ('lr','full') or plan['job_id'] != job_id(plan['run_id'],plan['phase']):
        raise ValueError('Job identity/phase changed')
    run=RUNS/plan['run_id']
    if plan['run_root'] != str(run) or any(p.is_symlink() for p in [run,*run.parents]):
        raise ValueError('Run escaped scope')
    paths=plan['execution_paths']
    expected=stage_commands(plan['phase'],run,paths['base_launcher'],paths['teacher_python'],paths['runtime_assets'])
    if plan['commands'] != expected or plan['storage']['reserve_bytes'] != 20*GiB:
        raise ValueError('Commands or storage bounds differ from the fixed production recipe')
    mandatory={str(p.absolute()) for p in required_paths(plan['phase'],read(CONFIG),paths['base_launcher'],paths['teacher_python'],paths['runtime_assets'])}
    if not mandatory.issubset({i['path'] for i in plan['required_files']}):
        raise ValueError('Source/input identity inventory is incomplete')
    if set(plan['gates']) != ({'wiring','metrics'} if plan['phase']=='lr' else {'wiring','metrics','teacher'}):
        raise ValueError('GPU gates are incomplete')
    return run


def storage_guard(commands, offset, reserve, free=None):
    needed = sum(n['additional_bytes'] for n in commands[offset:]) + reserve
    actual = shutil.disk_usage(ROOT).free if free is None else free
    if actual < needed: raise RuntimeError(f'Storage insufficient: {actual} available, {needed} required')
    return dict(available_bytes=actual,required_bytes=needed,reserve_bytes=reserve)


def verify_frozen(plan):
    for item in plan['required_files']: checked(item)


def file_limit(maximum):
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, maximum))


def signal_failure(signum, frame):
    raise KeyboardInterrupt('Broker/process stop signal ' + str(signum))


def run_plan(path):
    source = Path(path).absolute(); plan = read(source)
    run=validate_plan(plan)
    broker = admission(plan)
    verify_frozen(plan)
    for kind,item in plan['gates'].items(): validate_gate(kind,item,plan['gpu_uuid'])
    check_inputs()
    if plan['phase'] == 'full': verify_lr_completion(checked(plan['lr_completion']),run,plan['config'])
    else: run.mkdir(parents=True,exist_ok=False)
    out = run/(plan['phase']+'_phase')
    out.mkdir(exist_ok=False)  # A failed/running phase is never automatically replayed.
    write(out/'claim.json',dict(plan=identity(source),pid=os.getpid(),broker=broker),new=True)
    record = dict(schema='static_sr_local_phase_completion_v1',phase=plan['phase'],job_id=plan['job_id'],
        run_root=str(run),config=plan['config'],plan=identity(source),broker=broker,started_utc=now(),stages=[],artifacts=[])
    began = time.monotonic(); error = None
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg')
    old_handlers={s:signal.signal(s,signal_failure) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for index,node in enumerate(plan['commands']):
            verify_frozen(plan)
            gate = storage_guard(plan['commands'],index,plan['storage']['reserve_bytes'])
            stage = dict(name=node['name'],argv=node['argv'],storage=gate,started_utc=now())
            record['stages'].append(stage);write(out/'status.json',dict(record,status='running'))
            before=time.monotonic(); log=out/(node['name']+'.log')
            with log.open('x') as stream:
                child=subprocess.run(node['argv'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda: file_limit(node['maximum_file_bytes']))
            stage.update(returncode=child.returncode,seconds=time.monotonic()-before,log=identity(log))
            write(out/'status.json',dict(record,status='stage_completed' if child.returncode==0 else 'failed'))
            if child.returncode: raise RuntimeError(f"Stage {node['name']} failed with code {child.returncode}; no automatic retry")
        checkpoint=run/('lr' if plan['phase']=='lr' else 'sr')/'checkpoint.pt'
        completion=run/('lr' if plan['phase']=='lr' else 'sr')/'complete.json'
        complete=read(completion)
        expected='stage_completed' if plan['phase']=='lr' else 'completed_with_evaluation'
        if complete.get('status') != expected or complete.get('steps') != 30000 or complete.get('wiring_fixture'):
            raise ValueError('Production training did not complete all 30000 iterations')
        metric=run/'lr_evaluation/metrics.json' if plan['phase']=='lr' else run/'sr/evaluation/metrics.json'
        record['artifacts']=[identity(checkpoint),identity(completion),verify_metrics(metric,checkpoint)]
        verify_frozen(plan)
        record.update(status='completed',seconds=time.monotonic()-began,finished_utc=now())
        write(out/'status.json',record);write(out/'complete.json',record,new=True)
    except BaseException:
        error=traceback.format_exc();write(out/'status.json',dict(record,status='failed',seconds=time.monotonic()-began,error=error))
        raise
    finally:
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        detail=dict(status='failed' if error else 'completed',error=error,stages=record['stages'],seconds=time.monotonic()-began)
        if error and record['stages']:
            log=out/(record['stages'][-1]['name']+'.log')
            if log.is_file():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read(65536)
                detail['failure_log']=dict(**identity(log),tail=tail.decode('utf-8','replace'),tail_truncated=log.stat().st_size>65536)
        write(out/'diagnostics.json',detail)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--phase',choices=['lr','full'],required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--wiring-receipt',required=True);p.add_argument('--metric-receipt',required=True)
    p.add_argument('--teacher-receipt');p.add_argument('--lr-completion');p.add_argument('--out',required=True)
    p.add_argument('--base-launcher',default=RUNTIME/'launch_v2.py');p.add_argument('--teacher-python',default=RUNTIME/'stable_python')
    p.add_argument('--runtime-assets',default=ROOT/'output/static_sr_wu_20261022/splatsure_stablesr_assets/runtime_assets_v2.json')
    p=sub.add_parser('run');p.add_argument('--plan',required=True)
    args=parser.parse_args()
    if args.action=='plan':
        result=build_plan(args);out=Path(args.out)
        if out.resolve().is_relative_to(Path(result['run_root']).resolve()):
            raise ValueError('Keep the plan outside its fresh run directory')
        out.parent.mkdir(parents=True,exist_ok=True);write(out,result,new=True)
        print(json.dumps(dict(status=result['status'],plan=identity(out),job_id=result['job_id'],storage=result['storage'])))
    else:run_plan(args.plan)


if __name__=='__main__':main()
