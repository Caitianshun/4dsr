"""Serial remote Cook07 shared-SwinIR SplatSuRe continuation; never submits a job.

CPU plan -> broker-admitted contribution maps -> verified bundle -> fresh SR30k
with the frozen trainer's immediate complete cam00 evaluation. No teacher
inference, LR replay, remote access, deletion, or automatic retry is performed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deployment.static_sr_local_20261024 import run_splatsure as local
from deployment.static_sr_local_20261024 import stablesr_shards as remote
from experiments.static_sr_shared_teacher_20261024 import shared_cache as cache
from experiments.static_sr_shared_teacher_20261024 import build_splatsure_bundle as bundle

EXP, GiB = local.EXP, local.GiB
HERE = Path(__file__).resolve().parent
RUNS = ROOT / 'output/static_sr_wu_20261024/shared_splat_runs/Cook07'
PHASE = 'shared-sr'
SCHEMA = 'shared_swinir_splatsure_remote_plan_v1'
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'


def job_id(run_id):
    return local.job_id(run_id, PHASE).replace('ss-local-', 'ss-shared-', 1)


def fresh_run(run_id):
    job_id(run_id)
    run = cache.path(RUNS / run_id)
    if run.exists():
        raise FileExistsError('New shared-Splat output required; no replay')
    return run


def check_inputs(config_path, cache_path, lr_completion):
    cfg = bundle._base(config_path, ROOT)
    marker = cfg.get('information', {}).get('shared_teacher', {})
    original = cache.checked(marker['source_config'])
    if cfg != bundle.build_config(original, cache_path):
        raise ValueError('Only teacher identity/label/information may change')
    if original != local.CONFIG or cfg.get('teacher_variant') != cache.VARIANT:
        raise ValueError('This continuation requires the registered Cook07 SplatSuRe configuration')
    m, rows = cache.legal_manifest(cache.checked(cfg['manifest']))
    if (m['scene'] != 'cook_spinach' or m['resolutions'] != {'hr': [1344, 1008], 'lr': [336, 252]}
            or m['frame_indices'] != list(range(300)) or len(rows) != 6000
            or m['splits']['test'] != ['cam00'] or cfg['specification']['init_sr'] != 'fresh'
            or cfg['specification']['lr_steps'] != 30000 or cfg['specification']['sr_steps'] != 30000):
        raise ValueError('Require full Cook07 area4 inputs and fresh SR30000')
    parent, complete = bundle.verify_lr_parent(lr_completion, cfg)
    if complete.get('training_keys') != 6000:
        raise ValueError('LR parent does not identify all 6000 training observations')
    held = [o for o in m['observations'] if o['split'] == 'test']
    if len(held) != 300 or any(not cache.path(o['hr_path'], base=Path(m['_root'])).is_file() for o in held):
        raise ValueError('Complete later cam00 evaluation inputs are absent')
    return cfg, m, parent


def checked_profile(profile_path):
    """Environment metadata is separate from the scientific method/config."""
    p = cache.path(profile_path)
    value = cache.read(p)
    if (value.get('schema') != 'shared_splatsure_remote_runtime_v1'
            or value.get('status') != 'CPU_imports_verified' or value.get('broker') != 'remote'
            or value.get('project_id') != '4dsr' or value.get('root') != str(ROOT)
            or value.get('host') != socket.gethostname()
            or not ROOT.is_relative_to(Path(value['broker_project_root']))
            or local.normalize_uuid(value['gpu_uuid']) == local.normalize_uuid(PRO6000)
            or not value.get('visible_device') or ',' in value['visible_device'] or value['visible_device'] == '-1'):
        raise ValueError('Require this remote host/task and one explicit non-PRO GPU')
    files = value.get('required_files', [])
    paths = {str(local.checked(i).absolute()) for i in files}
    if not files or len(paths) != len(files):
        raise ValueError('Runtime profile requires unique verified environment files')
    mandatory = {str(Path(value['base_python']).absolute())}
    if value.get('base_launcher'):
        mandatory.add(str(cache.path(value['base_launcher'])))
    if not mandatory.issubset(paths) or not isinstance(value.get('launcher_arguments', []), list):
        raise ValueError('Actual interpreter and optional launcher must be frozen')
    if not value.get('base_launcher') and value.get('launcher_arguments'):
        raise ValueError('Launcher arguments require an explicit launcher')
    receipt = local.checked(value['CPU_acceptance'])
    if str(receipt.absolute()) not in paths:
        raise ValueError('Environment CPU acceptance must be in its frozen files')
    return value


def launch(profile, script, *args):
    argv = [str(Path(profile['base_python']).absolute()), '-s', '-B']
    if profile.get('base_launcher'):
        argv += [str(profile['base_launcher']), *profile.get('launcher_arguments', [])]
    return [*argv, str(script), *map(str, args)]


def commands(run, profile, config, shared_cache, lr_completion, parent):
    def call(script, *args):
        return launch(profile, script, *args)
    def node(name, argv, space, cap=GiB):
        return dict(name=name, argv=argv, additional_bytes=space, maximum_file_bytes=cap)
    # Each map is one float32 [252,336] tensor. 64 KiB/file covers the small
    # torch archive, rounded to 4096-byte allocation; no new SR PNG set is made.
    maps = 6000 * (((336 * 252 * 4 + 65536 + 4095) // 4096) * 4096)
    return [
        node('contribution', call(EXP/'prepare_priors.py', '--config', config,
            '--kind', 'splatsure_weights', '--checkpoint', parent,
            '--rasterizer-source', ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer',
            '--rasterizer-binary-root', ROOT/'output/static_sr_wu_20261022/runtime/splatsure',
            '--out', run/'contribution'), maps + GiB),
        node('bundle', call(HERE/'build_splatsure_bundle.py', 'bundle', '--config', config,
            '--shared-cache', shared_cache, '--weights-receipt', run/'contribution/receipt.json',
            '--lr-completion', lr_completion, '--out', run/'bundle.json'), GiB),
        node('bundle_gate', call(EXP/'train_stage.py', '--config', config, '--stage', 'sr',
            '--prior-bundle', run/'bundle.json', '--check-only', '--out', run/'bundle_gate'), GiB),
        node('fresh_sr_and_evaluation', call(EXP/'train_stage.py', '--config', config, '--stage', 'sr',
            '--prior-bundle', run/'bundle.json', '--out', run/'sr'),
            9 * GiB // 2 + 300 * local.png_bound(1344, 1008), 2 * GiB),
    ]


def source_paths(profile):
    # No local overlay or obsolete StableSR output is an execution dependency.
    paths = [Path(__file__), HERE/'shared_cache.py', HERE/'build_splatsure_bundle.py',
             Path(local.__file__), Path(remote.__file__), local.CONFIG]
    paths += sorted(EXP.rglob('*.py'))
    for directory in ('experiments/dynamic_sr_cull_retrain_20261009',
                      'experiments/dynamic_sr_multiview_footprint_20261007', 'experiments/dynamic_sr_20260918'):
        paths += sorted((ROOT/directory).glob('*.py'))
    for directory in ('third_party/4dgs_843d5ac_runtime', 'third_party/static_sr_eval_20261022'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.so')]
    paths += [local.checked(item) for item in profile['required_files']]
    raster = ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer'
    # Exact task-local Git exception only; no global ownership policy change.
    listing = subprocess.check_output(['git', '-c', f'safe.directory={raster}', '-C', str(raster), 'ls-files', '-z'])
    paths += [raster/p for p in listing.decode().split('\0') if p and (raster/p).is_file()]
    binary = ROOT/'output/static_sr_wu_20261022/runtime/splatsure'
    paths += [p for p in binary.rglob('*') if p.is_file() and p.suffix in ('.so', '.py', '.json')]
    return paths


def dependencies(config, cache_path, lr_completion, cfg, manifest, parent, profile):
    shared = cache.read(cache_path)
    paths = source_paths(profile) + [Path(config).absolute(), Path(cache_path).absolute(), Path(lr_completion).absolute()]
    paths += bundle.lr_dependency_paths(lr_completion)
    paths += [cache.checked(parent), cache.checked(cfg['manifest']), cache.checked(cfg['source_configuration'])]
    paths.append(cache.path(manifest['initialization']['npz_path'], base=Path(manifest['_root'])))
    paths += [cache.checked(shared[f]) for f in ('manifest', 'source_index', 'importer', 'validator')]
    paths += [cache.checked(i) for i in shared['producer'] + shared['model_identities']]
    provenance = shared['model_provenance']
    paths += [cache.checked(provenance[f]) for f in ('finetune_summary', 'base_weights', 'data_index')]
    if provenance['historical_training_code_current_identity']:
        paths.append(cache.checked(provenance['historical_training_code_current_identity']))
    for row in shared['entries']:
        paths += [cache.path(row['lr']['path']), cache.path(row['sr']['path'])]
    return list(dict.fromkeys(Path(p).absolute() for p in paths))


def outer(profile, plan_path):
    return launch(profile, Path(__file__).absolute(), 'run', '--plan', Path(plan_path).absolute())


def gate_files(gates):
    """Pin the actual accepted raster binaries and evaluator dependencies."""
    result = []
    for kind, item in gates.items():
        p = local.checked(item); r = local.read(p); result.append(p)
        items = [r['source']]
        if kind == 'wiring':
            items += list(r['source_snapshot'].values())
            for extension in r['extensions'].values():
                items += [extension['wrapper'], *extension['binaries']]
        else:
            assets = r['assets']
            items += [r['evaluator_source'], assets['source_receipt'], assets['ssim_implementation'],
                      *assets['implementations'], *assets['model_identities']]
        result += [local.checked(i) for i in items]
    return result


def build_plan(args):
    run = fresh_run(args.run_id)
    profile = checked_profile(args.runtime_profile)
    cfg, manifest, parent = check_inputs(args.config, args.shared_cache, args.lr_completion)
    gates = dict(wiring=local.identity(args.wiring_receipt), metrics=local.identity(args.metric_receipt))
    for kind, item in gates.items():
        local.validate_gate(kind, item, profile['gpu_uuid'], config=args.config)
    stages = commands(run, profile, Path(args.config).absolute(), Path(args.shared_cache).absolute(),
                      Path(args.lr_completion).absolute(), cache.checked(parent))
    paths = dependencies(args.config, args.shared_cache, args.lr_completion, cfg, manifest, parent, profile)
    paths += gate_files(gates) + [Path(args.runtime_profile).absolute()]
    block = os.statvfs(ROOT).f_frsize
    if block > 4096:
        raise ValueError('PNG/map allocation budget requires blocks no greater than 4096 bytes')
    needed = 20 * GiB + sum(n['additional_bytes'] for n in stages)
    free = shutil.disk_usage(ROOT).free
    return dict(schema=SCHEMA, status='CPU_plan_ready_requires_broker' if free >= needed else 'blocked_storage',
        phase=PHASE, run_id=args.run_id, job_id=job_id(args.run_id), run_root=str(run),
        host=socket.gethostname(), root=str(ROOT), broker='remote', broker_project_root=profile['broker_project_root'],
        gpu_uuid=profile['gpu_uuid'], visible_device=profile['visible_device'], config=local.identity(args.config),
        shared_cache=local.identity(args.shared_cache), lr_completion=local.identity(args.lr_completion), parent=parent,
        gates=gates, runtime_profile=local.identity(args.runtime_profile), commands=stages, plan_path=str(Path(args.out).absolute()),
        broker_entry_argv=outer(profile, args.out), required_files=[local.identity(p) for p in dict.fromkeys(paths)],
        storage=dict(reserve_bytes=20 * GiB, required_free_bytes=needed, observed_free_bytes=free,
            allocation_unit=block, checkpoint_cap_is_failure_boundary_not_valid_truncation=True,
            new_teacher_PNG_bytes=0), created_utc=local.now(),
        fresh_SR=True, formal_iterations=30000, optimizer_updates=29999, teacher_images=6000, evaluation_frames=300,
        teacher_generation=False, LR_updates=0, automatic_submission=False, automatic_replay=False,
        remote_access=False, asset_deletion=False, PRO6000_allowed=False)


def validate_plan(plan, source):
    if plan.get('schema') != SCHEMA or plan.get('status') != 'CPU_plan_ready_requires_broker':
        raise ValueError('Non-executable shared-Splat plan')
    run = fresh_run(plan['run_id'])
    profile = checked_profile(local.checked(plan['runtime_profile']))
    if (plan['run_root'] != str(run) or plan['phase'] != PHASE or plan['gpu_uuid'] != profile['gpu_uuid']
            or plan.get('root') != str(ROOT) or plan.get('broker') != 'remote'
            or plan.get('broker_project_root') != profile['broker_project_root']
            or plan.get('visible_device') != profile['visible_device'] or plan.get('host') != profile['host']
            or plan['job_id'] != job_id(plan['run_id'])
            or plan['plan_path'] != str(Path(source).absolute())
            or plan['broker_entry_argv'] != outer(profile, source)
            or set(plan['gates']) != {'wiring', 'metrics'} or plan['storage']['reserve_bytes'] != 20 * GiB):
        raise ValueError('Plan identity, paths, GPU, launcher or gates differ')
    config, cache_path, lr = [local.checked(plan[k]) for k in ('config', 'shared_cache', 'lr_completion')]
    cfg, manifest, parent = check_inputs(config, cache_path, lr)
    if parent != plan['parent'] or plan['commands'] != commands(run, profile, config, cache_path, lr, cache.checked(parent)):
        raise ValueError('Fixed recipe/parent/commands changed')
    paths = dependencies(config, cache_path, lr, cfg, manifest, parent, profile)
    paths += gate_files(plan['gates']) + [local.checked(plan['runtime_profile'])]
    frozen = plan['required_files']
    if len(frozen) != len({i['path'] for i in frozen}) or {str(p) for p in paths} != {i['path'] for i in frozen}:
        raise ValueError('Frozen source/input closure differs')
    return run, cfg, manifest


def admission(plan):
    if plan.get('broker') != 'remote' or local.normalize_uuid(plan.get('gpu_uuid')) == local.normalize_uuid(PRO6000):
        raise ValueError('This protocol currently excludes PRO6000 and requires the remote supervisor')
    proof = remote.admission(plan)
    context = proof['supervisor_context']
    job_dir = Path(plan['broker_project_root'])/'.claude-training/jobs'/plan['job_id']
    request_path = remote.no_symlink(job_dir/'request.json')
    request = local.read(request_path)
    if (context.get('project_id') != '4dsr' or request.get('project_id') != '4dsr'
            or request.get('operation') != 'run' or request.get('job_id') != plan['job_id']
            or request.get('target_job_id') != plan['job_id']
            or request.get('project_root') != plan['broker_project_root']
            or request.get('remote_cwd') != str(ROOT) or Path.cwd() != ROOT
            or request.get('argv') != plan['broker_entry_argv']
            or request.get('cuda_visible_devices') != plan['visible_device']):
        raise ValueError('Actual remote request project/job/cwd/command differs')
    proof['request'] = local.identity(request_path)
    return proof


def product_ids(name, run, plan, cfg, manifest):
    if name == 'contribution':
        _, rows = cache.legal_manifest(cache.checked(cfg['manifest']))
        maps, _ = bundle._maps(run/'contribution/receipt.json', plan['parent'], cfg, manifest, rows, ROOT)
        return [local.identity(run/'contribution/receipt.json'), *maps.values()]
    if name == 'bundle':
        p = run/'bundle.json'
        expected = bundle.build_bundle(local.checked(plan['config']), local.checked(plan['shared_cache']),
                                       run/'contribution/receipt.json', local.checked(plan['lr_completion']))
        if cache.read(p) != expected:
            raise ValueError('Generated bundle differs from genuine shared PNG/map evidence')
        return [local.identity(p)]
    if name == 'bundle_gate':
        p = run/'bundle_gate/preflight.json'; r = cache.read(p)
        if (r.get('status') != 'CPU_input_contract_passed' or r.get('training_keys') != 6000
                or r.get('stage') != 'sr' or local.checked(r['config']) != local.checked(plan['config'])
                or local.checked(r['prior_bundle']) != run/'bundle.json'
                or r.get('training_HR_read') is not False or r.get('heldout_pixels_read') is not False):
            raise ValueError('Full SR preflight did not pass')
        return [local.identity(p)]
    return []


def run_plan(source):
    source = Path(source).absolute(); plan = local.read(source)
    # Admission precedes importing/loading any model or creating output.
    broker = admission(plan)
    run, cfg, manifest = validate_plan(plan, source)
    local.verify_frozen(plan)
    for kind, item in plan['gates'].items():
        local.validate_gate(kind, item, plan['gpu_uuid'], config=local.checked(plan['config']))
    run.mkdir(parents=True, exist_ok=False)
    logs = run/'pipeline'; logs.mkdir()
    local.write(logs/'claim.json', dict(plan=local.identity(source), broker=broker, pid=os.getpid()), new=True)
    for name, item in (('broker_execution_context.json', broker['execution_context']), ('broker_request.json', broker['request'])):
        with (logs/name).open('xb') as target:
            target.write(local.checked(item).read_bytes())
    record = dict(schema='shared_swinir_splatsure_completion_v1', status='running', plan=local.identity(source),
        job_id=plan['job_id'], broker=broker, config=plan['config'], shared_cache=plan['shared_cache'],
        lr_completion=plan['lr_completion'], started_utc=local.now(), stages=[], artifacts=[])
    start = time.monotonic(); error = None; generated = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
    old = {s: signal.signal(s, local.signal_failure) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, node in enumerate(plan['commands']):
            local.verify_frozen(plan)
            for item in generated: local.checked(item)
            space = local.storage_guard(plan['commands'], index, 20 * GiB)
            stage = dict(name=node['name'], argv=node['argv'], storage=space, started_utc=local.now())
            record['stages'].append(stage); local.write(logs/'status.json', record)
            before = time.monotonic(); log = logs/(node['name'] + '.log')
            with log.open('x') as stream:
                child = subprocess.run(node['argv'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, preexec_fn=lambda: local.file_limit(node['maximum_file_bytes']))
            stage.update(returncode=child.returncode, seconds=time.monotonic()-before, log=local.identity(log))
            if child.returncode:
                raise RuntimeError(f"Stage {node['name']} failed ({child.returncode}); preserved without replay")
            generated += product_ids(node['name'], run, plan, cfg, manifest)
            local.write(logs/'status.json', record)
        complete_path = run/'sr/complete.json'; complete = cache.read(complete_path)
        checkpoint = run/'sr/checkpoint.pt'
        if (complete.get('status') != 'completed_with_evaluation' or complete.get('stage') != 'sr'
                or complete.get('steps') != 30000 or complete.get('wiring_fixture') is not False
                or complete.get('training_keys') != 6000 or local.checked(complete['config']) != local.checked(plan['config'])
                or local.checked(complete['prior_bundle']) != run/'bundle.json'
                or local.checked(complete['checkpoint']) != checkpoint):
            raise ValueError('Formal fresh SR30k and complete evaluation did not finish')
        record['artifacts'] = [*generated, local.identity(complete_path), local.identity(checkpoint),
                              local.verify_metrics(run/'sr/evaluation/metrics.json', checkpoint)]
        local.verify_frozen(plan)
        for item in generated: local.checked(item)
        record.update(status='completed', seconds=time.monotonic()-start, finished_utc=local.now())
        local.write(logs/'complete.json', record, new=True)
        local.write(logs/'status.json', record)
    except BaseException:
        error = traceback.format_exc()
        local.write(logs/'status.json', dict(record, status='failed_preserved_no_replay', error=error, seconds=time.monotonic()-start))
        raise
    finally:
        for signum, handler in old.items(): signal.signal(signum, handler)
        detail = dict(error=error, stages=record['stages'], seconds=time.monotonic()-start)
        if error and record['stages']:
            p = logs/(record['stages'][-1]['name'] + '.log')
            if p.is_file():
                with p.open('rb') as f: f.seek(max(0, p.stat().st_size-65536)); tail=f.read(65536)
                detail['failure_log'] = dict(**local.identity(p), tail=tail.decode('utf-8', 'replace'), tail_truncated=p.stat().st_size>65536)
        local.write(logs/'diagnostics.json', detail)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan')
    for name in ('config', 'shared-cache', 'lr-completion', 'run-id', 'runtime-profile', 'wiring-receipt', 'metric-receipt', 'out'):
        a.add_argument('--' + name, required=True)
    a = sub.add_parser('run'); a.add_argument('--plan', required=True)
    args = p.parse_args()
    if args.command == 'run':
        run_plan(args.plan); return
    result = build_plan(args); out = cache.path(args.out)
    if out.is_relative_to(Path(result['run_root'])):
        raise ValueError('Keep input plan outside the fresh output directory')
    cache.write_new(out, result)
    print(__import__('json').dumps(dict(status=result['status'], job_id=result['job_id'], plan=local.identity(out),
                                      broker_entry_argv=result['broker_entry_argv'], storage=result['storage'])))


if __name__ == '__main__': main()
