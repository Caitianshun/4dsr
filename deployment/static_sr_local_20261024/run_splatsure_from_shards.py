"""Continue completed Cook07 SplatSuRe LR using verified StableSR shards.

CPU planning never submits or initializes CUDA. The broker's outer command must
be the plan's broker_entry_argv: BASE_PYTHON -s launch_v2.py --mode base -- this
file run --plan PLAN. The same pinned launcher runs every production child.
This preserves the accepted private Git-ownership rules in broker isolation.
There is no teacher generation, LR replay, remote access, deletion or retry.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_splatsure as shared
import stablesr_shards as shards

ROOT, EXP, CONFIG, RUNTIME, RUNS = shared.ROOT, shared.EXP, shared.CONFIG, shared.RUNTIME, shared.RUNS
GiB = shared.GiB
SCHEMA = 'static_sr_local_splatsure_from_shards_plan_v1'
PHASE = 'shards-sr'
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
WEIGHTS = ROOT/'weight/static_sr_wu_20261022/stablesr'
sha, identity, checked, read, write, now = shared.sha, shared.identity, shared.checked, shared.read, shared.write, shared.now
OUTPUTS = ('full_phase', 'shards_sr_phase', 'teacher', 'contribution', 'bundle.json', 'bundle_gate', 'sr')


def same_file(left, right):
    return checked(left).resolve() == checked(right).resolve() and left['sha256'] == right['sha256']


def local_file(item):
    """Operational assets must be regular task-local files, never remote aliases."""
    p = shards.no_symlink(checked(item))
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError('Aggregate asset is outside the local project: ' + str(p))
    return p


def validate_aggregate(path, cfg):
    """Recompute the aggregate from original sealed execution and image bytes."""
    result = read(local_file(identity(path)))
    bridge = shards.load_bridge(ROOT)
    s = result.get('sharding', {})
    if (result.get('schema') != 'wu_splatsure_stablesr_teacher_v1' or result.get('status') != 'completed'
            or result.get('method') != 'splatsure' or result.get('recipe') != bridge.RECIPE
            or result.get('scene') != 'cook_spinach' or result.get('training_only') is not True
            or result.get('parameter_updates') != 0 or result.get('HR_images_read') != 0
            or result.get('heldout_images_read') != 0 or result.get('formal_prior_allowed') is False
            or s.get('schema') != 'stablesr_aggregated_shards_v1' or s.get('target_count') != 6000
            or s.get('original_receipts_preserved') is not True or s.get('rng_policy') != shards.RNG):
        raise ValueError('Complete formal Cook07 StableSR shard aggregate required')
    if local_file(result['manifest']) != checked(cfg['manifest']).absolute():
        raise ValueError('Aggregate belongs to different training inputs')
    expected_models = {role: (str(WEIGHTS/name), digest) for role, (name, digest) in bridge.WEIGHTS.items()}
    supplied = result['model_identities']
    actual = {i['role']: (str(local_file(i)), i['sha256']) for i in supplied}
    if len(supplied) != len(expected_models) or actual != expected_models:
        raise ValueError('Aggregate must bind the exact two local official weights')
    expected_producers = {str(Path(shards.__file__).absolute()), str(ROOT/shards.BRIDGE)}
    if {str(local_file(i)) for i in result.get('producer', [])} != expected_producers:
        raise ValueError('Aggregate producer sources differ')
    roster = local_file(s['roster'])
    parts = s['shards']
    if len(parts) != s.get('count') or not parts:
        raise ValueError('Aggregate shard count differs')
    seals = []
    for part in parts:
        seals.append(local_file(part['seal']))
        for field in ('returned_job_plan', 'returned_receipt', 'returned_execution'):
            local_file(part[field])
    # This read-only helper checks exact roster, portable target rows, LR SHA,
    # author argv/seed/tiling/runtime/model identities, successful execution,
    # per-shard seals, and all PNG SHA/encoding/dimensions. Original absolute
    # receipts remain untouched; only their explicit mappings are resolved.
    recomputed = shards.verify_aggregate(ROOT, checked(cfg['manifest']), roster, seals, s['mappings'])
    if recomputed != result:
        raise ValueError('Aggregate differs from independently verified original shard evidence')
    paths = [Path(path).absolute(), roster, *seals]
    paths += [local_file(i) for i in result['producer'] + supplied]
    for target in result['targets']:
        paths += [local_file(target['sr']), local_file(dict(path=target['lr_path'], sha256=target['lr_sha256']))]
    mapper = shards.Mapper(s['mappings'])
    # Close the exact returned provenance files plus their immutable execution
    # dependencies. Original remote paths are only read via explicit mappings.
    jobs = set()
    for part in parts:
        paths += [local_file(part[f]) for f in ('returned_job_plan', 'returned_receipt', 'returned_execution')]
        seal = read(part['seal']['path'])
        paths += [local_file(identity(mapper.checked(seal[f]))) for f in ('plan', 'inference_log')]
        if 'broker_context' in seal:
            paths.append(local_file(identity(mapper.checked(seal['broker_context']))))
        job_path = local_file(part['returned_job_plan'])
        if job_path in jobs:
            continue
        jobs.add(job_path)
        job = read(job_path)
        template_path = mapper.checked(job['template'])
        paths.append(local_file(identity(template_path)))
        template = read(template_path)
        dependencies = job['required_files'] + template['producer'] + template['model_identities']
        dependencies += [template['runtime_assets_identity']]
        dependencies += template['runtime_assets']['files'] + template['runtime_assets']['cache_mappings']
        dependencies += template['runtime_assets'].get('cache_metadata', [])
        for item in dependencies:
            paths.append(local_file(identity(mapper.checked(item))))
        ref = Path(template['runtime_assets']['cache_environment']['HUGGINGFACE_HUB_CACHE'])/bridge.OPENCLIP_CACHE_REPO/'refs/main'
        paths.append(local_file(identity(mapper.path(str(ref)))))
    return result, list(dict.fromkeys(paths))


def validate_lr(path, run, config_id):
    result = shared.verify_lr_completion(path, run, config_id)
    prior_path = checked(result['plan']); prior = read(prior_path)
    if shared.validate_plan(prior) != run or prior.get('config') != config_id:
        raise ValueError('Original formal LR plan/configuration differs')
    complete = read(run/'lr/complete.json')
    if not same_file(complete['config'], config_id) or checked(complete['checkpoint']).resolve() != (run/'lr/checkpoint.pt').resolve():
        raise ValueError('LR completion does not identify this configuration/checkpoint')
    metric_renders = [checked(row['render']) for row in read(run/'lr_evaluation/metrics.json')['frames']]
    return [Path(path).absolute(), prior_path, *[checked(i) for i in prior['required_files']],
            *[checked(i) for i in result['artifacts']], run/'lr/complete.json', run/'lr/checkpoint.pt',
            run/'lr_evaluation/metrics.json', *metric_renders]


def assert_fresh(run):
    if not run.is_dir() or any(p.is_symlink() for p in [run, *run.parents]):
        raise ValueError('Existing same-run LR directory required')
    if any((run/n).exists() or (run/n).is_symlink() for n in OUTPUTS):
        raise FileExistsError('Continuation/full outputs already exist; no replay or overwrite')


def stage_commands(run, launcher, aggregate):
    run = Path(run)
    def base(script, *args):
        return [str(shared.BASE_PYTHON), '-s', str(launcher), '--mode', 'base', '--', str(EXP/script), *map(str, args)]
    def node(name, argv, budget, cap=GiB):
        return dict(name=name, argv=argv, additional_bytes=budget, maximum_file_bytes=cap)
    return [
        node('contribution', base('prepare_priors.py', '--config', CONFIG, '--kind', 'splatsure_weights',
             '--checkpoint', run/'lr/checkpoint.pt', '--rasterizer-source', ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer',
             '--rasterizer-binary-root', ROOT/'output/static_sr_wu_20261022/runtime/splatsure', '--out', run/'contribution'),
             6000*((336*252*4+128+4095)//4096)*4096 + 9*GiB//8),
        node('bundle', base('bundle_priors.py', '--config', CONFIG, '--sr-receipt', aggregate,
             '--weights-receipt', run/'contribution/receipt.json', '--model', WEIGHTS/'stablesr_000117.ckpt',
             '--model', WEIGHTS/'vqgan_cfw_00011.ckpt', '--output', run/'bundle.json'), 9*GiB//8),
        node('bundle_gate', base('train_stage.py', '--config', CONFIG, '--stage', 'sr', '--prior-bundle', run/'bundle.json',
             '--check-only', '--out', run/'bundle_gate'), 9*GiB//8),
        node('sr_train_and_evaluate', base('train_stage.py', '--config', CONFIG, '--stage', 'sr',
             '--prior-bundle', run/'bundle.json', '--out', run/'sr'),
             9*GiB//2 + 300*shared.png_bound(1344,1008), 2*GiB),
    ]


def required_paths(cfg, launcher):
    paths = shared.required_paths('lr', cfg, launcher, None, None)
    paths += [Path(__file__), Path(shards.__file__), checked(cfg['manifest']).parent/read(checked(cfg['manifest']))['initialization']['npz_path']]
    # Contribution generation consumes this source/wrapper/binary; freeze all
    # tracked rasterizer source too, without making any environment changes.
    source = ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer'
    tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    paths += [source/p for p in tracked if p and (source/p).is_file()]
    binary = ROOT/'output/static_sr_wu_20261022/runtime/splatsure'
    paths += [p for p in binary.rglob('*') if p.is_file() and p.suffix in ('.so', '.py', '.json')]
    return list(dict.fromkeys(paths))


def outer_argv(launcher, plan_path):
    return [str(shared.BASE_PYTHON), '-s', str(launcher), '--mode', 'base', '--', str(Path(__file__).absolute()),
            'run', '--plan', str(Path(plan_path).absolute())]


def build_plan(args):
    cfg, manifest = shared.check_inputs()
    if args.gpu_uuid != PRO6000:
        raise ValueError('Continuation is restricted to the accepted local PRO6000 UUID')
    run = RUNS/args.run_id; jid = shared.job_id(args.run_id, PHASE)
    assert_fresh(run)
    lr_files = validate_lr(args.lr_completion, run, identity(CONFIG))
    aggregate, aggregate_files = validate_aggregate(args.aggregate_receipt, cfg)
    gates = dict(wiring=identity(args.wiring_receipt), metrics=identity(args.metric_receipt))
    for kind, item in gates.items(): shared.validate_gate(kind, item, args.gpu_uuid)
    launcher = Path(args.base_launcher).absolute()
    commands = stage_commands(run, launcher, Path(args.aggregate_receipt).absolute())
    paths = required_paths(cfg, launcher) + lr_files + aggregate_files
    block = os.statvfs(ROOT).f_frsize
    if block > 4096: raise ValueError('Storage budget requires allocation blocks <=4096 B')
    reserve = 20*GiB; needed = reserve + sum(n['additional_bytes'] for n in commands); free = shutil.disk_usage(ROOT).free
    return dict(schema=SCHEMA, status='CPU_plan_ready_requires_broker' if free >= needed else 'blocked_storage',
        phase=PHASE, run_id=args.run_id, job_id=jid, run_root=str(run), host=socket.gethostname(), gpu_uuid=args.gpu_uuid,
        config=identity(CONFIG), lr_completion=identity(args.lr_completion), aggregate_receipt=identity(args.aggregate_receipt),
        gates=gates, execution_paths=dict(base_launcher=str(launcher)), commands=commands,
        broker_entry_argv=outer_argv(launcher, args.out), plan_path=str(Path(args.out).absolute()),
        required_files=[identity(p) for p in dict.fromkeys(paths)],
        storage=dict(reserve_bytes=reserve, required_free_bytes=needed, observed_free_bytes=free, allocation_unit=block,
                     already_present_teacher_bytes=sum(Path(t['sr']['path']).stat().st_size for t in aggregate['targets']),
                     checkpoint_cap_is_failure_boundary_not_valid_truncation=True),
        created_utc=now(), formal_iterations=30000, optimizer_updates=29999, fresh_SR=True,
        teacher_images=6000, evaluation_frames=300, teacher_generation=False,
        automatic_submission=False, automatic_replay=False, asset_deletion=False, remote_access=False)


def validate_plan(plan, path):
    if plan.get('schema') != SCHEMA or plan.get('status') != 'CPU_plan_ready_requires_broker':
        raise ValueError('Plan is not executable')
    if (plan.get('phase') != PHASE or plan['job_id'] != shared.job_id(plan['run_id'], PHASE)
            or plan.get('gpu_uuid') != PRO6000 or plan.get('config') != identity(CONFIG)):
        raise ValueError('Formal identity, configuration or GPU changed')
    run = RUNS/plan['run_id']; launcher = plan['execution_paths']['base_launcher']
    if plan['run_root'] != str(run) or any(p.is_symlink() for p in [run, *run.parents]):
        raise ValueError('Run escaped scope')
    aggregate = checked(plan['aggregate_receipt'])
    if (plan['commands'] != stage_commands(run, launcher, aggregate) or plan['storage']['reserve_bytes'] != 20*GiB
            or set(plan['gates']) != {'wiring', 'metrics'} or plan['plan_path'] != str(Path(path).absolute())
            or plan['broker_entry_argv'] != outer_argv(launcher, path)):
        raise ValueError('Fixed commands, outer launcher, gates or remaining storage bounds changed')
    cfg = read(CONFIG)
    _, aggregate_files = validate_aggregate(aggregate, cfg)
    lr_files = validate_lr(checked(plan['lr_completion']), run, plan['config'])
    needed = {str(p.absolute()) for p in required_paths(cfg, launcher) + aggregate_files + lr_files}
    frozen = plan['required_files']
    if len({i['path'] for i in frozen}) != len(frozen) or needed != {i['path'] for i in frozen}:
        raise ValueError('Source/input/returned-provenance closure differs')
    return run


def validate_products(run, cfg, aggregate_id, stage):
    """Seal each newly generated dependency before the next child consumes it."""
    if stage == 'contribution':
        path = run/'contribution/receipt.json'; r = read(path)
        expected = {(o['camera_id'], o['frame_index']): o for o in read(checked(cfg['manifest']))['observations'] if o['split']=='train'}
        rows = r.get('records', []); actual = {(x['camera_id'], x['frame_index']): x for x in rows}
        if (r.get('schema') != 'wu_splatsure_weights_v1' or r.get('status') != 'prepared'
                or r.get('manifest_sha256') != cfg['manifest']['sha256'] or r.get('scene') != cfg['scene']
                or r.get('training_only') is not True or r.get('parameter_updates') != 0
                or r.get('requested_frame_indices') is not None or r.get('render_calls') != 18000
                or len(rows) != 6000 or len(actual) != len(rows) or set(actual) != set(expected)):
            raise ValueError('Contribution cache is incomplete or from another protocol')
        parent = r['provenance']['parent_checkpoint']
        if (checked(parent).resolve() != (run/'lr/checkpoint.pt').resolve()
                or r['provenance']['parent_checkpoint_sha256'] != parent['sha256']
                or checked(r['provenance']['source_config']).resolve() != CONFIG.resolve()):
            raise ValueError('Contribution map parent/configuration differs')
        paths = [path]
        for key, row in actual.items():
            expected_path = f'maps/{key[0]}_{key[1]:06d}.pt'
            if row['path'] != expected_path or row['time'] != expected[key]['time'] or row['shape'] != [252,336]:
                raise ValueError('Contribution map path/time/shape differs')
            paths.append(local_file(dict(path=str(run/'contribution'/row['path']), sha256=row['sha256'])))
        return [identity(p) for p in paths]
    if stage == 'bundle':
        sys.path.insert(0, str(EXP))
        from train_stage import preflight
        preflight(cfg, 'sr', str(run/'bundle.json'))
        b = read(run/'bundle.json')
        if (checked(b['teacher_producer_receipt']) != checked(aggregate_id)
                or checked(b['weight_producer_receipt']) != run/'contribution/receipt.json'):
            raise ValueError('Bundle references a different teacher or contribution receipt')
        targets = {(x['camera'],x['frame']):x for x in read(checked(aggregate_id))['targets']}
        maps = {(x['camera_id'],x['frame_index']):x for x in read(run/'contribution/receipt.json')['records']}
        for e in b['entries']:
            key = (e['camera'],e['frame']); target = targets[key]; weight = maps[key]
            if (e['time'] != target['time'] or e['lr_sha256'] != target['lr_sha256']
                    or not same_file(e['sr'],target['sr'])
                    or not same_file(e['weight'],dict(path=str(run/'contribution'/weight['path']),sha256=weight['sha256']))):
                raise ValueError('Bundle entry changed a sealed teacher/map/time identity')
        return [identity(run/'bundle.json')]
    if stage == 'bundle_gate':
        p = run/'bundle_gate/preflight.json'; r = read(p)
        if (r.get('status') != 'CPU_input_contract_passed' or r.get('stage') != 'sr' or r.get('training_keys') != 6000
                or not same_file(r['config'], identity(CONFIG)) or not same_file(r['prior_bundle'], identity(run/'bundle.json'))
                or r.get('training_HR_read') is not False or r.get('heldout_pixels_read') is not False):
            raise ValueError('Full fresh SR bundle gate did not pass')
        return [identity(p)]
    return []


def run_plan(path):
    path = Path(path).absolute(); plan = read(path)
    run = validate_plan(plan, path); broker = shared.admission(plan)
    shared.verify_frozen(plan)
    for kind, item in plan['gates'].items(): shared.validate_gate(kind, item, plan['gpu_uuid'])
    cfg, _ = shared.check_inputs(); assert_fresh(run)
    out = run/'shards_sr_phase'; out.mkdir(exist_ok=False)
    write(out/'claim.json', dict(plan=identity(path), pid=os.getpid(), broker=broker), new=True)
    record = dict(schema='static_sr_local_splatsure_from_shards_completion_v1', phase=PHASE, job_id=plan['job_id'],
        run_root=str(run), config=plan['config'], plan=identity(path), broker=broker, started_utc=now(),
        aggregate_receipt=plan['aggregate_receipt'], lr_completion=plan['lr_completion'], stages=[], artifacts=[])
    start = time.monotonic(); error = None; generated = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
    old = {s:signal.signal(s, shared.signal_failure) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for index, node in enumerate(plan['commands']):
            shared.verify_frozen(plan)
            for item in generated: checked(item)
            space = shared.storage_guard(plan['commands'], index, plan['storage']['reserve_bytes'])
            stage = dict(name=node['name'], argv=node['argv'], storage=space, started_utc=now()); record['stages'].append(stage)
            write(out/'status.json', dict(record, status='running')); before=time.monotonic(); log=out/(node['name']+'.log')
            with log.open('x') as stream:
                child = subprocess.run(node['argv'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, preexec_fn=lambda:shared.file_limit(node['maximum_file_bytes']))
            stage.update(returncode=child.returncode, seconds=time.monotonic()-before, log=identity(log))
            if child.returncode: raise RuntimeError(f"Stage {node['name']} failed with code {child.returncode}; no automatic retry")
            generated += validate_products(run, cfg, plan['aggregate_receipt'], node['name'])
            write(out/'status.json', dict(record, status='stage_completed'))
        checkpoint=run/'sr/checkpoint.pt'; complete=read(run/'sr/complete.json')
        if (complete.get('status') != 'completed_with_evaluation' or complete.get('stage') != 'sr'
                or complete.get('steps') != 30000 or complete.get('wiring_fixture') or complete.get('training_keys') != 6000
                or not same_file(complete['config'], identity(CONFIG)) or not same_file(complete['prior_bundle'], identity(run/'bundle.json'))
                or checked(complete['checkpoint']).resolve() != checkpoint.resolve()):
            raise ValueError('Formal fresh SR30000 plus evaluation did not complete')
        record['artifacts'] = [*generated, identity(checkpoint), identity(run/'sr/complete.json'),
                              shared.verify_metrics(run/'sr/evaluation/metrics.json',checkpoint)]
        shared.verify_frozen(plan)
        for item in generated: checked(item)
        record.update(status='completed', seconds=time.monotonic()-start, finished_utc=now())
        write(out/'status.json',record); write(out/'complete.json',record,new=True)
    except BaseException:
        error=traceback.format_exc()
        write(out/'status.json',dict(record,status='failed',seconds=time.monotonic()-start,error=error)); raise
    finally:
        for signum, handler in old.items(): signal.signal(signum,handler)
        detail=dict(status='failed' if error else 'completed', error=error, stages=record['stages'], seconds=time.monotonic()-start)
        if error and record['stages']:
            log=out/(record['stages'][-1]['name']+'.log')
            if log.is_file():
                with log.open('rb') as f: f.seek(max(0,log.stat().st_size-65536)); tail=f.read(65536)
                detail['failure_log']=dict(**identity(log),tail=tail.decode('utf-8','replace'),tail_truncated=log.stat().st_size>65536)
        write(out/'diagnostics.json',detail)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan')
    for name in ('run-id','lr-completion','aggregate-receipt','wiring-receipt','metric-receipt','gpu-uuid','out'):
        a.add_argument('--'+name,required=True)
    a.add_argument('--base-launcher',default=RUNTIME/'launch_v2.py')
    a=sub.add_parser('run'); a.add_argument('--plan',required=True)
    args=p.parse_args()
    if args.action=='run': run_plan(args.plan); return
    result=build_plan(args); out=shards.no_symlink(args.out)
    if out.is_relative_to(Path(result['run_root'])): raise ValueError('Keep plan outside the LR/continuation run directory')
    out.parent.mkdir(parents=True,exist_ok=True); write(out,result,new=True)
    print(__import__('json').dumps(dict(status=result['status'],plan=identity(out),job_id=result['job_id'],
                                     broker_entry_argv=result['broker_entry_argv'],storage=result['storage'])))


if __name__=='__main__': main()
