"""CPU-only full-scene DAG, capacity gates, and compact-result collection.

This tool never submits, trains, calls CUDA/SSH, or deletes assets. A blocked
plan contains reviewable commands but produces no executable run_job job file.
Run planning on the execution snapshot so disk checks refer to that filesystem.
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
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = 'experiments/static_sr_wu_20261022'
SELF = 'deployment/static_sr_cts_20261022/plan_scene.py'
METHODS = ('splatsure', 'iesrgs', 'sequence_matters', 'threedsr')
PRIORITY = ('Discussion07', 'Cook07', 'Cut07', 'Coffee07', 'Flame07', 'VRHeadset07')
SCENES = dict(zip(PRIORITY, ('meetroom_discussion', 'cook_spinach', 'cut_roasted_beef',
                           'coffee_martini', 'flame_steak', 'meetroom_vrheadset')))
MIB = 1 << 20
GIB = 1 << 30


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def resolve(path):
    p = Path(str(path).replace('{ROOT}', str(ROOT)))
    return p if p.is_absolute() else ROOT / p


def relative(path):
    return str(resolve(path).absolute().relative_to(ROOT))


def identity(path):
    p = resolve(path)
    return dict(path=relative(p), sha256=sha(p))


def checked(item):
    path = resolve(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Identity changed: ' + str(path))
    return path


def read(path):
    return json.loads(resolve(path).read_text())


def write_new(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def token(path):
    return '{ROOT}/' + relative(path)


def normalize_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def validate_acceptance(kind, receipt, gpu_uuid, hr_wh, check=checked):
    """Require actual completed inference/wiring, matching hardware and code."""
    if kind == 'wiring':
        expected = ('static_sr_wu_gpu_acceptance_v1', 'passed_bounded_GPU_wiring_only')
    elif kind == 'external':
        expected = ('iesrgs_external_teacher_GPU_acceptance_v1', 'passed_external_teacher_GPU_inference_only')
    elif kind == 'metrics':
        expected = ('static_sr_wu_metric_GPU_acceptance_v1', 'passed_GPU_metrics_and_LPIPS_API_parity_only')
    else:
        expected = ('static_sr_wu_teacher_acceptance_fixture_v1', 'fixture_passed')
    if (receipt.get('schema'), receipt.get('status')) != expected:
        raise ValueError(kind + ': not a completed real GPU acceptance')
    if not receipt.get('GPU_executed', receipt.get('gpu_executed', False)):
        raise ValueError(kind + ': GPU execution absent')
    actual_uuid = receipt.get('gpu_uuid', receipt.get('GPU_uuid', receipt.get('hardware', {}).get('gpu_uuid')))
    if normalize_uuid(actual_uuid) != normalize_uuid(gpu_uuid):
        raise ValueError(kind + ': acceptance belongs to another GPU')
    sources = []
    if kind == 'wiring':
        if not receipt.get('all_extension_modules_coexisted'):
            raise ValueError('Rasterizer coexistence not accepted')
        sources += [receipt['source'], *receipt['source_snapshot'].values()]
        scenes = {s['short'] for s in receipt['scenes'] if s['resolutions']['hr'] == list(hr_wh)}
        results = receipt.get('results', [])
        if not scenes or any(r.get('status') != 'passed' for r in results):
            raise ValueError('Wiring dimension or method acceptance missing/failed')
        if not any({r['method'] for r in results if r['scene'] == s} == set(METHODS) for s in scenes):
            raise ValueError('All four method updates must pass at the native dimensions')
        for value in receipt['extensions'].values():
            sources += [value['wrapper'], *value['binaries']]
    elif kind == 'external':
        if not receipt.get('teacher_networks_executed') or not receipt.get('strict_loading', {}).get('passed'):
            raise ValueError('External pretrained loading/inference not accepted')
        eligible = {s['short'] for s in receipt['samples'] if s['resolutions']['hr'] == list(hr_wh)}
        if not eligible or not any(r['short'] in eligible and r['status'] == 'passed' for r in receipt['results']):
            raise ValueError('External teacher native dimensions not accepted')
        if any(r.get('status') != 'passed' for r in receipt['results']):
            raise ValueError('External teacher has a failed sample')
        sources += [receipt['source'], receipt['production_producer'], *receipt['weights'].values()]
    elif kind == 'metrics':
        sizes = {s['short'] for s in receipt['samples'] if [4*x for x in s['original_lr_wh']] == list(hr_wh)}
        if (not sizes or not receipt.get('frozen_parameters_and_buffers_unchanged')
                or not any(r['short'] in sizes and r['status'] == 'passed' for r in receipt['results'])
                or any(r.get('status') != 'passed' or not r['convenience_class_parity']['passed'] for r in receipt['results'])):
            raise ValueError('Metric dimensions/API parity not accepted')
        assets = receipt['assets']
        sources += [receipt['source'], receipt['evaluator_source'], assets['source_receipt'],
                    assets['ssim_implementation'], *assets['implementations'], *assets['model_identities']]
    else:
        if receipt.get('method') != kind or not receipt.get('teacher_generated'):
            raise ValueError(kind + ': wrong teacher or no generated output')
        if kind == 'threedsr':
            rounds = receipt.get('rounds', [])
            if [r.get('round_index') for r in rounds] != list(range(4)):
                raise ValueError('3DSR must pass all four actual diffusion transitions')
            # The real fixture records each transition's outputs inside its
            # round, with no top-level outputs field.
            if any(not r.get('outputs') for r in rounds):
                raise ValueError('3DSR must retain output evidence from every diffusion transition')
            outputs = [o for r in rounds for o in r['outputs']]
        else:
            outputs = receipt.get('outputs', [])
        if not outputs or any(o['size_wh'] != list(hr_wh) for o in outputs):
            raise ValueError(kind + ': original teacher output dimensions not accepted')
        sources += [receipt['wrapper'], receipt['registered_records'], *receipt['adapter_sources'], *receipt['model_identities']]
        if kind != 'sequence_matters':
            sources.append(receipt['runtime_assets_registration'])
    for item in sources:
        check(item)


def build_dag(scene, run_id, prepared, sequence_plan, runtime_assets, base_python, stable_python, sequence_python,
              iesrgs_cache_format='npy'):
    """Authoritative interfaces only; no shortened iterations or fixture inputs."""
    run = f'output/static_sr_wu_20261022/cts_scene_runs/{scene}/{run_id}'
    configs = {m: token(Path(prepared)/scene/(m+'.json')) for m in METHODS}
    records = token(Path(prepared)/scene/'training_records.json')
    up = '{ROOT}/tmp/static_sr_wu_20261022/upstream'
    weights = '{ROOT}/weight/static_sr_wu_20261022'
    paths = {m: f'{run}/{m}' for m in METHODS}
    nodes = []
    if iesrgs_cache_format not in ('npy', 'npz_deflate'):
        raise ValueError('Unsupported IE lossless cache format')

    def script(python, name, *args):
        return [python, '-u', '{ROOT}/'+EXP+'/'+name, *map(str, args)]

    def add(name, method, depends, argv, out, storage_kind, note=''):
        nodes.append(dict(id=name, method=method, depends_on=depends, argv=argv,
                          output=out, storage_kind=storage_kind, note=note))

    def train(name, method, stage, depends, out, bundle=None):
        args = ['--config', configs[method], '--stage', stage, '--out', '{ROOT}/'+out]
        if bundle: args += ['--prior-bundle', '{ROOT}/'+bundle]
        add(name, method, depends, script(base_python, 'train_stage.py', *args), out,
            'training_and_evaluation' if stage == 'sr' else 'checkpoint_training',
            '30000 iterations / 29999 updates; fresh method-specific model. Final SR includes render_eval at successful exit.')

    ie = paths['iesrgs']; splat = paths['splatsure']; seq = paths['sequence_matters']; three = paths['threedsr']
    add('ie_external', 'iesrgs', [], script(base_python, 'prepare_priors.py', '--config', configs['iesrgs'],
        '--kind', 'iesrgs_external', '--upstream', up+'/iesrgs', '--swinir-checkpoint',
        '{ROOT}/third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
        '--depth-checkpoint', weights+'/depth_anything_v2_large/depth_anything_v2_vitl.pth',
        '--cache-format', iesrgs_cache_format,
        '--out', '{ROOT}/'+ie+'/external'), ie+'/external', 'ie_float_arrays' if iesrgs_cache_format=='npy' else 'ie_lossless_arrays')
    train('ie_internal_train', 'iesrgs', 'internal', [], ie+'/internal_train')
    add('ie_internal_cache', 'iesrgs', ['ie_internal_train'], script(base_python, 'prepare_priors.py',
        '--config', configs['iesrgs'], '--kind', 'iesrgs_internal', '--checkpoint',
        '{ROOT}/'+ie+'/internal_train/checkpoint.pt', '--cache-format', iesrgs_cache_format,
        '--out', '{ROOT}/'+ie+'/internal_cache'), ie+'/internal_cache', 'ie_float_arrays' if iesrgs_cache_format=='npy' else 'ie_lossless_arrays')
    add('ie_merge', 'iesrgs', ['ie_external', 'ie_internal_cache'], script(base_python, 'prepare_priors.py',
        '--config', configs['iesrgs'], '--kind', 'iesrgs_merge', '--external-index', '{ROOT}/'+ie+'/external/index.json',
        '--internal-index', '{ROOT}/'+ie+'/internal_cache/index.json', '--out', '{ROOT}/'+ie+'/bundle.json'),
        ie+'/bundle.json', 'metadata', 'Merge writes an index JSON; it does not copy float32 arrays.')
    add('splat_teacher_plan', 'splatsure', [], script(stable_python, 'methods/splatsure_stablesr.py', 'plan',
        '--manifest', '__REGISTERED_MANIFEST__', '--out', '{ROOT}/'+splat+'/teacher', '--source', up+'/stablesr',
        '--weight-root', weights+'/stablesr', '--python', stable_python, '--runtime-assets', token(runtime_assets)),
        splat+'/teacher', 'metadata')
    add('splat_teacher', 'splatsure', ['splat_teacher_plan'], script(stable_python, 'methods/splatsure_stablesr.py',
        'run', '--plan', '{ROOT}/'+splat+'/teacher/plan.json'), splat+'/teacher', 'png_teacher',
        'Main StableSR DDPM200 recipe; producer validates complete training-only target coverage.')
    train('splat_lr', 'splatsure', 'lr', [], splat+'/lr')
    add('splat_contribution', 'splatsure', ['splat_lr'], script(base_python, 'prepare_priors.py', '--config',
        configs['splatsure'], '--kind', 'splatsure_weights', '--checkpoint', '{ROOT}/'+splat+'/lr/checkpoint.pt',
        '--rasterizer-source', up+'/splatsure-rasterizer', '--rasterizer-binary-root',
        '{ROOT}/output/static_sr_wu_20261022/runtime/splatsure', '--out', '{ROOT}/'+splat+'/contribution'),
        splat+'/contribution', 'splat_weights')
    add('splat_bundle', 'splatsure', ['splat_teacher', 'splat_contribution'], script(base_python, 'bundle_priors.py',
        '--config', configs['splatsure'], '--sr-receipt', '{ROOT}/'+splat+'/teacher/receipt.json',
        '--weights-receipt', '{ROOT}/'+splat+'/contribution/receipt.json', '--model', weights+'/stablesr/stablesr_000117.ckpt',
        '--model', weights+'/stablesr/vqgan_cfw_00011.ckpt', '--output', '{ROOT}/'+splat+'/bundle.json'), splat+'/bundle.json', 'metadata')
    add('sequence_teacher', 'sequence_matters', [], script(sequence_python, 'methods/sequence_matters.py',
        '--manifest', records, '--upstream', up+'/sequence_matters', '--output', '{ROOT}/'+seq+'/teacher',
        '--psrt-checkpoint', weights+'/sequence_matters/PSRT_Vimeo.pth', '--spynet-checkpoint',
        weights+'/sequence_matters/spynet_sintel_final-3d2a1287.pth', '--sequence-plan', token(sequence_plan)),
        seq+'/teacher', 'png_teacher', 'Every original time and every training camera; full cached ALS order.')
    add('sequence_bundle', 'sequence_matters', ['sequence_teacher'], script(base_python, 'bundle_priors.py',
        '--config', configs['sequence_matters'], '--sr-receipt', '{ROOT}/'+seq+'/teacher/receipt.json',
        '--model', weights+'/sequence_matters/PSRT_Vimeo.pth', '--model', weights+'/sequence_matters/spynet_sintel_final-3d2a1287.pth',
        '--output', '{ROOT}/'+seq+'/bundle.json'), seq+'/bundle.json', 'metadata')
    gates = []
    for method, producer in [('iesrgs','ie_merge'), ('splatsure','splat_bundle'), ('sequence_matters','sequence_bundle')]:
        name = method+'_bundle_gate'; gates.append(name)
        add(name, method, [producer], script(base_python, 'train_stage.py', '--config', configs[method], '--stage', 'sr',
            '--prior-bundle', '{ROOT}/'+paths[method]+'/bundle.json', '--check-only', '--out',
            '{ROOT}/'+paths[method]+'/bundle_gate'), paths[method]+'/bundle_gate', 'metadata',
            'Full manifest/LR bytes/producer/model/target hashes verified by the production SR preflight.')
    train('threedsr_lr', 'threedsr', 'lr', [], three+'/lr')
    for method in ('iesrgs', 'splatsure', 'sequence_matters'):
        train(method+'_sr', method, 'sr', list(gates), paths[method]+'/sr', paths[method]+'/bundle.json')
    add('threedsr_coupled', 'threedsr', [*gates, 'threedsr_lr'], script(base_python, 'run_threedsr.py',
        '--config', configs['threedsr'], '--records', records, '--lr-checkpoint', '{ROOT}/'+three+'/lr/checkpoint.pt',
        '--upstream', up+'/threedsr', '--stable-checkpoint', weights+'/threedsr/stablesr_turbo.ckpt',
        '--vqgan-checkpoint', weights+'/stablesr/vqgan_cfw_00011.ckpt', '--teacher-python', stable_python,
        '--runtime-assets', token(runtime_assets), '--out', '{ROOT}/'+three+'/coupled'), three+'/coupled', 'coupled_diffusion_training',
        'Explicit exception: four actual diffusion/5000-iteration shared-Wu rounds; per-round target validation, 3D feedback, final render_eval.')
    seen = set()
    for n in nodes:
        if not set(n['depends_on']) <= seen: raise ValueError('DAG order is not topological')
        seen.add(n['id'])
    return run, nodes, paths, gates


def storage_estimates(nodes, observations, lr_wh, hr_wh, budget):
    """Known serialized payloads are lower bounds; unknown peaks stay unknown."""
    pixels = math.prod(hr_wh) * observations
    estimates = {}
    for n in nodes:
        kind = n['storage_kind']
        known = (pixels*16 + observations*2*128) if kind == 'ie_float_arrays' else 0
        if kind == 'splat_weights': known = math.prod(lr_wh)*observations*4
        entry = budget.get('stages', {}).get(n['id'], {}) if budget else {}
        allowance = entry.get('additional_bytes')
        if allowance is not None and (not isinstance(allowance, int) or isinstance(allowance, bool) or allowance < known):
            raise ValueError('Capacity budget is below known payload: '+n['id'])
        if allowance is not None and (entry.get('basis') not in ('measured', 'conservative_bound') or not entry.get('evidence')):
            raise ValueError('Budget requires a measured/conservative basis and hashed evidence: '+n['id'])
        for evidence in entry.get('evidence', []): checked(evidence)
        estimates[n['id']] = dict(known_payload_lower_bound_bytes=known, additional_bytes=allowance,
            uncompressed_float32_payload_bytes=pixels*16 if kind in ('ie_float_arrays','ie_lossless_arrays') else None,
            basis=entry.get('basis', 'unknown'), evidence=entry.get('evidence', []),
            limitations='Includes output growth and temporary peaks. PNG compression, checkpoint point count and diffusion latent overhead require a supplied bounded budget; no extrapolated compression guarantee.')
    return estimates


def check_disk(needed, reserve, path):
    probe = resolve(path)
    while not probe.exists(): probe = probe.parent
    usage = shutil.disk_usage(probe)
    return dict(path=str(probe), free_bytes=usage.free, required_additional_bytes=needed,
                reserve_bytes=reserve, passed=needed is not None and usage.free >= needed+reserve,
                observed_utc=datetime.now(timezone.utc).isoformat())


def cleanup_contract(run, paths, cameras):
    allow = []
    for m in METHODS:
        p = paths[m]
        if m == 'splatsure':
            allow += [p+'/teacher/generated', p+'/contribution/maps', p+'/lr/checkpoint.pt']
        if m == 'iesrgs':
            allow += [p+'/internal_train/checkpoint.pt']
            for name in ('external','internal_cache'):
                allow += [p+'/'+name+'/'+camera for camera in cameras]
        if m == 'sequence_matters': allow += [p+'/teacher/'+camera for camera in cameras]
        if m != 'threedsr': allow += [p+'/sr/checkpoint.pt', p+'/sr/evaluation/renders']
        else:
            allow += [p+'/lr/checkpoint.pt', p+'/coupled/latents', p+'/coupled/evaluation/renders']
            for index in range(4):
                allow += [p+f'/coupled/teacher_{index}/'+camera for camera in cameras]
                allow += [p+f'/coupled/reconstruction_{index}/checkpoint.pt']
                if index < 3: allow += [p+f'/coupled/reconstruction_{index}/feedback']
    if any(not Path(p).is_relative_to(run) or '..' in Path(p).parts for p in allow):
        raise ValueError('Cleanup allowlist escapes this task scene')
    return dict(execute=False, allowlist=allow,
        gates=['All four methods: completed_with_evaluation and exactly frames 0..299 of held-out cam00 with finite PSNR/SSIM/LPIPS',
               'Compact results collected with SHA256 and verified after transfer to the local controller',
               'No active job or writer uses these paths; root alone executes exact allowlist deletion'],
        preserve=['compact metrics/config/source and model identities/cost/receipt/log records', 'original data', 'generic weights',
                  'staged input hardlinks and small runtime/workdir references to generic assets', 'other scenes/tasks', 'shared environments/upstreams'],
        checkpoint_retention=False, png_retention=False, transfer_full_caches=False,
        note='Collection is not a verified return. No cleanup is authorized by remote completion alone.')


def create_plan(args):
    sys.path.insert(0, str(ROOT/EXP))
    from common import training_manifest, validate_records_bundle
    prepared = resolve(args.prepared)
    configs = {m: read(prepared/args.scene/(m+'.json')) for m in METHODS}
    cfg = configs['splatsure']; manifest_path = checked(cfg['manifest'])
    manifest, rows = training_manifest(manifest_path, verify_pixels=True)
    if manifest['scene'] != SCENES[args.scene] or manifest['frame_indices'] != list(range(300)):
        raise ValueError('Use this full original 300-frame scene, never a wiring fixture')
    if manifest['splits']['test'] != ['cam00'] or cfg.get('teacher_variant') != 'stablesr':
        raise ValueError('Require fixed cam00 test and StableSR main recipe')
    for method, c in configs.items():
        expected = (0,30000) if method == 'sequence_matters' else (30000,20000 if method == 'threedsr' else 30000)
        if (c['method'] != method or c['manifest'] != cfg['manifest'] or c['scene'] != cfg['scene']
                or (c['specification']['lr_steps'],c['specification']['sr_steps']) != expected):
            raise ValueError('Methods must retain own identities and full source schedules')
        checked(c['source_configuration'])
    record_path = prepared/args.scene/'training_records.json'
    validate_records_bundle(read(record_path), cfg['manifest'], manifest, rows)
    sequence_plan = args.sequence_plan or f'output/static_sr_wu_20261022/cts_cpu_prior_plans/{args.scene}_full300/receipt.json'
    run, nodes, paths, gates = build_dag(args.scene, args.run_id, prepared, sequence_plan, args.runtime_assets,
                                        args.base_python, args.stable_python, args.sequence_python,
                                        getattr(args,'iesrgs_cache_format','npy'))
    for node in nodes:
        node['argv'] = [token(manifest_path) if x == '__REGISTERED_MANIFEST__' else x for x in node['argv']]
    blockers, accepted = [], []
    receipts = [('wiring',args.wiring_receipt), ('external',args.external_receipt), ('metrics',args.metric_receipt)]
    teachers = {}
    for file in args.teacher_receipt:
        r = read(file)
        teachers.setdefault(r.get('method'), []).append(file)
    for kind in ('stablesr','sequence_matters','threedsr'):
        candidates = teachers.get(kind, [])
        if not candidates: receipts.append((kind,None))
        else:
            found = False; errors = []
            for file in candidates:
                try:
                    validate_acceptance(kind,read(file),args.gpu_uuid,manifest['resolutions']['hr'])
                    receipts.append((kind,file));found=True;break
                except (KeyError,ValueError,OSError) as exc: errors.append(str(exc))
            if not found: blockers.append(kind+': no valid receipt for these dimensions: '+'; '.join(errors))
    for kind, file in receipts:
        if not file: blockers.append(kind+': actual passing GPU receipt missing');continue
        try:
            validate_acceptance(kind,read(file),args.gpu_uuid,manifest['resolutions']['hr'])
            accepted.append(dict(kind=kind,**identity(file)))
        except (KeyError,ValueError,OSError) as exc: blockers.append(kind+': '+str(exc))
    for name, file in [('base_python',args.base_python), ('stable_python',args.stable_python), ('sequence_python',args.sequence_python),
                       ('sequence_plan',sequence_plan), ('runtime_assets',args.runtime_assets)]:
        if not resolve(file).is_file(): blockers.append(name+': missing '+str(file))
    if resolve(sequence_plan).is_file():
        from methods.sequence_matters import validate_sequence_plan
        validate_sequence_plan(read(sequence_plan), read(record_path), ROOT/'tmp/static_sr_wu_20261022/upstream/sequence_matters/vsr/utils_vsr.py')
    budget = read(args.budget) if args.budget else None
    if budget and (budget.get('schema') != 'static_sr_scene_capacity_budget_v1' or budget.get('scene') != args.scene):
        raise ValueError('Capacity budget belongs to a different scene/schema')
    storage = storage_estimates(nodes,len(rows),manifest['resolutions']['lr'],manifest['resolutions']['hr'],budget)
    unknown = [name for name,s in storage.items() if s['additional_bytes'] is None]
    if unknown: blockers.append('Unknown stage storage budgets: '+', '.join(unknown))
    total = sum(s['additional_bytes'] for s in storage.values()) if not unknown else None
    disk = check_disk(total,args.reserve_bytes,run)
    if not disk['passed']: blockers.append('Current snapshot filesystem does not pass full-scene retained-output capacity gate')
    if resolve(run).exists(): blockers.append('Run output already exists; preserve it and choose a new run-id')
    required = [identity(p) for p in sorted((ROOT/EXP).rglob('*.py'))]
    required += [identity(ROOT/SELF),identity(ROOT/'deployment/static_sr_cts_20261022/run_job.py'),
                 identity(manifest_path),identity(record_path),*[identity(prepared/args.scene/(m+'.json')) for m in METHODS],
                 *[dict(path=r['path'],sha256=r['sha256']) for r in accepted]]
    compact_metadata = [dict(role='manifest', **identity(manifest_path)),
                        dict(role='training_records', **identity(record_path))]
    compact_metadata += [dict(role='source_configuration:'+m, **identity(checked(c['source_configuration'])))
                         for m,c in configs.items()]
    for role,file in [('sequence_plan',sequence_plan),('runtime_assets',args.runtime_assets)]:
        if resolve(file).is_file(): compact_metadata.append(dict(role=role, **identity(file)))
    compact_metadata += [dict(role='GPU_acceptance:'+r['kind'], path=r['path'],sha256=r['sha256']) for r in accepted]
    required += [dict(path=r['path'],sha256=r['sha256']) for r in compact_metadata]
    if budget: required += [identity(args.budget)]
    result = dict(schema='static_sr_scene_execution_plan_v1',status='blocked' if blockers else 'executable_after_resource_admission',
        scene=args.scene, scene_name=manifest['scene'], run_id=args.run_id, run_root=run, root=str(ROOT),
        scene_priority=list(PRIORITY), one_scene_at_a_time=True, blockers=blockers,
        gpu_uuid=args.gpu_uuid, training_observations=len(rows), frames=300, train_cameras=manifest['splits']['train'],
        heldout_cameras=['cam00'], LR_identity=cfg['manifest'], required_files=required,
        accepted_receipts=accepted, storage=storage, disk=disk, reserve_bytes=args.reserve_bytes,
        dag=nodes, static_bundle_barrier=gates, separate_LR_parents=True,
        threeDSR_exception='Four validated per-round diffusion caches interleave with 4x5000 shared-Wu iterations; standalone final SR is forbidden.',
        evaluation='Production train_stage/run_threedsr invoke render_eval immediately after successful final training; do not duplicate scoring.',
        cleanup=cleanup_contract(run,paths,manifest['splits']['train']),
        final_outputs={m:paths[m]+('/coupled' if m=='threedsr' else '/sr') for m in METHODS},
        configs={m:identity(prepared/args.scene/(m+'.json')) for m in METHODS},
        compact_metadata=compact_metadata,
        return_policy='Compact metrics, identities, costs and logs only; no checkpoint, PNG, NPY, latent or raw cache transfer.',
        env=dict(PYTHONDONTWRITEBYTECODE='1',TORCH_HOME='{ROOT}/weight/static_sr_wu_20261022/evaluation/torch',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
        GPU_calls=0, submitted=False, cleanup_executed=False)
    return result


def materialize(args, plan):
    out = resolve(args.out);out.mkdir(parents=True,exist_ok=False)
    plan_path = out/'plan.json';write_new(plan_path,plan)
    template = dict(schema='static_sr_scene_capacity_budget_v1',scene=plan['scene'],
        note='Fill all stages with measured or conservative-bound additional bytes and hashed supporting evidence; unknown is deliberately blocked.',
        stages={n['id']:dict(additional_bytes=None,basis='unknown',evidence=[],
            known_payload_lower_bound_bytes=plan['storage'][n['id']]['known_payload_lower_bound_bytes']) for n in plan['dag']})
    write_new(out/'capacity_budget_template.json',template)
    if plan['status'] == 'blocked': return
    commands=[]
    for node in plan['dag']:
        commands.append(dict(name=node['id']+'_disk_guard',argv=[args.base_python,'{ROOT}/'+SELF,'guard','--plan',token(plan_path),'--stage',node['id']]))
        commands.append(dict(name=node['id'],argv=node['argv']))
    commands.append(dict(name='collect_compact_results',argv=[args.base_python,'{ROOT}/'+SELF,'collect','--plan',token(plan_path)]))
    job_out=plan['run_root']+'/job_logs'
    job=dict(job_id='static-sr-'+plan['scene'].lower()+'-'+plan['run_id'],method='four_static_SR_Wu_adaptations',scene=plan['scene'],
        gpu_uuid=plan['gpu_uuid'],shared_project_root=args.shared_project_root,training_only_inputs=True,no_proposed_modules=True,
        heldout_HR_use='evaluation_only',out=job_out,env=plan['env'],required_files=plan['required_files']+[identity(plan_path)],
        commands=commands,return_paths=[plan['run_root']+'/compact',job_out+'/status.json',job_out+'/complete.json',job_out+'/job.json',job_out+'/diagnostics.json'])
    write_new(out/'job.json',job)


def guard(plan_path, stage):
    plan=read(plan_path)
    if plan['status']!='executable_after_resource_admission': raise ValueError('Blocked plan cannot run')
    for item in plan['required_files']: checked(item)
    node=next(n for n in plan['dag'] if n['id']==stage)
    free=check_disk(plan['storage'][stage]['additional_bytes'],plan['reserve_bytes'],plan['run_root'])
    if not free['passed']: raise RuntimeError('Disk guard refused stage: '+json.dumps(free))
    # Source producer/trainer CLIs enforce all dependency content/complete-cache contracts.
    print(json.dumps(dict(stage=node['id'],status='disk_guard_passed',**free)))


def verify_final_metrics(plan):
    outputs={}
    for method, directory in plan['final_outputs'].items():
        complete=read(directory+'/complete.json');metrics=read(directory+'/evaluation/metrics.json')
        if complete.get('status')!='completed_with_evaluation' or complete.get('wiring_fixture') is not False:
            raise ValueError(method+': final training/evaluation incomplete')
        if complete.get('config') != plan['configs'][method]:
            raise ValueError(method+': final result belongs to another registered method/configuration')
        if method=='threedsr':
            rounds=complete.get('rounds',[])
            if [r.get('round') for r in rounds]!=list(range(4)):
                raise ValueError('3DSR: four coupled rounds did not complete')
            final_checkpoint=rounds[-1]['checkpoint']
        else:
            if complete.get('steps')!=30000:raise ValueError(method+': abbreviated training is not a final result')
            final_checkpoint=complete.get('checkpoint')
        if not final_checkpoint or metrics.get('checkpoint')!=final_checkpoint:
            raise ValueError(method+': evaluator checkpoint differs from final training checkpoint')
        if (metrics.get('schema')!='static_sr_wu_uniform_png_metrics_v1' or metrics.get('scene')!=plan['scene_name']
                or metrics.get('camera')!='cam00' or sorted(r['frame'] for r in metrics['frames'])!=list(range(300))):
            raise ValueError(method+': full cam00 metric coverage missing')
        if any(not math.isfinite(float(row[k])) for row in [metrics['metrics'],*metrics['frames']] for k in ('PSNR','SSIM','LPIPS')):
            raise ValueError(method+': nonfinite metrics')
        for key in ('PSNR','SSIM','LPIPS'):
            mean=sum(float(row[key]) for row in metrics['frames'])/300
            if not math.isclose(mean,float(metrics['metrics'][key]),rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError(method+': aggregate metrics differ from complete per-frame results')
        outputs[method]=dict(completion=identity(directory+'/complete.json'),metrics=identity(directory+'/evaluation/metrics.json'),
                             values=metrics['metrics'],checkpoint_identity=metrics['checkpoint'])
    return outputs


def collect(plan_path):
    plan=read(plan_path)
    if plan['status']!='executable_after_resource_admission':raise ValueError('Blocked plan cannot collect formal results')
    outputs=verify_final_metrics(plan)
    if not check_disk(512*MIB,plan['reserve_bytes'],plan['run_root'])['passed']:
        raise RuntimeError('Insufficient disk for bounded compact collection')
    metadata=plan['compact_metadata']
    roles={v['role'] for v in metadata}
    if not {'manifest','training_records',*('source_configuration:'+m for m in METHODS)} <= roles:
        raise ValueError('Compact collection requires original manifest, training records and all source configurations')
    metadata_paths=[checked(v) for v in metadata]
    if any(path.suffix!='.json' or path.stat().st_size>64*MIB for path in metadata_paths):
        raise ValueError('Compact input metadata must be bounded JSON records, never large assets')
    root=resolve(plan['run_root']);out=root/'compact';out.mkdir(exist_ok=False)
    selected=[resolve(plan_path),*[checked(v) for v in plan['configs'].values()],*metadata_paths]
    allowed={'.json','.jsonl','.log'}; omitted=[]
    for path in root.rglob('*'):
        if not path.is_file() or path.is_relative_to(out) or path.suffix not in allowed: continue
        # run_job writes its terminal status only after this subprocess returns;
        # that final status is returned separately by submit_remote.
        if path==root/'job_logs/status.json': continue
        if path.is_symlink() or any(part in ('execution_workdir','inputs') or part.endswith('_runtime') for part in path.relative_to(root).parts): continue
        if path.stat().st_size>64*MIB: omitted.append(dict(path=relative(path),bytes=path.stat().st_size,sha256=sha(path),reason='log larger than compact per-file cap'));continue
        selected.append(path)
    unique={str(p):p for p in selected}
    if sum(p.stat().st_size for p in unique.values())>512*MIB:
        raise RuntimeError('Compact records exceed 512 MiB; explicitly review log retention, never copy large caches')
    copied=[]
    for path in unique.values():
        destination=out/'records'/relative(path);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        if sha(path)!=sha(destination):raise RuntimeError('Compact copy identity mismatch')
        copied.append(dict(path=str(destination.relative_to(out)),bytes=destination.stat().st_size,sha256=sha(destination),original_path=relative(path)))
    write_new(out/'index.json',dict(schema='static_sr_scene_compact_results_v1',status='remote_collected_not_return_verified',
        scene=plan['scene'],run_id=plan['run_id'],metrics=outputs,files=copied,omitted_large_logs=omitted,
        input_metadata=metadata,job_terminal_status='Returned separately after collect exits; in-progress job_logs/status.json is excluded',
        bytes=sum(r['bytes'] for r in copied),local_return_verified=False,cleanup_allowed=False))
    print(json.dumps(dict(status='compact_collected',path=str(out),files=len(copied))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--scene',choices=PRIORITY,default='Discussion07');p.add_argument('--run-id',required=True)
    p.add_argument('--prepared',default='output/static_sr_wu_20261022/cts_prepared');p.add_argument('--out',required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--shared-project-root',default='/home/cts/Project/4DSR')
    p.add_argument('--base-python',required=True);p.add_argument('--stable-python',default=str(ROOT/'output/static_sr_wu_20261022/runtime/stablesr_env/bin/python'))
    p.add_argument('--sequence-python',default=str(ROOT/'output/static_sr_wu_20261022/runtime/sequence_env/bin/python'))
    p.add_argument('--runtime-assets',default='output/static_sr_wu_20261022/runtime/cts_bootstrap/runtime_assets_cts.json')
    p.add_argument('--sequence-plan');p.add_argument('--wiring-receipt');p.add_argument('--external-receipt');p.add_argument('--metric-receipt')
    p.add_argument('--teacher-receipt',action='append',default=[]);p.add_argument('--budget');p.add_argument('--reserve-bytes',type=int,default=20*GIB)
    p.add_argument('--iesrgs-cache-format',choices=['npy','npz_deflate'],default='npy')
    for action in ('guard','collect'):
        q=sub.add_parser(action);q.add_argument('--plan',required=True)
        if action=='guard':q.add_argument('--stage',required=True)
    args=parser.parse_args()
    if args.action=='guard':guard(args.plan,args.stage)
    elif args.action=='collect':collect(args.plan)
    else:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) or args.reserve_bytes<0:raise ValueError('Unsafe run-id or negative reserve')
        relative(args.out)
        plan=create_plan(args);materialize(args,plan)
        print(json.dumps(dict(status=plan['status'],scene=plan['scene'],out=args.out,blockers=plan['blockers'])))
        if plan['status']=='blocked':raise SystemExit(2)


if __name__=='__main__':main()
