"""CPU-only final audit of the completed four-branch MeetRoom control batch.

Refuses an absent/nonfinal run complete.json before importing torch or loading
any checkpoint. Reads saved artifacts only: no GPU, rendering, or training.
Writes checkpoint_index.json and audit.json after the completion gate passes.
"""
from __future__ import annotations

import argparse
import ast
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'output/dynamic_sr_20260923/meeting_controls_v1'
BRANCHES = list('ABCD')
STEPS = [1200, 3000, 6000]
POINTS = 88944
GPU0 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
GPU1 = 'GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec'
EXPECTED_PARENT_SHA = '21e14ab525507832e79fc0449277cf36e1da17c5ed867f8cbb20847e5be45e6e'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(a, b, label):
    """Exact tree comparison, with CPU tensor values and no tolerance."""
    import numpy as np
    import torch
    if torch.is_tensor(a):
        require(torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), label)
    elif isinstance(a, np.ndarray):
        require(isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), label)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), label + ': keys')
        for k in a:
            equal(a[k], b[k], f'{label}.{k}')
    elif isinstance(a, (tuple, list)):
        require(isinstance(b, (tuple, list)) and len(a) == len(b), label + ': length')
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f'{label}[{i}]')
    else:
        require(a == b, label)


def tensor_finiteness(tree, label):
    import torch
    count = elements = 0
    def visit(value, path):
        nonlocal count, elements
        if torch.is_tensor(value):
            require(value.device.type == 'cpu', f'{path}: non-CPU tensor')
            require(bool(torch.isfinite(value).all()), f'{path}: nonfinite tensor')
            count += 1
            elements += value.numel()
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(v, f'{path}.{k}')
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                visit(v, f'{path}[{i}]')
        elif isinstance(value, float):
            require(math.isfinite(value), f'{path}: nonfinite scalar')
    visit(tree, label)
    return {'all_finite': True, 'cpu_tensor_count': count, 'tensor_elements': elements}


def capture_layout(source):
    tree = ast.parse(Path(source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'capture')
    result = next(n for n in ast.walk(method) if isinstance(n, ast.Return))
    fields = [ast.unparse(n) for n in result.value.elts]
    require(len(fields) == 14 and fields.index('self.optimizer.state_dict()') == 12,
            'Unexpected historical capture tuple; inspect source before auditing')
    return fields


def replay_sampling(manifest, seed, prior_cameras):
    # The existing loader preserves manifest observation order; no image load.
    records = [o for o in manifest['observations'] if o['split'] == 'train']
    ids = [i for i, o in enumerate(records) if o['camera_id'] in set(prior_cameras.split(','))]
    require(len(records) == 660 and len(ids) == 240, 'Unexpected sampling population')
    lr, sr = random.Random(seed + 177), random.Random(seed + 211)
    exposure = collections.Counter()
    digest = hashlib.sha256()
    result = {}
    for step in range(1, max(STEPS) + 1):
        li, si = lr.randrange(len(records)), sr.choice(ids)
        digest.update(f'{li},{si}\n'.encode())
        o = records[si]
        exposure[f'{o["camera_id"]}/{o["frame_index"]}'] += 1
        if step in STEPS:
            result[step] = {'draw_sha256': digest.hexdigest(), 'lr': lr.getstate(), 'sr': sr.getstate(),
                            'source_prefix_exposure': {}, 'additional_exposure': dict(exposure)}
    return result


def eval_hardware_evidence(run, branch, step, provisional=False):
    prefix = 'provisional_3090' if provisional else 'eval'
    receipt = run / 'parallel_events' / f'parallel_{branch}_{prefix}_{step}.json'
    expected = GPU1 if provisional else GPU0
    if receipt.exists():
        data = load_json(receipt)
        require(data.get('status') == 'completed' and data.get('returncode') == 0, f'Unfinished evaluation receipt: {receipt}')
        require(data.get('gpu') == expected, f'Evaluation GPU differs: {receipt}')
        command = data['command']
        require(Path(command[command.index('--checkpoint') + 1]).resolve() == (run / branch / f'checkpoint_{step}.pt').resolve(),
                f'Evaluation receipt references a different checkpoint: {receipt}')
        require(Path(command[command.index('--out') + 1]).resolve() == (run / branch / f'{prefix}_{step}').resolve(),
                f'Evaluation receipt references a different output: {receipt}')
        return {'gpu_uuid': data['gpu'], 'evidence_path': str(receipt), 'evidence_sha256': sha(receipt)}
    require(not provisional and branch in ['A', 'B'], f'Missing parallel evaluation receipt: {receipt}')
    receipt = run / 'status_serial_snapshot.json'
    data = load_json(receipt)
    require(data.get('gpu_uuid') == GPU0, 'Serial evaluator GPU not recorded as GPU0')
    event = next((e for e in data['events'] if e['label'] == f'{branch}_eval_{step}'), None)
    require(event is not None and event.get('returncode') == 0, f'Missing successful serial evaluation event {branch}/{step}')
    command = event['command']
    require(Path(command[command.index('--checkpoint') + 1]).resolve() == (run / branch / f'checkpoint_{step}.pt').resolve(),
            'Serial evaluation checkpoint differs')
    require(Path(command[command.index('--out') + 1]).resolve() == (run / branch / f'eval_{step}').resolve(),
            'Serial evaluation output differs')
    return {'gpu_uuid': GPU0, 'evidence_path': str(receipt), 'evidence_sha256': sha(receipt), 'event_label': event['label']}


def validate_metrics(path, checkpoint, checkpoint_sha, metadata, manifest_sha):
    m = load_json(path)
    require(Path(m['checkpoint']).resolve() == checkpoint.resolve(), f'Eval checkpoint path mismatch: {path}')
    require(m['checkpoint_sha256'] == checkpoint_sha, f'Eval checkpoint SHA mismatch: {path}')
    require(m['manifest_sha256'] == manifest_sha and m['scene'] == 'meetroom_discussion', f'Eval manifest mismatch: {path}')
    require(m['gaussian_count'] == POINTS and m['test_camera'] == 'cam00', f'Eval model/split mismatch: {path}')
    require(m['frame_indices'] == list(range(0, 120, 2)), f'Eval frame declaration mismatch: {path}')
    require([r['frame_index'] for r in m['rows']] == list(range(0, 120, 2)), f'Eval actual rows mismatch: {path}')
    require(len(list((path.parent / 'predictions').glob('*.png'))) == 60, f'Incomplete predictions: {path}')
    equal(m['checkpoint_metadata'], metadata, f'{path}: checkpoint metadata')
    for key in ['psnr_mean', 'ssim_mean', 'lpips_alex_mean']:
        require(isinstance(m['aggregate']['full'][key], (int, float)) and math.isfinite(m['aggregate']['full'][key]),
                f'Missing/nonfinite full metric {key}: {path}')
    return m


def metric_difference(unified, provisional):
    def numeric_deltas(a, b, prefix=''):
        result = {}
        for key in a.keys() & b.keys():
            x, y = a[key], b[key]
            p = f'{prefix}.{key}' if prefix else key
            if isinstance(x, dict) and isinstance(y, dict):
                result.update(numeric_deltas(x, y, p))
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                require(math.isfinite(x) and math.isfinite(y), f'Nonfinite metric {p}')
                result[p] = x - y
        return result
    frames = []
    for x, y in zip(unified['rows'], provisional['rows']):
        require(x['frame_index'] == y['frame_index'], 'Hardware comparison frame ordering mismatch')
        frames.append({'frame': x['frame_index'], **{key: x['spatial']['full'][key] - y['spatial']['full'][key]
                                                    for key in ['psnr', 'ssim', 'lpips_alex']}})
    return {'sign': 'unified_GPU0 minus provisional_RTX3090; same trained checkpoint',
            'spatial_aggregate_deltas': numeric_deltas(unified['aggregate'], provisional['aggregate']),
            'temporal_aggregate_deltas': numeric_deltas(unified['temporal_aggregate'], provisional['temporal_aggregate']),
            'per_frame_full_metric_deltas': frames,
            'maximum_absolute_per_frame_difference': {k: max(abs(r[k]) for r in frames) for k in ['psnr', 'ssim', 'lpips_alex']}}


def audit(run, output, completion):
    # Only after completed_and_evaluated gate. Do not import common, GaussianModel,
    # or CUDA extensions: torch.load(map_location='cpu') is sufficient.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'CUDA already initialized')
    started = time.monotonic()
    index, details, hardware_diffs, branch_config = [], [], [], {}
    outcome = {'status': 'running', 'run': str(run), 'started_utc': now(), 'source_sha256': sha(__file__),
               'run_complete_sha256': sha(run / 'complete.json'), 'cpu_only': True, 'no_hr_images_read': True,
               'completion_record': completion,
               'optimizer_steps': 0, 'checkpoint_writes': 0, 'global_rng_cross_branch_equality_required': False,
               'training_hardware_confound': {'present': True, 'A_B_C': 'GPU0 RTX PRO 6000 Blackwell', 'D': 'GPU1 RTX 3090',
                   'interpretation': 'Unified GPU0 evaluation controls evaluation hardware only. It does not remove effects of different training hardware on floating-point reductions and optimization trajectories. Small D-vs-other gains cannot be attributed exclusively to the teacher without a same-training-hardware control.'},
               'environment': {'python': sys.executable, 'python_version': platform.python_version(), 'torch': str(torch.__version__)}}
    try:
        # A recovered dispatcher may leave failed.json and an old failure snapshot.
        # Those preserve the incident; successful final receipts are authoritative.
        recovery_path = run / 'recovery.json'
        if recovery_path.exists():
            recovery = load_json(recovery_path)
            require(recovery['status'] == 'completed_and_evaluated', 'Evaluation recovery is unfinished')
            outcome['evaluation_recovery'] = {'path': str(recovery_path), 'sha256': sha(recovery_path),
                                               'record': recovery, 'old_dispatcher_success_required': False}
        configs = {b: load_json(run / b / 'config.json') for b in BRANCHES}
        baseline = configs['A']
        parent_path = Path(baseline['checkpoint']).resolve()
        parent_sha = sha(parent_path)
        require(parent_sha == EXPECTED_PARENT_SHA, 'Registered LR parent identity differs')
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        manifest_path = Path(baseline['manifest']).resolve()
        manifest_sha = sha(manifest_path)
        manifest = load_json(manifest_path)
        cache_path = Path(baseline['cache_manifest']).resolve()
        cache_sha = sha(cache_path)
        calibration_path = Path(baseline['calibration']).resolve()
        calibration = load_json(calibration_path)
        require(calibration['checkpoint_sha256'] == parent_sha and calibration['cache_manifest_sha256'] == cache_sha,
                'Calibration input identities differ')
        require(parent['metadata']['manifest_sha'] == manifest_sha, 'Parent/manifest mismatch')
        sampling = replay_sampling(manifest, baseline['seed'], baseline['prior_cameras'])
        comparison_states = {}
        outcome['inputs'] = {'parent_checkpoint': str(parent_path), 'parent_sha256': parent_sha,
                             'manifest': str(manifest_path), 'manifest_sha256': manifest_sha,
                             'cache_manifest': str(cache_path), 'cache_manifest_sha256': cache_sha,
                             'calibration': str(calibration_path), 'calibration_sha256': sha(calibration_path)}
        source_identity = None
        for branch in BRANCHES:
            config = configs[branch]
            folder = run / branch
            done = load_json(folder / 'complete.json')
            require(done['parameter_updates'] == 6000 and done['full_sampler_states_saved'] and done['source_unchanged'],
                    f'{branch}: training not fully completed')
            require(config['branch'] == branch and config['steps'] == 6000, f'{branch}: branch config mismatch')
            require(config['initial_points'] == POINTS and not config['topology_changes'] and config['new_parameters'] == 0,
                    f'{branch}: model topology config differs')
            for key in ['checkpoint', 'manifest', 'cache_manifest', 'calibration', 'seed', 'prior_cameras', 'milestones', 'scheduler_offset', 'sh_degree']:
                equal(config[key], baseline[key], f'{branch}: common config {key}')
            require(config['weight'] == (calibration['weight_B'] if branch == 'B' else .1), f'{branch}: teacher weight differs')
            require(config['parent_sha256'] == parent_sha and config['manifest_sha256'] == manifest_sha and
                    config['cache_manifest_sha256'] == cache_sha, f'{branch}: input SHA mismatch')
            expected_gpu = GPU1 if branch == 'D' else GPU0
            expected_name = 'RTX 3090' if branch == 'D' else 'RTX PRO 6000'
            require(config['visible_cuda'] == expected_gpu and expected_name in config['gpu'], f'{branch}: unexpected training hardware')
            identities = [(s['path'], s['sha256']) for s in config['sources']]
            if source_identity is None:
                source_identity = identities
            equal(identities, source_identity, f'{branch}: training source versions')
            model_source = None
            for i, record in enumerate(config['sources']):
                snapshot = folder / 'sources' / f'{i}_{Path(record["path"]).name}'
                require(sha(snapshot) == record['sha256'], f'{branch}: source snapshot corrupted: {snapshot}')
                if record['path'].endswith('/scene/gaussian_model.py'):
                    model_source = snapshot
            require(model_source is not None, 'Missing original GaussianModel capture source')
            fields = capture_layout(model_source)
            optimizer_index = fields.index('self.optimizer.state_dict()')
            parent_opt = parent['model'][optimizer_index]
            branch_config[branch] = {'gpu_name': config['gpu'], 'gpu_uuid': config['visible_cuda'],
                                     'config': str(folder / 'config.json'), 'config_sha256': sha(folder / 'config.json'),
                                     'weight': config['weight'], 'source_snapshots_verified': len(identities)}
            require((folder / 'checkpoint_final.pt').resolve() == (folder / 'checkpoint_6000.pt').resolve(),
                    f'{branch}: final checkpoint does not reference fixed 6000 endpoint')
            for step in STEPS:
                path = (folder / f'checkpoint_{step}.pt').resolve()
                digest = sha(path)
                entry = {'branch': branch, 'intervention_step': step, 'path': str(path), 'sha256': digest,
                         'bytes': path.stat().st_size, 'status': 'checking', 'training_gpu': config['gpu'],
                         'training_gpu_uuid': config['visible_cuda'], 'evaluation_gpu_uuid': GPU0}
                index.append(entry)
                ck = torch.load(path, map_location='cpu', weights_only=False)
                require({'model', 'hidden', 'optim', 'metadata', 'rng', 'samplers'} <= ck.keys(), f'{path}: missing checkpoint fields')
                require(len(ck['model']) == len(fields), f'{path}: invalid capture layout')
                require(ck['model'][1].shape[0] == POINTS and ck['model'][0] == parent['model'][0], f'{path}: point/SH change')
                equal(ck['hidden'], parent['hidden'], f'{path}: hidden')
                equal(ck['optim'], parent['optim'], f'{path}: optim config')
                meta = ck['metadata']
                require(meta['scene'] == 'meetroom_discussion' and meta['stage'] == 'temporal_sharing_control', f'{path}: scene/stage')
                require(meta['intervention_step'] == step and meta['step'] == parent['metadata']['step'] + step and meta['points'] == POINTS,
                        f'{path}: step/point metadata')
                require(meta['parent_sha'] == parent_sha and meta['manifest_sha'] == manifest_sha, f'{path}: lineage metadata')
                for key, value in meta['args'].items():
                    equal(value, config[key], f'{path}: argument {key}')
                finite_model = tensor_finiteness(ck['model'], 'model')
                opt = ck['model'][optimizer_index]
                finite_adam = tensor_finiteness(opt, 'Adam')
                require(opt['state'].keys() == parent_opt['state'].keys(), f'{path}: Adam state identities changed')
                adam_steps = []
                for key, value in opt['state'].items():
                    delta = float(value['step']) - float(parent_opt['state'][key]['step'])
                    require(delta == step, f'{path}: Adam update count mismatch for parameter {key}: {delta}')
                    adam_steps.append(float(value['step']))
                replay = sampling[step]
                for key in ['lr', 'sr', 'source_prefix_exposure', 'additional_exposure']:
                    equal(ck['samplers'][key], replay[key], f'{path}: replayed sampler {key}')
                require(meta['draw_sha256'] == replay['draw_sha256'], f'{path}: draw hash replay mismatch')
                require(sum(ck['samplers']['additional_exposure'].values()) == step, f'{path}: exposure count mismatch')
                if step not in comparison_states:
                    comparison_states[step] = {'samplers': ck['samplers'], 'draw_sha256': meta['draw_sha256'],
                                               'optimizer_param_groups': opt['param_groups']}
                else:
                    equal(ck['samplers'], comparison_states[step]['samplers'], f'{path}: cross-branch sampler/exposure')
                    equal(meta['draw_sha256'], comparison_states[step]['draw_sha256'], f'{path}: cross-branch draw hash')
                    equal(opt['param_groups'], comparison_states[step]['optimizer_param_groups'], f'{path}: optimizer groups/schedule')
                if step == 6000:
                    equal(ck['samplers']['additional_exposure'], load_json(folder / 'exposure.json'), f'{path}: final exposure file')
                    require(done['draw_sha256'] == meta['draw_sha256'], f'{path}: final completion draw hash')
                metric_path = folder / f'eval_{step}' / 'metrics.json'
                metrics = validate_metrics(metric_path, path, digest, meta, manifest_sha)
                evidence = eval_hardware_evidence(run, branch, step)
                require((folder / f'fit_{step}.json').is_file() and (folder / f'own_target_fit_{step}.json').is_file(),
                        f'{path}: missing fixed train-view diagnostics')
                entry.update(status='verified', step=meta['step'], points=POINTS,
                             draw_sha256=meta['draw_sha256'], evaluation_metrics=str(metric_path),
                             evaluation_metrics_sha256=sha(metric_path))
                details.append({'branch': branch, 'intervention_step': step, 'model_finiteness': finite_model,
                                'adam_finiteness': finite_adam, 'adam_step_min': min(adam_steps), 'adam_step_max': max(adam_steps),
                                'model_and_optimizer_config_match_parent': True, 'sampler_replay_exact': True,
                                'exposure_total': step, 'evaluation_identity_match': True, 'evaluation_hardware': evidence,
                                'global_rng_comparison': 'Not required; checkpoint global RNG is preserved but hardware-dependent equality is not a fairness condition here'})
                if branch == 'D':
                    provisional_path = folder / f'provisional_3090_{step}' / 'metrics.json'
                    provisional = validate_metrics(provisional_path, path, digest, meta, manifest_sha)
                    provisional_evidence = eval_hardware_evidence(run, branch, step, provisional=True)
                    hardware_diffs.append({'step': step, 'checkpoint_sha256': digest,
                                           'unified_metrics': str(metric_path), 'provisional_metrics': str(provisional_path),
                                           'provisional_metrics_sha256': sha(provisional_path),
                                           'provisional_hardware': provisional_evidence, **metric_difference(metrics, provisional)})
                print(json.dumps({'event': 'checkpoint_verified', 'branch': branch, 'step': step, 'sha256': digest}), flush=True)
                del ck
        require(len(index) == 12 and all(r['status'] == 'verified' for r in index), 'Expected all twelve checkpoints')
        require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
        outcome.update(status='passed', checkpoints=details, training_hardware=branch_config,
                       cross_branch_sampling='Exact LR/SR random state, draw hash, and exposure equality at all three endpoints, independently verified by manifest-order replay',
                       D_evaluation_hardware_differences=hardware_diffs,
                       capture_fields=fields, optimizer_capture_index=optimizer_index, tensor_loading='torch.load(map_location=cpu, weights_only=False)',
                       elapsed_seconds=time.monotonic() - started, finished_utc=now())
    except Exception:
        if index and index[-1]['status'] == 'checking':
            index[-1]['status'] = 'failed'
        outcome.update(status='failed', traceback=traceback.format_exc(), checkpoints=details,
                       training_hardware=branch_config, D_evaluation_hardware_differences=hardware_diffs,
                       elapsed_seconds=time.monotonic() - started, finished_utc=now())
        raise
    finally:
        write(output / 'checkpoint_index.json', {'schema': 'dynamic_sr_checkpoint_index_v1', 'run': str(run),
              'run_status': completion['status'], 'audit_status': outcome['status'], 'entries': index,
              'training_hardware_confound': outcome['training_hardware_confound']})
        outcome['checkpoint_index_sha256'] = sha(output / 'checkpoint_index.json')
        write(output / 'audit.json', outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, help='Defaults to run directory; refuses existing audit artifacts')
    args = parser.parse_args()
    run = args.run.resolve()
    marker = run / 'complete.json'
    if not marker.is_file():
        raise SystemExit('REFUSED: run complete.json is absent; no checkpoints were loaded and no audit artifacts were written.')
    completion = load_json(marker)
    if completion.get('status') != 'completed_and_evaluated':
        raise SystemExit('REFUSED: run is not completed_and_evaluated; no checkpoints were loaded and no audit artifacts were written.')
    output = args.out.resolve() if args.out else run
    if any((output / name).exists() for name in ['audit.json', 'checkpoint_index.json']):
        raise SystemExit('REFUSED: audit artifacts already exist; choose a new --out directory to preserve them.')
    output.mkdir(parents=True, exist_ok=True)
    audit(run, output, completion)


if __name__ == '__main__':
    main()
