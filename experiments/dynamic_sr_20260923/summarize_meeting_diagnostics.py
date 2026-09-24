"""Posthoc CPU summaries of a completed A/B/C/D development experiment.

Reads JSON/JSONL only, after the completed_and_evaluated gate. No image,
checkpoint, torch, CUDA, training, metric rerendering, or rule fitting occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'output/dynamic_sr_20260923/meeting_controls_v1'
BRANCHES = 'ABCD'
STEPS = [1200, 3000, 6000]
AUDIT_STEPS = [1, 1200, 3000, 6000]
FRAME_GROUPS = {'interior': [40, 80], 'boundary': [0, 118], 'all': [0, 40, 80, 118]}
FIT_METRICS = ['psnr', 'ssim', 'lpips_alex', 'mse']
IDENTITY_KEYS = ['script_sha256', 'cache_key', 'metric_definitions',
                 'manifest_sha256', 'frame_indices', 'test_camera', 'scene', 'version', 'versions']
GRADIENT_KEYS = ['gradient_norms', 'lr_plus_reg_gradient_norms', 'weighted_sr_gradient_norms']
CANONICAL_GROUPS = {'_xyz': 'xyz', '_features_dc': 'f_dc', '_features_rest': 'f_rest',
                    '_scaling': 'scaling', '_rotation': 'rotation', '_opacity': 'opacity'}
OPTIMIZER_GROUPS = set(CANONICAL_GROUPS.values()) | {'deformation', 'grid'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'Nonfinite or nonnumeric {label}: {value}')
    return float(value)


def average(values):
    require(bool(values), 'Cannot average an empty group')
    return math.fsum(values) / len(values)


def ratio(a, b):
    return a / b if b != 0 else None


def euclidean(values):
    return math.sqrt(math.fsum(v * v for v in values))


def parameter_group(name):
    if name in CANONICAL_GROUPS:
        return CANONICAL_GROUPS[name]
    if name.startswith('deformation.'):
        # Mirrors frozen get_mlp_parameters/get_grid_parameters: all grid names
        # belong to grid; deformation MLP and timenet belong to deformation.
        return 'grid' if 'grid' in name else 'deformation'
    raise ValueError(f'Unknown named parameter, refusing to guess its group: {name}')


def unique_observations(rows, expected, label):
    index = {(r['camera_id'], r['frame_index']): r for r in rows}
    require(len(index) == len(rows), f'Duplicate observation in {label}')
    require(set(index) == expected, f'Unexpected observation identities in {label}')
    return index


def fit_groups(branch, step, fit, own, cameras):
    require(fit['step'] == step, f'{branch}/{step}: fit step mismatch')
    require(own['parameter_updates'] == 0 and own['target_clipped_for_metrics'] is True
            and own['training_uses_raw_float32'] is True, f'{branch}/{step}: own-fit semantics changed')
    expected = {(camera, frame) for camera in cameras for frame in FRAME_GROUPS['all']}
    fi = unique_observations(fit['rows'], expected, f'{branch}/{step}/fit')
    oi = unique_observations(own['rows'], expected, f'{branch}/{step}/own')
    result = {'branch': branch, 'step': step, 'groups': {}}
    for group, frames in FRAME_GROUPS.items():
        keys = sorted(key for key in expected if key[1] in frames)
        row = {'observations': len(keys), 'frames': frames, 'cameras': sorted(cameras),
               'train_LR_psnr_mean': average([finite(fi[k]['lr_psnr'], 'lr_psnr') for k in keys])}
        for field in ['render_hr', 'render_prior']:
            row[field] = {metric: average([finite(fi[k][field][metric], f'{field}/{metric}') for k in keys])
                          for metric in FIT_METRICS}
        row['own_target'] = {metric: average([finite(oi[k]['metrics'][metric], f'own/{metric}') for k in keys])
                             for metric in FIT_METRICS}
        result['groups'][group] = row
    return result


def gradient_summary(branch, records):
    audit = {}
    steps = []
    for row in records:
        step = row['step']
        steps.append(step)
        if step in AUDIT_STEPS:
            require(step not in audit, f'{branch}: duplicate gradient audit step {step}')
            groups = {}
            for key in GRADIENT_KEYS:
                values = {k: finite(v, f'{branch}/{step}/{key}/{k}') for k, v in row[key].items()}
                require(set(values) == OPTIMIZER_GROUPS and min(values.values()) >= 0,
                        f'{branch}/{step}: unexpected/nonpositive group norms in {key}')
                groups[key] = {'by_optimizer_group': values, 'all_group_l2': euclidean(values.values())}
            parameters = {k: finite(v, f'{branch}/{step}/delta/{k}')
                          for k, v in row['actual_parameter_delta_norms'].items()}
            require(bool(parameters) and min(parameters.values()) >= 0, 'Invalid parameter displacement norms')
            grouped = {group: [] for group in OPTIMIZER_GROUPS}
            for name, value in parameters.items():
                grouped[parameter_group(name)].append(value)
            require(all(grouped.values()), f'{branch}/{step}: empty parameter group in displacement mapping')
            group_delta = {k: euclidean(v) for k, v in sorted(grouped.items())}
            audit[step] = {'branch': branch, 'step': step, 'weight': finite(row['weight'], 'weight'),
                           'draw_sha256': row['draw_sha256'], **groups,
                           'actual_parameter_displacement': {
                               'by_optimizer_group': group_delta,
                               'all_parameter_l2': euclidean(parameters.values()),
                               'by_named_parameter': parameters,
                               'nonzero_parameter_tensors': sum(v > 0 for v in parameters.values()),
                               'parameter_tensors': len(parameters)}}
    require(steps == sorted(set(steps)) and steps[-1] == 6000, f'{branch}: incomplete or reordered training log')
    require(set(audit) == set(AUDIT_STEPS), f'{branch}: missing audit steps')
    return [audit[step] for step in AUDIT_STEPS]


def main(run, output):
    started = time.monotonic()
    run = run.resolve()
    output = (output or run / 'supplemental_diagnostics').resolve()
    completion_path = run / 'complete.json'
    # This is deliberately the first experiment artifact read. Do not inspect
    # metrics, fits or training logs before final unified evaluation completes.
    require(completion_path.is_file(), 'Run complete.json is required; no results were read')
    completion_digest = sha(completion_path)
    completion = json.loads(completion_path.read_text())
    require(completion.get('status') == 'completed_and_evaluated',
            'Run is not completed_and_evaluated; no results were read')
    require(not output.exists(), f'Refusing to overwrite: {output}')
    inputs = {str(completion_path): completion_digest, str(Path(__file__).resolve()): sha(__file__)}

    def read(path, lines=False):
        path = Path(path).resolve()
        before = sha(path)
        text = path.read_text()
        require(sha(path) == before, f'Artifact changed during read: {path}')
        inputs[str(path)] = before
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)

    configs = {b: read(run / b / 'config.json') for b in BRANCHES}
    cameras = set(configs['A']['prior_cameras'].split(','))
    require(cameras == {'cam02', 'cam04', 'cam08', 'cam12'}, 'Expected four registered teacher cameras')
    calibration = read(Path(configs['A']['calibration']))
    calibration_summary = {k: calibration[k] for k in ['weight_A', 'weight_B', 'ratio', 'norm_A', 'norm_D', 'rule',
                                                       'checkpoint_sha256', 'cache_manifest_sha256']}
    identities, fit_rows, gradient_rows = [], [], []
    reference = None
    for branch in BRANCHES:
        config = configs[branch]
        require(config['branch'] == branch and config['steps'] == 6000, f'{branch}: wrong branch/length')
        require(set(config['prior_cameras'].split(',')) == cameras, f'{branch}: teacher cameras differ')
        require(config['calibration'] == configs['A']['calibration'], f'{branch}: calibration identity differs')
        require(config['parent_sha256'] == calibration['checkpoint_sha256']
                and config['cache_manifest_sha256'] == calibration['cache_manifest_sha256'],
                f'{branch}: calibration input identities differ')
        expected_weight = calibration['weight_B'] if branch == 'B' else calibration['weight_A']
        require(config['weight'] == expected_weight, f'{branch}: actual weight differs from calibration')
        for step in STEPS:
            metric_path = run / branch / f'eval_{step}' / 'metrics.json'
            metric = read(metric_path)
            identity = {key: metric[key] for key in IDENTITY_KEYS}
            if reference is None:
                reference = identity
            require(identity == reference, f'{branch}/{step}: evaluation identity differs')
            expected_frames = list(range(0, 120, 2))
            require(identity['frame_indices'] == expected_frames and identity['test_camera'] == 'cam00'
                    and identity['scene'] == 'meetroom_discussion', f'{branch}/{step}: wrong evaluation data')
            require([r['frame_index'] for r in metric['rows']] == expected_frames,
                    f'{branch}/{step}: incomplete, duplicate or reordered actual rows')
            require(metric['manifest_sha256'] == config['manifest_sha256'], f'{branch}/{step}: manifest differs')
            require(Path(metric['checkpoint']).resolve() == (run / branch / f'checkpoint_{step}.pt').resolve(),
                    f'{branch}/{step}: checkpoint path differs')
            identities.append({'branch': branch, 'step': step, 'path': str(metric_path),
                               'sha256': inputs[str(metric_path.resolve())], 'checkpoint_sha256': metric['checkpoint_sha256'],
                               'rows': len(metric['rows'])})
            fit_rows.append(fit_groups(branch, step, read(run / branch / f'fit_{step}.json'),
                                       read(run / branch / f'own_target_fit_{step}.json'), cameras))
        gradient_rows.extend(gradient_summary(branch, read(run / branch / 'training.jsonl', lines=True)))

    gradient_index = {(row['branch'], row['step']): row for row in gradient_rows}
    ratios = []
    for step in AUDIT_STEPS:
        a = gradient_index['A', step]
        for branch in 'BCD':
            row = gradient_index[branch, step]
            require(row['draw_sha256'] == a['draw_sha256'], f'{branch}/{step}: sampled draw prefix differs')
            ratios.append({'comparison': f'{branch}/A', 'step': step,
                           'gradient_l2_ratios': {key: ratio(row[key]['all_group_l2'], a[key]['all_group_l2'])
                                                 for key in GRADIENT_KEYS},
                           'actual_displacement_l2_ratio': ratio(row['actual_parameter_displacement']['all_parameter_l2'],
                                                                a['actual_parameter_displacement']['all_parameter_l2']),
                           'displacement_by_group_ratios': {
                               group: ratio(row['actual_parameter_displacement']['by_optimizer_group'][group],
                                            a['actual_parameter_displacement']['by_optimizer_group'][group])
                               for group in sorted(OPTIMIZER_GROUPS)}})
    result = {
        'role': 'posthoc descriptive diagnostics on one completed development scene; no new pass/fail gate',
        'run': str(run), 'run_status': completion['status'], 'cpu_only': True,
        'hr_images_read': False, 'checkpoint_tensors_read': False, 'parameter_updates': 0,
        'evaluation_identity': {'all_twelve_match': len(identities) == 12, 'shared': reference, 'files': identities},
        'calibration': calibration_summary,
        'training_hardware': {b: {key: configs[b][key] for key in ['gpu', 'visible_cuda']} for b in BRANCHES},
        'fit_by_interior_and_boundary': fit_rows,
        'gradient_and_actual_step_displacement': gradient_rows,
        'gradient_and_displacement_ratios_to_A': ratios,
        'parameter_group_mapping': {'canonical': CANONICAL_GROUPS,
                                    'deformation_prefix': 'grid iff grid substring is present; otherwise deformation',
                                    'source': 'common.named_parameters and deformation.get_mlp_parameters/get_grid_parameters'},
        'limitations': [
            'Interior 40/80 and boundary 0/118 are fixed diagnostics, eight observations per group; neither is independent test data.',
            'Boundary C/D targets equal A, but their renders can still change through parameters trained at other times.',
            'Own-target fit uses different targets across branches and clipped image metrics; it is not a common-target quality score or the raw training L1 objective.',
            'Calibration matches an RMS raw SR gradient norm over eight fixed observations once, not total LR+SR gradients, Adam update direction, accumulated optimization or compute.',
            'Recorded parameter displacement is the norm of one actual optimizer step, not cumulative displacement from the parent; later steps use already different model/Adam states.',
            'The aggregate Euclidean norms mix different parameter types and are descriptive; equal magnitudes do not imply equal directions or functional changes.',
            'Unified evaluation hardware does not remove the different D training hardware or give a bound on its optimization effect.',
            'Frame/group comparisons are correlated and single-seed; no iid significance, new thresholds or causal attribution is inferred.',
            'Metric-file identities are validated here; checkpoint tensor lineage/state validation belongs to audit_meeting_checkpoints.py.',
        ],
        'input_sha256': inputs,
        'elapsed_seconds': time.monotonic() - started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    # Detect concurrent replacements before publishing a completed diagnostic.
    for path, digest in inputs.items():
        require(sha(path) == digest, f'Input changed during analysis: {path}')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'source.py')
    write(output / 'summary.json', result)
    write(output / 'complete.json', {'status': 'completed_cpu_diagnostics',
                                    'summary_sha256': sha(output / 'summary.json'),
                                    'source_sha256': sha(__file__), 'parameter_updates': 0,
                                    'cpu_only': True, 'hr_images_read': False,
                                    'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'output': str(output / 'summary.json'), 'metric_files': len(identities),
                      'fit_endpoints': len(fit_rows), 'gradient_audits': len(gradient_rows),
                      'cpu_only': True, 'parameter_updates': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, help='New output directory; defaults to run/supplemental_diagnostics')
    args = parser.parse_args()
    main(args.run, args.out)
