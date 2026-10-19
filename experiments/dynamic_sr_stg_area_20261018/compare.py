#!/usr/bin/env python3
"""CPU-only aggregation of six native STG segment evaluations and Wu controls.

Missing, short, failed, or hash-inconsistent results cannot become full300
metrics. Inter-segment temporal pairs are recomputed from saved raw endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('areaL0', 'areaTm', 'areaOracle')
METRICS = ('psnr', 'ssim', 'lpips', 'mse', 'LR_diagnostic_psnr', 'low_psnr', 'mid_psnr', 'high_psnr')
PATH_MAP = []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Full evaluations can be added later; each aggregation is an explicit snapshot.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_file(entry):
    path = Path(entry['path'])
    for old, new in PATH_MAP:
        if path.is_relative_to(old):
            path = new / path.relative_to(old)
            break
    if not path.is_file() or sha(path) != entry['sha256']:
        raise ValueError(f'Artifact missing or hash changed: {path}')
    return path


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def mean(rows, key):
    values = [r.get(key) for r in rows]
    return statistics.fmean(values) if values and all(finite(v) for v in values) else None


def metrics(rows):
    result = {'observations': len(rows), **{key: mean(rows, key) for key in METRICS}}
    result['PSNR_of_pooled_MSE'] = (-10 * math.log10(max(result['mse'], 1e-12))
                                   if result['mse'] is not None else None)
    return result


def boundary_pair(left, right, manifest_path, manifest, before, after):
    import numpy as np
    from PIL import Image
    lookup = {(r['camera_id'], r['frame_index']): r for r in manifest['observations']}
    predictions, references = [], []
    for summary, frame in ((left, before), (right, after)):
        endpoint = summary['endpoints'][str(frame)]
        row = lookup['cam00', frame]
        if (endpoint.get('frame') != frame or endpoint.get('camera') != 'cam00'
                or endpoint.get('reference_hr_sha256') != row['hr_sha256']):
            raise ValueError('Boundary endpoint identity mismatch')
        with np.load(checked_file(endpoint)) as saved:
            raw = saved['rgb_raw']
            if raw.shape != (3, *manifest['resolutions']['hr'][::-1]) or not np.isfinite(raw).all():
                raise ValueError('Boundary endpoint shape or finite check failed')
            predictions.append(np.clip(raw, 0., 1.).transpose(1, 2, 0))
        hr = (manifest_path.parent / row['hr_path']).resolve()
        if not hr.is_relative_to(manifest_path.parent.resolve()) or sha(hr) != row['hr_sha256']:
            raise ValueError('Boundary HR reference identity mismatch')
        with Image.open(hr) as image:
            references.append(np.asarray(image.convert('RGB'), dtype=np.float32) / 255.)
    residual = (predictions[1] - predictions[0]) - (references[1] - references[0])
    return {'previous_frame': before, 'frame': after, 'boundary': True,
            'frame_difference_residual_l1': float(np.abs(residual).mean()),
            'frame_difference_residual_mse': float((residual ** 2).mean())}


def arm_result(arm, paths, scene, output):
    records, summaries, problems, all_rows, temporal = [], {}, [], [], []
    expected_manifest = scene['manifest']['sha256']
    metric_sources, cam00_mask = None, None
    for path in paths:
        path = Path(path)
        if not path.is_file():
            problems.append({'path': str(path), 'reason': 'summary_missing'})
            continue
        record = {**identity(path), 'status': 'incomplete'}
        records.append(record)
        try:
            summary = read(path)
            start = summary.get('segment_start')
            if (summary.get('status') != 'completed' or summary.get('method') != arm
                    or summary.get('model') != 'STG' or summary.get('manifest', {}).get('sha256') != expected_manifest
                    or summary.get('duration') != 50 or start not in range(0, 300, 50)
                    or summary.get('frame_indices') != list(range(start, start + 50))
                    or summary.get('frames') != 50 or summary.get('lpips_status') != 'alex_v0.1'):
                raise ValueError('Requires completed50-frame STG segment, matching arm/manifest and full LPIPS')
            if start in summaries:
                raise ValueError('Duplicate segment start')
            current_sources = [(Path(s['path']).name, s['sha256']) for s in summary['metric_sources']]
            current_mask = summary['mask_receipts']['cam00']['mask_pixel_sha256']
            if metric_sources is not None and current_sources != metric_sources:
                raise ValueError('Metric implementation changed across segments')
            if cam00_mask is not None and current_mask != cam00_mask:
                raise ValueError('Full300 reference mask changed across segments')
            metric_sources, cam00_mask = current_sources, current_mask
            rows_path = checked_file(summary['per_observation'])
            rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
            tests = [r for r in rows if r['scope'] == 'test']
            if [(r['camera'], r['frame']) for r in tests] != [('cam00', f) for f in range(start, start + 50)]:
                raise ValueError('Test row identities do not cover the registered segment exactly')
            if not all(finite(r.get(k)) for r in tests for k in ('psnr', 'ssim', 'lpips', 'mse')):
                raise ValueError('Missing or nonfinite spatial metric')
            pairs = read(checked_file(summary['temporal_per_pair']))
            if [(r['previous_frame'], r['frame']) for r in pairs] != [(f, f + 1) for f in range(start, start + 49)]:
                raise ValueError('Missing internal consecutive temporal pairs')
            if not all(finite(r.get(k)) for r in pairs for k in ('frame_difference_residual_l1', 'frame_difference_residual_mse')):
                raise ValueError('Missing or nonfinite temporal metric')
            summaries[start] = summary
            all_rows.extend(rows)
            temporal.extend(pairs)
            record.update(status='completed', segment_start=start,
                          checkpoint=summary['checkpoint'], decoder=summary['decoder'],
                          capacity_points=summary.get('capacity_points'), checkpoint_bytes=summary.get('checkpoint_bytes'),
                          gpu=summary.get('gpu'), seconds=summary.get('seconds'))
        except (ValueError, KeyError, TypeError, OSError) as error:
            problems.append({'path': str(path), 'reason': str(error)})
    missing = sorted(set(range(0, 300, 50)) - set(summaries))
    tests = sorted((r for r in all_rows if r['scope'] == 'test'), key=lambda r: r['frame'])
    training = [r for r in all_rows if r['scope'] == 'train_diagnostic']
    base = {'schema': 'stg_area_aggregate_v1', 'model': 'STG', 'method': arm, 'scene': scene['scene'],
            'manifest': scene['manifest'], 'segments': records, 'completed_segments': len(summaries),
            'missing_segment_starts': missing, 'problems': problems,
            'frames': len(tests), 'test': None, 'temporal': None, 'status': 'incomplete',
            'posthoc_cull': False, 'visual_acceptance': 'not_performed',
            'note': 'Six independent50-frame models; not a single300-frame representation or matched-capacity comparison.'}
    if missing or problems or len(tests) != 300:
        write(output / arm / 'summary.json', base)
        return base
    try:
        manifest_path = checked_file(scene['manifest'])
        manifest = read(manifest_path)
        boundaries = [boundary_pair(summaries[end - 50], summaries[end], manifest_path, manifest, end - 1, end)
                      for end in range(50, 300, 50)]
        temporal.extend(boundaries)
        temporal.sort(key=lambda r: r['frame'])
        if [(r['previous_frame'], r['frame']) for r in temporal] != [(f, f + 1) for f in range(299)]:
            raise ValueError('Combined internal/boundary temporal sequence is not all299 pairs')
        expected_train = {(c, f) for c in scene['train_cameras'] for f in (0, 150, 299)}
        if len(training) != len(expected_train) or {(r['camera'], r['frame']) for r in training} != expected_train:
            raise ValueError('Training diagnostics do not cover every registered camera at0/150/299')
        out = output / arm
        write(out / 'temporal_per_pair.json', temporal)
        write(out / 'per_observation.json', all_rows)
        roi_names = sorted(set.intersection(*(set(r.get('fixed_rois', {})) for r in tests)))
        base.update(status='completed', test=metrics(tests), train_diagnostic=metrics(training),
                    training_by_camera={c: metrics([r for r in training if r['camera'] == c]) for c in scene['train_cameras']},
                    temporal={'pairs': 299, 'boundary_pairs': 5,
                              **{key: mean(temporal, key) for key in ('frame_difference_residual_l1', 'frame_difference_residual_mse')},
                              'boundary': {key: mean(boundaries, key) for key in ('frame_difference_residual_l1', 'frame_difference_residual_mse')}},
                    region_aggregate={name: {key: mean([r['regions'][name] for r in tests], key)
                                            for key in ('psnr', 'ssim', 'mse', 'lpips_alex_spatial_mask')}
                                      for name in ('dynamic', 'static')},
                    fixed_roi_aggregate={name: {key: mean([r['fixed_rois'][name] for r in tests], key)
                                               for key in ('psnr', 'ssim', 'mse')} for name in roi_names},
                    per_observation=identity(out / 'per_observation.json'),
                    temporal_per_pair=identity(out / 'temporal_per_pair.json'),
                    total_points=sum(s['capacity_points'] for s in summaries.values()),
                    total_checkpoint_bytes=sum(s['checkpoint_bytes'] for s in summaries.values()),
                    evaluation_seconds=sum(s['seconds'] for s in summaries.values()),
                    gpu=sorted({s['gpu'] for s in summaries.values()}))
    except (ValueError, KeyError, TypeError, OSError) as error:
        base['problems'].append({'reason': str(error), 'stage': 'full300_aggregation'})
    write(output / arm / 'summary.json', base)
    return base


def difference(left, right):
    if left.get('status') != 'completed' or right.get('status') != 'completed':
        return {'status': 'incomplete', 'delta': None}
    a, b = left.get('test', left.get('metrics')), right.get('test', right.get('metrics'))
    if not a or not b:
        return {'status': 'incomplete', 'delta': None}
    return {'status': 'completed', 'delta': {k: a[k] - b[k] for k in ('psnr', 'ssim', 'lpips')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'output/dynamic_sr_stg_area_20261018/protocol.json')
    parser.add_argument('--scene', choices=('Cut07', 'Cook07'), required=True)
    parser.add_argument('--segments-json', type=Path, help='JSON mapping each arm to explicit segment summary paths')
    parser.add_argument('--run-root', type=Path, help='Default layout: ROOT/areaTm/f000_049/eval/summary.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW',
                        help='Resolve copied artifact paths without modifying original provenance JSON')
    args = parser.parse_args()
    for mapping in args.path_map:
        old, separator, new = mapping.partition('=')
        if not separator or not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error('--path-map requires absolute OLD=NEW prefixes')
        PATH_MAP.append((Path(old), Path(new)))
    PATH_MAP.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    if not args.segments_json and not args.run_root:
        parser.error('One of --segments-json or --run-root is required')
    protocol = read(args.protocol)
    scene = next(s for s in protocol['scenes'] if s['tag'] == args.scene)
    supplied = read(args.segments_json) if args.segments_json else {
        arm: [str(args.run_root / arm / s['segment_id'] / 'eval/summary.json') for s in scene['segments']] for arm in ARMS}
    stg = {arm: arm_result(arm, supplied.get(arm, []), scene, args.out) for arm in ARMS}
    # Registered source summaries are re-hashed before using their metrics.
    wu = scene['Wu_references']
    for entry in wu.values():
        if entry.get('status') == 'completed':
            checked_file(entry)
    control = wu[scene['Wu_controlled_teacher_arm']]
    within = {'STG_teacher_gain': difference(stg['areaTm'], stg['areaL0']),
              'STG_oracle_gap': difference(stg['areaOracle'], stg['areaTm']),
              'Wu_teacher_gain': difference(control, wu['areaL0']),
              'Wu_oracle_gap': difference(wu['areaOracle'], control)}
    cross = {arm: difference(stg[arm], wu[arm]) for arm in ARMS}
    cross['STG_matched_minus_Wu_complete'] = difference(stg['areaTm'], wu['complete'])
    complete = all(stg[a]['status'] == 'completed' for a in ARMS)
    gate = {'status': 'incomplete', 'trigger_Cook07': None, 'criterion_values': None}
    if complete and cross['areaTm']['status'] == 'completed' and within['Wu_teacher_gain']['status'] == 'completed':
        values = {'absolute_matched_PSNR_difference': abs(cross['areaTm']['delta']['psnr']),
                  'absolute_matched_LPIPS_difference': abs(cross['areaTm']['delta']['lpips']),
                  'absolute_teacher_gain_PSNR_difference': abs(within['STG_teacher_gain']['delta']['psnr'] - within['Wu_teacher_gain']['delta']['psnr'])}
        gate = {'status': 'completed', 'trigger_Cook07': (values['absolute_matched_PSNR_difference'] >= .05
                  or values['absolute_matched_LPIPS_difference'] >= .003 or values['absolute_teacher_gain_PSNR_difference'] >= .05),
                'criterion_values': values, 'interpretation': 'Preregistered development confirmation trigger, not significance or human acceptance.'}
    result = {'schema': 'stg_area_comparison_v1', 'status': 'completed' if complete else 'incomplete',
              'scene': args.scene, 'protocol': identity(args.protocol), 'STG': stg, 'Wu': wu,
              'within_model': within, 'cross_model': cross, 'Cook07_gate': gate,
              'artifact_path_mapping': [{'source_prefix': str(old), 'local_prefix': str(new)} for old, new in PATH_MAP],
              'limitations': protocol['comparison_limits'], 'source': identity(__file__)}
    write(args.out / 'comparison.json', result)
    print(json.dumps({'status': result['status'], 'comparison': str(args.out / 'comparison.json'), 'Cook07_gate': gate}))


if __name__ == '__main__':
    main()
