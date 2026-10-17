#!/usr/bin/env python3
"""Audit complete Multi4D area4 arms against identified Wu raw summaries.

Example: assess.py --scene cut_roasted_beef --multi-root OUTPUT/eval --out report.json
Missing/failed/incomplete arms never become zero-valued metrics. Native Multi4D
and posthoc-cull Wu are explicitly different complete pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WU = ROOT / 'output/dynamic_sr_physlr_20261016/eval'
REFERENCES = {
    'cut_roasted_beef': {
        'areaL0': 'Cut07_area4_areaL0_cam00_posthoc_cull',
        'areaTm': 'Cut07_area4_P7areaTm_cam00_posthoc_cull',
        'controlled_areaTm': 'Cut07_area4_areaTm_cam00_posthoc_cull',
        'complete': 'Cut07_area4_P7areaTm_calib_cam00_posthoc_cull',
        'areaOracle': 'Cut07_area4_oracle_cam00_posthoc_cull',
    },
    'cook_spinach': {
        'areaL0': 'Cook07_area4_areaL0_cam00_posthoc_cull',
        'areaTm': 'Cook07_area4_areaTm_cam00_posthoc_cull',
        'complete': 'Cook07_area4_areaTm_calib_cam00_posthoc_cull',
        'best_development_geo_calib': 'Cook07_area4_areaTm_geo_calib_cam00_posthoc_cull',
        'areaOracle': 'Cook07_area4_oracle_cam00_posthoc_cull',
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_result(path, method, model):
    path = Path(path).resolve()
    result = dict(path=str(path), model=model, method=method,
        supervision='training_view_HR_diagnostic_only' if method == 'areaOracle' else 'legal_training_LR_and_optional_frozen_teacher',
        posthoc_cull=(model == 'Wu'), metrics=None)
    if not path.exists():
        return dict(result, status='incomplete', reason='summary_missing')
    result['sha256'] = sha(path)
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeError) as error:
        return dict(result, status='failed', reason='invalid_summary_json', error=str(error))
    result.update(source_status=value.get('status'), frames=value.get('frames'), gpu=value.get('gpu'),
        scene=value.get('scene'), checkpoint=value.get('checkpoint'), manifest=value.get('manifest'),
        temporal=value.get('temporal'), visual_acceptance=value.get('visual_acceptance', 'not_recorded'))
    if model == 'Multi4D' and value.get('method') != method:
        return dict(result, status='failed', reason='summary_method_does_not_match_requested_arm')
    if value.get('status') == 'failed':
        return dict(result, status='failed', reason=value.get('error', 'evaluation_failed'))
    metrics = value.get('test')
    if value.get('status') != 'completed' or value.get('frames') != 300 or not isinstance(metrics, dict):
        return dict(result, status='incomplete', reason='requires_completed_cam00_all300')
    keys = ('psnr', 'ssim', 'lpips')
    if not all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) for k in keys):
        return dict(result, status='incomplete', reason='required_spatial_metrics_missing')
    result['metrics'] = {key: metrics.get(key) for key in
        ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr')}
    result['posthoc_cull'] = bool(value.get('posthoc_cull', model == 'Wu'))
    return dict(result, status='completed')


def difference(left, right):
    if left.get('status') != 'completed' or right.get('status') != 'completed':
        return dict(status='incomplete', delta=None, required=[left['path'], right['path']])
    left_manifest, right_manifest = left.get('manifest'), right.get('manifest')
    if not isinstance(left_manifest, dict) or not isinstance(right_manifest, dict) or not left_manifest.get('sha256') or left_manifest['sha256'] != right_manifest.get('sha256'):
        return dict(status='failed', delta=None, reason='manifest_identity_mismatch', required=[left['path'], right['path']])
    delta = {key: left['metrics'][key] - right['metrics'][key]
             if left['metrics'].get(key) is not None and right['metrics'].get(key) is not None else None
             for key in ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr')}
    return dict(status='completed', delta=delta,
        sources=dict(left=left['path'], right=right['path']),
        evaluation_gpus=dict(left=left.get('gpu'), right=right.get('gpu')),
        same_evaluation_gpu=left.get('gpu') == right.get('gpu'),
        interpretation='positive PSNR/SSIM and negative LPIPS favour the left arm; this arithmetic alone does not establish a paired training comparison')


def model_contrasts(arms):
    matched = arms.get('controlled_areaTm', arms['areaTm'])
    return dict(matched_source_for_controlled_contrasts=matched['path'],
        SR_gain_matched_minus_LR=difference(matched, arms['areaL0']),
        remaining_oracle_gap=difference(arms['areaOracle'], matched),
        perfect_teacher_gain_over_LR=difference(arms['areaOracle'], arms['areaL0']))


def build(args):
    multi = {}
    for name in ('areaL0', 'areaTm', 'areaOracle'):
        supplied = getattr(args, name)
        path = supplied or (args.multi_root / name / 'summary.json')
        multi[name] = read_result(path, name, 'Multi4D')
    wu = {name: read_result(WU / folder / 'summary.json', name, 'Wu')
          for name, folder in REFERENCES[args.scene].items()}
    cross = {name: difference(multi[name], wu[name]) for name in ('areaL0', 'areaTm', 'areaOracle')}
    cross['Multi4D_matched_minus_Wu_complete'] = difference(multi['areaTm'], wu['complete'])
    required = list(multi.values()) + [wu[k] for k in ('areaL0', 'areaTm', 'complete', 'areaOracle')]
    status = 'completed' if all(r['status'] == 'completed' for r in required) else 'incomplete'
    if any(value['status'] == 'failed' for value in cross.values()):
        status = 'failed'
    legal_comparison = cross['Multi4D_matched_minus_Wu_complete']
    winner = 'undetermined'
    if legal_comparison['status'] == 'completed':
        d = legal_comparison['delta']
        if d['psnr'] >= .05 and d['ssim'] >= -.001 and d['lpips'] <= .003:
            winner = 'Multi4D_matched_preliminary'
        elif d['psnr'] <= -.05 and d['ssim'] <= .001 and d['lpips'] >= -.003:
            winner = 'Wu_complete_preliminary'
        else:
            winner = 'mixed_or_small_difference'
    return dict(schema='multi4d_area_assessment_v1', status=status, scene=args.scene,
        execution_state=('prepared_waiting_all_claude_idle_and_no_claude_gpu_use'
            if not any(r['status'] == 'completed' for r in multi.values()) else 'existing_results_read_only'),
        execution_gate='User instruction: start GPU work only after all Claude conversations are inactive and Claude uses no GPU; this report does not verify that live condition.',
        Multi4D=multi, Wu=wu, within_model=dict(Multi4D=model_contrasts(multi), Wu=model_contrasts(wu)),
        cross_model=cross, preliminary_spatial_winner=winner,
        winner_limit='Full-pipeline spatial comparison only; fixed ROI, temporal and human blind acceptability are separate checks. No universal backbone-strength conclusion.',
        protocol_notes=[
            'Multi4D is native three-branch inference; Wu reference summaries use training-camera-based posthoc culling.',
            'Cross-model full-pipeline quality does not isolate representation, training schedule, hardware, or culling.',
            'areaOracle reads training-view HR and is a diagnostic, never a legal competing method.',
            'SR gain and oracle-minus-matched gap are different quantities and may move in opposite directions.',
            'Cut PRO6000 areaTm32.5623278669 and calib32.9685805868 are kept as identified raw summaries; no cross-card subtraction.',
            'Cut within-Wu SR gain/oracle gap use the separate RTX3090 areaTm32.640 source alongside its RTX3090 areaL0/oracle controls; PRO6000 values remain the current absolute comparison.',
            'Cook geo_calib is an extra development combination and is not silently substituted for the selected three-component method.',
            'Results missing or failing completeness remain null; no missing metric is replaced by zero.',
            'All current Cook/Cut/MeetRoom scenes are previously used development scenes.',
        ], assessment_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--multi-root', type=Path, required=True)
    parser.add_argument('--areaL0', type=Path, help='Explicit evaluated summary override')
    parser.add_argument('--areaTm', type=Path, help='Explicit evaluated summary override')
    parser.add_argument('--areaOracle', type=Path, help='Explicit diagnostic summary override')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps(dict(status=result['status'], preliminary_spatial_winner=result['preliminary_spatial_winner'], path=str(args.out))))


if __name__ == '__main__':
    main()
