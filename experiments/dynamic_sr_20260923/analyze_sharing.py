"""Descriptive, group-aware analysis of the locked P1 teacher diagnostic.

This script never reads HR image files, fits a sharing rule, or changes the
pre-registered P1 gate. It requires evaluation_complete.json before execution.
The response variable is fixed-eta full-target MSE improvement, not high-pass
error improvement or new-view reconstruction quality. No iid p-values are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'output/dynamic_sr_20260923/temporal_sharing_v1'
CONTROL_NAMES = ['mean_r', 'log_texture_plus_1e-12', 'motion', 'absolute_offset']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def correlation(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(x, y)/denominator, -1., 1.))


def rank(values):
    # Unit-range average ranks make the residual-variance audit comparable.
    return (rankdata(values, method='average')-.5) / max(len(values), 1)


def group_key(row):
    return (row['scene'], row['camera'], int(row['anchor']))


def partial_rank(rows):
    if len(rows) < 8:
        return dict(correlation=None, reason='fewer_than_8_pairs', n=len(rows))
    controls = np.asarray([[r['mean_r'], np.log(max(r['texture'], 0.)+1e-12),
                            r['motion'], abs(r['offset'])] for r in rows], dtype=np.float64)
    groups = sorted({group_key(r) for r in rows})
    design_columns = [np.ones(len(rows))]
    design_columns += [rank(controls[:, i]) for i in range(controls.shape[1])]
    # Intercept plus all but one group dummy: fixed scene/camera/anchor effects.
    design_columns += [np.asarray([group_key(r) == g for r in rows], dtype=float) for g in groups[1:]]
    design = np.stack(design_columns, axis=1)
    x, y = rank([r['p'] for r in rows]), rank([r['gain'] for r in rows])
    beta_x, _, design_rank, singular_values = np.linalg.lstsq(design, x, rcond=None)
    beta_y = np.linalg.lstsq(design, y, rcond=None)[0]
    rx, ry = x-design@beta_x, y-design@beta_y
    dof = len(rows)-int(design_rank)
    usable = dof >= 3
    value = correlation(rx, ry) if usable else None
    return dict(correlation=value, n=len(rows), observation_groups=len(groups),
                design_columns=design.shape[1], design_rank=int(design_rank), residual_dof=dof,
                rank_controls=CONTROL_NAMES, group_effect='scene/camera/anchor intercept',
                p_residual_variance=float(np.var(rx)), gain_residual_variance=float(np.var(ry)),
                rank_deficient=bool(design_rank < design.shape[1]),
                reason=None if value is not None else ('insufficient_residual_dof' if not usable else 'constant_residual'),
                singular_values=[float(s) for s in singular_values])


def summarize(rows):
    n = len(rows)
    raw = correlation(rank([r['p'] for r in rows]), rank([r['gain'] for r in rows])) if n else None
    return dict(n=n, raw_spearman=raw, conditional_rank=partial_rank(rows),
                mean_fixed_eta_full_target_mse_gain=float(np.mean([r['gain'] for r in rows])) if n else None,
                beneficial_fraction=float(np.mean([r['gain'] > 0 for r in rows])) if n else None)


def observation_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    output = []
    for (scene, camera, anchor), block in sorted(groups.items()):
        output.append(dict(scene=scene, camera=camera, anchor=anchor,
                           split=block[0]['split'], **summarize(block)))
    return output


def sign_summary(observations):
    correlations = [r['raw_spearman'] for r in observations if r['raw_spearman'] is not None]
    return dict(valid_groups=len(correlations), total_groups=len(observations),
                positive_groups=sum(c > 0 for c in correlations),
                negative_groups=sum(c < 0 for c in correlations),
                equal_group_mean=float(np.mean(correlations)) if correlations else None,
                equal_group_median=float(np.median(correlations)) if correlations else None,
                interpretation='Descriptive group summaries, not independent trials or a significance test.')


def cpu_checks(module_path):
    """Exercise the actual P1 lift/warp implementation on synthetic arrays."""
    spec = importlib.util.spec_from_file_location('p1_temporal_sharing_checked', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.set_num_threads(4)
    module.cv2.setNumThreads(2)
    h, w = 24, 32
    hh, ww = h*4, w*4
    yy, xx = np.mgrid[:hh, :ww].astype(np.float32)
    ramp = lambda x, y: np.stack([x/(ww-1), y/(hh-1), (x+y)/(ww+hh-2)], axis=-1)
    base = ramp(xx, yy)
    flow_lr = np.zeros((h, w, 2), dtype=np.float32)
    # Fractional LR shifts become half-integer HR shifts. Cubic interpolation
    # exactly reproduces these linear ramp samples away from its border.
    flow_lr[..., 0], flow_lr[..., 1] = .125, -.375
    flow_hr = module.lift(flow_lr, (hh, ww))*4
    neighbor = ramp(xx-.5, yy+1.5)
    interior = np.s_[8:-8, 8:-8]
    correct = float(np.max(np.abs(module.warp(neighbor, flow_hr)[interior]-base[interior])))
    wrong_sign = float(np.max(np.abs(module.warp(neighbor, -flow_hr)[interior]-base[interior])))
    omitted_scale = float(np.max(np.abs(module.warp(neighbor, module.lift(flow_lr, (hh, ww)))[interior]-base[interior])))
    assert correct < 2e-6, correct
    assert wrong_sign > 1e-3 and omitted_scale > 1e-3, (wrong_sign, omitted_scale)
    # A spatially varying affine flow additionally checks half-pixel resizing.
    ly, lx = np.mgrid[:h, :w].astype(np.float32)
    affine = lambda x, y: np.stack([.125+.01*x+.005*y, -.375+.007*x-.004*y], axis=-1)
    actual = module.lift(affine(lx, ly), (hh, ww))*4
    expected = affine((xx+.5)/4-.5, (yy+.5)/4-.5)*4
    lift_error = float(np.max(np.abs(actual[4:-4, 4:-4]-expected[4:-4, 4:-4])))
    assert lift_error < 2e-6, lift_error
    rng = np.random.default_rng(20260923)
    a, b = [rng.normal(.5, .5, (32, 40, 3)).astype(np.float32) for _ in range(2)]
    linear_error = float(np.max(np.abs(module.high(.7*a-.2*b, (8, 10))-
                                      (.7*module.high(a, (8, 10))-.2*module.high(b, (8, 10))))))
    assert linear_error < 3e-6, linear_error
    return dict(cpu_only=True, synthetic_only=True, passed=True,
                lr_fractional_shift=[.125, -.375], hr_shift=[.5, -1.5],
                lift4_fractional_pull_warp_max_abs=correct,
                wrong_sign_max_abs=wrong_sign, omitted_x4_max_abs=omitted_scale,
                affine_flow_half_pixel_lift_max_abs=lift_error,
                high_operator_linearity_max_abs=linear_error,
                limitation='Verifies coordinate conventions, not real-video flow accuracy or visibility.')


def main(out):
    complete_path = out/'evaluation_complete.json'
    if not complete_path.is_file():
        raise SystemExit('P1 evaluation_complete.json is required; no evaluation images or labels were read.')
    complete = json.loads(complete_path.read_text())
    assert sha(out/'summary.json') == complete['summary_sha256']
    assert sha(out/'targets_locked.json') == complete['targets_lock_sha256']
    module_path = Path(__file__).with_name('temporal_sharing.py')
    assert sha(module_path) == complete['source_sha256'], 'P1 source changed after evaluated lock'
    output_path = out/'conditional_analysis.json'
    if output_path.exists():
        raise SystemExit(f'Refusing to overwrite existing analysis: {output_path}')
    paths = [complete_path, out/'pair_evaluation.json', out/'summary.json', out/'decision.json',
             out/'targets_locked.json', module_path, Path(__file__)]
    input_hashes = {str(path): sha(path) for path in paths}
    checks = cpu_checks(module_path)
    all_rows = json.loads((out/'pair_evaluation.json').read_text())
    rejected = defaultdict(int)
    eligible = []
    for row in all_rows:
        if not row.get('exposure_known', False):
            rejected['unknown_exposure'] += 1
        elif row['reliable_fraction'] < .75:
            rejected['reliable_fraction_below_.75'] += 1
        elif not all(np.isfinite(row[k]) for k in ['p', 'gain', 'mean_r', 'texture', 'motion', 'offset']):
            rejected['nonfinite_feature_or_label'] += 1
        else:
            eligible.append(row)
    by_observation = observation_summary(eligible)
    by_scene_split, by_camera_split = [], []
    for scene in sorted({r['scene'] for r in all_rows}):
        for split in ['calibration', 'check']:
            selected = [r for r in eligible if r['scene'] == scene and r['split'] == split]
            observation_rows = [r for r in by_observation if r['scene'] == scene and r['split'] == split]
            by_scene_split.append(dict(scene=scene, split=split, **summarize(selected),
                                       observation_signs=sign_summary(observation_rows)))
            for camera in sorted({r['camera'] for r in all_rows if r['scene'] == scene}):
                camera_rows = [r for r in selected if r['camera'] == camera]
                by_camera_split.append(dict(scene=scene, camera=camera, split=split, **summarize(camera_rows)))
    conclusions = []
    for row in by_scene_split:
        if row['split'] != 'check':
            continue
        partial = row['conditional_rank']['correlation']
        direction = '不可估计' if partial is None else ('正向' if partial > 0 else '非正向')
        signs = row['observation_signs']
        conclusions.append(f"{row['scene']}：check 有效关系 {row['n']}；控制对应质量、纹理、运动、时间间隔和观察组后，rank 残差相关为{direction}（{partial}）；观察组原始相关 {signs['positive_groups']}/{signs['valid_groups']} 为正。")
    original_decision = json.loads((out/'decision.json').read_text())
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=input_hashes,
                  outcome='fixed eta=0.5 full-target clipped MSE gain against the current teacher; NOT high-pass-only gain',
                  selection=dict(required_exposure_known=True, minimum_reliable_fraction=.75,
                                 original_pairs=len(all_rows), eligible_pairs=len(eligible), exclusions=dict(rejected)),
                  rank_residualization='Average ranks of p, gain and four controls; OLS projection onto controls plus scene/camera/anchor fixed effects; correlate residuals. No training rule is fitted.',
                  controls=CONTROL_NAMES, by_observation=by_observation,
                  by_scene_split=by_scene_split, by_camera_split=by_camera_split,
                  cpu_checks=checks, original_proceed_p2=original_decision['proceed_p2'],
                  gate_override=False, conclusions_zh=conclusions,
                  limitations=[
                      '同一图块多个邻居、同相机相邻时刻及同场景仍相关；不报告 iid p 值、显著性或独立试验数。',
                      'rank 线性残差化只消除所列代理的线性秩关联；不能排除非线性混杂、光流误差或隐藏外观因素。',
                      '图块经 LR 配额选择，有效关系再按对应覆盖筛选；结论条件于该集合，不代表整图自然频率。',
                      '相关为正不保证实际聚合收益；须结合已锁定 A/C/D/G 操作结果、覆盖和失败案例。',
                      'check 仍是已使用开发场景中的固定片段，不是独立场景确认或重建训练成绩。',
                      '该分析不修改原始门槛；原始 gate 通过也不自动证明 p 提供因果增量。'])
    # Confirm that no concurrent writer changed the evaluated artifacts.
    for path, digest in input_hashes.items():
        assert sha(path) == digest, f'Input changed during analysis: {path}'
    write(output_path, result)
    write(out/'conditional_analysis_complete.json', dict(analysis_sha256=sha(output_path),
          source_sha256=sha(__file__), finished_utc=datetime.now(timezone.utc).isoformat(),
          cpu_only=True, hr_images_read=False, parameter_updates=0))
    print(json.dumps(dict(output=str(output_path), checks=checks, conclusions=conclusions,
                          original_proceed_p2=original_decision['proceed_p2'], gate_override=False),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)
