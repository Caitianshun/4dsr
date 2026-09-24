#!/usr/bin/env python3
"""CPU-only analysis of completed diagnostics; never renders or selects endpoints.

Usage: /home/cai_tianshun/Project/4dgs/.venv/bin/python \
    experiments/dynamic_sr_20260919/audit_generalization.py
Reads 72 train fit JSONs, 18 final evaluations and 18 intermediate evaluations.
Existing final-checkpoint verification is reused. Intermediate hashes are read
once. The only writes are this audit's JSON, generated table appendix and plots.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/dynamic_sr_20260919'
DEST = BASE / 'final_assessment'
SCENES = ['cook_spinach', 'cut_roasted_beef', 'meetroom_discussion']
CASES = ['lr_long', 'sr_w01', 'sr_w10', 'hr_oracle_w10', 'sr_w10_dense', 'lr_dense']
STEPS = [1200, 6000, 12000, 18000]
METRICS = ['psnr', 'ssim', 'lpips']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def train_metrics(d):
    return dict(psnr=d['psnr'], ssim=d['ssim'], lpips=d['lpips_alex'])


def test_metrics(d, region):
    return dict(psnr=d['psnr_mean'], ssim=d['ssim_mean'],
                lpips=d['lpips_alex_mean' if region == 'full' else 'lpips_alex_spatial_mask_mean'])


def delta(end, start):
    return {k: end[k] - start[k] for k in METRICS}


def triple(d, signed=False):
    return ' / '.join(format(d[k], ('+' if signed else '') + ('.3f' if k == 'psnr' else '.5f'))
                      for k in METRICS)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    DEST.mkdir(exist_ok=True)
    trajectory_path = BASE / 'generalization_trajectory/metrics.json'
    trajectory = read(trajectory_path)
    assert trajectory['no_checkpoint_selection'] is True
    output = dict(version='cpu_generalization_audit_v1', no_checkpoint_selection=True,
                  source_script=str(Path(__file__)), source_script_sha256=sha(__file__),
                  trajectory_source=str(trajectory_path), trajectory_source_sha256=sha(trajectory_path),
                  steps=STEPS, metric_order=METRICS,
                  metric_directions=dict(psnr='higher', ssim='higher', lpips='lower'),
                  definitions=dict(train='16 fixed train images: render against HR; separate render against SR and downsampled-render against LR',
                                   test='60 fixed cam00 frames, per-frame metrics averaged',
                                   regional_lpips='official spatial LPIPS map averaged over a fixed mask; distinct from standard full-image scalar LPIPS',
                                   delta='end minus start; LPIPS negative indicates improvement',
                                   uncertainty='descriptive only, correlated frames and one seed; no independent statistical significance claim'),
                  provenance={}, scenes={})
    for scene in SCENES:
        cfgs = {c: read(BASE / f'{scene}_{c}/config.json') for c in CASES}
        verified = read(BASE / f'completion_events/{scene}_verification.json')
        assert verified['status'] == 'passed'
        final_hashes = {x['case']: x['checkpoint_sha256'] for x in verified['checks']}
        first = cfgs[CASES[0]]
        keys = ['manifest_sha256', 'parent_sha256', 'seed', 'prior_cameras', 'scheduler_offset', 'scheduler_max_steps']
        assert all(cfgs[c][k] == first[k] for c in CASES for k in keys)
        assert sha(first['manifest']) == first['manifest_sha256']
        scene_result = {}
        evidence = []
        fit_observations = None
        source_versions = {}
        source_fit_bodies = set()
        for case in CASES:
            cfg = cfgs[case]
            snapshot = Path(cfg['sources'][0]['copy'])
            assert sha(snapshot) == cfg['sources'][0]['sha256']
            source_versions[case] = cfg['sources'][0]['sha256']
            source_text = snapshot.read_text()
            source_fit_bodies.add(source_text[source_text.index('@torch.no_grad()'):])
            fits = {}
            for step in STEPS:
                path = BASE / f'{scene}_{case}/fit_{step}.json'
                f = read(path)
                observations = sorted((r['camera_id'], r['frame_index']) for r in f['rows'])
                if fit_observations is None:
                    fit_observations = observations
                assert f['step'] == step and observations == fit_observations and len(observations) == 16
                for kind in ['render_hr', 'render_prior']:
                    for key in ['psnr', 'ssim', 'lpips_alex']:
                        assert math.isclose(f['aggregate'][kind][key], mean(r[kind][key] for r in f['rows']), abs_tol=1e-12)
                fits[str(step)] = dict(hr=train_metrics(f['aggregate']['render_hr']),
                                       sr=train_metrics(f['aggregate']['render_prior']), lr_psnr=f['lr_psnr'])
            final = read(BASE / f'{scene}_{case}_evaluation/metrics.json')
            assert final['checkpoint_sha256'] == final_hashes[case]
            evidence.append(final)
            points = {}
            if case in trajectory['scenes'].get(scene, {}):
                for step in STEPS:
                    tr = trajectory['scenes'][scene][case][str(step)]
                    ev = read(tr['metrics'])
                    assert tr['aggregate'] == ev['aggregate']
                    assert tr['temporal'] == ev['temporal_aggregate']
                    assert tr['checkpoint_sha256'] == ev['checkpoint_sha256']
                    assert ev['checkpoint_metadata']['intervention_step'] == step
                    assert ev['checkpoint_metadata']['step'] == cfg['scheduler_offset'] + step
                    assert ev['checkpoint_metadata']['parent_sha'] == cfg['parent_sha256']
                    if step != 18000:
                        assert sha(ev['checkpoint']) == ev['checkpoint_sha256']
                        evidence.append(ev)
                    else:
                        assert ev == final
                    points[str(step)] = dict(
                        hr={r: test_metrics(ev['aggregate'][r], r) for r in ['full', 'dynamic', 'static']},
                        mse={r: ev['aggregate'][r]['mse_mean'] for r in ['full', 'dynamic', 'static']},
                        checkpoint_sha256=ev['checkpoint_sha256'],
                        temporal_gt_relative_l1={r: ev['temporal_aggregate'][r]['gt_relative_warp_l1_mean']
                                                 for r in ['full', 'dynamic', 'static']})
            intervals = {}
            for start in [1200, 6000, 12000]:
                a, z = str(start), '18000'
                entry = dict(train_hr=delta(fits[z]['hr'], fits[a]['hr']),
                             train_sr=delta(fits[z]['sr'], fits[a]['sr']),
                             train_lr_psnr=fits[z]['lr_psnr'] - fits[a]['lr_psnr'])
                if points:
                    entry['test_hr'] = {r: delta(points[z]['hr'][r], points[a]['hr'][r])
                                        for r in ['full', 'dynamic', 'static']}
                    entry['gap_change_train_minus_test'] = delta(entry['train_hr'], entry['test_hr']['full'])
                    fraction = final['dynamic_fraction']
                    entry['test_mse_change_decomposition'] = dict(
                        full=points[z]['mse']['full'] - points[a]['mse']['full'],
                        weighted_dynamic=fraction * (points[z]['mse']['dynamic'] - points[a]['mse']['dynamic']),
                        weighted_static=(1-fraction) * (points[z]['mse']['static'] - points[a]['mse']['static']))
                intervals[f'{start}_to_18000'] = entry
            scene_result[case] = dict(train=fits, test_trajectory=points, intervals=intervals,
                                     test_final_hr={r: test_metrics(final['aggregate'][r], r)
                                                    for r in ['full', 'dynamic', 'static']},
                                     initial_points=cfg['initial_points'], final_points=final['gaussian_count'])
        assert len(source_fit_bodies) == 1, 'Training/diagnostic implementation differs beyond durability wrappers'
        canonical = evidence[0]
        identity_keys = ['manifest_sha256', 'frame_indices', 'test_camera', 'cache_key', 'evaluation_cache',
                         'script_sha256', 'metric_definitions', 'dynamic_fraction', 'dynamic_threshold']
        assert all(ev[k] == canonical[k] for ev in evidence for k in identity_keys)
        assert all([r['frame_index'] for r in ev['rows']] == canonical['frame_indices'] for ev in evidence)
        assert all(ev['manifest_sha256'] == first['manifest_sha256'] for ev in evidence)
        cache = read(Path(canonical['evaluation_cache']) / 'cache.json')
        cache_hash = hashlib.sha256(json.dumps(cache['identity'], sort_keys=True).encode()).hexdigest()[:16]
        assert cache_hash == canonical['cache_key'] == cache['cache_key']
        assert cache['identity']['manifest_sha256'] == first['manifest_sha256']
        assert len(cache['identity']['gt_files']) == 60
        output['provenance'][scene] = dict(**{k: first[k] for k in keys},
            train_observations=fit_observations, test_camera=canonical['test_camera'],
            test_frame_indices=canonical['frame_indices'], test_evaluations_compared=len(evidence),
            cache_key=canonical['cache_key'], evaluate_script_sha256=canonical['script_sha256'],
            dynamic_fraction=canonical['dynamic_fraction'], training_source_versions=source_versions,
            shared_training_and_diagnostic_body=True, checks='passed',
            final_checkpoint_hashes='reused successful completion_events verification',
            intermediate_checkpoints_hashed=9 if scene in SCENES[:2] else 0,
            fit_checkpoint_binding='fit JSON has no checkpoint hash; association relies on saved source execution order, directory and exact step')
        output['scenes'][scene] = scene_result
    output['limitations'] = [
        'Novel-view trajectories exist only for two kitchen scenes and three fixed-capacity branches.',
        'Train diagnostics have full-image metrics only; regional training trajectories were not evaluated.',
        'No significance test: 60 adjacent frames are not 60 independent scenes.',
        'Different train/test cameras and image counts prevent interpreting absolute gaps as a controlled geometry error.',
        'Dense changes point placement and optimization as well as count; no pure capacity isolation.',
        'Old code/new code source hash difference is fsync persistence wrappers, with identical training and fit function bodies.',
        'Fixed 18000 endpoint retained; these development curves do not select a checkpoint.'
    ]
    (DEST / 'trajectory_audit.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')

    tables = []
    for scene in SCENES:
        tables += [f'## {scene}：训练 HR 完整轨迹',
            '每格依次 PSNR↑ / SSIM↑ / 标准 LPIPS↓；16 张固定训练图像。',
            table(['分支'] + [str(s) for s in STEPS],
                  [[c] + [triple(output['scenes'][scene][c]['train'][str(s)]['hr']) for s in STEPS] for c in CASES])]
    for scene in SCENES[:2]:
        tables += [f'## {scene}：训练与 cam00 全图轨迹',
            table(['分支', '步', '训练 PSNR', 'cam00 PSNR', '训练 SSIM', 'cam00 SSIM', '训练 LPIPS', 'cam00 LPIPS'],
                [[c, s] + [f'{value:.5f}' for k in METRICS for value in
                   [output['scenes'][scene][c]['train'][str(s)]['hr'][k],
                    output['scenes'][scene][c]['test_trajectory'][str(s)]['hr']['full'][k]]]
                 for c in CASES[:3] for s in STEPS])]
        for region in ['dynamic', 'static']:
            tables += [f'### cam00 {region} 区域',
                '每格依次 PSNR↑ / SSIM↑ / 区域空间 LPIPS↓（与全图标准 LPIPS 定义不同）。',
                table(['分支'] + [str(s) for s in STEPS],
                     [[c] + [triple(output['scenes'][scene][c]['test_trajectory'][str(s)]['hr'][region])
                             for s in STEPS] for c in CASES[:3]])]
    (DEST / 'trajectory_tables.md').write_text('\n\n'.join(tables) + '\n')
    plot(output)
    print(json.dumps(dict(status='passed', output=str(DEST / 'trajectory_audit.json'),
                          train_diagnostics=72, novel_view_evaluations=36,
                          intermediate_checkpoints_hashed=18)))


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#64748b', '#2563eb', '#c2410c']
    labels = ['LR long', 'SR 0.1', 'SR 1.0']
    for kind in ['train_test', 'region']:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for row, scene in enumerate(SCENES[:2]):
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                for case, color, label in zip(CASES[:3], colors, labels):
                    entry = output['scenes'][scene][case]
                    if kind == 'train_test':
                        first = [entry['train'][str(s)]['hr'][metric] for s in STEPS]
                        second = [entry['test_trajectory'][str(s)]['hr']['full'][metric] for s in STEPS]
                        pair = ['train (16)', 'cam00 (60)']
                    else:
                        first = [entry['test_trajectory'][str(s)]['hr']['dynamic'][metric] for s in STEPS]
                        second = [entry['test_trajectory'][str(s)]['hr']['static'][metric] for s in STEPS]
                        pair = ['changing', 'non-changing']
                    ax.plot(STEPS, first, marker='o', color=color, label=f'{label}: {pair[0]}')
                    ax.plot(STEPS, second, marker='s', ls='--', color=color, label=f'{label}: {pair[1]}')
                name = metric.upper()
                if metric == 'lpips' and kind == 'region':
                    name = 'Spatial LPIPS (region)'
                ax.set_title(f'{scene.replace("_", " ")} / {name}')
                ax.set_xticks(STEPS, ['1.2k', '6k', '12k', '18k'])
                ax.set_xlabel('Added optimization steps')
                ax.grid(alpha=.22)
                if col == 0:
                    ax.legend(fontsize=7)
        fig.suptitle('Development diagnosis; fixed 18k endpoint; no checkpoint selection', fontsize=13)
        fig.savefig(DEST / f'trajectory_{kind}.png', dpi=180)
        fig.savefig(DEST / f'trajectory_{kind}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
