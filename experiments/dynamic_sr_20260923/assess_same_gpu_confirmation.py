"""CPU audit and fixed-endpoint assessment of the completed same-GPU D repeat.

Requires both batch completion gates and the original 12-checkpoint audit.
Reads no intermediate training logs, performs no rendering/training, and refuses
existing output. New results are written below confirmation/assessment by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from audit_meeting_checkpoints import (
    ROOT, GPU0, POINTS, STEPS, EXPECTED_PARENT_SHA, capture_layout, equal,
    load_json, now, replay_sampling, require, sha, tensor_finiteness,
    validate_metrics, write,
)

ORIGINAL = ROOT / 'output/dynamic_sr_20260923/meeting_controls_v1'
CONFIRMATION = ROOT / 'output/dynamic_sr_20260923/meeting_D_gpu0_confirmation_v1'
BRANCHES = ['A', 'B', 'C', 'D3090', 'D_GPU0']
PROTOCOL_KEYS = ['version', 'manifest_sha256', 'script_sha256', 'test_camera',
                 'frame_indices', 'cache_key', 'metric_definitions', 'lpips_status', 'versions']
FIT_KEYS = ['psnr', 'ssim', 'lpips_alex', 'mse']
SCORE_KEYS = ['psnr', 'ssim', 'lpips', 'temporal', 'dynamic_lpips', 'static_lpips']
CONFIG_EXCEPTIONS = {'out', 'visible_cuda', 'gpu'}
ROI = {'time_changing_proxy': (504, 420, 672, 588),
       'mostly_static': (1092, 504, 1260, 672)}


def protocol_identity(metrics, reference, label):
    for key in PROTOCOL_KEYS:
        require(key in metrics and key in reference, f'{label}: missing protocol field {key}')
        equal(metrics[key], reference[key], f'{label}: evaluation protocol {key}')


def receipt(complete, label, checkpoint=None, out=None, step=None):
    require(complete['gpu'] == GPU0, 'Confirmation dispatcher did not bind GPU0')
    events = [e for e in complete['events'] if e['label'] == label]
    require(len(events) == 1 and events[0].get('returncode') == 0, f'Missing successful receipt {label}')
    event = events[0]
    command = event['command']
    if checkpoint is not None:
        require(Path(command[command.index('--checkpoint') + 1]).resolve() == checkpoint.resolve(), f'{label}: checkpoint command')
    if out is not None:
        flag = '--run' if label.startswith('fit_') else '--out'
        require(Path(command[command.index(flag) + 1]).resolve() == out.resolve(), f'{label}: output command')
    if step is not None:
        require(int(command[command.index('--step') + 1]) == step, f'{label}: fit step')
    return event


def summary(branch, step, m):
    return dict(branch=branch, step=step,
                psnr=m['aggregate']['full']['psnr_mean'], ssim=m['aggregate']['full']['ssim_mean'],
                lpips=m['aggregate']['full']['lpips_alex_mean'],
                temporal=m['temporal_aggregate']['full']['gt_relative_warp_l1_mean'],
                dynamic_lpips=m['aggregate']['dynamic']['lpips_alex_spatial_mask_mean'],
                static_lpips=m['aggregate']['static']['lpips_alex_spatial_mask_mean'],
                spatial_aggregate=m['aggregate'], temporal_aggregate=m['temporal_aggregate'])


def fit_summary(folder, branch, step):
    import numpy as np
    fit_path, own_path = folder / f'fit_{step}.json', folder / f'own_target_fit_{step}.json'
    fit, own = load_json(fit_path), load_json(own_path)
    require(fit['step'] == step and own['parameter_updates'] == 0, f'{branch}/{step}: diagnostic metadata')
    expected = {(c, f) for c in ['cam02', 'cam04', 'cam08', 'cam12'] for f in [0, 40, 80, 118]}
    keyed = lambda rows: {(r['camera_id'], r['frame_index']): r for r in rows}
    fr, tr = keyed(fit['rows']), keyed(own['rows'])
    require(len(fit['rows']) == len(own['rows']) == 16 and fr.keys() == tr.keys() == expected,
            f'{branch}/{step}: train diagnostic samples differ')
    groups = {}
    partitions = {'all': {0, 40, 80, 118}, 'interior_40_80': {40, 80}, 'boundary_0_118': {0, 118},
                  **{f'frame_{f}': {f} for f in [0, 40, 80, 118]}}
    for name, frames in partitions.items():
        selected = sorted(k for k in fr if k[1] in frames)
        values = dict(count=len(selected), frames=sorted(frames),
                      train_HR={k: float(np.mean([fr[i]['render_hr'][k] for i in selected])) for k in FIT_KEYS},
                      train_original_teacher={k: float(np.mean([fr[i]['render_prior'][k] for i in selected])) for k in FIT_KEYS},
                      train_LR_psnr_mean=float(np.mean([fr[i]['lr_psnr'] for i in selected])),
                      own_target={k: float(np.mean([tr[i]['metrics'][k] for i in selected])) for k in FIT_KEYS})
        groups[name] = values
    for kind, label in [('render_hr', 'train_HR'), ('render_prior', 'train_original_teacher')]:
        for key in FIT_KEYS:
            require(math.isclose(groups['all'][label][key], fit['aggregate'][kind][key], rel_tol=1e-12, abs_tol=1e-12),
                    f'{branch}/{step}: inconsistent fit aggregate {kind}/{key}')
    for key in FIT_KEYS:
        require(math.isclose(groups['all']['own_target'][key], own['aggregate'][key], rel_tol=1e-12, abs_tol=1e-12),
                f'{branch}/{step}: inconsistent own-target aggregate {key}')
    require(own['target_clipped_for_metrics'] and own['training_uses_raw_float32'], 'Own target clipping disclosure differs')
    return dict(branch=branch, step=step, groups=groups, original_fit_rows=fit['rows'], own_target_rows=own['rows'],
                evidence=[dict(path=str(p), sha256=sha(p)) for p in [fit_path, own_path]],
                own_target_comparability='Different branch targets are different objectives; own-target fit is not a common-target quality ranking.',
                artifact_identity_limit='Fit JSON has no checkpoint hash. Association is via fixed endpoint path and dispatcher command; main eval identity is hash-bound.')


def pairwise(step, new, baseline, new_summary, base_summary):
    deltas = []
    for nr, br in zip(new['rows'], baseline['rows']):
        require(nr['frame_index'] == br['frame_index'], 'Comparison frame order differs')
        deltas.append(dict(frame=nr['frame_index'], **{
            k: nr['spatial']['full'][k] - br['spatial']['full'][k] for k in ['psnr', 'ssim', 'lpips_alex']}))
    counts = {}
    for key in ['psnr', 'ssim', 'lpips_alex']:
        sign = -1 if key == 'lpips_alex' else 1
        counts[key] = dict(wins=sum(sign * p[key] > 0 for p in deltas),
                           ties=sum(p[key] == 0 for p in deltas), losses=sum(sign * p[key] < 0 for p in deltas))
    return dict(step=step, comparison=f'D_GPU0-{base_summary["branch"]}',
                mean={k: new_summary[k] - base_summary[k] for k in SCORE_KEYS},
                frames_improved={k: v['wins'] for k, v in counts.items()}, frame_counts=counts, frame_deltas=deltas,
                worst_psnr_delta=min(p['psnr'] for p in deltas), best_psnr_delta=max(p['psnr'] for p in deltas))


def fit_difference(new, baseline):
    groups = {}
    for name, value in new['groups'].items():
        b = baseline['groups'][name]
        require(value['count'] == b['count'] and value['frames'] == b['frames'], 'Fit subset mismatch')
        groups[name] = dict(count=value['count'], frames=value['frames'],
                           train_LR_psnr_mean=value['train_LR_psnr_mean'] - b['train_LR_psnr_mean'],
                           **{kind: {k: value[kind][k] - b[kind][k] for k in FIT_KEYS}
                              for kind in ['train_HR', 'train_original_teacher', 'own_target']})
    return dict(step=new['step'], comparison=f'D_GPU0-{baseline["branch"]}', groups=groups)


def figures(output, folders, rows):
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from temporal_sharing import high, rgb

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), layout='constrained')
    for ax, key, title in zip(axes, ['psnr', 'ssim', 'lpips', 'temporal'],
                              ['PSNR (higher)', 'SSIM (higher)', 'LPIPS (lower)', 'GT-relative temporal L1 (lower)']):
        for branch in BRANCHES:
            rr = [r for r in rows if r['branch'] == branch]
            ax.plot([r['step'] for r in rr], [r[key] for r in rr], marker='o', label=branch,
                    linestyle='--' if branch == 'D3090' else '-', linewidth=2 if branch == 'D_GPU0' else 1.3)
        ax.set_title(title); ax.set_xlabel('Additional training steps'); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('MeetRoom discussion: fixed endpoints; all evaluated on GPU0')
    fig.savefig(output / 'quality_trajectory.png', dpi=170); plt.close(fig)
    gt_path = ROOT / 'data/dynamic_sr/meetroom_prepared/discussion/hr/cam00/0040.png'
    paths = {'HR': gt_path, **{b: f / 'eval_6000/predictions/0040.png' for b, f in folders.items()}}
    images = {b: rgb(p) for b, p in paths.items()}
    high_images = {b: high(im, (180, 320)) for b, im in images.items()}
    roi_rows = []
    for name, (x0, y0, x1, y1) in ROI.items():
        sl = np.s_[y0:y1, x0:x1]
        for suffix, columns in [('', ['HR', 'A', 'B', 'C', 'D_GPU0']),
                                ('_with_D3090', ['HR', 'A', 'B', 'C', 'D3090', 'D_GPU0'])]:
            panel = Image.new('RGB', (252 * len(columns), 296), 'white'); draw = ImageDraw.Draw(panel)
            for i, b in enumerate(columns):
                crop = np.round(np.clip(images[b][sl], 0, 1) * 255).astype(np.uint8)
                panel.paste(Image.fromarray(crop).resize((252, 252), Image.Resampling.NEAREST), (252 * i, 44))
                draw.text((252 * i + 5, 6), b, fill='black')
            draw.text((5, 25), f'{name}; fixed frame40; ROI={x0,y0,x1,y1}; saved PNG diagnostic', fill='black')
            panel.save(output / f'{name}_frame40{suffix}.png')
        for b in BRANCHES:
            ph, gh = high_images[b][sl], high_images['HR'][sl]
            roi_rows.append(dict(region=name, branch=b, xyxy=[x0, y0, x1, y1], frame=40,
                mse=float(np.mean((images[b][sl] - images['HR'][sl]) ** 2)), high_mse=float(np.mean((ph - gh) ** 2)),
                high_energy_ratio=float(np.mean(ph ** 2) / max(np.mean(gh ** 2), 1e-20)),
                high_correlation=float(np.corrcoef(ph.ravel(), gh.ravel())[0, 1])))
    fixed = dict(rows=roi_rows, selection='Same fixed frame40 and two regions from 2026-09-20 audit, reused before this repeat finished.',
                 quantization='Saved uint8 PNG diagnostic; main metrics use full float renders.',
                 high_operator='linear H=x-U(D_raw(x)); not orthogonal and not evidence of geometry truth',
                 inputs={b: dict(path=str(p), sha256=sha(p)) for b, p in paths.items()})
    write(output / 'fixed_roi.json', fixed)
    return dict(fixed_roi=sha(output / 'fixed_roi.json'),
                images={p.name: sha(p) for p in sorted(output.glob('*.png'))},
                operator_source_sha256=sha(ROOT / 'experiments/dynamic_sr_20260923/temporal_sharing.py'))


def assess(original, confirmation, output, complete):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    tick = time.monotonic()
    entries, audits = [], []
    result = dict(status='running', started_utc=now(), source_sha256=sha(__file__), cpu_only=True,
                  audit_helpers_source_sha256=sha(Path(__file__).with_name('audit_meeting_checkpoints.py')),
                  environment=dict(python=sys.executable, torch=str(torch.__version__)),
                  run=str(confirmation), original=str(original), run_complete_sha256=sha(confirmation / 'complete.json'),
                  checkpoint_writes=0, optimizer_steps=0, intermediate_training_logs_read=False,
                  global_rng_equality_required=False,
                  interpretation='A/B/C/D_GPU0 share training and evaluation GPU. D3090 is retained to measure the same-seed hardware-associated change; one repeat cannot isolate hardware from nondeterministic optimization or establish statistical significance.')
    try:
        prior_audit = load_json(original / 'audit.json')
        require(prior_audit['status'] == 'passed', 'Original checkpoint audit did not pass')
        require(sha(original / 'checkpoint_index.json') == prior_audit['checkpoint_index_sha256'], 'Original index changed after audit')
        prior_index = load_json(original / 'checkpoint_index.json')
        require(prior_index['audit_status'] == 'passed' and len(prior_index['entries']) == 12, 'Original index incomplete')
        prior_entries = {(r['branch'], r['intervention_step']): r for r in prior_index['entries']}
        require(len(prior_entries) == 12 and all(r['status'] == 'verified' for r in prior_entries.values()), 'Original identities incomplete')
        folder = confirmation / 'D'
        config, old_config = load_json(folder / 'config.json'), load_json(original / 'D/config.json')
        equal({k: v for k, v in config.items() if k not in CONFIG_EXCEPTIONS},
              {k: v for k, v in old_config.items() if k not in CONFIG_EXCEPTIONS}, 'D repeat config excluding permitted hardware/output fields')
        require(Path(config['out']).resolve() == folder and config['visible_cuda'] == GPU0 and 'RTX PRO 6000' in config['gpu'], 'D repeat hardware/output differ')
        require(config['parent_sha256'] == EXPECTED_PARENT_SHA and config['initial_points'] == POINTS and config['steps'] == 6000,
                'Unregistered parent/topology/steps')
        for key, hash_key in [('checkpoint', 'parent_sha256'), ('manifest', 'manifest_sha256'), ('cache_manifest', 'cache_manifest_sha256')]:
            require(sha(config[key]) == config[hash_key], f'Changed input {key}')
        for branch in 'ABCD':
            require(sha(original / branch / 'config.json') == prior_audit['training_hardware'][branch]['config_sha256'],
                    f'Original {branch} config changed after audit')
        model_source = None
        for i, record in enumerate(config['sources']):
            name = f'{i}_{Path(record["path"]).name}'
            for archive in [folder / 'sources' / name, original / 'D/sources' / name]:
                require(sha(archive) == record['sha256'], f'Training source snapshot changed: {archive}')
            if record['path'].endswith('/scene/gaussian_model.py'):
                model_source = folder / 'sources' / name
        require(model_source is not None, 'Missing capture source')
        fields = capture_layout(model_source); optimizer_index = fields.index('self.optimizer.state_dict()')
        parent = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        sampling = replay_sampling(load_json(config['manifest']), config['seed'], config['prior_cameras'])
        done = load_json(folder / 'complete.json')
        require(done['parameter_updates'] == 6000 and done['source_unchanged'] and done['full_sampler_states_saved'], 'D repeat training incomplete')
        require((folder / 'checkpoint_final.pt').resolve() == (folder / 'checkpoint_6000.pt').resolve(), 'Final checkpoint differs')
        receipt(complete, 'train', checkpoint=Path(config['checkpoint']), out=folder)
        folders = {'A': original / 'A', 'B': original / 'B', 'C': original / 'C', 'D3090': original / 'D', 'D_GPU0': folder}
        scores, summaries, fit_rows = {}, [], []
        protocol_reference = load_json(original / 'A/eval_1200/metrics.json')
        result['protocol_identity_fields'] = PROTOCOL_KEYS
        result['inputs'] = dict(config_sha256=sha(folder / 'config.json'), original_audit_sha256=sha(original / 'audit.json'),
                                original_index_sha256=sha(original / 'checkpoint_index.json'), parent_sha256=config['parent_sha256'],
                                manifest_sha256=config['manifest_sha256'], cache_manifest_sha256=config['cache_manifest_sha256'])
        # Verify all original identities against their already-passed audit before comparison.
        for step in STEPS:
            for branch in ['A', 'B', 'C', 'D3090']:
                saved = prior_entries[('D' if branch == 'D3090' else branch, step)]
                old_path = folders[branch] / f'checkpoint_{step}.pt'
                metric_path = folders[branch] / f'eval_{step}/metrics.json'
                require(old_path.stat().st_size == saved['bytes'] and sha(old_path) == saved['sha256'], f'{branch}/{step}: original checkpoint changed')
                require(sha(metric_path) == saved['evaluation_metrics_sha256'], f'{branch}/{step}: original evaluation changed')
                m = load_json(metric_path)
                require(m['checkpoint_sha256'] == saved['sha256'], 'Original evaluation checkpoint binding differs')
                protocol_identity(m, protocol_reference, f'{branch}/{step}')
                scores[(branch, step)] = m
            path = folder / f'checkpoint_{step}.pt'
            digest = sha(path)
            entry = dict(branch='D_GPU0', path=str(path.resolve()), sha256=digest, bytes=path.stat().st_size,
                         intervention_step=step, status='checking', training_gpu_uuid=GPU0, evaluation_gpu_uuid=GPU0)
            entries.append(entry)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            a = torch.load(original / 'A' / f'checkpoint_{step}.pt', map_location='cpu', weights_only=False)
            require({'model', 'hidden', 'optim', 'metadata', 'rng', 'samplers'} <= ck.keys(), 'Checkpoint missing fields')
            require(len(ck['model']) == len(fields) and ck['model'][1].shape[0] == POINTS and ck['model'][0] == parent['model'][0], 'Capture/point/SH mismatch')
            for key in ['hidden', 'optim']:
                equal(ck[key], parent[key], f'D_GPU0/{step}: parent {key}')
                equal(ck[key], a[key], f'D_GPU0/{step}: A {key}')
            finite_model = tensor_finiteness(ck['model'], 'model')
            finite_adam = tensor_finiteness(ck['model'][optimizer_index], 'Adam')
            opt, parent_opt = ck['model'][optimizer_index], parent['model'][optimizer_index]
            require(opt['state'].keys() == parent_opt['state'].keys(), 'Adam state keys changed')
            for k, state in opt['state'].items():
                require(float(state['step']) - float(parent_opt['state'][k]['step']) == step, f'Adam step count {k}')
            equal(opt['param_groups'], a['model'][optimizer_index]['param_groups'], 'Same-step optimizer groups/schedule')
            equal(ck['samplers'], a['samplers'], f'D_GPU0/{step}: exact A sampling/exposure')
            for k in ['lr', 'sr', 'source_prefix_exposure', 'additional_exposure']:
                equal(ck['samplers'][k], sampling[step][k], f'D_GPU0/{step}: replay {k}')
            meta = ck['metadata']
            require(meta['intervention_step'] == step and meta['step'] == parent['metadata']['step'] + step and meta['points'] == POINTS,
                    'Checkpoint step/point metadata')
            require(meta['draw_sha256'] == a['metadata']['draw_sha256'] == sampling[step]['draw_sha256'], 'Draw hash mismatch')
            require(meta['parent_sha'] == config['parent_sha256'] and meta['manifest_sha'] == config['manifest_sha256'], 'Lineage metadata differs')
            require(meta['scene'] == 'meetroom_discussion' and meta['stage'] == 'temporal_sharing_control', 'Checkpoint scene/stage differs')
            for k, v in meta['args'].items():
                equal(v, config[k], f'Checkpoint args {k}')
            old_meta = scores[('D3090', step)]['checkpoint_metadata']
            equal(meta['extent'], old_meta['extent'], 'Original D scene extent')
            equal({k: v for k, v in meta['args'].items() if k != 'out'},
                  {k: v for k, v in old_meta['args'].items() if k != 'out'}, 'Original D checkpoint arguments')
            if step == 6000:
                equal(ck['samplers']['additional_exposure'], load_json(folder / 'exposure.json'), 'Final exposure file')
                require(meta['draw_sha256'] == done['draw_sha256'], 'Final completion draw hash')
            metric_path = folder / f'eval_{step}/metrics.json'
            m = validate_metrics(metric_path, path, digest, meta, config['manifest_sha256'])
            protocol_identity(m, protocol_reference, f'D_GPU0/{step}')
            receipt(complete, f'eval_{step}', checkpoint=path, out=metric_path.parent)
            receipt(complete, f'fit_{step}', out=folder, step=step)
            scores[('D_GPU0', step)] = m
            entry.update(status='verified', step=meta['step'], points=POINTS, draw_sha256=meta['draw_sha256'],
                         evaluation_metrics=str(metric_path), evaluation_metrics_sha256=sha(metric_path))
            audits.append(dict(step=step, model_finiteness=finite_model, adam_finiteness=finite_adam,
                               exact_A_sampler_and_exposure=True, exact_manifest_replay=True, exact_training_config_except_allowed_fields=True,
                               checkpoint_eval_hash_match=True, evaluation_protocol_identity=True))
            for branch in BRANCHES:
                summaries.append(summary(branch, step, scores[(branch, step)]))
                fit_rows.append(fit_summary(folders[branch], branch, step))
            del ck, a
        by_summary = {(r['branch'], r['step']): r for r in summaries}
        by_fit = {(r['branch'], r['step']): r for r in fit_rows}
        deltas, fit_deltas = [], []
        for step in STEPS:
            for branch in ['A', 'B', 'C', 'D3090']:
                deltas.append(pairwise(step, scores[('D_GPU0', step)], scores[(branch, step)],
                                      by_summary[('D_GPU0', step)], by_summary[(branch, step)]))
                fit_deltas.append(fit_difference(by_fit[('D_GPU0', step)], by_fit[(branch, step)]))
        combined = dict(rows=summaries, deltas=deltas, fits=fit_rows, fit_deltas=fit_deltas, main_endpoint=6000,
                        fixed_endpoints=STEPS, endpoint_selection='All three predeclared endpoints; no best-test selection.',
                        scope='One development scene, one seed; 60 correlated frames per endpoint. Frame wins are descriptive counts, not independent trials or a significance test.',
                        labels={'D3090': 'Original D trained on RTX3090 and evaluated on GPU0',
                                'D_GPU0': 'Same D parameters/seed/source trained and evaluated on GPU0 RTX PRO6000'},
                        same_gpu_comparison='A/B/C/D_GPU0', training_hardware_note=result['interpretation'],
                        diagnostics='Interior train frames40/80 have temporal targets; boundary0/118 use direct A fallback. All train diagnostics contain four prior cameras; HR is evaluation-only.')
        write(output / 'combined_comparison.json', combined)
        result['figures'] = figures(output, folders, summaries)
        require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization during assessment')
        result.update(status='passed', checkpoint_audits=audits, checkpoints_verified=3,
                      original_checkpoints_rehashed=12, combined_comparison_sha256=sha(output / 'combined_comparison.json'),
                      finished_utc=now(), elapsed_seconds=time.monotonic() - tick)
    except Exception:
        if entries and entries[-1]['status'] == 'checking':
            entries[-1]['status'] = 'failed'
        result.update(status='failed', traceback=traceback.format_exc(), checkpoint_audits=audits,
                      finished_utc=now(), elapsed_seconds=time.monotonic() - tick)
        raise
    finally:
        write(output / 'checkpoint_index.json', dict(schema='dynamic_sr_same_gpu_checkpoint_index_v1',
              run=str(confirmation), audit_status=result['status'], entries=entries))
        result['checkpoint_index_sha256'] = sha(output / 'checkpoint_index.json')
        write(output / 'audit.json', result)
    print(json.dumps(dict(status=result['status'], output=str(output), checkpoints=3)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    parser.add_argument('--confirmation', type=Path, default=CONFIRMATION)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    original, confirmation = args.original.resolve(), args.confirmation.resolve()
    for path in [confirmation, original]:
        marker = path / 'complete.json'
        if not marker.is_file() or load_json(marker).get('status') != 'completed_and_evaluated':
            raise SystemExit(f'REFUSED: {path} is not completed_and_evaluated; no torch import, checkpoint read, or output write.')
    prior_audit = original / 'audit.json'
    if not prior_audit.is_file() or load_json(prior_audit).get('status') != 'passed':
        raise SystemExit('REFUSED: original 12-checkpoint CPU audit must pass first.')
    output = args.out.resolve() if args.out else confirmation / 'assessment'
    if output.exists():
        raise SystemExit('REFUSED: assessment output exists; preserve it and choose a new --out.')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'source.py')
    assess(original, confirmation, output, load_json(confirmation / 'complete.json'))


if __name__ == '__main__':
    main()
