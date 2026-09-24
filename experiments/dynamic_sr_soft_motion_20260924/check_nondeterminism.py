#!/usr/bin/env python3
"""Run exactly one extra B update and describe B/B versus B/S0 differences.

This never changes the failed exact-parity result or grants a tolerance-based
pass. GPU binding is inherited from the caller and must match the first check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from check_parity import ExactComparison, ROOT, OLD, require, sha, write


def stamp():
    return datetime.now(timezone.utc).isoformat()


def flatten_tensors(checkpoint):
    """Stable logical groups, retaining per-tensor detail for both comparisons."""
    values = {}
    def add(group, name, tensor):
        values[group + '/' + name] = (group, tensor)
    model = checkpoint['model']
    for index, name in ((1, 'xyz'), (4, 'f_dc'), (5, 'f_rest'), (6, 'scaling'), (7, 'rotation'), (8, 'opacity')):
        add('parameter.base.' + name, name, model[index])
    for name, tensor in model[2].items():
        add('state.deformation.' + ('grid' if '.grid.' in name else 'network'), name, tensor)
    for index, name in ((3, 'deformation_table'), (9, 'max_radii2D'), (10, 'xyz_gradient_accum'), (11, 'denom')):
        add('buffer.base', name, model[index])
    child_names = {'offset_raw', 'logscale', 'quaternion', 'opacity', 'sh_dc', 'sh_rest'}
    for name, tensor in checkpoint['motion_refinement']['children'].items():
        group = 'parameter.child.' + name if name in child_names else 'buffer.child'
        add(group, name, tensor)
    for owner, optimizer in (('base', model[12]), ('child', checkpoint['motion_refinement']['child_optimizer'])):
        for group in optimizer['param_groups']:
            for position, pid in enumerate(group['params']):
                for state_name, tensor in optimizer['state'].get(pid, {}).items():
                    import torch
                    if torch.is_tensor(tensor):
                        logical = f"optimizer.{owner}.{group['name']}.{state_name}"
                        add(logical, str(position), tensor)
    return values


def tensor_difference(left, right):
    import torch
    require(left.shape == right.shape and left.dtype == right.dtype, 'Checkpoint tensor shape/dtype changed')
    a, b = left.detach().double(), right.detach().double()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite checkpoint tensor')
    difference = (a - b).abs()
    n = difference.numel()
    sign_flip = ((a > 0) & (b < 0)) | ((a < 0) & (b > 0))
    return dict(elements=n, unequal_elements=int((a != b).sum()),
        max_abs=float(difference.max()) if n else 0., sum_squared=float(difference.square().sum()),
        sum_abs=float(difference.sum()), opposite_sign_elements=int(sign_flip.sum()),
        max_value_at_opposite_sign=float(torch.maximum(a.abs(), b.abs())[sign_flip].max())
        if bool(sign_flip.any()) else 0.)


def finish_stats(row):
    n = row['elements']
    row['rms'] = (row['sum_squared'] / n) ** .5 if n else 0.
    row['mean_abs'] = row['sum_abs'] / n if n else 0.
    row['exact'] = row['unequal_elements'] == 0
    return row


def difference_report(first, other):
    left, right = flatten_tensors(first), flatten_tensors(other)
    require(left.keys() == right.keys(), 'Checkpoint tensor inventory differs')
    per_tensor, groups = {}, {}
    for name, (group, tensor) in left.items():
        row = tensor_difference(tensor, right[name][1])
        per_tensor[name] = finish_stats(dict(row))
        if group not in groups:
            groups[group] = dict(row)
        else:
            total = groups[group]
            for key in ('elements', 'unequal_elements', 'sum_squared', 'sum_abs', 'opposite_sign_elements'):
                total[key] += row[key]
            for key in ('max_abs', 'max_value_at_opposite_sign'):
                total[key] = max(total[key], row[key])
    controls = ExactComparison()
    for key in ('rng', 'samplers', 'hidden', 'optim'):
        controls.compare(first[key], other[key], key)
    controls.compare(first['model'][12]['param_groups'], other['model'][12]['param_groups'], 'base_optimizer_groups')
    controls.compare(first['motion_refinement']['child_optimizer']['param_groups'],
                     other['motion_refinement']['child_optimizer']['param_groups'], 'child_optimizer_groups')
    for key in ('step', 'intervention_step', 'parent_sha', 'selection_sha256', 'manifest_sha', 'draw_sha256'):
        controls.compare(first['metadata'][key], other['metadata'][key], 'metadata/' + key)
    return dict(groups={k:finish_stats(v) for k,v in sorted(groups.items())}, tensors=per_tensor,
                controls_rng_samplers=controls.result())


def relative_observation(repeat, soft):
    return {metric: {'B1_B2': repeat[metric], 'B1_S0': soft[metric],
                     'S0_greater_than_this_single_repeat': soft[metric] > repeat[metric],
                     'S0_to_repeat_ratio': soft[metric] / repeat[metric] if repeat[metric] > 0 else None}
            for metric in ('max_abs', 'rms', 'mean_abs')}


def source_evidence():
    import torch
    spec = importlib.util.find_spec('diff_gaussian_rasterization')
    require(spec is not None and spec.origin, 'Cannot locate installed rasterizer')
    base = Path(spec.origin).parent.parent
    torch_root = Path(torch.__file__).parent
    paths = [(base / 'cuda_rasterizer/backward.cu', 'atomicAdd'),
             (torch_root / 'include/ATen/native/cuda/GridSampler.cuh', 'fastAtomicAdd'),
             (Path(os.environ.get('FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs'))
              / 'scene/hexplane.py', 'grid_sample')]
    rows = []
    for path, token in paths:
        require(path.is_file(), f'Expected local source missing: {path}')
        text = path.read_text().splitlines()
        rows.append(dict(path=str(path), sha256=sha(path),
            matches=[dict(line=i+1, text=line.strip()) for i,line in enumerate(text) if token in line]))
    distribution = importlib.metadata.distribution('diff_gaussian_rasterization')
    return dict(torch_version=str(torch.__version__), installed_rasterizer_python=str(spec.origin),
        installed_rasterizer_direct_url=json.loads(distribution.read_text('direct_url.json') or '{}'),
        sources=rows, interpretation='Rasterizer backward and CUDA grid gradient accumulation use atomic additions. '
        'Floating-point sum ordering can vary without consuming RNG. This is a possible mechanism, not proof '
        'that every observed B/S0 difference is caused by atomic ordering; no kernel or optimizer is modified.')


def render_comparisons(parity, checkpoint_paths):
    import torch
    sys.path.insert(0, str(OLD))
    from motion_model import load_model, render_model
    from common import resized_camera
    from n3dv_data import load_manifest
    from run_experiment import load_training
    manifest = load_manifest(parity['args']['manifest'])
    records, cameras = load_training(manifest)
    camera_id, frame = parity['initial_forward']['camera'], parity['initial_forward']['frame']
    matches = [i for i,r in enumerate(records) if r['camera_id'] == camera_id and int(r['frame_index']) == frame]
    require(len(matches) == 1, 'Original training render observation must exist exactly once')
    width, height = manifest['resolutions']['hr']
    camera = resized_camera(cameras[matches[0]], height, width)
    rendered = {}
    for name, path in checkpoint_paths.items():
        model = load_model(path)
        with torch.no_grad():
            rendered[name] = render_model(model, camera)['render'].cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    rows = {}
    for name in ('B2', 'S0'):
        difference = (rendered['B1'].double() - rendered[name].double()).abs()
        require(bool(torch.isfinite(difference).all()), 'Nonfinite render difference')
        rows['B1_' + name] = dict(mean_abs=float(difference.mean()), max_abs=float(difference.max()),
            rms=float(difference.square().mean().sqrt()),
            component_fraction_above_1_over_255=float((difference > 1/255).double().mean()),
            pixel_fraction_any_channel_above_1_over_255=float((difference > 1/255).any(0).double().mean()),
            exact=bool(torch.equal(rendered['B1'], rendered[name])))
    return dict(camera=camera_id, frame=frame, precision='unclipped float render, before PNG quantization',
                threshold_note='1/255 is a descriptive display increment, never a parity pass threshold', **rows)


def run(args, result):
    parity = json.loads((args.parity_dir / 'checks.json').read_text())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(visible) and ',' not in visible and visible != '-1', 'Bind one GPU explicitly')
    require(visible == parity['visible_cuda'], 'Repeat must retain the original parity GPU binding')
    original_folder = args.parity_dir / 'old_B_one_step'
    config = json.loads((original_folder / 'config.json').read_text())
    for source in config['sources']:
        require(sha(source['path']) == source['sha256'], f'B training source changed: {source["path"]}')
    for key, field in (('manifest', 'manifest_sha256'), ('checkpoint', 'parent_sha256'), ('selection', 'selection_sha256')):
        require(sha(parity['args'][key]) == config[field], f'Original input changed: {key}')
    command = list(parity['runs'][0]['command'])
    require(Path(command[1]).resolve() == (OLD / 'train.py').resolve(), 'Original command is not the old B trainer')
    require(command[command.index('--steps')+1] == '1' and '--smoke' not in command, 'Only one B update is allowed')
    require(command[command.index('--branch')+1] == 'ordinary_split', 'Expected ordinary_split B')
    folder = args.out / 'B2_one_step'
    command[command.index('--out')+1] = str(folder)
    result.update(original_parity=str(args.parity_dir), original_strict_parity_unchanged=True,
                  command=command, visible_cuda=visible, args_inherited=parity['args'])
    write(args.out / 'status.json', result)
    with (args.out / 'B2_one_step.log').open('w') as stream:
        child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
    require(child.returncode == 0, 'B2 one-step run failed; see B2_one_step.log')
    done = json.loads((folder / 'complete.json').read_text())
    require(done.get('status') == 'completed' and done.get('parameter_updates') == 1 and not done.get('smoke'),
            'B2 must complete exactly one update')
    import torch
    torch.set_num_threads(4)
    require(str(torch.__version__) == config['torch'], 'Analysis environment differs from original Torch version')
    checkpoint_paths = {'B1': original_folder / 'checkpoint_1.pt',
        'B2': folder / 'checkpoint_1.pt', 'S0': args.parity_dir / 'soft_zero_one_step/checkpoint_1.pt'}
    checkpoints = {name:torch.load(path, map_location='cpu', weights_only=False) for name,path in checkpoint_paths.items()}
    result['checkpoint_sha256'] = {name:sha(path) for name,path in checkpoint_paths.items()}
    result['B1_B2'] = difference_report(checkpoints['B1'], checkpoints['B2'])
    result['B1_S0'] = difference_report(checkpoints['B1'], checkpoints['S0'])
    del checkpoints
    gc.collect()
    result['group_comparison'] = {name: relative_observation(result['B1_B2']['groups'][name], row)
                                 for name,row in result['B1_S0']['groups'].items()}
    result['renders'] = render_comparisons(parity, checkpoint_paths)
    result['render_comparison'] = relative_observation(result['renders']['B1_B2'], result['renders']['B1_S0'])
    result['cuda_backward_evidence'] = source_evidence()
    repeat_nonzero = any(not g['exact'] for g in result['B1_B2']['groups'].values())
    result['interpretation'] = dict(B_repeat_is_bitwise_nondeterministic=repeat_nonzero,
        S0_render_mean_exceeds_observed_repeat=result['renders']['B1_S0']['mean_abs'] > result['renders']['B1_B2']['mean_abs'],
        S0_render_max_exceeds_observed_repeat=result['renders']['B1_S0']['max_abs'] > result['renders']['B1_B2']['max_abs'],
        S0_render_rms_exceeds_observed_repeat=result['renders']['B1_S0']['rms'] > result['renders']['B1_B2']['rms'],
        no_automatic_parity_pass=True,
        limitation='One repeated B run gives one observed difference, not a noise distribution, upper bound, '
        'statistical equivalence test, or proof of B/S0 implementation identity. Strict parity remains failed; '
        'compare parameter/optimizer groups and rendered magnitudes together before a human engineering decision.')
    write(args.out / 'comparison.json', result)
    for name in ('B1_B2', 'B1_S0'):
        require(result[name]['controls_rng_samplers']['exact'], f'RNG/samplers/config controls differ in {name}')
    lines = ['# One-step nondeterminism comparison', '',
        'Exactly one extra old-B update. Original strict parity is retained; this report grants no relaxed pass.', '',
        '| Comparison | Render mean absolute | RMS | Maximum | RGB fraction > 1/255 | Pixel fraction > 1/255 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ('B1_B2', 'B1_S0'):
        r = result['renders'][name]
        lines.append(f"| {name} | {r['mean_abs']:.9g} | {r['rms']:.9g} | {r['max_abs']:.9g} | "
                     f"{r['component_fraction_above_1_over_255']:.9g} | {r['pixel_fraction_any_channel_above_1_over_255']:.9g} |")
    lines += ['', 'Full tensor, parameter-group, Adam-state, sign-change and source details: [comparison.json](comparison.json).', '',
        'Atomic accumulation is present in the installed rasterizer source and CUDA grid-sampler header. '
        'It can vary floating-point reduction order without changing RNG. This is a plausible cause, not a complete causal attribution.', '',
        result['interpretation']['limitation'], '']
    (args.out / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.parity_dir, args.out = args.parity_dir.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(status='running_nondeterminism_check', started_utc=stamp(),
                  source_sha256=sha(__file__), extra_training_updates=1)
    try:
        run(args, result)
        result.update(status='completed_nondeterminism_check', finished_utc=stamp())
        write(args.out / 'complete.json', result)
        write(args.out / 'status.json', result)
        print(json.dumps({'status':result['status'], 'out':str(args.out), 'interpretation':result['interpretation']}))
    except BaseException:
        result.update(status='failed_nondeterminism_check', finished_utc=stamp(), traceback=traceback.format_exc())
        write(args.out / 'failed.json', result)
        write(args.out / 'status.json', result)
        raise


if __name__ == '__main__':
    main()
