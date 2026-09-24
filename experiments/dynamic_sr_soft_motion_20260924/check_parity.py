#!/usr/bin/env python3
"""One-step B / S(lambda=0) engineering parity; not a scientific training run.

The caller must bind one GPU via CUDA_VISIBLE_DEVICES. This script runs the
unchanged old and new train entry points sequentially for exactly one update,
then checks saved state and small forward/gradient invariants. No HR is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/dynamic_sr_motion_bound_20260923'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def require(value, message):
    if not value:
        raise AssertionError(message)


class ExactComparison:
    """All tensor/array/scalar leaves checked; numerical differences never waived."""
    def __init__(self):
        import torch
        import numpy as np
        self.torch, self.np = torch, np
        self.tensors = self.arrays = self.scalars = self.mismatch_count = 0
        self.max_abs = 0.0
        self.examples = []

    def mismatch(self, path, reason, maximum=None):
        self.mismatch_count += 1
        if maximum is not None:
            self.max_abs = max(self.max_abs, maximum)
        if len(self.examples) < 30:
            self.examples.append(dict(path=path, reason=reason, max_abs=maximum))

    def compare(self, a, b, path):
        torch, np = self.torch, self.np
        if torch.is_tensor(a) or torch.is_tensor(b):
            self.tensors += 1
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                self.mismatch(path, 'tensor/type mismatch')
            elif a.shape != b.shape or a.dtype != b.dtype:
                self.mismatch(path, f'shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}')
            elif not torch.equal(a, b):
                delta = (a.detach().cpu().double() - b.detach().cpu().double()).abs()
                maximum = float(delta.max()) if delta.numel() and bool(torch.isfinite(delta).all()) else None
                self.mismatch(path, 'not bitwise equal' if maximum is not None else 'nonfinite mismatch', maximum)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            self.arrays += 1
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                    and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
                self.mismatch(path, 'numpy array mismatch')
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                self.mismatch(path, f'dict keys differ: {set(a) ^ set(b)}')
            for key in a.keys() & b.keys():
                self.compare(a[key], b[key], f'{path}/{key}')
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                self.mismatch(path, 'sequence type/length mismatch')
            for i, (left, right) in enumerate(zip(a, b)):
                self.compare(left, right, f'{path}/{i}')
        else:
            self.scalars += 1
            if type(a) is not type(b) or a != b:
                self.mismatch(path, f'scalar/type mismatch: {a!r} vs {b!r}')

    def result(self):
        return dict(exact=self.mismatch_count == 0, tensor_leaves=self.tensors,
                    numpy_array_leaves=self.arrays, scalar_leaves=self.scalars,
                    mismatches=self.mismatch_count, max_finite_absolute_difference=self.max_abs,
                    mismatch_examples=self.examples, tolerance='zero; no automatic relaxation')


def rng_state():
    import numpy as np
    import torch
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def child_run(command, folder, log_path):
    with log_path.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'One-step command failed ({result.returncode}); see {log_path}')
    done = json.loads((folder / 'complete.json').read_text())
    require(done.get('status') == 'completed' and done.get('parameter_updates') == 1
            and not done.get('smoke'), f'Not an exact one-update run: {folder}')
    return dict(command=command, log=str(log_path), status='completed_one_step',
                checkpoint=str(folder / 'checkpoint_1.pt'), draw_sha256=done['draw_sha256'])


def check_identity(args):
    import torch
    state = torch.load(args.reference, map_location='cpu', weights_only=False)
    calibration = json.loads(args.calibration.read_text())
    identity = {key: sha(path) for key, path in (
        ('parent_sha256', args.checkpoint), ('manifest_sha256', args.manifest),
        ('selection_sha256', args.selection))}
    for key, value in identity.items():
        require(state['metadata'][key] == value, f'Reference identity mismatch: {key}')
        require(calibration[key] == value, f'Calibration identity mismatch: {key}')
    require(calibration['status'] == 'calibrated', 'Calibration is not complete')
    require(calibration['reference_sha256'] == sha(args.reference), 'Calibration/reference SHA mismatch')
    require(state['frames'] == list(range(0, 120, 2)), 'Expected the registered 60 training times')
    require(state['metadata']['reference_frame'] == 60, 'Expected frame60 anchor')
    for name in ('reference_child_centers', 'child_length', 'delta0', 'initial_offset_raw'):
        require(not state[name].requires_grad and state[name].grad_fn is None,
                f'Reference constant has autograd connection: {name}')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    require(parent['metadata'].get('mode') == 'lr_integrated', 'Reference source must be pure LR integrated parent')
    require('motion_refinement' not in parent, 'Reference source must be unsplit')
    return dict(**identity, reference_sha256=sha(args.reference), calibration_sha256=sha(args.calibration),
                reference_frame=60, training_frames=60, reference_constants_detached=True,
                reference_source_mode=parent['metadata']['mode'])


def probes(args, report):
    import gc
    import torch
    from soft_model import FrozenReference, child_centers, render_with_centers
    from motion_model import load_model, make_model, render_model
    from common import load_checkpoint, named_parameters, resized_camera
    from n3dv_data import load_manifest
    from run_experiment import load_training

    torch.set_num_threads(4)
    manifest = load_manifest(args.manifest)
    records, cameras = load_training(manifest)
    camera_id = args.prior_cameras.split(',')[0]
    selected = [i for i, r in enumerate(records)
                if r['camera_id'] == camera_id and int(r['frame_index']) == 0]
    require(len(selected) == 1, 'Need one fixed first-prior-camera/frame0 training observation')
    index = selected[0]
    require(records[index]['split'] == 'train', 'Probe must use training observation only')
    width, height = manifest['resolutions']['hr']
    camera = resized_camera(cameras[index], height, width)
    g, h, o, ck = load_checkpoint(args.checkpoint)
    selection = torch.load(args.selection, map_location='cpu', weights_only=False)
    model = make_model(g, h, o, ck, manifest, 'ordinary_split', selection)
    reference = FrozenReference(args.reference)
    reference.validate(model, report['identity']['parent_sha256'], report['identity']['selection_sha256'],
                       report['identity']['manifest_sha256'])
    before = rng_state()
    old = render_model(model, camera)
    new = render_with_centers(model, camera)
    initial = ExactComparison()
    for key in ('render', 'radii', 'visibility_filter', 'depth'):
        initial.compare(old[key].detach(), new[key].detach(), f'initial/{key}')
    report['initial_forward'] = dict(camera=camera_id, frame=0, **initial.result())
    b = new['child_centers']
    br = child_centers(model, reference.reference_time)
    loss = reference.loss(b, br, 0)
    require(bool(torch.isfinite(loss)), 'Frame0 motion loss is nonfinite')
    current_grad, anchor_grad = torch.autograd.grad(loss, (b, br), retain_graph=True)
    require(all(bool(torch.isfinite(t).all()) for t in (current_grad, anchor_grad)),
            'Motion gradients must reach both current and anchor centers and be finite')
    # This also tests that the exposed center tensor is an ancestor of the
    # rendered image, rather than a new slice disconnected from the image.
    image_grad = torch.autograd.grad(new['render'].mean(), b)[0]
    require(bool(torch.isfinite(image_grad).all()), 'Image/current-center graph is invalid')
    at60 = child_centers(model, reference.reference_time)
    anchor60 = child_centers(model, reference.reference_time)
    zero_loss = reference.loss(at60, anchor60, 60)
    require(float(zero_loss) == 0.0, 'Frame60 motion loss must be exactly zero')
    require(not reference.centers.requires_grad and not reference.length.requires_grad
            and reference.centers.grad is None and reference.length.grad is None,
            'Frozen reference must not receive gradients')
    parameters = list(named_parameters(g).values()) + list(model.children.parameters())
    require(all(p.grad is None for p in parameters), 'Output-gradient probes accumulated parameter gradients')
    report['motion_graph'] = dict(frame0_loss=float(loss), frame60_loss=float(zero_loss),
        current_center_gradient_norm=float(current_grad.double().norm()),
        anchor_center_gradient_norm=float(anchor_grad.double().norm()),
        opposite_center_gradients_exact=bool(torch.equal(current_grad, -anchor_grad)),
        image_center_gradient_norm=float(image_grad.double().norm()),
        both_center_gradients_connected=True, all_gradients_finite=True,
        reference_has_no_grad=True, parameter_grads_untouched=True)
    queries = ExactComparison()
    queries.compare(before, rng_state(), 'render_and_extra_queries_rng')
    report['query_rng'] = queries.result()
    del old, new, b, br, loss, current_grad, anchor_grad, image_grad, at60, anchor60, zero_loss
    del model, g, h, o, ck, selection, parameters, reference
    gc.collect()
    torch.cuda.empty_cache()

    updated_images = []
    for folder in (args.out / 'old_B_one_step', args.out / 'soft_zero_one_step'):
        updated = load_model(folder / 'checkpoint_1.pt')
        with torch.no_grad():
            updated_images.append(render_model(updated, camera)['render'].cpu())
        del updated
        gc.collect()
        torch.cuda.empty_cache()
    comparison = ExactComparison()
    comparison.compare(*updated_images, 'updated_render')
    report['updated_render'] = dict(camera=camera_id, frame=0, **comparison.result())


def run(args, report):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(visible) and ',' not in visible and visible != '-1',
            'Caller must bind one GPU using CUDA_VISIBLE_DEVICES')
    report['visible_cuda'] = visible
    report['identity'] = check_identity(args)
    common = ['--manifest', str(args.manifest), '--checkpoint', str(args.checkpoint),
              '--selection', str(args.selection), '--prior-cameras', args.prior_cameras,
              '--steps', '1', '--milestones', '1', '--seed', '20260923', '--sr-weight', '0.1']
    before_hashes = {str(path): sha(path) for path in
                     (Path(__file__), OLD / 'train.py', OLD / 'motion_model.py',
                      HERE / 'train.py', HERE / 'soft_model.py', HERE / 'prepare_reference.py')}
    report['source_sha256'] = before_hashes
    report['runs'] = []
    for label, script, extras in (
        ('old_B_one_step', OLD / 'train.py', ['--branch', 'ordinary_split']),
        ('soft_zero_one_step', HERE / 'train.py', ['--lambda-override', '0',
             '--reference', str(args.reference), '--calibration', str(args.calibration)])):
        folder = args.out / label
        command = [sys.executable, str(script), *common, '--out', str(folder), *extras]
        report['runs'].append(child_run(command, folder, args.out / f'{label}.log'))
        write(args.out / 'status.json', report)
    import torch
    left, right = [torch.load(args.out / name / 'checkpoint_1.pt', map_location='cpu', weights_only=False)
                   for name in ('old_B_one_step', 'soft_zero_one_step')]
    comparison = ExactComparison()
    # Capture includes all Gaussian tensors, shared deformation state, Adam
    # groups/moments/steps and buffers. The refinement adds all child fields.
    for key in ('model', 'hidden', 'optim', 'motion_refinement', 'samplers', 'rng'):
        comparison.compare(left[key], right[key], key)
    for key in ('step', 'intervention_step', 'parent_sha', 'selection_sha256', 'manifest_sha',
                'points', 'capacity', 'draw_sha256'):
        comparison.compare(left['metadata'][key], right['metadata'][key], 'metadata/' + key)
    report['one_step_state'] = comparison.result()
    del left, right
    probes(args, report)
    for path, expected in before_hashes.items():
        require(sha(path) == expected, f'Code changed during parity check: {path}')
    write(args.out / 'checks.json', report)
    for name in ('one_step_state', 'initial_forward', 'query_rng', 'updated_render'):
        require(report[name]['exact'], f'Exact parity failed: {name}; see checks.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'checkpoint', 'selection', 'reference', 'calibration', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--prior-cameras', required=True)
    args = parser.parse_args()
    for key in ('manifest', 'checkpoint', 'selection', 'reference', 'calibration', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running_parity', started_utc=stamp(), args={k:str(v) for k,v in vars(args).items()},
                  scope='Engineering only: two sequential one-update jobs, no HR reads, no scientific result',
                  strictness='All saved numerical states and rendered outputs require exact equality')
    write(args.out / 'status.json', report)
    try:
        run(args, report)
        report.update(status='passed_parity', finished_utc=stamp())
        write(args.out / 'complete.json', report)
        write(args.out / 'status.json', report)
        print(json.dumps({'status':report['status'], 'out':str(args.out)}, ensure_ascii=False))
    except BaseException:
        report.update(status='failed_parity', finished_utc=stamp(), traceback=traceback.format_exc())
        write(args.out / 'failed.json', report)
        write(args.out / 'status.json', report)
        raise


if __name__ == '__main__':
    main()
