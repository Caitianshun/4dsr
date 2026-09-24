"""P0 checkpoint/runtime audit: two LR parents, one LR image backward, no updates.

Run after activate_local.sh with only physical GPU0 exposed by UUID. This is a
read-only model/data audit; only the new output directory is written. The
checkpoint tuple layout is derived from the installed capture() implementation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import inspect
import io
import json
import os
import platform
import random
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
PARENTS = [
    ('cook_spinach', ROOT / 'output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/checkpoint_final.pt',
     '65cc81028121de0dc37f48d351a281d5a8dd525f642878c6281f02f870565945'),
    ('meetroom_discussion', ROOT / 'output/dynamic_sr_20260919/meetroom_discussion_integrated_parent/checkpoint_final.pt',
     '21e14ab525507832e79fc0449277cf36e1da17c5ed867f8cbb20847e5be45e6e'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def gpu_snapshot():
    def query(arg):
        raw = subprocess.check_output(['nvidia-smi', arg, '--format=csv,noheader,nounits'], text=True)
        return list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    devices = query('--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu')
    processes = query('--query-compute-apps=gpu_uuid,pid,process_name,used_memory')
    return {'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'devices': devices, 'compute_processes': processes}


def require_gpu0_available(snapshot):
    device = [x for x in snapshot['devices'] if x[0] == '0']
    if len(device) != 1 or device[0][2] != GPU_UUID or 'RTX PRO 6000' not in device[0][1]:
        raise RuntimeError('Physical GPU0 identity changed; refusing to choose another GPU')
    unexpected = [x for x in snapshot['compute_processes']
                  if x[0] == GPU_UUID and 'ToDesk' not in x[2] and int(x[1]) != os.getpid()]
    if unexpected:
        raise RuntimeError(f'GPU0 has unexpected compute processes: {unexpected}')


def compare_tree(reference, actual, prefix='root'):
    """Exact values/dtypes/shapes, normalizing device only; no tolerance hiding."""
    import numpy as np
    import torch
    count, mismatches = 0, []
    def walk(a, b, p):
        nonlocal count
        count += 1
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
                mismatches.append({'path': p, 'reason': 'tensor_shape_or_dtype'})
            elif not torch.equal(a.detach().cpu(), b.detach().cpu()):
                delta = (a.detach().cpu().double() - b.detach().cpu().double()).abs()
                mismatches.append({'path': p, 'reason': 'tensor_values', 'max_abs': float(delta.max())})
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
                mismatches.append({'path': p, 'reason': 'numpy_values'})
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                mismatches.append({'path': p, 'reason': 'dict_keys'})
            else:
                for k in a:
                    walk(a[k], b[k], f'{p}.{k}')
        elif isinstance(a, (tuple, list)):
            if not isinstance(b, (tuple, list)) or len(a) != len(b):
                mismatches.append({'path': p, 'reason': 'sequence_length'})
            else:
                for i, (aa, bb) in enumerate(zip(a, b)):
                    walk(aa, bb, f'{p}[{i}]')
        elif a != b:
            mismatches.append({'path': p, 'reason': 'scalar_value', 'reference': str(a), 'actual': str(b)})
    walk(reference, actual, prefix)
    return {'exact_equal': not mismatches, 'nodes_checked': count, 'mismatches': mismatches}


def rng_state():
    import numpy as np
    import torch
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def gradients(g, named):
    import torch
    by_id = {id(p): n for n, p in named.items()}
    def describe(params):
        values = [(n, p, p.grad) for n, p in params]
        with_grad = [(n, p, d) for n, p, d in values if d is not None]
        return {'parameter_tensors': len(values), 'parameter_elements': sum(p.numel() for _, p, _ in values),
                'requires_grad_tensors': sum(p.requires_grad for _, p, _ in values),
                'with_gradient_tensors': len(with_grad),
                'nonzero_gradient_tensors': sum(bool(torch.count_nonzero(d)) for _, _, d in with_grad),
                'all_gradients_finite': all(bool(torch.isfinite(d).all()) for _, _, d in with_grad),
                'gradient_l2': sum(float(d.double().square().sum()) for _, _, d in with_grad) ** .5,
                'gradient_max_abs': max([float(d.abs().max()) for _, _, d in with_grad] or [0.]),
                'missing_gradient_names': [n for n, _, d in values if d is None]}
    groups = {p['name']: describe([(by_id[id(x)], x) for x in p['params']]) for p in g.optimizer.param_groups}
    heads = {key: describe([(n, p) for n, p in named.items() if f'.{key}.' in n])
             for key in ['pos_deform', 'scales_deform', 'rotations_deform', 'opacity_deform', 'shs_deform', 'feature_out', 'timenet']}
    return {'optimizer_groups': groups, 'deformation_heads': heads}


def run(out):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
    import common
    from run_experiment import load_training
    from n3dv_data import load_manifest

    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Expose only physical GPU0 using its UUID before importing CUDA')
    pre = gpu_snapshot()
    require_gpu0_available(pre)
    write(out / 'gpu_before.json', pre)
    if torch.cuda.device_count() != 1 or 'RTX PRO 6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Unexpected visible CUDA device')

    capture_source = inspect.getsource(common.GaussianModel.capture)
    capture_ast = ast.parse(textwrap.dedent(capture_source))
    capture_return = next(n for n in ast.walk(capture_ast) if isinstance(n, ast.Return))
    fields = [ast.unparse(x) for x in capture_return.value.elts]
    opt_index = fields.index('self.optimizer.state_dict()')
    if len(fields) != 14 or opt_index != 12:
        raise RuntimeError('Unexpected capture schema; review original implementation')
    sources = [Path(__file__), Path(common.__file__), ROOT / 'experiments/dynamic_sr_20260918/run_experiment.py',
               ROOT / 'experiments/dynamic_sr_20260918/n3dv_data.py',
               common.UPSTREAM / 'scene/gaussian_model.py', common.UPSTREAM / 'scene/deformation.py',
               common.UPSTREAM / 'scene/hexplane.py', common.UPSTREAM / 'gaussian_renderer/__init__.py',
               common.UPSTREAM / 'arguments/__init__.py', common.UPSTREAM / 'activate_local.sh']
    source_hashes = {str(p): sha(p) for p in sources}
    write(out / 'source_identity.json', {'sources': source_hashes, 'capture_fields': fields,
                                       'optimizer_tuple_index': opt_index, 'capture_source': capture_source,
                                       'restore_source': inspect.getsource(common.GaussianModel.restore)})
    runtime = {'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
               'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
               'host': platform.node(), 'pid': os.getpid(), 'gpu_name': torch.cuda.get_device_name(0),
               'gpu_uuid': GPU_UUID, 'gpu_logical_index': 0, 'gpu_physical_index': 0,
               'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
               'CONDA_PREFIX': os.environ.get('CONDA_PREFIX'), 'CUDA_HOME': os.environ.get('CUDA_HOME'),
               'optimizer_steps': 0, 'checkpoint_writes': 0, 'hr_images_opened': 0,
               'operator': 'HR-resolution render -> bicubic antialiased downsample -> train LR L1; no HR image target',
               'scenes': []}
    write(out / 'runtime.json', runtime)

    for scene, checkpoint, expected_sha in PARENTS:
        started = time.monotonic()
        scene_out = out / scene
        scene_out.mkdir()
        print(json.dumps({'event': 'start_scene', 'scene': scene}), flush=True)
        checkpoint_sha = sha(checkpoint)
        if checkpoint_sha != expected_sha:
            raise AssertionError(f'Checkpoint hash changed: {scene}')
        cpu = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if len(cpu['model']) != len(fields):
            raise AssertionError('Checkpoint schema does not match capture')
        saved_opt = cpu['model'][opt_index]
        manifest_path = Path(cpu['metadata']['manifest'])
        manifest_sha = sha(manifest_path)
        if manifest_sha != cpu['metadata']['manifest_sha']:
            raise AssertionError('Parent manifest identity changed')
        cpuid = {'scene': scene, 'checkpoint': str(checkpoint), 'sha256': checkpoint_sha,
                 'expected_sha256': expected_sha, 'bytes': checkpoint.stat().st_size,
                 'checkpoint_keys': list(cpu), 'metadata': cpu['metadata'], 'hidden': cpu['hidden'],
                 'optim': cpu['optim'], 'capture_fields': fields, 'optimizer_tuple_index': opt_index,
                 'points': int(cpu['model'][1].shape[0]), 'active_sh_degree': cpu['model'][0],
                 'manifest': str(manifest_path), 'manifest_sha256': manifest_sha,
                 'saved_rng_keys': list(cpu.get('rng', {})),
                 'saved_cuda_rng_count': len(cpu.get('rng', {}).get('cuda', [])),
                 'camera_sampler_state_saved': False,
                 'optimizer_group_summary': [{k: v for k, v in p.items() if k != 'params'} | {'parameter_ids': p['params']}
                                             for p in saved_opt['param_groups']],
                 'adam_state_entries': len(saved_opt['state'])}
        write(scene_out / 'checkpoint_identity_cpu.json', cpuid)
        require_gpu0_available(gpu_snapshot())
        torch.cuda.reset_peak_memory_stats()
        global_before = rng_state()
        g, h, o, loaded = common.load_checkpoint(checkpoint)
        after_restore = rng_state()
        checks = {'capture_vs_cpu_checkpoint': compare_tree(cpu['model'], g.capture()),
                  'optimizer_vs_cpu_checkpoint': compare_tree(saved_opt, g.optimizer.state_dict()),
                  'hidden_vs_checkpoint': compare_tree(cpu['hidden'], vars(h)),
                  'optim_vs_checkpoint': compare_tree(cpu['optim'], vars(o))}
        for label, check in checks.items():
            if not check['exact_equal']:
                raise AssertionError(f'{label}: {check}')
        rng_audit = {'loader_restores_saved_rng': False,
                     'reason': 'common.load_checkpoint restores model/Adam only; no RNG setters; model construction consumes torch RNG',
                     'loader_rng_changed_from_entry': not compare_tree(global_before, after_restore)['exact_equal'],
                     'loader_rng_matches_saved': compare_tree(cpu['rng'], after_restore),
                     'camera_sampler_state_saved': False,
                     'camera_sampler_boundary': 'branch creates separate random.Random(seed+177/211); their getstate is not in saved rng. Exact continuation requires explicit sampler state or validated replay.'}
        random.setstate(cpu['rng']['python'])
        np.random.set_state(cpu['rng']['numpy'])
        torch.set_rng_state(cpu['rng']['torch'].cpu())
        if len(cpu['rng']['cuda']) != 1:
            raise RuntimeError('Saved CUDA RNG device mapping ambiguous')
        torch.cuda.set_rng_state(cpu['rng']['cuda'][0].cpu(), device=0)
        rng_audit['explicit_global_restore_check'] = compare_tree(cpu['rng'], rng_state())
        if not rng_audit['explicit_global_restore_check']['exact_equal']:
            raise AssertionError('Explicit global RNG restoration failed')
        manifest = load_manifest(manifest_path)
        records, cameras = load_training(manifest)
        target_id = next(i for i, r in enumerate(records) if r['camera_id'] == 'cam02' and r['frame_index'] == 32)
        rec, cam = records[target_id], cameras[target_id]
        if any(r['split'] != 'train' or r['camera_id'] in ['cam00', 'cam01'] for r in records):
            raise AssertionError('Training split contains excluded camera')
        named = common.named_parameters(g)
        flags = {k: getattr(h, k) for k in ['no_dx', 'no_ds', 'no_dr', 'no_do', 'no_dshs']}
        net_flags = {k: getattr(g._deformation.deformation_net.args, k) for k in flags}
        if flags != net_flags:
            raise AssertionError('Effective deformation flags differ from loaded hidden config')
        hooks, calls = [], {}
        for key in ['pos_deform', 'scales_deform', 'rotations_deform', 'opacity_deform', 'shs_deform']:
            module = getattr(g._deformation.deformation_net, key)
            def hook(_module, _inputs, output, key=key):
                calls[key] = {'calls': calls.get(key, {}).get('calls', 0) + 1,
                              'output_shape': list(output.shape),
                              'output_finite': bool(torch.isfinite(output.detach()).all()),
                              'output_abs_mean': float(output.detach().abs().mean())}
            hooks.append(module.register_forward_hook(hook))
        target = rec['image'].cuda()
        hr_w, hr_h = manifest['resolutions']['hr']
        render_cam = common.resized_camera(cam, hr_h, hr_w)
        g.optimizer.zero_grad(set_to_none=True)
        package = common.render_image(g, render_cam)
        pred = common.downsample(package['render'], target.shape[-2:])
        loss = (pred - target).abs().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite LR loss')
        loss.backward()
        torch.cuda.synchronize()
        image_grad = gradients(g, named)
        reg = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
        reg.backward()
        torch.cuda.synchronize()
        joint_grad = gradients(g, named)
        for entry in image_grad['optimizer_groups'].values():
            if not entry['all_gradients_finite']:
                raise AssertionError('Nonfinite image gradient')
        checks['capture_unchanged_after_backward'] = compare_tree(cpu['model'], g.capture())
        checks['optimizer_unchanged_after_backward'] = compare_tree(saved_opt, g.optimizer.state_dict())
        if not all(x['exact_equal'] for x in checks.values()):
            raise AssertionError('Captured model or optimizer changed without an update')
        for flag, head in [('no_dx', 'pos_deform'), ('no_do', 'opacity_deform'), ('no_dshs', 'shs_deform')]:
            if not flags[flag] and (head not in calls or image_grad['deformation_heads'][head]['nonzero_gradient_tensors'] == 0):
                raise AssertionError(f'Enabled {head} lacks forward use/nonzero LR image gradient')
        result = {'scene': scene, 'status': 'passed_model_optimizer_and_backward', 'checks': checks,
                  'rng_audit': rng_audit, 'effective_flags': flags, 'deformation_net_flags': net_flags,
                  'sh_head_exists': hasattr(g._deformation.deformation_net, 'shs_deform'),
                  'sh_head_parameters': {n: list(p.shape) for n, p in named.items() if '.shs_deform.' in n},
                  'forward_head_calls': calls, 'lr_image_only_gradients': image_grad,
                  'lr_image_plus_regularization_gradients': joint_grad,
                  'lr_l1': float(loss.detach()), 'regularization': float(reg.detach()),
                  'train_record_count': len(records), 'train_cameras': sorted({r['camera_id'] for r in records}),
                  'selected_record': {k: rec[k] for k in ['camera_id', 'frame_index', 'time', 'image_path', 'width', 'height', 'split']},
                  'selected_lr_sha256': sha(rec['image_path']), 'selected_lr_K': rec['K'].tolist(),
                  'render_K': render_cam.intrinsics.tolist(), 'render_size_wh': [hr_w, hr_h],
                  'model_time': float(cam.time), 'render_finite': bool(torch.isfinite(package['render']).all()),
                  'visible_points': int(package['visibility_filter'].sum()), 'point_count': len(g._xyz),
                  'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'max_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'elapsed_seconds': time.monotonic() - started, 'optimizer_steps': 0, 'hr_images_opened': 0}
        write(scene_out / 'gpu_runtime.json', result)
        runtime['scenes'].append({'scene': scene, 'result': str(scene_out / 'gpu_runtime.json'),
                                  'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                                  'max_allocated_bytes': result['max_allocated_bytes']})
        write(out / 'runtime.json', runtime)
        print(json.dumps({'event': 'scene_complete', 'scene': scene, 'lr_l1': result['lr_l1'],
                          'flags': flags, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        for hook in hooks:
            hook.remove()
        del package, pred, loss, reg, target, g, loaded, cpu, records, cameras, named, saved_opt
        gc.collect()
        torch.cuda.empty_cache()

    source_end = {str(p): sha(p) for p in sources}
    if source_end != source_hashes:
        raise AssertionError('Relevant source changed while runtime audit was running')
    write(out / 'gpu_after.json', gpu_snapshot())
    write(out / 'complete.json', {'status': 'passed_model_optimizer_and_backward', 'runtime': str(out / 'runtime.json'),
                                'scenes': [s[0] for s in PARENTS], 'optimizer_steps': 0,
                                'source_unchanged': True, 'checkpoint_hashes_rechecked': {str(p): sha(p) for _, p, _ in PARENTS},
                                'limitation': 'Legacy loader does not restore saved global RNG, and legacy camera samplers are not serialized. Explicit global RNG restoration passed; exact legacy sampler continuation is not claimed.',
                                'p1_or_training_permission': 'No conclusion; this script performs P0 runtime verification only.'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'output/dynamic_sr_20260923/p0_runtime_v1')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.out)
    except Exception:
        write(args.out / 'failed.json', {'status': 'failed', 'traceback': traceback.format_exc(),
                                         'optimizer_steps': 0, 'checkpoint_writes': 0})
        raise


if __name__ == '__main__':
    main()
