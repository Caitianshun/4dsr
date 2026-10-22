"""Offline PSNR/SSIM/LPIPS GPU acceptance on complete training LR images.

Default is CPU planning, without importing torch. --run requires one explicitly
admitted GPU. No Wu model is constructed, no model parameters are updated, and no
HR or held-out pixels are opened. This is metric wiring acceptance, not a score
for any reconstruction method. Original LR dimensions are retained without crop.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from unittest import mock

from common import ROOT, checked, identity, read, resolve, sha, training_manifest
from prepare_suite import SCENES, DEFAULT_PREPARED
from gpu_acceptance import save_receipt

EVALUATOR = ROOT / 'third_party/static_sr_eval_20261022'
CACHE = ROOT / 'weight/static_sr_wu_20261022/evaluation/torch/hub'
WU_RUNTIME = ROOT / 'third_party/4dgs_843d5ac_runtime'
SOURCE_SHA = {
    'lpipsPyTorch/__init__.py': 'a657ce7355782eb970f95fd0d2e26eec5b9023212d2b655814da6e15b70510b4',
    'lpipsPyTorch/modules/lpips.py': '3bece0b9cf9943af5b043026458819d259df9bfe7c2a1d3ffc6c905e7e5aa2b4',
    'lpipsPyTorch/modules/networks.py': 'dfa6f152b0e3fbc23ac3bfaea52b46e9473e64ad539ea22ab030c57af51c7f14',
    'lpipsPyTorch/modules/utils.py': '1bd4a7d4e7b43215497675ed852936fe4f27e7ea3068afe0b0e1f7cfb6c48570',
}
WEIGHT_SHA = {
    'alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'alex.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
}
SSIM_SHA = '980f64ffa391d9b673a9412d0f4b38ff6b0aac3618e237b079f8a5107c33810f'
UPSTREAM_COMMIT = '843d5ac636c37e4b611242287754f3d4ed150144'
PARITY_ATOL = 1e-6


def verify_assets(evaluator=EVALUATOR, cache=CACHE, runtime=WU_RUNTIME):
    evaluator, cache, runtime = map(resolve, (evaluator, cache, runtime))
    source_receipt = evaluator / 'source_receipt.json'
    receipt = read(source_receipt)
    if (receipt.get('schema') != 'static_sr_lpips_evaluator_assets_v1'
            or receipt['upstream']['git_commit'] != UPSTREAM_COMMIT
            or receipt['network_type'] != 'alex' or receipt['version'] != '0.1'):
        raise ValueError('LPIPS source registration differs from pinned Wu evaluator')
    sources = [{'path': str(evaluator / name), 'sha256': digest} for name, digest in SOURCE_SHA.items()]
    weights = [{'path': str(cache / 'checkpoints' / name), 'sha256': digest} for name, digest in WEIGHT_SHA.items()]
    ssim = {'path': str(runtime / 'utils/loss_utils.py'), 'sha256': SSIM_SHA}
    for entry in sources + weights + [ssim]:
        checked(entry)
    return dict(source_receipt=identity(source_receipt), upstream_commit=UPSTREAM_COMMIT,
        implementations=sources, model_identities=weights, ssim_implementation=ssim,
        evaluator_root=str(evaluator), torch_hub_cache=str(cache), wu_runtime_root=str(runtime))


def select_pair(manifest, rows, frame, cameras=('cam01', 'cam02')):
    if len(cameras) != 2 or len(set(cameras)) != 2 or 'cam00' in cameras:
        raise ValueError('Require two distinct training cameras, excluding cam00')
    if frame not in manifest['frame_indices'] or any(c not in manifest['splits']['train'] for c in cameras):
        raise ValueError('Camera/frame absent from registered training split')
    selected = []
    for camera in cameras:
        found = [r for r in rows if r['camera_id'] == camera and r['frame_index'] == frame and r['split'] == 'train']
        if len(found) != 1 or found[0]['time'] != frame / 300:
            raise ValueError('Metric fixture requires one original frame/300 observation per camera')
        selected.append(found[0])
    return selected


def make_plan(prepared, scenes, frame, cameras):
    if len(set(scenes)) != len(scenes) or set(scenes) - dict(SCENES).keys():
        raise ValueError('Use unique names from the current six-scene suite')
    assets = verify_assets()
    samples = []
    for short in scenes:
        config = Path(prepared) / short / 'iesrgs.json'
        cfg = read(config)
        manifest, rows = training_manifest(checked(cfg['manifest']), verify_pixels=False)
        if manifest['scene'] != dict(SCENES)[short] or cfg['scene'] != manifest['scene']:
            raise ValueError('Scene identity differs')
        pair = []
        for row in select_pair(manifest, rows, frame, cameras):
            lr = Path(manifest['_root']) / row['lr_path']
            if sha(lr) != row['lr_sha256']:
                raise ValueError('Metric fixture LR bytes changed')
            pair.append(dict(camera=row['camera_id'], frame=frame, time=row['time'], lr=identity(lr)))
        samples.append(dict(short=short, scene=manifest['scene'], configuration=identity(config),
            manifest=cfg['manifest'], original_lr_wh=manifest['resolutions']['lr'], pair=pair))
    if not (any(s['scene'].startswith('meetroom_') for s in samples)
            and any(not s['scene'].startswith('meetroom_') for s in samples)):
        raise ValueError('Metric acceptance must cover N3DV and MeetRoom full LR dimensions')
    return dict(schema='static_sr_wu_metric_GPU_acceptance_v1', status='pending_GPU_execution',
        source=identity(__file__), evaluator_source=identity(ROOT / 'experiments/static_sr_wu_20261022/render_eval.py'),
        assets=assets, samples=samples, GPU_executed=False, parameter_updates=0,
        Wu_model_constructed=False, training_HR_read=False, heldout_pixels_read=False,
        downloaded_assets=False, metric_input='PNG uint8 decoded to float32 CHW / 255, unchanged [0,1] Wu convention',
        image_size='Full original training LR; no crop, resize, or synthetic HR',
        pairs='Each sample: cam01 self-comparison and cam01 versus cam02 at the same original timestamp',
        thresholds=dict(self_PSNR=120., self_PSNR_atol=1e-5, self_SSIM=1., self_SSIM_atol=1e-6,
            self_LPIPS=0., self_LPIPS_atol=1e-8, convenience_vs_reused_class_absolute_atol=PARITY_ATOL,
            convenience_vs_reused_class_relative_atol=0.),
        validation_scope='GPU metric execution and old/new LPIPS API parity only; no reconstruction-quality or training claim')


def require_metric_contract(self_metrics, different_metrics, class_value, function_value):
    values = list(self_metrics.values()) + list(different_metrics.values()) + [class_value, function_value]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Metric output is non-finite')
    if abs(self_metrics['PSNR'] - 120.) > 1e-5 or abs(self_metrics['SSIM'] - 1.) > 1e-6 or abs(self_metrics['LPIPS']) > 1e-8:
        raise ValueError('Fixed self-comparison metric contract failed')
    if not (0 <= different_metrics['PSNR'] < 120. and -1. <= different_metrics['SSIM'] <= 1. + 1e-6
            and different_metrics['LPIPS'] > 1e-8):
        raise ValueError('Distinct-image metric contract failed')
    delta = abs(class_value - function_value)
    if delta > PARITY_ATOL:
        raise ValueError('Convenience versus reused LPIPS class differs by more than strict absolute 1e-6')
    return dict(passed=True, absolute_difference=delta, absolute_tolerance=PARITY_ATOL, relative_tolerance=0.)


@contextlib.contextmanager
def forbid_downloads(torch):
    """A missing cache is a failure; neither constructor may fetch a replacement."""
    def reject(*args, **kwargs):
        raise RuntimeError('Network/download attempted during offline metric acceptance')
    with mock.patch.object(torch.hub, 'download_url_to_file', reject), \
            mock.patch('urllib.request.urlopen', reject), mock.patch('urllib.request.urlretrieve', reject):
        yield


def run(plan, output):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        raise ValueError('Explicit one-GPU resource admission required')
    import numpy as np
    import torch
    from PIL import Image
    from render_eval import write_rgb
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one admitted GPU must be visible')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20261007)
    assets = plan['assets']
    for item in assets['implementations'] + assets['model_identities'] + [assets['ssim_implementation']]:
        checked(item)
    sys.path[:0] = [assets['evaluator_root'], assets['wu_runtime_root']]
    lpips_module = importlib.import_module('lpipsPyTorch')
    loss_module = importlib.import_module('utils.loss_utils')
    for module, file in [(lpips_module, Path(assets['evaluator_root']) / 'lpipsPyTorch/__init__.py'),
                         (loss_module, Path(assets['wu_runtime_root']) / 'utils/loss_utils.py')]:
        if Path(module.__file__).resolve() != file.resolve():
            raise RuntimeError('An earlier import shadowed the pinned metric module')
    for name in ('lpips', 'networks', 'utils'):
        module = importlib.import_module('lpipsPyTorch.modules.' + name)
        if Path(module.__file__).resolve() != Path(assets['evaluator_root']) / 'lpipsPyTorch/modules' / (name + '.py'):
            raise RuntimeError('LPIPS dependency submodule was shadowed')
    torch.hub.set_dir(assets['torch_hub_cache'])
    begun = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    plan.update(status='running', GPU_executed=True, results=[], host=socket.gethostname(),
        started_utc=datetime.now(timezone.utc).isoformat(), visible_gpu=visible,
        GPU_name=torch.cuda.get_device_name(), GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
        torchvision=importlib.import_module('torchvision').__version__,
        math_settings=dict(cuda_matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False),
        imported_lpips=identity(lpips_module.__file__), imported_ssim=identity(loss_module.__file__))
    save_receipt(output, plan)
    with forbid_downloads(torch), torch.no_grad():
        loading = time.perf_counter()
        criterion = lpips_module.LPIPS(net_type='alex', version='0.1').to('cuda').eval()
        torch.cuda.synchronize()
        plan['reused_class_constructions'] = 1
        plan['constructor_seconds'] = time.perf_counter() - loading
        if any(p.requires_grad for p in criterion.parameters()):
            raise RuntimeError('Pinned LPIPS parameters should all be frozen')
        before = {name: value.detach().cpu().clone() for name, value in criterion.state_dict().items()}
        for sample in plan['samples']:
            sample_start = time.perf_counter()
            arrays, tensors, pngs = [], [], []
            for pair in sample['pair']:
                path = checked(pair['lr'])
                with Image.open(path) as image:
                    if list(image.size) != sample['original_lr_wh']:
                        raise ValueError('Original full LR size changed')
                    array = np.array(image.convert('RGB'), copy=True)
                tensor = torch.from_numpy(array).permute(2, 0, 1).float().cuda() / 255
                png = output / 'NONFORMAL_training_LR_pngs' / sample['short'] / (pair['camera'] + '.png')
                write_rgb(png, tensor)
                with Image.open(png) as image:
                    decoded = np.array(image.convert('RGB'), copy=True)
                    if image.format != 'PNG' or list(image.size) != sample['original_lr_wh']:
                        raise ValueError('PNG full-size serialization failed')
                if not np.array_equal(decoded, array):
                    raise ValueError('PNG roundtrip changed original uint8 RGB values')
                recovered = torch.from_numpy(decoded).permute(2, 0, 1).float().cuda() / 255
                if not torch.equal(tensor, recovered):
                    raise ValueError('PNG float32 decode roundtrip changed metric input')
                arrays.append(array); tensors.append(recovered)
                pngs.append(dict(camera=pair['camera'], path=identity(png), size_wh=sample['original_lr_wh'],
                    uint8_pixels_equal=True, float32_tensor_equal=True))
            if np.array_equal(*arrays):
                raise ValueError('The two real camera observations are identical; not a non-self fixture')
            first, second = tensors
            def metrics(a, b):
                return dict(PSNR=float(-10 * torch.log10((a-b).square().mean().clamp_min(1e-12))),
                    SSIM=float(loss_module.ssim(a, b)), LPIPS=float(criterion(a[None], b[None]).mean()))
            self_metrics, different_metrics = metrics(first, first), metrics(first, second)
            convenience = float(lpips_module.lpips(first[None], second[None], net_type='alex', version='0.1').mean())
            parity = require_metric_contract(self_metrics, different_metrics, different_metrics['LPIPS'], convenience)
            torch.cuda.synchronize()
            plan['results'].append(dict(short=sample['short'], frame=sample['pair'][0]['frame'],
                time=sample['pair'][0]['time'], PNG_roundtrips=pngs, self_comparison=self_metrics,
                distinct_camera_comparison=different_metrics, convenience_LPIPS=convenience,
                convenience_class_parity=parity, elapsed_seconds=time.perf_counter()-sample_start,
                distinct_real_inputs=True, status='passed'))
            save_receipt(output, plan)
        for name, value in criterion.state_dict().items():
            if not torch.equal(value.detach().cpu(), before[name]):
                raise RuntimeError('Metric execution changed frozen parameters/buffers')
        for item in assets['implementations'] + assets['model_identities'] + [assets['ssim_implementation']]:
            checked(item)
    plan.update(status='passed_GPU_metrics_and_LPIPS_API_parity_only',
        frozen_parameters_and_buffers_unchanged=True, convenience_constructions=len(plan['samples']),
        elapsed_seconds=time.perf_counter()-begun, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), finished_utc=datetime.now(timezone.utc).isoformat())
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--scenes', nargs='+', choices=[s for s, _ in SCENES], default=['Cook07', 'Discussion07'])
    parser.add_argument('--frame', type=int, default=149)
    parser.add_argument('--cameras', nargs=2, default=['cam01', 'cam02'])
    parser.add_argument('--out', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true'); mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = make_plan(resolve(args.prepared), args.scenes, args.frame, args.cameras)
    output = resolve(args.out); output.mkdir(parents=True, exist_ok=False)
    save_receipt(output, plan)
    if args.run:
        try:
            run(plan, output)
        except BaseException:
            plan.update(status='failed', error=traceback.format_exc())
            save_receipt(output, plan)
            raise
    else:
        plan['torch_imported'] = 'torch' in sys.modules
    save_receipt(output, plan)
    print({'status': plan['status'], 'receipt': identity(output / 'receipt.json')})


if __name__ == '__main__':
    main()
