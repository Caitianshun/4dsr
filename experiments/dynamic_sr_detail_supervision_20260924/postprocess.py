#!/usr/bin/env python3
"""Legal B4 render -> original floating D -> frozen SwinIR reference.

No observed target-view pixels enter this API. Evaluation may read those pixels
after inference for metrics. This is an additional inference network, never F's
forward. Padding/colour/model match the frozen teacher, without PNG rounding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/dynamic_sr_20260918'


def infer_float(model, image):
    """Original tile=0, overlap=32 tensor path; CHW RGB float -> CHW float."""
    import torch
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.float32:
        raise ValueError('SwinIR post input must be float32 RGB CHW')
    if not bool(torch.isfinite(image).all()) or float(image.min()) < 0 or float(image.max()) > 1:
        raise ValueError('SwinIR post input must be finite in [0,1]')
    x = image[None]
    height, width = x.shape[-2:]
    hp, wp = (height // 8 + 1) * 8 - height, (width // 8 + 1) * 8 - width
    x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, :height + hp, :]
    x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, :width + wp]
    with torch.inference_mode():
        return model(x)[0, :, :height * 4, :width * 4].clamp(0, 1)


class FrozenPostprocess:
    def __init__(self, network=None, checkpoint=None, tile=0, overlap=32):
        import torch
        sys.path.insert(0, str(OLD / 'vendor'))
        sys.path.insert(0, str(OLD))
        spec = importlib.util.spec_from_file_location('detail_post_original_swinir', OLD / 'generate_prior.py')
        original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        from common import downsample
        if tile != 0 or overlap != 32:
            raise ValueError('This bounded post reference retains original full-image tile=0, overlap=32')
        network, checkpoint = Path(network or original.DEFAULT_NETWORK), Path(checkpoint or original.DEFAULT_CHECKPOINT)
        for path, expected in [(network, original.NETWORK_SHA256), (checkpoint, original.CHECKPOINT_SHA256)]:
            if original.sha256(path) != expected:
                raise ValueError(f'Frozen SwinIR identity changed: {path}')
        self.model = original.build_model(network, checkpoint, 'cuda')
        self.downsample, self.records = downsample, []
        self.protocol = {
            'name': 'B4_render_downsample_frozen_SwinIR', 'extra_inference_network': True,
            'input': 'Float32 HR rendering -> unchanged common.downsample bicubic antialias=True align_corners=False then clamp[0,1]; NEVER observed LR',
            'colour': 'RGB [0,1], float32; no BGR conversion, gamma transform, input/output uint8 conversion or rounding',
            'padding': 'Mirror-concatenate to next multiple of8, adding8 when already divisible; crop to4x original LR size',
            'tile': 0, 'overlap': 32, 'output': 'float32 RGB clamped[0,1], no quantization before metrics',
            'precision': 'no autocast; TF32 matmul/cudnn False, cudnn benchmark False during SwinIR only; restored afterwards',
            'model_parameter_count': sum(p.numel() for p in self.model.parameters()),
            'network_tensor_bytes': sum(p.numel() * p.element_size() for p in self.model.parameters())
                                    + sum(b.numel() * b.element_size() for b in self.model.buffers()),
            'checkpoint_file_bytes': checkpoint.stat().st_size,
            'checkpoint_sha256': original.sha256(checkpoint), 'network_sha256': original.sha256(network),
            'original_generator_sha256': original.sha256(Path(original.__file__)),
            'postprocess_script_sha256': original.sha256(Path(__file__)),
            'downsample_source_sha256': original.sha256(OLD / 'common.py'),
        }

    def __call__(self, rendered_hr, lr_size):
        import torch
        if tuple(rendered_hr.shape[-2:]) != tuple(int(x) * 4 for x in lr_size):
            raise ValueError('Post reference requires exact registered x4 dimensions')
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        low = self.downsample(rendered_hr.float(), lr_size)
        torch.cuda.synchronize()
        degraded = time.monotonic()
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            output = infer_float(self.model, low)
            torch.cuda.synchronize()
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags
        stopped = time.monotonic()
        peak = torch.cuda.max_memory_allocated()
        record = {'downsample_seconds': degraded-started, 'network_seconds': stopped-degraded,
                  'extra_seconds': stopped-started, 'allocated_baseline_bytes': baseline,
                  'peak_allocated_bytes': peak, 'additional_peak_over_loaded_model_bytes': max(0, peak-baseline)}
        self.records.append(record)
        return output, record

    def cost_summary(self):
        n = len(self.records)
        if not n:
            return None
        return {'frames': n, 'network_parameter_count': self.protocol['model_parameter_count'],
                'network_tensor_bytes': self.protocol['network_tensor_bytes'],
                'checkpoint_file_bytes': self.protocol['checkpoint_file_bytes'],
                **{key: sum(r[key] for r in self.records) for key in ['downsample_seconds', 'network_seconds', 'extra_seconds']},
                'extra_seconds_per_frame': sum(r['extra_seconds'] for r in self.records)/n,
                'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in self.records),
                'additional_peak_over_loaded_model_bytes': max(r['additional_peak_over_loaded_model_bytes'] for r in self.records),
                'timing_note': 'CUDA synchronized, includes first call, D+network only; excludes base3D render, model load, metrics, copies and I/O. Not end-to-end FPS.',
                'memory_note': 'Absolute process allocation includes3D and SR models plus evaluator tensors. Increment is temporary allocation above already-loaded models, NOT total SR-network memory; network tensor bytes separately reported.'}


if __name__ == '__main__':
    # Only the entry selects this mode. The same evaluator computes all metrics.
    from evaluate import main
    main([*sys.argv[1:], '--postrender-sr'])
