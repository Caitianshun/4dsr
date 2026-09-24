"""Fixed signed residual loss; exactly the current LR closure downsampler.

The downsampler includes its existing clamp. H is therefore piecewise linear,
not a strict linear high-pass filter or an observation null-space projector.
"""
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
from common import downsample

OPERATOR = {
    'definition': 'H(X)=X-Up(D(X)); signed, no residual clipping',
    'downsample': 'existing common.downsample: bicubic, align_corners=False, antialias=True, then clamp[0,1]',
    'upsample': 'bicubic to original HR size, align_corners=False, antialias=False, no clamp',
    'boundary': 'PyTorch interpolate boundary handling; no custom padding/crop',
    'gradient': 'complete autograd through D and Up, including existing clamp derivative',
    'interpretation': 'signed downsampling residual; nonlinear at D saturation, not strict high-pass or null-space',
    'teacher_H': 'computed online under no_grad from the same frozen uint8 PNG teacher; no separate H cache',
}


def highpass(image, lr_size):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'Expected CHW RGB, got {tuple(image.shape)}')
    low = downsample(image, tuple(lr_size))
    up = F.interpolate(low[None], size=image.shape[-2:], mode='bicubic',
                       align_corners=False, antialias=False)[0]
    return image - up


def teacher_loss(prediction, target, method, alpha, lr_size):
    """F(alpha=0) takes the identical U branch, with no zero-weight graph."""
    if target.requires_grad:
        raise ValueError('Teacher must be frozen')
    rgb = (prediction - target).abs().mean()
    weight = .2 if method == 'W' else .1
    if method not in ('U', 'W', 'F'):
        raise ValueError(method)
    if method != 'F' or alpha == 0:
        return weight * rgb, rgb, None
    with torch.no_grad():
        ht = highpass(target, lr_size)
    detail = (highpass(prediction, lr_size) - ht).abs().mean()
    return weight * rgb + alpha * detail, rgb, detail
