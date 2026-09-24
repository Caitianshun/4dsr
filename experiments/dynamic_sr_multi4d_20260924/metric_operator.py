"""Identical RGB residual operator without importing the Wu renderer/environment."""
import torch.nn.functional as F

OPERATOR = {'definition':'H(X)=X-Up(D(X)); signed, no residual clipping',
    'downsample':'bicubic, align_corners=False, antialias=True, then clamp[0,1]',
    'upsample':'bicubic, align_corners=False, antialias=False, no clamp',
    'implementation_scope':'Same operations as detail_supervision_20260924/detail_loss.py; detached from Wu imports'}


def highpass(image,lr_size):
    low=F.interpolate(image[None],size=lr_size,mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
    return image-F.interpolate(low[None],size=image.shape[-2:],mode='bicubic',align_corners=False,antialias=False)[0]
