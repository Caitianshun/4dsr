"""The registered L1 teacher, LR objective competitor and six-edge X loss.

No masks or soft weights are learned. Empty edges keep their zero contribution
in the fixed six-edge denominator. RGB channels enter the inner denominator.
"""
from __future__ import annotations
import torch

CHARBONNIER_DELTA = .01
EPSILON = 1e-8
DIRECTED_EDGES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def plain(value):
    """Persist every computed diagnostic rather than dropping tensor fields."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [plain(item) for item in value]
    return value


def degradation(rgb, size):
    import torch.nn.functional as F
    return F.interpolate(rgb[None], size=tuple(size), mode='bicubic',
                         align_corners=False, antialias=True)[0].clamp(0, 1)


def lr_loss(rgb, lr):
    return (degradation(rgb, lr.shape[-2:]) - lr).abs().mean()


def sr_loss(rgb, teacher):
    return (rgb - teacher).abs().mean()


def e_loss(rgb, lr, kappa):
    error = degradation(rgb, lr.shape[-2:]) - lr
    return .8 * error.abs().mean() + .2 * float(kappa) * error.square().mean()


def charbonnier(error, delta=CHARBONNIER_DELTA):
    return torch.sqrt(error.square() + float(delta)**2) - float(delta)


def edge_loss(prediction, lr, valid_lr, weight_lr, delta=CHARBONNIER_DELTA):
    """All arguments are finite before weighted reduction; no NaN*0 hiding."""
    if prediction.shape != lr.shape or prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError('X requires matched RGB CHW images')
    valid = valid_lr.detach().bool()
    weight = weight_lr.detach().to(device=prediction.device, dtype=prediction.dtype)
    if valid.shape != prediction.shape[-2:] or weight.shape != valid.shape:
        raise ValueError('X mask and frozen weight must have target LR shape')
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(lr).all()):
        raise ValueError('Invalid RGB reached X before reduction')
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError('Frozen X weights must be finite and nonnegative')
    weighted = weight * valid.to(weight.dtype)
    mass = weighted.sum()
    count = int(valid.sum().item())
    if count == 0 or float(mass) == 0:
        return prediction.sum() * 0., dict(empty=True, valid_pixels=0, weight_mass=0.,
            effective_area_fraction=0., weighted_mean=None, weighted_quantiles=None,
            residual_l1=None, residual_mse=None)
    error = prediction - lr
    value = (charbonnier(error, delta) * weighted[None]).sum() / (3 * mass + EPSILON)
    selected = weight[valid]
    quantiles = torch.quantile(selected.float(), selected.new_tensor([.25, .5, .75]).float())
    stats = dict(empty=False, valid_pixels=count, weight_mass=float(mass.detach()),
        effective_area_fraction=count / valid.numel(), weighted_mean=float(selected.mean()),
        weighted_quantiles=[float(x) for x in quantiles],
        residual_l1=float((error.detach().abs()*weighted[None]).sum()/(3*mass)),
        residual_mse=float((error.detach().square()*weighted[None]).sum()/(3*mass)))
    return value, stats


def x_loss(rgbs, moments, cameras, lrs, keys, support, footprint=None):
    """Three renders and three moments are reused by all six directed edges.

    moments may be render_moments dictionaries or raw HR (3,H,W) tensors.
    X keeps source RGB and target HR axial-depth paths differentiable.
    """
    if not all(len(x) == 3 for x in (rgbs, moments, cameras, lrs, keys)):
        raise ValueError('Exactly three registered observations are required')
    if len({key[0] for key in keys}) != 3 or len({int(key[1]) for key in keys}) != 1:
        raise ValueError('X requires distinct cameras at the same original frame')
    if footprint is None:
        import footprint as footprint
    from support_cache import normalize_hr_moments
    normalized = []
    for value in moments:
        raw = value['hr_moments'] if isinstance(value, dict) else value
        if raw.ndim == 2:
            valid = (torch.isfinite(raw.detach()) & (raw.detach() > 0)).detach()
            normalized.append(dict(z=torch.where(valid, raw, torch.ones_like(raw)), valid_hr=valid))
        else:
            normalized.append(normalize_hr_moments(raw))
    depths = [value['z'] for value in normalized]
    source_support = [(support.observation(*key, device=rgbs[0].device)['valid_hr'] & value['valid_hr']).detach()
                      for key, value in zip(keys, normalized)]
    # This is rebuilt from current RGB on every step and retains autograd.
    pyramids = [footprint.build_area_mipmap(rgb, valid) for rgb, valid in zip(rgbs, source_support)]
    values, diagnostics = [], []
    for i, j in DIRECTED_EDGES:
        frozen = support.edge(keys[i], keys[j], device=rgbs[i].device)
        result = footprint.edge_prediction(rgbs[i], depths[j], cameras[i], cameras[j],
            lrs[j].shape[-2:], frozen_mask_hr=(frozen['mask_hr'] & normalized[j]['valid_hr']).detach(),
            source_valid_hr=source_support[i], pyramid=pyramids[i])
        value, stats = edge_loss(result['prediction'], lrs[j], result['valid_lr'], frozen['weight_lr'])
        stats.update(source=list(keys[i]), target=list(keys[j]), footprint=plain(result['diagnostics']))
        stats['frozen_valid_area_fraction'] = float(frozen['frozen_valid_lr_fraction'])
        stats['support_decrease_fraction'] = max(0., stats['frozen_valid_area_fraction'] - stats['effective_area_fraction'])
        values.append(value); diagnostics.append(stats)
    total = torch.stack(values).sum() / 6
    return total, dict(edges=diagnostics, registered_edges=6,
        effective_edges=sum(not row['empty'] for row in diagnostics),
        empty_edges=sum(row['empty'] for row in diagnostics),
        effective_area_fraction=sum(row['effective_area_fraction'] for row in diagnostics)/6,
        support_decrease_fraction=sum(row['support_decrease_fraction'] for row in diagnostics)/6,
        loss=float(total.detach()), delta=CHARBONNIER_DELTA, outer_denominator=6)


def summed_xyz_gradient_rms(loss, xyz_tensors, retain_graph=True):
    """Sum same-Gaussian-index derivatives BEFORE computing the RMS.

    Three same-time effective states can share one tensor or be separate
    tensors. A shared tensor is queried once; three independent states have
    their N-by-3 gradients summed, never concatenated into 3N-by-3.
    """
    tensors, ids = [], set()
    for xyz in xyz_tensors:
        if id(xyz) not in ids:
            ids.add(id(xyz)); tensors.append(xyz)
    if not tensors or any(x.shape != tensors[0].shape for x in tensors):
        raise ValueError('All effective_xyz tensors must use the same Gaussian indices')
    gradients = torch.autograd.grad(loss, tensors, retain_graph=retain_graph, allow_unused=True)
    summed = torch.zeros_like(tensors[0])
    for gradient in gradients:
        if gradient is not None:
            summed = summed + gradient
    if not bool(torch.isfinite(summed).all()):
        raise ValueError('Nonfinite effective_xyz gradient')
    return float(summed.detach().double().square().mean().sqrt()), summed
