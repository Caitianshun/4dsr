"""SR-stage module kit, part 5: monocular depth prior (Claude, session 4dsr-e9, 2026-10-17). New file; modkit4.py (hash pinned
below) and everything it imports are imported, never modified.

  modkit5.py [--depth-prior W] [--depth-patch P] <modkit4 / modkit2 / modkit options> -- --mode fix cullretrain sr <SR args>

User request (10-17 11:20 PDT): constrain depth with a depth prior or another method (multi-view fusion failed on the 4DGS
depth misregistration). CSC-G (modkit's csc entry) constrains depth through cross-view self-consistency of the rendered depth;
this adds an external prior. 10-06 tested an ordinal (ranking) loss with the same network under the old bicubic protocol
(slightly worse); this is the correlation form used by FSGS / SparseGS, under the area4 recipe and on top of self-calibration.

--depth-prior W : Depth Anything V2 Small (official code a561b849, weights sha256 715fade1..., copies under
                  output/dynamic_sr_modules_20261010/geoprobe_20261017/third_party/) predicts relative inverse depth on the teacher
                  image of each teacher view (views b and c of every SR step). Frozen network, inference only; the image is
                  resized (bicubic, antialiased) so that the short side is 518 px and both sides are multiples of 14 (the
                  official 'lower_bound' rule; the official code resizes with cv2 without antialiasing), ImageNet normalisation.
                  The rendered depth of the same view (rasteriser depth, alpha-blended camera z) becomes inverse depth (z clamped
                  to >= 0.05 x its median), area-resized to the network grid. Both maps are divided by their image-wide std,
                  then compared patch-wise: L = mean over P x P patches (default 32) of 1 - Pearson(render, prior), skipping
                  patches whose prior std is below 0.05 (in units of the image-wide std). Pearson is invariant to scale and
                  shift on both sides (relative depth). The term enters the objective with weight W per teacher view: the SR
                  loss value of the view is raised by W / (lambda_sr (1 - dssim)) L, so that after the registered lambda_sr /
                  D-SSIM mixing the objective gains exactly W L. Its gradient reaches positions, scales, rotations, opacities
                  and the deformation network through the rasteriser's depth backward. The rendered depth is not passed through
                  the calibsr warp (at most ~0.6 HR px, < 0.3 px on the 518 grid). Evaluation renders are unchanged.
                  cullretrain entry only (the csc entry renders extra views per step).
modkit5.json in --out records the switches, the mean prior loss and valid-patch share over the first and last 500 steps, the
mean prior time per call, and source / weight hashes.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
for _p in (MOD, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import modkit4  # noqa: E402

PINNED = {'modkit4.py': 'b344093040e5f857'}
THIRD = ROOT / 'output/dynamic_sr_modules_20261010/geoprobe_20261017/third_party'
DA_DIR = THIRD / 'Depth-Anything-V2'
DA_WEIGHT = THIRD / 'depth_anything_v2_vits.pth'
DA_SHA = '715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
MIN_STD = 0.05


def load_prior(device):
    import torch
    try:
        import cv2  # noqa: F401  (only the unused official transforms need it)
    except ImportError:
        stub = types.ModuleType('cv2')
        stub.__getattr__ = lambda name: 0
        sys.modules['cv2'] = stub
    if str(DA_DIR) not in sys.path:
        sys.path.insert(0, str(DA_DIR))
    from depth_anything_v2.dpt import DepthAnythingV2
    if modkit4.modkit2.modkit.calibsr_v2.sha(DA_WEIGHT) != DA_SHA:
        raise SystemExit('Depth Anything V2 weights changed')
    net = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    net.load_state_dict(torch.load(DA_WEIGHT, map_location='cpu', weights_only=True))
    return net.to(device).eval()


def network_grid(h, w, size=518, multiple=14):
    """official Resize(keep_aspect_ratio, ensure_multiple_of=14, resize_method='lower_bound')"""
    import math
    s = max(size / h, size / w)

    def constrain(x, lo):
        y = round(x / multiple) * multiple
        return int(math.ceil(x / multiple) * multiple) if y < lo else int(y)
    return constrain(s * h, size), constrain(s * w, size)


def prior_inverse_depth(net, image):
    """image 3 x H x W in [0, 1] -> relative inverse depth on the network grid (no grad)"""
    import torch
    import torch.nn.functional as F
    nh, nw = network_grid(*image.shape[-2:])
    x = F.interpolate(image[None].float(), size=(nh, nw), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    with torch.no_grad():
        return net((x - mean) / std)[0].float()


def render_inverse_depth(depth, grid):
    import torch.nn.functional as F
    z = depth.reshape(depth.shape[-2:]).float()
    inv = 1.0 / z.clamp_min(0.05 * float(z.detach().median().clamp_min(1e-6)))
    return F.interpolate(inv[None, None], size=grid, mode='area')[0, 0]


def patch_pearson_loss(render, prior, patch):
    """mean over patch x patch blocks of 1 - Pearson(render, prior); both maps scaled by their image-wide std first"""
    import torch
    h, w = render.shape
    h2, w2 = h // patch * patch, w // patch * patch
    a = render[:h2, :w2] / (render.detach().std() + 1e-8)
    b = prior[:h2, :w2] / (prior.std() + 1e-8)

    def blocks(x):
        return x.reshape(h2 // patch, patch, w2 // patch, patch).permute(0, 2, 1, 3).reshape(-1, patch * patch)
    a, b = blocks(a), blocks(b)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    sa, sb = a.pow(2).mean(1).add(1e-6).sqrt(), b.pow(2).mean(1).sqrt()
    valid = sb >= MIN_STD
    if not bool(valid.any()):
        return render.sum() * 0, 0.0
    corr = (a * b).mean(1) / (sa * sb + 1e-6)
    return (1 - corr[valid]).mean(), float(valid.float().mean())


def install(native, losses, weight, patch, lam, dssim, stats):
    """wrap native.native_rgb (keep the depth of this step's grad-enabled renders) and losses.sr_loss (add the prior term).
    cullretrain's SR step renders views a, b, c, then calls sr_loss for b and c; any other pattern disables the prior."""
    import torch
    original_rgb, original_sr = native.native_rgb, losses.sr_loss
    factor = weight / (lam * (1 - dssim))
    buf = dict(depths=[], used=0)
    holder = {}

    def native_rgb(camera, g, st, pipe, background, raster_api=None):
        packet = original_rgb(camera, g, st, pipe, background, raster_api)
        if stats['enabled'] and torch.is_grad_enabled() and st['opacity'].requires_grad:
            if buf['used'] or len(buf['depths']) >= 3:
                if len(buf['depths']) >= 3 and buf['used'] < 2:
                    stats['enabled'] = False
                    stats['error'] = f"render pattern: {len(buf['depths'])} renders, {buf['used']} sr_loss calls before a new render"
                buf['depths'], buf['used'], buf['vs'] = [], 0, []
            buf['depths'].append(packet['depth'])
            buf.setdefault('vs', []).append(packet.get('viewspace_points'))
            holder['g'] = g
        return packet

    def sr_loss(rgb, teacher):
        value = original_sr(rgb, teacher)
        if not (stats['enabled'] and torch.is_grad_enabled() and rgb.requires_grad):
            return value
        if len(buf['depths']) != 3 or buf['used'] >= 2:
            stats['enabled'] = False
            stats['error'] = f"sr_loss with {len(buf['depths'])} renders and {buf['used']} earlier calls"
            return value
        depth = buf['depths'][1 + buf['used']]
        buf['used'] += 1
        if 'net' not in holder:
            holder['net'] = load_prior(rgb.device)
        tick = time.time()
        prior = prior_inverse_depth(holder['net'], teacher.detach())
        stats['prior_seconds'] += time.time() - tick
        loss, share = patch_pearson_loss(render_inverse_depth(depth, prior.shape[-2:]), prior, patch)
        if stats['calls'] == 0:
            # one-time check that the rasteriser's depth backward reaches the Gaussian centres; autograd.grad also fills the
            # .grad of retain_grad screen-space points (densification statistics), so those are reset before the real backward
            if not loss.requires_grad:
                stats['enabled'] = False
                stats['error'] = 'the rendered depth carries no gradient'
                return value
            grad = torch.autograd.grad(loss, [holder['g']._xyz], retain_graph=True, allow_unused=True)[0]
            stats['first_call_xyz_grad_norm'] = None if grad is None else float(grad.norm())
            for v in buf.get('vs', []):
                if v is not None:
                    v.grad = None
        stats['calls'] += 1
        stats['rows'].append((float(loss.detach()), share))
        return value + factor * loss

    native.native_rgb = native_rgb
    losses.sr_loss = sr_loss


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    for name, digest in PINNED.items():
        if not modkit4.modkit2.modkit.calibsr_v2.sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(depth_prior=0.0, depth_patch=32)
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] == '--depth-prior':
            s['depth_prior'] = float(own[i + 1]); i += 2
        elif own[i] == '--depth-patch':
            s['depth_patch'] = int(own[i + 1]); i += 2
        else:
            passthrough.append(own[i]); i += 1
    if not 0 <= s['depth_prior'] <= 1 or not 8 <= s['depth_patch'] <= 128:
        raise SystemExit('switch out of range')
    if len(rest) < 4 or rest[0] != '--mode' or rest[3] != 'sr':
        raise SystemExit('expected -- --mode fix {cullretrain|csc} sr ...')
    if s['depth_prior'] and rest[2] != 'cullretrain':
        raise SystemExit('--depth-prior needs the cullretrain entry')
    sr_args = rest[3:]
    lam = float(sr_args[sr_args.index('--lambda-sr') + 1]) if '--lambda-sr' in sr_args else 0.05
    dssim = float(sr_args[sr_args.index('--dssim') + 1]) if '--dssim' in sr_args else 0.0
    if s['depth_prior'] and (lam <= 0 or dssim >= 1):
        raise SystemExit('--depth-prior needs lambda_sr > 0 and dssim < 1')
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402
    cr = srfix.cr
    losses = sys.modules['losses']
    stats = dict(enabled=bool(s['depth_prior']), calls=0, prior_seconds=0.0, rows=[], error=None, first_call_xyz_grad_norm=None)
    real_main = srfix.main

    def depth_main():
        """called innermost by the modkit4 / modkit2 / modkit wrappers, after modkit has installed its render hooks"""
        if s['depth_prior']:
            install(cr.native, losses, s['depth_prior'], s['depth_patch'], lam, dssim, stats)
        return real_main()

    srfix.main = depth_main
    sys.argv = [str(HERE / 'modkit4.py')] + passthrough + ['--'] + rest
    modkit4.main()
    rows = stats.pop('rows')

    def window(part):
        return dict(loss=sum(r[0] for r in part) / len(part), valid_share=sum(r[1] for r in part) / len(part)) if part else None
    record = dict(switches=s, lambda_sr=lam, dssim=dssim, passthrough=passthrough, argv=rest, stats=stats,
                  first_1000_calls=window(rows[:1000]), last_1000_calls=window(rows[-1000:]),
                  prior_seconds_per_call=stats['prior_seconds'] / max(stats['calls'], 1),
                  weights=dict(path=str(DA_WEIGHT.relative_to(ROOT)), sha256=DA_SHA),
                  sources={str(Path(f).resolve().relative_to(ROOT)): modkit4.modkit2.modkit.calibsr_v2.sha(f)
                           for f in (__file__, modkit4.__file__, DA_DIR / 'depth_anything_v2' / 'dpt.py')})
    (out / 'modkit5.json').write_text(json.dumps(record, indent=1))
    if stats['error']:
        print(json.dumps(dict(modkit5_error=stats['error'])), flush=True)
    print(json.dumps(dict(modkit5=s, calls=stats['calls']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
