"""SR stage with geometry-shaping switches (Claude, session 4dsr-e9, 2026-10-17). New file; calibsr_v2.py (byte-frozen,
sha256 900c3f63...), srfix.py, cullretrain.py and losses.py are imported, never modified.

  geosr.py --lr-operator {bicaa,area} [--calib {none,shift,full}] [--warp-lr 2e-3] [--teacher-zeros H W]
           [--densify-scale F] [--xyz-lr-scale F] [--deform-lr-scale F] [--teacher-lowpass SIGMA] -- <srfix.py args>

Question (user, 10-17 07:50 PDT): about three quarters of the matched-teacher -> HR-teacher gap on Cook07 area4 is geometry
(frozen-geometry re-projection, A100). Can the SR stage let the teacher and the multi-view LR shape the geometry better?
In the registered SR stage (cullretrain.run_sr) the scheduled learning rates continue the LR-stage schedule
(update_learning_rate(14000 + cursor), position_lr_max_steps 20000), so they sit at 1-4x their final values: positions
6.4e-6 -> 1.6e-6, deformation MLP 3.2e-5 -> 1.6e-5, HexPlane grid 3.2e-4 -> 1.6e-4 (times spatial_lr_scale). Densification
runs at the fixed gradient threshold 2e-4 every 100 steps. The HR-teacher model ends with 7% more points than the
matched-teacher model (178k vs 166k).

--densify-scale F   : densify_grad_threshold_fine_init and _after times F (F < 1: more clone/split).
--xyz-lr-scale F    : position_lr_init and position_lr_final times F (the whole SR-stage position schedule times F).
--deform-lr-scale F : deformation_lr_init/_final and grid_lr_init/_final times F.
  These three wrap cullretrain.configuration, which builds fresh namespaces on every call; the plan dictionaries and the
  registry checks are untouched. GaussianModel.restore calls training_setup with the patched values, and
  update_learning_rate sets the scheduled groups from them at every step.
--teacher-lowpass S : the teacher terms compare Gaussian-blurred (sigma S HR px, separable, reflect padding) render and
  teacher, so the teacher no longer constrains the top band (amplitude at r = 0.25 cycles/px is 0.29 for S = 1, 0.73 at
  r = 0.125). losses.sr_loss (used only for the teacher views) is wrapped; the D-SSIM hook wraps utils.loss_utils.ssim (run_sr
  imports it at call time) and blurs only inputs whose larger side exceeds 700 px, i.e. the HR-size teacher views, never the
  LR-size closure term.
--lr-operator, --calib, --warp-lr, --teacher-zeros: as calibsr_v2.py (exact 4x4 box LR operator; learned per-camera warp).
geosr.json in --out records the switches and source hashes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
for _p in (MOD, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import calibsr_v2  # noqa: E402

CALIBSR_V2_SHA = '900c3f6305e92718a1643c4fe9f80baf5b8c4d0b15e62c0cbecbbcb19f2130f5'
HR_MIN_SIDE = 700


def scaled_configuration(original, densify=1.0, xyz=1.0, deform=1.0):
    def configuration(plan):
        h, opt, pipe, sh = original(plan)
        opt.densify_grad_threshold_fine_init *= densify
        opt.densify_grad_threshold_after *= densify
        opt.position_lr_init *= xyz
        opt.position_lr_final *= xyz
        for name in ('deformation_lr_init', 'deformation_lr_final', 'grid_lr_init', 'grid_lr_final'):
            setattr(opt, name, getattr(opt, name) * deform)
        return h, opt, pipe, sh
    return configuration


_kernels = {}


def gaussian_blur(x, sigma):
    """separable Gaussian blur over the last two axes, reflect padding; x: (..., H, W)"""
    import torch
    import torch.nn.functional as F
    key = (float(sigma), x.device, x.dtype)
    if key not in _kernels:
        r = int(math.ceil(3 * sigma))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
        _kernels[key] = (k / k.sum(), r)
    k, r = _kernels[key]
    shape = x.shape
    y = x.reshape(-1, 1, shape[-2], shape[-1])
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))
    return y.reshape(shape)


def lowpass_teacher_terms(losses, loss_utils, sigma):
    sr_loss, ssim = losses.sr_loss, loss_utils.ssim

    def sr_loss_lp(rgb, teacher):
        return sr_loss(gaussian_blur(rgb, sigma), gaussian_blur(teacher, sigma))

    def ssim_lp(img1, img2, *args, **kwargs):
        if max(img1.shape[-2:]) > HR_MIN_SIDE:
            img1, img2 = gaussian_blur(img1, sigma), gaussian_blur(img2, sigma)
        return ssim(img1, img2, *args, **kwargs)

    losses.sr_loss, loss_utils.ssim = sr_loss_lp, ssim_lp


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    if calibsr_v2.sha(calibsr_v2.__file__) != CALIBSR_V2_SHA:
        raise SystemExit('calibsr_v2.py changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(op=None, calib='none', warp_lr=2e-3, zeros=None, densify=1.0, xyz=1.0, deform=1.0, lowpass=0.0)
    flags = {'--lr-operator': ('op', str), '--calib': ('calib', str), '--warp-lr': ('warp_lr', float),
             '--densify-scale': ('densify', float), '--xyz-lr-scale': ('xyz', float), '--deform-lr-scale': ('deform', float),
             '--teacher-lowpass': ('lowpass', float)}
    i = 0
    while i < len(own):
        if own[i] in flags:
            name, cast = flags[own[i]]
            s[name] = cast(own[i + 1]); i += 2
        elif own[i] == '--teacher-zeros':
            s['zeros'] = (int(own[i + 1]), int(own[i + 2])); i += 3
        else:
            raise SystemExit(f'unknown option {own[i]}')
    if s['op'] not in ('bicaa', 'area') or s['calib'] not in ('none', 'shift', 'full'):
        raise SystemExit(__doc__)
    if not all(0 < s[n] <= 100 for n in ('densify', 'xyz', 'deform')) or not 0 <= s['lowpass'] <= 4:
        raise SystemExit('scales must be in (0, 100], --teacher-lowpass in [0, 4]')
    if s['calib'] != 'none' and '--affine' not in rest:
        raise SystemExit('--calib needs --affine in the srfix arguments (the warp rides on the colour-affine hook)')
    if s['zeros'] is not None:
        lam = float(rest[rest.index('--lambda-sr') + 1]) if '--lambda-sr' in rest else 0.05
        if lam != 0:
            raise SystemExit('--teacher-zeros requires --lambda-sr 0')
    import srfix  # noqa: E402  (imports cullretrain, which imports losses)
    import torch
    import utils.loss_utils as loss_utils
    cr = srfix.cr
    losses = sys.modules['losses']
    if s['op'] == 'area':
        losses.degradation = calibsr_v2.area_degradation
    if s['calib'] != 'none':
        cr.CameraAffine = calibsr_v2.make_calib_affine(cr, s['calib'], s['warp_lr'])
    if s['zeros'] is not None:
        h, w = s['zeros']
        cr.read_teacher = lambda path: torch.zeros(3, h, w)
    if (s['densify'], s['xyz'], s['deform']) != (1.0, 1.0, 1.0):
        cr.configuration = scaled_configuration(cr.configuration, s['densify'], s['xyz'], s['deform'])
    if s['lowpass'] > 0:
        lowpass_teacher_terms(losses, loss_utils, s['lowpass'])
    sys.argv = [str(MOD / 'srfix.py')] + rest
    srfix.main()
    out = Path(rest[rest.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    record = dict(switches=s, argv=rest, degradation_function=f'{losses.degradation.__module__}.{losses.degradation.__name__}',
                  sources={str(Path(f).resolve().relative_to(ROOT)): calibsr_v2.sha(f)
                           for f in (__file__, calibsr_v2.__file__, srfix.__file__, cr.__file__, losses.__file__)})
    (out / 'geosr.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(geosr=s), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
