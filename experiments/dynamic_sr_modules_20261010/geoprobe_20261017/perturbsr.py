"""LR and SR stages with injected per-camera sub-pixel misregistration (Claude, session 4dsr-e9, 2026-10-17). New file;
calibsr_v2.py (byte-frozen, sha256 900c3f63...), srfix.py and cullretrain.py are imported, never modified.

  perturbsr.py --perturb-sigma S [--perturb-seed N] --hr-width W --lr-operator {bicaa,area} [--calib {none,shift,full}]
               [--warp-lr 2e-3] [--teacher-zeros H W] -- <srfix.py args>            (SR stage, e.g. --mode fix cullretrain sr ...)
  perturbsr.py --perturb-sigma S [--perturb-seed N] --hr-width W [--calib ...] -- lr <cullretrain lr args>   (LR fine stage)

Question (user, 10-17): does a weaker base model leave SR less usable headroom? The frozen-geometry probe varied the geometry
source; this varies the registration quality in real training (a controlled "weaker base"). Each training camera i gets a fixed
image-space shift eps_i, drawn per axis from N(0, S^2) HR px (numpy RandomState(N), cameras in sorted-name order) and
mean-centred over the training cameras (the gauge of the colour affine and of calibsr). Every training-view render is resampled
at q = p + eps_i (+ the learned calibsr warp when --calib is not none) before the colour affine and the loss -- the LR term, the
LR operator and the teacher term of the SR stage, or the L1/D-SSIM term of the LR fine stage -- i.e. the model's camera i
carries a principal-point error while all supervision (LR, teacher, HR oracle) stays in the true frame. Test views (cam00) are
rendered without any warp. S = 0 adds an identity shift (calibsr_v2 behaviour).

All warp quantities are in HR px: eps and every calibsr term (tx, ty, s, theta, k1 = displacement at the half-diagonal) are
multiplied by w / W for a render of width w (W = --hr-width), so the LR fine stage (renders at LR resolution) and the SR stage
(HR renders) see the same camera error, and a warp learned in the LR stage carries over to the SR stage. (calibsr_v2 itself
applies its parameters in the pixels of whatever it is given; it is only used in the SR stage, where w = W.)
The colour affine is cullretrain.CameraAffine; the warp machinery is calibsr_v2.make_calib_affine (bicubic grid_sample, border
padding). With --calib none the warp parameters are frozen at zero and only eps is applied. A checkpoint's affine state stores
perturb_px; loading a state whose perturb_px differs from this run's eps raises (stages must use the same S and seed).
The LR coarse stage has no colour affine and is not perturbed (use the shared coarse-end checkpoint as --fine0).
"""
from __future__ import annotations

import json
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


def perturbation(names, sigma, seed):
    import numpy as np
    order = sorted(names)
    draw = np.random.RandomState(seed).normal(0.0, sigma, size=(len(order), 2)) if sigma > 0 else np.zeros((len(order), 2))
    draw = draw - draw.mean(0)
    by_name = {c: draw[i] for i, c in enumerate(order)}
    return [[float(by_name[c][0]), float(by_name[c][1])] for c in names]


def make_perturbed_affine(cr, calib, warp_lr, sigma, seed, hr_width):
    import torch
    import torch.nn.functional as F
    original = cr.CameraAffine
    base = calibsr_v2.make_calib_affine(cr, 'shift' if calib == 'none' else calib, warp_lr)
    learn = calib != 'none'

    class PerturbedAffine(base):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            self.eps_list = perturbation(self.names, sigma, seed)
            self.eps = torch.tensor(self.eps_list, dtype=torch.float32, device=device)
            self.perturb_sigma, self.perturb_seed, self.learn_warp, self.hr_width = sigma, seed, learn, hr_width
            if not learn:
                self.warp.requires_grad_(False)

        def warp_image(self, image, camera_id):
            i = self.index[camera_id]
            h, w = image.shape[-2:]
            k = w / self.hr_width
            d = (self.warp[i] - self.warp.mean(0)) * k
            px, py, R = self.base_grid(h, w, image.device)
            tx, ty = d[0] + self.eps[i, 0] * k, d[1] + self.eps[i, 1] * k
            if d.numel() == 5:
                s, th, k1 = d[2] / R, d[3] / R, d[4] / R
                radial = 1 + k1 * (px * px + py * py) / (R * R)
                c, sn = torch.cos(th), torch.sin(th)
                qx = (1 + s) * (c * px - sn * py) * radial + tx
                qy = (1 + s) * (sn * px + c * py) * radial + ty
            else:
                qx, qy = px + tx, py + ty
            grid = torch.stack([(qx + w / 2) * (2 / w) - 1, (qy + h / 2) * (2 / h) - 1], -1)
            return F.grid_sample(image[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]

        def step(self, clock):
            if self.learn_warp:
                super().step(clock)
            else:
                original.step(self, clock)  # colour affine only; the frozen warp stays at zero

        def state_dict(self):
            value = super().state_dict()
            value.update(perturb_px=self.eps.detach().clone(), perturb_sigma=self.perturb_sigma, perturb_seed=self.perturb_seed,
                         learn_warp=self.learn_warp, warp_units='HR px', hr_width=self.hr_width)
            return value

        def load_state_dict(self, value):
            original.load_state_dict(self, value)  # colour affine (delta + optimizer), camera-list check
            if 'perturb_px' in value and not torch.allclose(value['perturb_px'].to(self.eps.device), self.eps, atol=1e-6):
                raise ValueError('parent affine state carries a different perturbation (sigma/seed)')
            if value.get('learn_warp') and 'warp' in value:  # a warp learned by an earlier perturbsr stage (HR px units)
                src = value['warp'].to(self.warp.device)
                n = min(src.shape[1], self.warp.shape[1])
                with torch.no_grad():
                    self.warp[:, :n].copy_(src[:, :n])

        def summary(self):
            value = super().summary()
            for i, c in enumerate(self.names):
                value[c]['perturb_px'] = self.eps_list[i]
            return value

    return PerturbedAffine


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    if calibsr_v2.sha(calibsr_v2.__file__) != CALIBSR_V2_SHA:
        raise SystemExit('calibsr_v2.py changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    op, calib, warp_lr, zeros, sigma, seed, hr_width = None, 'none', 2e-3, None, None, 20261017, None
    i = 0
    while i < len(own):
        if own[i] == '--lr-operator':
            op = own[i + 1]; i += 2
        elif own[i] == '--calib':
            calib = own[i + 1]; i += 2
        elif own[i] == '--warp-lr':
            warp_lr = float(own[i + 1]); i += 2
        elif own[i] == '--perturb-sigma':
            sigma = float(own[i + 1]); i += 2
        elif own[i] == '--perturb-seed':
            seed = int(own[i + 1]); i += 2
        elif own[i] == '--hr-width':
            hr_width = int(own[i + 1]); i += 2
        elif own[i] == '--teacher-zeros':
            zeros = (int(own[i + 1]), int(own[i + 2])); i += 3
        else:
            raise SystemExit(f'unknown option {own[i]}')
    lr_stage = bool(rest) and rest[0] == 'lr'
    if calib not in ('none', 'shift', 'full') or sigma is None or not 0 <= sigma <= 2 or not hr_width:
        raise SystemExit(__doc__)
    if lr_stage and (op is not None or zeros is not None or '--coarse-only' in rest):
        raise SystemExit('the LR fine stage takes no --lr-operator / --teacher-zeros and cannot be --coarse-only')
    if not lr_stage and op not in ('bicaa', 'area'):
        raise SystemExit(__doc__)
    if '--affine' not in rest:
        raise SystemExit('perturbsr needs --affine in the stage arguments (the warp rides on the colour-affine hook)')
    if zeros is not None:
        lam = float(rest[rest.index('--lambda-sr') + 1]) if '--lambda-sr' in rest else 0.05
        if lam != 0:
            raise SystemExit('--teacher-zeros requires --lambda-sr 0')
    import srfix  # noqa: E402  (imports cullretrain, which imports losses; srfix patches nothing at import)
    import torch
    cr = srfix.cr
    losses = sys.modules['losses']
    if op == 'area':
        losses.degradation = calibsr_v2.area_degradation
    cr.CameraAffine = make_perturbed_affine(cr, calib, warp_lr, sigma, seed, hr_width)
    if zeros is not None:
        h, w = zeros
        cr.read_teacher = lambda path: torch.zeros(3, h, w)
    if lr_stage:
        cr.main(rest)
        sources = (__file__, calibsr_v2.__file__, cr.__file__)
    else:
        sys.argv = [str(MOD / 'srfix.py')] + rest
        srfix.main()
        sources = (__file__, calibsr_v2.__file__, srfix.__file__, cr.__file__, losses.__file__)
    out = Path(rest[rest.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    record = dict(stage='lr_fine' if lr_stage else 'sr', lr_operator=op, calib=calib, warp_lr=warp_lr, teacher_zeros=zeros,
                  perturb_sigma=sigma, perturb_seed=seed, hr_width=hr_width, argv=rest,
                  degradation_function=f'{losses.degradation.__module__}.{losses.degradation.__name__}',
                  sources={str(Path(f).resolve().relative_to(ROOT)): calibsr_v2.sha(f) for f in sources})
    (out / 'perturbsr.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(perturbsr=dict(stage=record['stage'], op=op, calib=calib, sigma=sigma, seed=seed)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
