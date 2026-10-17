"""SR stage with a selectable LR forward operator and optional per-camera sub-pixel self-calibration (Claude, session 4dsr-e9,
2026-10-17). New file; srfix.py / cullretrain.py are imported, never modified.
v2 of calibsr.py (v1 is byte-frozen on the remotes): adds --route-teacher and --sh-cap; v1 behaviour is unchanged otherwise.

  calibsr.py --lr-operator {bicaa,area} [--calib {none,shift,full}] [--warp-lr 2e-3] [--teacher-zeros H W]
             [--route-teacher {all,geometry}] [--sh-cap N] -- <srfix.py args>

Motivation (tmp/claude_beyond_sr_20261017 simulation, 10-17): under the pixel-area protocol the multi-view headroom collapses once
registration errors reach ~0.4 HR px, and every N3DV camera shares one pinhole K with the principal point at the exact image
centre and no distortion. Intrinsic errors (principal point, focal, in-plane rotation, radial distortion) are exact image-space
warps, independent of depth, so they can be absorbed by a per-camera warp of the rendered image.

--calib shift : per training camera (tx, ty) in HR px.
--calib full  : (tx, ty, s, theta, k1), all parameterised as the displacement in HR px at the half-diagonal R:
                q = c + (1 + s/R) Rot(theta/R) p (1 + (k1/R) |p|^2 / R^2) + t,  p = pixel centre - image centre c.
The mean over training cameras is fixed to the identity (same gauge as the colour affine), so the scene frame is unchanged.
The warp is applied to every training-view render before the colour affine and before both the LR operator and the teacher
term (bicubic grid_sample, border padding; identity parameters reproduce the render exactly). Test views (cam00) are rendered
without warp. Implementation: cullretrain.CameraAffine is replaced by a subclass (requires --affine in the srfix arguments);
the warp has its own Adam (lr --warp-lr, exponential decay to 0.1x over the SR steps). The checkpoint's affine state carries
the warp, and calibsr.json in --out records the switches and source hashes.
--lr-operator area replaces losses.degradation by the exact 4x4 box average (as physsr.py, 4dsr-7c).
--teacher-zeros H W serves zero teachers (lambda_SR must be 0): lets pure multi-view arms run where teacher images are absent.
--route-teacher geometry: the teacher views (b, c of each same-time Bsync row) are rendered with the deformed SH detached, so
the teacher term updates only geometry (centres, scales, rotations, opacities, deformation trunk and geometry heads) and the
densification statistics; SH colours and the SH-deformation head learn from the LR term alone. Motivation: the frozen-geometry
probe (output/dynamic_sr_geoprobe_20261017) -- with HR-quality geometry, LR-only appearance is within ~0.5 dB of HR-supervised
appearance in the mid/high bands. Implemented by wrapping native.native_effective_state (called once per step, resets a counter)
and native.native_rgb (2nd and 3rd call of the step get the detached SH).
--sh-cap N: after the parent is restored the active SH degree is set to min(current, N) and oneupSHdegree() stops at N, so the
SR stage (and the saved checkpoint, hence its evaluation) renders with SH degree N; coefficients above N are kept but unused.
Motivation: in the frozen-geometry probe, SH degree 0 under LR-only supervision raised cam00 PSNR by +0.35/+0.37 dB (low band
+0.67/+0.90, mid/high -0.2) on both geometries.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
if str(MOD) not in sys.path:
    sys.path.insert(0, str(MOD))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def area_degradation(rgb, size):
    import torch.nn.functional as F
    h, w = rgb.shape[-2:]
    fy, fx = h // int(size[0]), w // int(size[1])
    if fy != fx or fy * int(size[0]) != h or fx * int(size[1]) != w:
        raise ValueError(f'area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}')
    return F.avg_pool2d(rgb[None], fy)[0].clamp(0, 1)


def make_calib_affine(cr, mode, warp_lr):
    import torch
    import torch.nn.functional as F
    n_terms = {'shift': 2, 'full': 5}[mode]

    class CalibAffine(cr.CameraAffine):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            self.warp = torch.zeros(len(self.names), n_terms, device=device, requires_grad=True)
            self.warp_optimizer = torch.optim.Adam([self.warp], lr=warp_lr)
            self.warp_lr, self.calib_mode, self.grids = warp_lr, mode, {}

        def base_grid(self, h, w, device):
            key = (h, w)
            if key not in self.grids:
                ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5 - h / 2
                xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5 - w / 2
                py, px = torch.meshgrid(ys, xs, indexing='ij')
                self.grids[key] = (px, py, math.hypot(h / 2, w / 2))
            return self.grids[key]

        def warp_image(self, image, camera_id):
            d = self.warp[self.index[camera_id]] - self.warp.mean(0)
            h, w = image.shape[-2:]
            px, py, R = self.base_grid(h, w, image.device)
            tx, ty = d[0], d[1]
            if n_terms == 5:
                s, th, k1 = d[2] / R, d[3] / R, d[4] / R
                radial = 1 + k1 * (px * px + py * py) / (R * R)
                c, sn = torch.cos(th), torch.sin(th)
                qx = (1 + s) * (c * px - sn * py) * radial + tx
                qy = (1 + s) * (sn * px + c * py) * radial + ty
            else:
                qx, qy = px + tx, py + ty
            grid = torch.stack([(qx + w / 2) * (2 / w) - 1, (qy + h / 2) * (2 / h) - 1], -1)
            return F.grid_sample(image[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]

        def apply(self, image, camera_id):
            return super().apply(self.warp_image(image, camera_id), camera_id)

        def step(self, clock):
            super().step(clock)
            t = min(max(clock / self.max_steps, 0.), 1.)
            for group in self.warp_optimizer.param_groups:
                group['lr'] = self.warp_lr * (0.1 ** t)
            self.warp_optimizer.step()
            self.warp_optimizer.zero_grad(set_to_none=True)

        def state_dict(self):
            value = super().state_dict()
            value.update(calib_mode=self.calib_mode, warp=self.warp.detach().clone(),
                         warp_optimizer=self.warp_optimizer.state_dict(), warp_lr=self.warp_lr)
            return value

        def load_state_dict(self, value):
            super().load_state_dict(value)  # parent LR checkpoints carry the colour affine only; the warp starts at identity
            if 'warp' in value:
                with torch.no_grad():
                    self.warp.copy_(value['warp'])

        def summary(self):
            value = super().summary()
            d = (self.warp - self.warp.mean(0)).detach().cpu()
            for i, c in enumerate(self.names):
                value[c]['warp_px_at_R'] = d[i].tolist()
            return value

    return CalibAffine


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    op, calib, warp_lr, zeros, route, sh_cap = None, 'none', 2e-3, None, 'all', None
    i = 0
    while i < len(own):
        if own[i] == '--lr-operator':
            op = own[i + 1]; i += 2
        elif own[i] == '--calib':
            calib = own[i + 1]; i += 2
        elif own[i] == '--warp-lr':
            warp_lr = float(own[i + 1]); i += 2
        elif own[i] == '--sh-cap':
            sh_cap = int(own[i + 1]); i += 2
        elif own[i] == '--route-teacher':
            route = own[i + 1]; i += 2
        elif own[i] == '--teacher-zeros':
            zeros = (int(own[i + 1]), int(own[i + 2])); i += 3
        else:
            raise SystemExit(f'unknown option {own[i]}')
    if op not in ('bicaa', 'area') or calib not in ('none', 'shift', 'full') or route not in ('all', 'geometry'):
        raise SystemExit(__doc__)
    if calib != 'none' and '--affine' not in rest:
        raise SystemExit('--calib needs --affine in the srfix arguments (the warp rides on the colour-affine hook)')
    if zeros is not None:
        lam = float(rest[rest.index('--lambda-sr') + 1]) if '--lambda-sr' in rest else 0.05
        if lam != 0:
            raise SystemExit('--teacher-zeros requires --lambda-sr 0')
    import srfix  # noqa: E402  (imports cullretrain, which imports losses)
    import torch
    cr = srfix.cr
    losses = sys.modules['losses']
    if op == 'area':
        losses.degradation = area_degradation
    if calib != 'none':
        cr.CameraAffine = make_calib_affine(cr, calib, warp_lr)
    if zeros is not None:
        h, w = zeros
        cr.read_teacher = lambda path: torch.zeros(3, h, w)
    if route == 'geometry':
        native = cr.native
        orig_state, orig_rgb, count = native.native_effective_state, native.native_rgb, [0]

        def state_fn(g, time_value, torch_module=None):
            count[0] = 0
            return orig_state(g, time_value, torch_module)

        def rgb_fn(camera, g, state, pipe, background, raster_api=None):
            k = count[0]
            count[0] += 1
            if k in (1, 2):
                state = dict(state, sh=state['sh'].detach())
            return orig_rgb(camera, g, state, pipe, background, raster_api)

        native.native_effective_state, native.native_rgb = state_fn, rgb_fn
    if sh_cap is not None:
        from scene.gaussian_model import GaussianModel
        orig_restore, orig_up = GaussianModel.restore, GaussianModel.oneupSHdegree

        def restore(self, model_args, training_args):
            orig_restore(self, model_args, training_args)
            self.active_sh_degree = min(self.active_sh_degree, sh_cap)

        def oneup(self):
            if self.active_sh_degree < sh_cap:
                orig_up(self)

        GaussianModel.restore, GaussianModel.oneupSHdegree = restore, oneup
    sys.argv = [str(MOD / 'srfix.py')] + rest
    srfix.main()
    out = Path(rest[rest.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    record = dict(lr_operator=op, calib=calib, warp_lr=warp_lr, teacher_zeros=zeros, route_teacher=route, sh_cap=sh_cap, argv=rest,
                  degradation_function=f'{losses.degradation.__module__}.{losses.degradation.__name__}',
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, srfix.__file__, cr.__file__, losses.__file__)})
    (out / 'calibsr.json').write_text(json.dumps(record, indent=1))  # same file name as v1; record carries the v2 source hash
    print(json.dumps(dict(calibsr=dict(op=op, calib=calib)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
