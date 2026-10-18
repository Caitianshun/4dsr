"""SR-stage module kit, part 4: per-camera point-spread function (Claude, session 4dsr-e9, 2026-10-17). New file; modkit2.py (hash
pinned below) and everything it imports are imported, never modified.

  modkit4.py [--psf K] [--psf-lr X] <modkit2.py options> -- --mode fix {cullretrain|csc} sr <SR args>

Motivation (4dsr-7c, CPU probe 10-17 14:47, Cook07): after HR optical-flow registration, cam06 is blurrier than its five
neighbours by about sigma 1-1.25 HR px and cam16 by about 0.5, i.e. the training cameras differ in sharpness (focus / lens). One
shared radiance field then fits a compromise sharpness, and the blurry cameras (and their teachers) pull the scene toward blur.

--psf K   : every training camera c gets a K x K kernel k_c = delta + E_c (K odd, 3..9), applied to the HR render with reflect
            padding, the same kernel for the three channels. E_c = D_c - mean_taps(D_c), then minus the mean over training cameras,
            so every kernel sums to one and the mean kernel over the training cameras is the delta (the scene-sharpness gauge, as the
            colour affine's mean-identity gauge). D zero init; own Adam (lr --psf-lr, default 1e-3, exponential decay to 0.1x over
            the SR steps). Applied first in the camera chain (before the modkit2 warp field / time offsets, the calibsr warp and the
            colour affine). Test views (cam00) use the delta, i.e. the scene at the training cameras' mean sharpness.
            Requires --affine in the SR arguments.
modkit4.json in --out records the switches, per camera the kernel's second moment (sigma^2 in HR px^2 of an equivalent isotropic
blur; negative = sharpening relative to the mean camera) and source hashes.
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
import modkit2  # noqa: E402

PINNED = {'modkit2.py': 'f2e05b7c47a84d96'}
REG = {}


def make_psf_affine(base, ksize, psf_lr):
    import torch
    import torch.nn.functional as F
    r = ksize // 2

    class PsfAffine(base):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            self.psf_d = torch.zeros(len(self.names), ksize, ksize, device=device, requires_grad=True)
            self.psf_optimizer = torch.optim.Adam([self.psf_d], lr=psf_lr)
            self.psf_lr, self.psf_k = psf_lr, ksize
            self.psf_delta = torch.zeros(ksize, ksize, device=device)  # not 'delta': that is the colour affine's parameter
            self.psf_delta[r, r] = 1.0
            REG['affine'] = self

        def kernels(self):
            e = self.psf_d - self.psf_d.mean((1, 2), keepdim=True)
            e = e - e.mean(0, keepdim=True)
            return self.psf_delta.to(e.device) + e

        def psf(self, image, camera_id):
            if camera_id not in self.index:
                return image
            k = self.kernels()[self.index[camera_id]]
            x = F.pad(image[None], (r, r, r, r), mode='reflect')
            return F.conv2d(x, k.expand(image.shape[0], 1, ksize, ksize), groups=image.shape[0])[0]

        def apply(self, image, camera_id):
            return super().apply(self.psf(image, camera_id), camera_id)

        def step(self, clock):
            super().step(clock)
            t = min(max(clock / self.max_steps, 0.), 1.)
            for group in self.psf_optimizer.param_groups:
                group['lr'] = self.psf_lr * (0.1 ** t)
            self.psf_optimizer.step()
            self.psf_optimizer.zero_grad(set_to_none=True)

        def state_dict(self):
            value = super().state_dict()
            value.update(psf_d=self.psf_d.detach().clone(), psf_optimizer=self.psf_optimizer.state_dict(), psf_k=self.psf_k,
                         psf_lr=self.psf_lr)
            return value

        def load_state_dict(self, value):
            super().load_state_dict(value)
            if 'psf_d' in value and tuple(value['psf_d'].shape) == tuple(self.psf_d.shape):
                with torch.no_grad():
                    self.psf_d.copy_(value['psf_d'])

        def summary(self):
            value = super().summary()
            k = self.kernels().detach().cpu()
            ys, xs = torch.meshgrid(torch.arange(ksize) - r, torch.arange(ksize) - r, indexing='ij')
            rr = (xs * xs + ys * ys).float()
            for i, c in enumerate(self.names):
                value[c]['psf_sigma2_px2'] = float((k[i] * rr).sum() / 2)
                value[c]['psf_center'] = float(k[i, r, r])
            return value

    return PsfAffine


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    for name, digest in PINNED.items():
        if not modkit2.modkit.calibsr_v2.sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(psf=0, psf_lr=1e-3)
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] == '--psf':
            s['psf'] = int(own[i + 1]); i += 2
        elif own[i] == '--psf-lr':
            s['psf_lr'] = float(own[i + 1]); i += 2
        else:
            passthrough.append(own[i]); i += 1
    if s['psf'] and (s['psf'] % 2 == 0 or not 3 <= s['psf'] <= 9) or not 0 < s['psf_lr'] <= 1:
        raise SystemExit('switch out of range')
    if s['psf'] and '--affine' not in rest:
        raise SystemExit('--psf needs --affine in the SR arguments')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402
    cr = srfix.cr
    real_main = srfix.main

    def psf_main():
        """called by modkit2's wrapper after modkit / modkit2 have set cr.CameraAffine"""
        if s['psf']:
            cr.CameraAffine = make_psf_affine(cr.CameraAffine, s['psf'], s['psf_lr'])
        return real_main()

    srfix.main = psf_main
    sys.argv = [str(HERE / 'modkit2.py')] + passthrough + ['--'] + rest
    modkit2.main()
    aff = REG.get('affine')
    summ = aff.summary() if aff is not None else {}
    record = dict(switches=s, passthrough=passthrough, argv=rest,
                  psf_sigma2_px2={c: v.get('psf_sigma2_px2') for c, v in summ.items() if 'psf_sigma2_px2' in v},
                  sources={str(Path(f).resolve().relative_to(ROOT)): modkit2.modkit.calibsr_v2.sha(f) for f in (__file__, modkit2.__file__)})
    (out / 'modkit4.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit4=s), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
