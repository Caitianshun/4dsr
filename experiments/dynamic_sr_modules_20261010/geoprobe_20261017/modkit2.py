"""SR-stage module kit, part 2: per-camera time offsets and per-camera smooth warp fields (Claude, session 4dsr-e9, 2026-10-17).
New file; modkit.py (byte-frozen on the remotes, hash pinned below) and everything it imports are imported, never modified.

  modkit2.py [--time-offset LR] [--warp-field G] [--warp-field-lr X] <modkit.py options> -- --mode fix {cullretrain|csc} sr <SR args>

Motivation (camres_probe.py, Cook07 area4 matched teacher, 10-17): about half of each training camera's LR residual is
persistent, and the learned calibsr_v2 warps on Cut07 are dominated by the radial term (rms 0.59 HR px at the half-diagonal).
Two per-camera error codes that the existing kit cannot express:

--time-offset LR : a learnable time offset per training camera, in frames (dt = the median frame spacing of the manifest), mean
                   over training cameras fixed to zero (scene clock gauge), zero init, Adam (lr LR, exponential decay to 0.1x
                   over the SR steps). Every grad-enabled training render of the SR model uses its own deformation state at
                   time + offset (the registered loop shares one state per same-time batch; with offsets each camera gets its own
                   deformation call, so the time input is differentiable). Test views keep offset 0; evaluation is unchanged.
                   Prior art: Sync-NeRF (AAAI 2024) learns per-video time offsets for unsynchronised multi-view video.
--warp-field G   : a smooth per-camera displacement field (HR px) on a G x G grid (bilinear, align_corners), zero init, mean over
                   training cameras removed, Adam (lr --warp-field-lr, default 2e-3, decay to 0.1x); applied to the render
                   (bicubic grid_sample, border) before the rest of the camera model (calibsr_v2 warp, colour affine, colour
                   field). Absorbs distortion residuals that the 5-term calibsr_v2 model cannot. Test views use the identity.
Both live in the camera-affine object (state saved in the checkpoint's affine payload) and need --affine in the SR arguments.
modkit2.json in --out records the switches, the learned offsets / field rms and source hashes.
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
import modkit  # noqa: E402

PINNED = {'modkit.py': '611381b59d7488da'}
REG = {}


def make_ext_affine(base, time_lr, wf_grid, wf_lr, dt):
    import torch
    import torch.nn.functional as F

    class ExtAffine(base):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            n = len(self.names)
            self.ext = dict(time_lr=time_lr, wf_grid=wf_grid, wf_lr=wf_lr, dt=dt)
            self.toff = torch.zeros(n, device=device, requires_grad=True)
            self.toff_optimizer = torch.optim.Adam([self.toff], lr=time_lr or 1e-3)
            self.wfield = torch.zeros(n, 2, max(wf_grid, 1), max(wf_grid, 1), device=device, requires_grad=True)
            self.wfield_optimizer = torch.optim.Adam([self.wfield], lr=wf_lr)
            self.wf_grids = {}
            REG['affine'] = self

        def time_offset(self, camera_id):
            """offset in scene-time units (frames * dt); 0 for cameras outside the training list"""
            if not self.ext['time_lr'] or camera_id not in self.index:
                return None
            return (self.toff[self.index[camera_id]] - self.toff.mean()) * self.ext['dt']

        def warp_field(self, image, camera_id):
            if not self.ext['wf_grid'] or camera_id not in self.index:
                return image
            h, w = image.shape[-2:]
            key = (h, w, image.device)
            if key not in self.wf_grids:
                py, px = torch.meshgrid(torch.arange(h, device=image.device, dtype=torch.float32) + 0.5,
                                        torch.arange(w, device=image.device, dtype=torch.float32) + 0.5, indexing='ij')
                self.wf_grids[key] = (px, py)
            px, py = self.wf_grids[key]
            d = self.wfield[self.index[camera_id]] - self.wfield.mean(0)
            up = F.interpolate(d[None], size=(h, w), mode='bilinear', align_corners=True)[0]
            grid = torch.stack([(px + up[0]) * (2 / w) - 1, (py + up[1]) * (2 / h) - 1], -1)
            return F.grid_sample(image[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]

        def apply(self, image, camera_id):
            return super().apply(self.warp_field(image, camera_id), camera_id)

        def step(self, clock):
            super().step(clock)
            t = min(max(clock / self.max_steps, 0.), 1.)
            if self.ext['time_lr']:
                for group in self.toff_optimizer.param_groups:
                    group['lr'] = self.ext['time_lr'] * (0.1 ** t)
                self.toff_optimizer.step()
            if self.ext['wf_grid']:
                for group in self.wfield_optimizer.param_groups:
                    group['lr'] = self.ext['wf_lr'] * (0.1 ** t)
                self.wfield_optimizer.step()
            self.toff_optimizer.zero_grad(set_to_none=True)
            self.wfield_optimizer.zero_grad(set_to_none=True)

        def state_dict(self):
            value = super().state_dict()
            value.update(ext=dict(self.ext), toff=self.toff.detach().clone(), wfield=self.wfield.detach().clone(),
                         toff_optimizer=self.toff_optimizer.state_dict(), wfield_optimizer=self.wfield_optimizer.state_dict())
            return value

        def load_state_dict(self, value):
            super().load_state_dict(value)  # LR parents carry the colour affine only; offsets and field start at zero
            with torch.no_grad():
                if 'toff' in value and tuple(value['toff'].shape) == tuple(self.toff.shape):
                    self.toff.copy_(value['toff'])
                if 'wfield' in value and tuple(value['wfield'].shape) == tuple(self.wfield.shape):
                    self.wfield.copy_(value['wfield'])

        def summary(self):
            value = super().summary()
            to = (self.toff - self.toff.mean()).detach().cpu()
            wf = (self.wfield - self.wfield.mean(0)).detach().cpu()
            for i, c in enumerate(self.names):
                if self.ext['time_lr']:
                    value[c]['time_offset_frames'] = float(to[i])
                if self.ext['wf_grid']:
                    value[c]['warp_field_rms_px'] = float(wf[i].pow(2).sum(0).mean().sqrt())
            return value

    return ExtAffine


def install_time_hooks(native, state):
    """per-camera deformation state for grad-enabled renders of the SR model (cameras tagged by hr_camera)"""
    import torch
    hr_camera, rgb = native.hr_camera, native.native_rgb
    state['time'] = dict(renders=0)

    def hr_camera_(observation, uid, manifest):
        camera = hr_camera(observation, uid, manifest)
        camera.mk_camera_id = observation['camera_id']
        return camera

    def native_rgb(camera, g, st, pipe, background, raster_api=None):
        aff = REG.get('affine')
        if aff is not None and g is state.get('g') and torch.is_grad_enabled() and hasattr(camera, 'mk_camera_id'):
            off = aff.time_offset(camera.mk_camera_id)
            if off is not None:
                n = g.get_xyz.shape[0]
                tt = (torch.tensor(float(camera.time), dtype=torch.float32, device=g.get_xyz.device) + off).reshape(1, 1).repeat(n, 1)
                xyz, scaling, rotation, opacity, sh = g._deformation(g.get_xyz, g._scaling, g._rotation, g._opacity, g.get_features, tt)
                st = dict(xyz=xyz, scales=g.scaling_activation(scaling), rotation=g.rotation_activation(rotation),
                          opacity=g.opacity_activation(opacity), sh=sh)
                state['time']['renders'] += 1
        return rgb(camera, g, st, pipe, background, raster_api)

    native.hr_camera, native.native_rgb = hr_camera_, native_rgb


def manifest_dt(cr, sr_args):
    import numpy as np
    import torch
    parent = Path(sr_args[sr_args.index('--parent-checkpoint') + 1])
    parent = parent if parent.is_absolute() else ROOT / parent
    plan = torch.load(parent, map_location='cpu', weights_only=False)['plan']
    _, manifest = cr.plan_scene(plan)
    cams = manifest['splits']['train']
    dts = []
    for c in cams:
        t = sorted(float(o['time']) for o in manifest['observations'] if o['camera_id'] == c)
        dts.append(float(np.median(np.diff(t))))
    return float(np.median(dts))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    for name, digest in PINNED.items():
        if not modkit.calibsr_v2.sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(time_lr=0.0, wf_grid=0, wf_lr=2e-3)
    flags = {'--time-offset': ('time_lr', float), '--warp-field': ('wf_grid', int), '--warp-field-lr': ('wf_lr', float)}
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] in flags:
            name, cast = flags[own[i]]
            s[name] = cast(own[i + 1]); i += 2
        else:
            passthrough.append(own[i]); i += 1
    if not 0 <= s['time_lr'] <= 1 or not 0 <= s['wf_grid'] <= 16 or not 0 < s['wf_lr'] <= 1:
        raise SystemExit('switch out of range')
    if (s['time_lr'] or s['wf_grid']) and '--affine' not in rest:
        raise SystemExit('--time-offset / --warp-field need --affine in the SR arguments')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402  (the module object modkit.main imports again)
    cr = srfix.cr
    native = cr.native
    original_main = srfix.main
    state = dict(switches=s)

    def srfix_main():
        """runs after modkit.main has installed its switches (cr.CameraAffine is final here)"""
        if s['time_lr'] or s['wf_grid']:
            s['dt'] = manifest_dt(cr, sr_args) if s['time_lr'] else None
            cr.CameraAffine = make_ext_affine(cr.CameraAffine, s['time_lr'], s['wf_grid'], s['wf_lr'], s['dt'] or 0.0)
            from scene.gaussian_model import GaussianModel
            restore = GaussianModel.restore

            def restore_(self, model_args, training_args):
                restore(self, model_args, training_args)
                state['g'] = self
            GaussianModel.restore = restore_
            if s['time_lr']:
                install_time_hooks(native, state)
        return original_main()

    srfix.main = srfix_main
    sys.argv = [str(HERE / 'modkit.py')] + passthrough + ['--'] + rest
    modkit.main()
    aff = REG.get('affine')
    summ = aff.summary() if aff is not None else {}
    record = dict(switches=s, passthrough=passthrough, argv=rest, time=state.get('time'),
                  time_offsets_frames={c: v.get('time_offset_frames') for c, v in summ.items() if 'time_offset_frames' in v},
                  warp_field_rms_px={c: v.get('warp_field_rms_px') for c, v in summ.items() if 'warp_field_rms_px' in v},
                  sources={str(Path(f).resolve().relative_to(ROOT)): modkit.calibsr_v2.sha(f) for f in (__file__, modkit.__file__)})
    (out / 'modkit2.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit2=s, time=state.get('time')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
