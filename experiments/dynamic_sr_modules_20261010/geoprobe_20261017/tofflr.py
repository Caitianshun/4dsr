"""LR fine stage with learnable per-camera time offsets (Claude, session 4dsr-e9, 2026-10-17). New file; modkit2.py, cullretrain.py
and the pinned 4DGaussians runtime are imported, never modified.

  tofflr.py --time-offset LR [--time-init FILE.json] -- lr <cullretrain lr args (must include --affine)>

Motivation: camres_probe2.py found sub-frame (Cut07) to >= 1 frame (Cook07 cam12, Discussion07 cam09) synchronisation errors
between training cameras. modkit2.py --time-offset learns them in the SR stage only (6000 steps at the end of the learning-rate
schedule); on Cut07 it gave +0.31 dB in cam00's dynamic region on top of self-calibration. The LR fine stage (14000 steps,
deformation network at full learning rate) is where motion is learned, so the offsets are applied there as well.

The time offsets live in the camera-affine object (modkit2.make_ext_affine on cullretrain.CameraAffine): one value per training
camera in frames (dt = median frame spacing of the manifest), mean over training cameras fixed to zero, Adam (lr LR, exponential
decay to 0.1x over the fine stage), zero init or --time-init {camera: frames}. During the fine stage every grad-enabled training
render of gaussian_renderer.render (camera id from image_name "camXX/FFFF") passes time + offset to the deformation network
(deform_network.forward wrapper; the time input stays differentiable). The coarse stage (no deformation) is unchanged. The offsets
are saved with the affine state of the LR checkpoint; an SR stage run with modkit2.py --time-offset inherits them.
tofflr.json in --out records the switches, the learned offsets and source hashes.
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
CTX = {}


def install(cr, time_lr, init, dt):
    import torch
    import gaussian_renderer as GR
    import scene.deformation as D
    base = cr.CameraAffine
    Ext = modkit2.make_ext_affine(base, time_lr, 0, 2e-3, dt)

    class InitExt(Ext):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            if init:
                with torch.no_grad():
                    for c, v in init.items():
                        if c in self.index:
                            self.toff[self.index[c]] = float(v)
    cr.CameraAffine = InitExt

    original_render = GR.render

    def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None, stage='fine', cam_type=None):
        aff = modkit2.REG.get('affine')
        if aff is not None and stage == 'fine' and torch.is_grad_enabled():
            cid = str(getattr(viewpoint_camera, 'image_name', '')).split('/')[0]
            off = aff.time_offset(cid)
            if off is not None:
                CTX['off'] = off
                CTX['renders'] = CTX.get('renders', 0) + 1
        try:
            return original_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type)
        finally:
            CTX.pop('off', None)

    GR.render = render
    original_forward = D.deform_network.forward

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        off = CTX.get('off')
        if off is not None and times_sel is not None:
            times_sel = times_sel + off
        return original_forward(self, point, scales, rotations, opacity, shs, times_sel)

    D.deform_network.forward = forward


def manifest_dt(cr, lr_args):
    import numpy as np
    parent = Path(lr_args[lr_args.index('--parent') + 1])
    parent = parent if parent.is_absolute() else ROOT / parent
    plan = json.loads((parent / 'config.json').read_text())
    _, manifest = cr.plan_scene(plan)
    dts = []
    for c in manifest['splits']['train']:
        t = sorted(float(o['time']) for o in manifest['observations'] if o['camera_id'] == c)
        dts.append(float(np.median(np.diff(t))))
    return float(np.median(dts))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    for name, digest in PINNED.items():
        if not modkit2.modkit.calibsr_v2.sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    time_lr, init_file = 0.0, None
    i = 0
    while i < len(own):
        if own[i] == '--time-offset':
            time_lr = float(own[i + 1]); i += 2
        elif own[i] == '--time-init':
            init_file = own[i + 1]; i += 2
        else:
            raise SystemExit(f'unknown option {own[i]}')
    if not rest or rest[0] != 'lr' or '--affine' not in rest or '--coarse-only' in rest:
        raise SystemExit('expected -- lr <args> with --affine (fine stage)')
    if not 0 < time_lr <= 1:
        raise SystemExit('--time-offset must be in (0, 1]')
    init = json.loads((ROOT / init_file).read_text()) if init_file else None
    import srfix  # noqa: E402  (imports cullretrain and the pinned runtime)
    cr = srfix.cr
    dt = manifest_dt(cr, rest)
    install(cr, time_lr, init, dt)
    cr.main(rest)
    out = Path(rest[rest.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    aff = modkit2.REG.get('affine')
    summ = aff.summary() if aff is not None else {}
    record = dict(stage='lr_fine', time_lr=time_lr, time_init=init, dt=dt, argv=rest, renders_with_offset=CTX.get('renders', 0),
                  time_offsets_frames={c: v.get('time_offset_frames') for c, v in summ.items()},
                  sources={str(Path(f).resolve().relative_to(ROOT)): modkit2.modkit.calibsr_v2.sha(f)
                           for f in (__file__, modkit2.__file__, cr.__file__)})
    (out / 'tofflr.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(tofflr=dict(time_lr=time_lr, dt=dt, renders=CTX.get('renders', 0))), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
