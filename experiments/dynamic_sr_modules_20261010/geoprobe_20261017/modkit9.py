"""SR-stage module kit, part 9: a clean combination of the 5-term calibration and the per-camera warp field (Claude, session 4dsr-e9,
2026-10-18). New file; modkit8.py (hash pinned below) and everything it imports are imported, never modified.

  modkit9.py [--compose-warp] [--field-residual] <modkit8 / ... / modkit options> -- --mode fix cullretrain sr <SR args>

Context (10-18): with calib full + warp field, modkit2's ExtAffine resamples the render twice (bicubic warp field, then calibsr_v2's
bicubic 5-term warp), and the 4x4 field can represent the 5-term displacement itself, so the two corrections share degrees of
freedom with different learning rates. Discussion07 lost 0.33 dB with this stack (field alone +0.18), VRHeadset07 gained 0.52.
--compose-warp    : one bicubic sampling at the composed coordinate x(p) = q(p) + u(q(p)) instead of two (q = the 5-term calib
                    mapping of calibsr_v2, u = the bilinear warp field evaluated at q); the same geometry as the double sampling.
--field-residual  : the field (re-centred over cameras as before) is projected onto the orthogonal complement of the calibration's
                    displacement basis at the field nodes (translation x / y, scale, in-plane rotation, radial r^2 term, linearised
                    at the identity; translations only for calib shift), so the calibration carries the global parametric part
                    and the field only the residual. With the warp field but no calib this switch removes nothing.
Both switches only change the forward model of training cameras with calib and field active; cameras outside the training list
(cam00, and the identity evaluation of a held-out camera) are unchanged. modkit9.json in --out records the switches, the number of
composed / projected applications and source hashes.
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

PINNED = {'modkit8.py': '1bc6b05f758628e8'}


def calib_basis(h, w, g, mode, device):
    """displacement basis of the calibration at the g x g field nodes (node vector layout: channel x then y, row-major), 2 g g x k"""
    import math
    import torch
    rows = (torch.arange(g, device=device, dtype=torch.float64) * (h - 1) / (g - 1) + 0.5) - h / 2
    cols = (torch.arange(g, device=device, dtype=torch.float64) * (w - 1) / (g - 1) + 0.5) - w / 2
    py, px = torch.meshgrid(rows, cols, indexing='ij')
    R = math.hypot(h / 2, w / 2)
    one, zero = torch.ones_like(px), torch.zeros_like(px)
    fields = [(one, zero), (zero, one)]
    if mode == 'full':
        r2 = (px * px + py * py) / (R * R)
        fields += [(px / R, py / R), (-py / R, px / R), (px * r2 / R, py * r2 / R)]
    return torch.stack([torch.cat([fx.reshape(-1), fy.reshape(-1)]) for fx, fy in fields], 1)


def residual_projector(h, w, g, mode, device):
    import torch
    B = calib_basis(h, w, g, mode, device)
    return (torch.eye(B.shape[0], device=device, dtype=torch.float64) - B @ torch.linalg.solve(B.T @ B, B.T)).float()


def make_clean(base, compose, residual, stats):
    """subclass of modkit2's ExtAffine class `base` with the two switches"""
    import torch
    import torch.nn.functional as F

    class CleanAffine(base):
        def _colour(self, image, camera_id):
            for k in type(self).__mro__:
                if k.__name__ == 'CameraAffine':
                    return k.apply(self, image, camera_id)
            raise RuntimeError('CameraAffine not found')

        def _field(self, camera_id, h, w):
            f = self.wfield[self.index[camera_id]] - self.wfield.mean(0)
            if residual and hasattr(self, 'warp'):
                g = f.shape[-1]
                key = ('P', h, w, g, f.device)
                if key not in self.wf_grids:
                    self.wf_grids[key] = residual_projector(h, w, g, self.calib_mode, f.device)
                f = (self.wf_grids[key] @ f.reshape(-1)).view_as(f)
                stats['projected'] += 1
            return f

        def warp_field(self, image, camera_id):
            if not residual or not self.ext['wf_grid'] or camera_id not in self.index:
                return super().warp_field(image, camera_id)
            h, w = image.shape[-2:]
            key = (h, w, image.device)
            if key not in self.wf_grids:
                py, px = torch.meshgrid(torch.arange(h, device=image.device, dtype=torch.float32) + 0.5,
                                        torch.arange(w, device=image.device, dtype=torch.float32) + 0.5, indexing='ij')
                self.wf_grids[key] = (px, py)
            px, py = self.wf_grids[key]
            up = F.interpolate(self._field(camera_id, h, w)[None], size=(h, w), mode='bilinear', align_corners=True)[0]
            grid = torch.stack([(px + up[0]) * (2 / w) - 1, (py + up[1]) * (2 / h) - 1], -1)
            return F.grid_sample(image[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]

        def apply(self, image, camera_id):
            if not (compose and self.ext['wf_grid'] and hasattr(self, 'warp') and camera_id in self.index):
                return super().apply(image, camera_id)
            h, w = image.shape[-2:]
            d = self.warp[self.index[camera_id]] - self.warp.mean(0)
            px, py, R = self.base_grid(h, w, image.device)
            tx, ty = d[0], d[1]
            if self.calib_mode == 'full':
                s, th, k1 = d[2] / R, d[3] / R, d[4] / R
                radial = 1 + k1 * (px * px + py * py) / (R * R)
                c, sn = torch.cos(th), torch.sin(th)
                qx = (1 + s) * (c * px - sn * py) * radial + tx
                qy = (1 + s) * (sn * px + c * py) * radial + ty
            else:
                qx, qy = px + tx, py + ty
            X, Y = qx + w / 2, qy + h / 2                       # absolute pixel coordinates (pixel centres at k + 0.5)
            f = self._field(camera_id, h, w)
            node = torch.stack([(X - 0.5) / (w - 1) * 2 - 1, (Y - 0.5) / (h - 1) * 2 - 1], -1)
            u = F.grid_sample(f[None], node[None], mode='bilinear', padding_mode='border', align_corners=True)[0]
            grid = torch.stack([(X + u[0]) * (2 / w) - 1, (Y + u[1]) * (2 / h) - 1], -1)
            stats['composed'] += 1
            out = F.grid_sample(image[None], grid[None], mode='bicubic', padding_mode='border', align_corners=False)[0]
            return self._colour(out, camera_id)

    return CleanAffine


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    import modkit8  # noqa: E402
    sha = modkit8.modkit6.modkit5.modkit4.modkit2.modkit.calibsr_v2.sha
    for name, digest in PINNED.items():
        if not sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(compose_warp=False, field_residual=False)
    passthrough = []
    for x in own:
        if x == '--compose-warp':
            s['compose_warp'] = True
        elif x == '--field-residual':
            s['field_residual'] = True
        else:
            passthrough.append(x)
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    modkit2 = modkit8.modkit6.modkit5.modkit4.modkit2
    stats = dict(composed=0, projected=0)
    if s['compose_warp'] or s['field_residual']:
        original = modkit2.make_ext_affine

        def make_ext_affine(base, time_lr, wf_grid, wf_lr, dt):
            return make_clean(original(base, time_lr, wf_grid, wf_lr, dt), s['compose_warp'], s['field_residual'], stats)
        modkit2.make_ext_affine = make_ext_affine     # modkit2.main looks the name up when srfix starts
    sys.argv = [str(HERE / 'modkit8.py')] + passthrough + ['--'] + rest
    modkit8.main()
    error = None
    if s['compose_warp'] and not stats['composed']:
        error = 'compose requested but never applied (needs calib and warp field)'
    if s['field_residual'] and not stats['projected']:
        error = 'residual requested but never applied (needs calib and warp field)'
    record = dict(switches=s, passthrough=passthrough, argv=rest, stats=stats, error=error,
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, modkit8.__file__)})
    (out / 'modkit9.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit9=s, stats=stats, error=error), ensure_ascii=False), flush=True)
    if error:
        raise SystemExit(error)


if __name__ == '__main__':
    main()
