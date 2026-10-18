"""SR-stage module kit (Claude, session 4dsr-e9, 2026-10-17). New file; calibsr_v2.py and geosr.py (byte-frozen, hashes
pinned below), srfix.py, cullretrain.py, csc.py and losses.py are imported, never modified.

User request (10-17 ~11:20 PDT): under the pixel-area (area4) protocol, build effective SR modules -- re-check DropGaussian and
CSC-G, depth/geometry constraints, per-camera error codes, gradient conflict -- and explore broadly. This entry composes the
switches on the registered SR stage (srfix --mode fix, then the cullretrain or csc SR loop):

  modkit.py --lr-operator {bicaa,area} [--calib {none,shift,full}] [--warp-lr X] [--teacher-zeros H W]
            [--densify-scale F] [--xyz-lr-scale F] [--deform-lr-scale F] [--teacher-lowpass S]
            [--drop-rate R] [--drop-seed N] [--cam-field G] [--cam-field-lr X] [--grad-probe K]
            -- --mode fix {cullretrain|csc} sr <SR args>

--lr-operator, --calib, --warp-lr, --teacher-zeros : as calibsr_v2.py (exact 4x4 box operator; per-camera image warp).
--densify-scale, --xyz-lr-scale, --deform-lr-scale, --teacher-lowpass : as geosr.py (its functions are imported).
--drop-rate R  : DropGaussian (CVPR 2025) as selfens_sr2.py --drop-max: on every grad-enabled training render each Gaussian's
                 opacity is multiplied by m / (1 - r_t), m ~ Bernoulli(1 - r_t), r_t = R * cursor / stop; own CUDA generator
                 (--drop-seed, default 20261016), so the registered RNG streams are untouched; evaluation renders unchanged.
--cam-field G  : per-training-camera error code -- a smooth spatial colour-affine field on top of the per-camera 3x4 colour
                 affine: out' = out + M(x) out + b(x), (M, b) bilinear (align_corners) from a G x G grid of 12 numbers per
                 camera, zero init, mean over training cameras removed (same gauge as the affine), own Adam (lr --cam-field-lr,
                 default 1e-3, constant). Absorbs persistent, low-frequency per-camera colour errors (vignetting, codec
                 bias) that a global affine cannot; test views use the identity. Requires --affine.
--grad-probe K : diagnostic only, the trajectory is unchanged. Every K SR steps, the gradients of the LR term (anchor view,
                 L1 part) and of the teacher terms (views b, c, L1 parts) are computed separately (torch.autograd.grad,
                 retain_graph) for xyz, f_dc, opacity and scaling; the global cosine, the fraction of Gaussians whose
                 per-Gaussian dot product is negative and the gradient-norm share of those are written to grad_probe.jsonl.
                 torch.autograd.grad also fills the .grad of retain_grad tensors (the screen-space points used by the
                 densification statistics), so the probe resets them to None before the step's own backward.
                 A probe failure disables the probe; it never stops training.
csc entry      : CSC-G (experiments/dynamic_sr_csc_20261010/csc.py, geometry-only cross-view consistency through rendered
                 world-point maps; --geo-weight, --geo-source ...). Its colour reference inverts only the global affine.
modkit.json in --out records the switches, the probe summary and source hashes.
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
import geosr  # noqa: E402

PINNED = {'calibsr_v2.py': '900c3f6305e92718a1643c4fe9f80baf5b8c4d0b15e62c0cbecbbcb19f2130f5',
          'geosr.py': '45f9cfe47a8c9a00'}


def make_field_affine(base, grid, field_lr):
    import torch
    import torch.nn.functional as F

    class FieldAffine(base):
        def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
            super().__init__(cameras, device=device, lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)
            self.field = torch.zeros(len(self.names), 12, grid, grid, device=device, requires_grad=True)
            self.field_optimizer = torch.optim.Adam([self.field], lr=field_lr)
            self.field_grid, self.field_lr = grid, field_lr

        def field_at(self, camera_id, h, w):
            f = self.field[self.index[camera_id]] - self.field.mean(0)
            return F.interpolate(f[None], size=(h, w), mode='bilinear', align_corners=True)[0]

        def apply(self, image, camera_id):
            out = super().apply(image, camera_id)
            up = self.field_at(camera_id, *out.shape[-2:])
            m = up[:9].reshape(3, 3, *out.shape[-2:])
            return out + torch.einsum('ijhw,jhw->ihw', m, out) + up[9:]

        def step(self, clock):
            super().step(clock)
            self.field_optimizer.step()
            self.field_optimizer.zero_grad(set_to_none=True)

        def state_dict(self):
            value = super().state_dict()
            value.update(field=self.field.detach().clone(), field_optimizer=self.field_optimizer.state_dict(),
                         field_grid=self.field_grid, field_lr=self.field_lr)
            return value

        def load_state_dict(self, value):
            super().load_state_dict(value)
            if 'field' in value and tuple(value['field'].shape) == tuple(self.field.shape):
                with torch.no_grad():
                    self.field.copy_(value['field'])

        def summary(self):
            value = super().summary()
            d = (self.field - self.field.mean(0)).detach().cpu()
            for i, c in enumerate(self.names):
                value[c]['field_rms'] = dict(matrix=float(d[i, :9].pow(2).mean().sqrt()), offset=float(d[i, 9:].pow(2).mean().sqrt()))
            return value

    return FieldAffine


def install_drop(native, GaussianModel, cr, rate, seed, stop, state):
    import torch
    original = native.native_rgb
    gens = {}
    state['drop'] = dict(renders=0, dropped=0, kept=0)

    def native_rgb(camera, g, st, pipe, background, raster_api=None):
        if g is state.get('g') and torch.is_grad_enabled() and st['opacity'].requires_grad:
            r = rate * state.get('cursor', 0) / stop
            if r > 0:
                dev = st['opacity'].device
                if dev not in gens:
                    gens[dev] = torch.Generator(device=dev)
                    gens[dev].manual_seed(seed)
                keep = (torch.rand(st['opacity'].shape[0], 1, device=dev, generator=gens[dev]) >= r).to(st['opacity'].dtype)
                st = dict(st, opacity=st['opacity'] * keep / (1 - r))
                state['drop']['renders'] += 1
                state['drop']['dropped'] += int((keep == 0).sum())
                state['drop']['kept'] += int(keep.sum())
        return original(camera, g, st, pipe, background, raster_api)

    native.native_rgb = native_rgb


def install_tracking(GaussianModel, cr, native, state):
    """state['g'] = the model restored for the SR stage; state['cursor'] = SR step (from update_learning_rate);
    state['vs'] = this step's screen-space point tensors of the training renders (retain_grad tensors of native_rgb)."""
    import torch
    restore, ulr, rgb = GaussianModel.restore, GaussianModel.update_learning_rate, native.native_rgb

    def native_rgb(camera, g, st, pipe, background, raster_api=None):
        packet = rgb(camera, g, st, pipe, background, raster_api)
        if g is state.get('g') and torch.is_grad_enabled():
            state.setdefault('vs', []).append(packet['viewspace_points'])
        return packet

    def restore_(self, model_args, training_args):
        restore(self, model_args, training_args)
        state['g'] = self

    def update_learning_rate(self, iteration):
        if self is state.get('g'):
            state['cursor'] = int(iteration) - cr.LR_OFFSET
            state['terms'] = dict(lr=[], sr=[])
            state['vs'] = []
        return ulr(self, iteration)

    GaussianModel.restore, GaussianModel.update_learning_rate = restore_, update_learning_rate
    native.native_rgb = native_rgb


def install_grad_probe(GaussianModel, losses, state, every, out):
    import torch
    lr_loss, sr_loss, regulation = losses.lr_loss, losses.sr_loss, GaussianModel.compute_regulation
    state['probe'] = dict(every=every, rows=0, errors=0, disabled=False)
    log = dict(f=None)

    def lr_loss_(rgb, lr):
        v = lr_loss(rgb, lr)
        if 'terms' in state:
            state['terms']['lr'].append(v)
        return v

    def sr_loss_(rgb, teacher):
        v = sr_loss(rgb, teacher)
        if 'terms' in state:
            state['terms']['sr'].append(v)
        return v

    def compute_regulation(self, *args, **kwargs):
        value = regulation(self, *args, **kwargs)
        cur = state.get('cursor', 0)
        if self is state.get('g') and not state['probe']['disabled'] and cur > 0 and cur % every == 0:
            try:
                terms = state.get('terms', {})
                if len(terms.get('lr', [])) >= 1 and len(terms.get('sr', [])) >= 2:
                    params = dict(xyz=self._xyz, f_dc=self._features_dc, opacity=self._opacity, scaling=self._scaling)
                    names = [k for k, p in params.items() if p.requires_grad]
                    ps = [params[k] for k in names]
                    gl = torch.autograd.grad(terms['lr'][0], ps, retain_graph=True, allow_unused=True)
                    gs = torch.autograd.grad(terms['sr'][-2] + terms['sr'][-1], ps, retain_graph=True, allow_unused=True)
                    for t in state.get('vs', []):  # autograd.grad fills retain_grad tensors; the densification
                        t.grad = None              # statistics must see only the step's own backward
                    row = dict(cursor=cur)
                    for k, a, b in zip(names, gl, gs):
                        if a is None or b is None:
                            continue
                        a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
                        dot = (a2 * b2).sum(1)
                        both = (a2.norm(dim=1) > 0) & (b2.norm(dim=1) > 0)
                        neg = both & (dot < 0)
                        w = b2.norm(dim=1)
                        row[k] = dict(cos=float((a2 * b2).sum() / (a2.norm() * b2.norm() + 1e-30)),
                                      neg_frac=float(neg.sum() / both.sum().clamp_min(1)),
                                      neg_sr_norm_share=float(w[neg].sum() / w[both].sum().clamp_min(1e-30)),
                                      lr_norm=float(a2.norm()), sr_norm=float(b2.norm()))
                    if log['f'] is None:
                        log['f'] = (Path(out) / 'grad_probe.jsonl').open('a', buffering=1)
                    log['f'].write(json.dumps(row) + '\n')
                    state['probe']['rows'] += 1
            except Exception as e:  # never stop training for a diagnostic
                state['probe']['errors'] += 1
                state['probe']['disabled'] = True
                state['probe']['error'] = repr(e)[:500]
        return value

    losses.lr_loss, losses.sr_loss, GaussianModel.compute_regulation = lr_loss_, sr_loss_, compute_regulation


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    for name, digest in PINNED.items():
        if not calibsr_v2.sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(op=None, calib='none', warp_lr=2e-3, zeros=None, densify=1.0, xyz=1.0, deform=1.0, lowpass=0.0,
             drop=0.0, drop_seed=20261016, field=0, field_lr=1e-3, probe=0)
    flags = {'--lr-operator': ('op', str), '--calib': ('calib', str), '--warp-lr': ('warp_lr', float),
             '--densify-scale': ('densify', float), '--xyz-lr-scale': ('xyz', float), '--deform-lr-scale': ('deform', float),
             '--teacher-lowpass': ('lowpass', float), '--drop-rate': ('drop', float), '--drop-seed': ('drop_seed', int),
             '--cam-field': ('field', int), '--cam-field-lr': ('field_lr', float), '--grad-probe': ('probe', int)}
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
    if not all(0 < s[n] <= 100 for n in ('densify', 'xyz', 'deform')) or not 0 <= s['lowpass'] <= 4 or not 0 <= s['drop'] < 1 \
            or not 0 <= s['field'] <= 32 or s['probe'] < 0:
        raise SystemExit('switch out of range')
    if (s['calib'] != 'none' or s['field']) and '--affine' not in rest:
        raise SystemExit('--calib / --cam-field need --affine in the SR arguments')
    if len(rest) < 4 or rest[0] != '--mode' or rest[2] not in ('cullretrain', 'csc') or rest[3] != 'sr':
        raise SystemExit('expected -- --mode fix {cullretrain|csc} sr ...')
    sr_args = rest[3:]
    if s['zeros'] is not None:
        lam = float(sr_args[sr_args.index('--lambda-sr') + 1]) if '--lambda-sr' in sr_args else 0.05
        if lam != 0:
            raise SystemExit('--teacher-zeros requires --lambda-sr 0')
    import srfix  # noqa: E402  (imports cullretrain, which imports losses)
    import torch
    import utils.loss_utils as loss_utils
    from scene.gaussian_model import GaussianModel
    cr = srfix.cr
    losses = sys.modules['losses']
    native = cr.native
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    stop = int(sr_args[sr_args.index('--stop') + 1]) if '--stop' in sr_args else cr.SR_STEPS
    if s['op'] == 'area':
        losses.degradation = calibsr_v2.area_degradation
    affine = cr.CameraAffine
    if s['calib'] != 'none':
        affine = calibsr_v2.make_calib_affine(cr, s['calib'], s['warp_lr'])
    if s['field']:
        affine = make_field_affine(affine, s['field'], s['field_lr'])
    cr.CameraAffine = affine
    if s['zeros'] is not None:
        h, w = s['zeros']
        cr.read_teacher = lambda path: torch.zeros(3, h, w)
    if (s['densify'], s['xyz'], s['deform']) != (1.0, 1.0, 1.0):
        cr.configuration = geosr.scaled_configuration(cr.configuration, s['densify'], s['xyz'], s['deform'])
    if s['lowpass'] > 0:
        geosr.lowpass_teacher_terms(losses, loss_utils, s['lowpass'])
    state = {}
    install_tracking(GaussianModel, cr, native, state)
    if s['drop'] > 0:
        install_drop(native, GaussianModel, cr, s['drop'], s['drop_seed'], stop, state)
    if s['probe']:
        install_grad_probe(GaussianModel, losses, state, s['probe'], out)
    sys.argv = [str(MOD / 'srfix.py')] + rest
    srfix.main()
    files = [__file__, calibsr_v2.__file__, geosr.__file__, srfix.__file__, cr.__file__, losses.__file__]
    if rest[2] == 'csc':
        files.append(sys.modules['csc'].__file__)
    record = dict(switches=s, entry=rest[2], argv=rest, drop=state.get('drop'), probe=state.get('probe'),
                  degradation_function=f'{losses.degradation.__module__}.{losses.degradation.__name__}',
                  sources={str(Path(f).resolve().relative_to(ROOT)): calibsr_v2.sha(f) for f in files})
    (out / 'modkit.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit=s, drop=state.get('drop'), probe=state.get('probe')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
