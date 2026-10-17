"""Combined SR-stage wrapper for the physical-LR method (Claude, 4dsr-7c, 2026-10-17). Wraps srfix.py; nothing is modified.

  physcomb.py [--lr-operator {area,bicaa}] [--calib {none,shift,full}] [--warp-lr W] [--drop-max R [--drop-seed S]]
              -- --mode fix cullretrain sr <cullretrain sr arguments incl. --affine when --calib is used>

--lr-operator area  losses.degradation -> exact integer-factor box average (physsr.area_degradation).
--calib             per-training-camera image-space warp of the render (tx, ty [, s, theta, k1]) before the colour affine,
                    the LR operator and the teacher term; mean over training cameras fixed to the identity; cam00 unwarped.
                    The CameraAffine subclass is imported unchanged from peer 4dsr-e9's frozen calibsr_v2.py (sha256 pinned).
--drop-max R        DropGaussian-style dropout on the training renders, as selfens_sr2.py: opacity * m / (1 - r_t),
                    m ~ Bernoulli(1 - r_t) per Gaussian and per render, r_t = R * cursor / stop; own CUDA generator
                    (--drop-seed) so the registered torch/numpy streams are not consumed; no-grad renders untouched.
Defaults (area, no calib, no drop) reproduce physsr.py --lr-operator area. physcomb.json in --out records the switches,
the drop counts and the source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOD = ROOT / "experiments/dynamic_sr_modules_20261010"
CALIB = MOD / "geoprobe_20261017/calibsr_v2.py"
CALIB_SHA = "900c3f6305e92718a1643c4fe9f80baf5b8c4d0b15e62c0cbecbbcb19f2130f5"
for p in (MOD, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(prog="physcomb.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr-operator", choices=("area", "bicaa"), default="area")
    p.add_argument("--calib", choices=("none", "shift", "full"), default="none")
    p.add_argument("--warp-lr", type=float, default=5e-3)
    p.add_argument("--drop-max", type=float)
    p.add_argument("--drop-seed", type=int, default=20261016)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    if rest[:4] != ["--mode", "fix", "cullretrain", "sr"]:
        raise SystemExit("expected: physcomb.py [options] -- --mode fix cullretrain sr <args>")
    if a.calib != "none" and "--affine" not in rest:
        raise SystemExit("--calib needs --affine (the warp rides on the colour-affine hook)")
    if a.drop_max is not None and not 0 < a.drop_max < 1:
        raise SystemExit("--drop-max must lie in (0, 1)")
    import srfix  # noqa: E402  (imports cullretrain -> losses, torch, author code)
    import physsr  # noqa: E402
    import torch
    cr = srfix.cr
    native = cr.native
    losses = sys.modules["losses"]
    from scene.gaussian_model import GaussianModel
    patches = []
    if a.lr_operator == "area":
        losses.degradation = physsr.area_degradation
        patches.append("losses.degradation=area")
    if a.calib != "none":
        if sha(CALIB) != CALIB_SHA:
            raise SystemExit("calibsr_v2.py differs from the pinned sha256")
        spec = importlib.util.spec_from_file_location("calibsr_v2_frozen", CALIB)
        cal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cal)
        cr.CameraAffine = cal.make_calib_affine(cr, a.calib, a.warp_lr)
        patches.append(f"CameraAffine=calibsr_v2.make_calib_affine({a.calib}, warp_lr={a.warp_lr})")
    sr_args = rest[4:]
    stop = int(sr_args[sr_args.index("--stop") + 1]) if "--stop" in sr_args else cr.SR_STEPS
    state = dict(g=None, cursor=0, drop=dict(renders=0, dropped=0, kept=0))
    originals = dict(rgb=native.native_rgb, ulr=GaussianModel.update_learning_rate)
    if a.drop_max is not None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(a.drop_seed)

        def update_learning_rate(self, iteration):
            result = originals["ulr"](self, iteration)
            if state["g"] is None:
                state["g"] = self
            if self is state["g"]:
                state["cursor"] = iteration - cr.LR_OFFSET
            return result

        def native_rgb(camera, g, st, pipe, background, raster_api=None):
            if g is state["g"] and torch.is_grad_enabled() and st["opacity"].requires_grad:
                r = a.drop_max * state["cursor"] / stop
                if r > 0:
                    keep = (torch.rand(st["opacity"].shape[0], 1, device=st["opacity"].device, generator=gen) >= r).to(st["opacity"].dtype)
                    st = dict(st, opacity=st["opacity"] * keep / (1 - r))
                    state["drop"]["renders"] += 1
                    state["drop"]["dropped"] += int((keep == 0).sum())
                    state["drop"]["kept"] += int(keep.sum())
            return originals["rgb"](camera, g, st, pipe, background, raster_api)

        GaussianModel.update_learning_rate = update_learning_rate
        native.native_rgb = native_rgb
        patches.append(f"DropGaussian(drop_max={a.drop_max}, seed={a.drop_seed})")
    sys.argv = [str(MOD / "srfix.py")] + rest
    try:
        srfix.main()
    finally:
        GaussianModel.update_learning_rate = originals["ulr"]
        native.native_rgb = originals["rgb"]
    out = Path(sr_args[sr_args.index("--out") + 1])
    out = out if out.is_absolute() else ROOT / out
    if a.drop_max is not None and state["drop"]["renders"] == 0:
        raise SystemExit("drop hook never fired")
    record = dict(options={k: v for k, v in vars(a).items() if k != "rest"}, patches=patches, argv=rest, stop=stop,
                  drop=state["drop"], sources={str(Path(f).resolve().relative_to(ROOT)): sha(f)
                                               for f in (__file__, physsr.__file__, srfix.__file__, cr.__file__, losses.__file__, CALIB)})
    (out / "physcomb.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physcomb=patches, drop=state["drop"])), flush=True)


if __name__ == "__main__":
    main()
