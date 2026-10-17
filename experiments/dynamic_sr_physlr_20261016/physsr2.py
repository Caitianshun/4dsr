"""Restructured SR stage for physically sampled LR (Claude, 4dsr-7c, 2026-10-16). Wraps srfix.py -> modsr.py (or refs.py);
nothing is modified. Module-attribute patches only, all recorded in physsr2.json.

  physsr2.py [--lr-operator {bicaa,area}] [--teacher-band {all,high}] [--teacher-cut C] [--steps N] <srfix.py arguments>

--lr-operator area   losses.degradation -> exact integer-factor box average (pixel-area forward model), as physsr.py.
--teacher-band high  the teacher term only acts above C cycles per HR pixel (default 0.25 = first zero of the x4 box):
                     losses.sr_loss -> mean |HP_C(render - teacher)| (radial FFT mask per channel), and the teacher D-SSIM
                     term is switched off (utils.loss_utils.ssim returns 1 for HR-size inputs; the LR D-SSIM term, at LR
                     size, is unchanged). Below C the solution is left to the multi-view LR term.
--steps N            SR-stage length N (default 6000 = registered schedule): the registered same-time rows are cycled in
                     their order (rows[i % 6000]); learning-rate and densification clocks follow the author schedule.
Defaults reproduce srfix/modsr exactly (no patch applied).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOD = ROOT / "experiments/dynamic_sr_modules_20261010"
if str(MOD) not in sys.path:
    sys.path.insert(0, str(MOD))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def highpass_l1_factory(cut):
    import torch
    cache = {}

    def mask(h, w, device):
        key = (h, w, str(device))
        if key not in cache:
            fy = torch.fft.fftfreq(h, device=device)[:, None]
            fx = torch.fft.rfftfreq(w, device=device)[None, :]
            cache[key] = (torch.sqrt(fy ** 2 + fx ** 2) >= cut).float()
        return cache[key]

    def sr_loss(rgb, teacher):
        d = rgb - teacher
        m = mask(d.shape[-2], d.shape[-1], d.device)
        hp = torch.fft.irfft2(torch.fft.rfft2(d) * m, s=d.shape[-2:])
        return hp.abs().mean()
    return sr_loss


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lr-operator", choices=("bicaa", "area"), default="bicaa")
    p.add_argument("--teacher-band", choices=("all", "high"), default="all")
    p.add_argument("--teacher-cut", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=6000)
    a, rest = p.parse_known_args(sys.argv[1:])
    import srfix  # noqa: E402  (imports cullretrain -> losses, torch, author code)
    import physsr  # noqa: E402
    cr = srfix.cr
    losses = sys.modules["losses"]
    patches = []
    if a.lr_operator == "area":
        losses.degradation = physsr.area_degradation
        patches.append("losses.degradation=area")
    if a.teacher_band == "high":
        import utils.loss_utils as lu
        import torch
        k = rest.index("--baseline-sr")
        base = cr.read(Path(rest[k + 1]) / "config.json" if Path(rest[k + 1]).is_absolute() else ROOT / rest[k + 1] / "config.json")
        man = cr.read(ROOT / base["data"]["manifest"]["path"])
        hr_w, hr_h = man["resolutions"]["hr"]
        original_ssim = lu.ssim

        def ssim(img1, img2, *args, **kw):
            if tuple(img1.shape[-2:]) == (hr_h, hr_w):
                return torch.ones((), device=img1.device, dtype=img1.dtype)  # teacher D-SSIM off
            return original_ssim(img1, img2, *args, **kw)
        lu.ssim = ssim
        losses.sr_loss = highpass_l1_factory(a.teacher_cut)
        patches.append(f"losses.sr_loss=highpass_l1(cut={a.teacher_cut}); teacher D-SSIM off at HR size {hr_w}x{hr_h}")
    if a.steps != cr.SR_STEPS:
        k = rest.index("--baseline-sr")
        base_dir = Path(rest[k + 1]) if Path(rest[k + 1]).is_absolute() else ROOT / rest[k + 1]
        base = cr.read(base_dir / "config.json")
        schedule = (ROOT / base["dependencies"]["schedule"]["path"]).resolve()
        original_read = cr.read
        n = a.steps

        def read(path):
            value = original_read(path)
            if Path(path).resolve() == schedule:
                key = base["definition"]["schedule"]
                rows = value[key]
                value = dict(value)
                value[key] = [rows[i % len(rows)] for i in range(n)]
            return value
        cr.read = read
        cr.SR_STEPS = n
        patches.append(f"SR_STEPS={n}; schedule rows cycled")
    sys.argv = [str(MOD / "srfix.py")] + rest
    srfix.main()
    if "--out" in rest:
        out = Path(rest[rest.index("--out") + 1])
        out = out if out.is_absolute() else ROOT / out
        record = dict(options=vars(a), patches=patches, argv=rest,
                      sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, physsr.__file__, losses.__file__)})
        (out / "physsr2.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physsr2=patches)), flush=True)


if __name__ == "__main__":
    main()
