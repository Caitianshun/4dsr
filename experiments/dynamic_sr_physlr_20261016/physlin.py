"""SR stage with the linear-light area operator (Claude, 4dsr-7c, 2026-10-18). Colour-space sensitivity control of the area4
protocol: on area4lin scenes (physlr.py), whose LR averages the HR in linear light, the matching forward operator is

    render -> sRGB decode -> 4x4 box mean -> sRGB encode -> clamp [0, 1]   (IEC 61966-2-1 transfer, as physlr.py area4lin)

  physlin.py physsr --lr-operator area <physsr.py arguments ...>
  physlin.py calib  --lr-operator area <calibsr_v2.py arguments ...>

Wraps physsr.py (4dsr-7c) or calibsr_v2.py (4dsr-e9); nothing is modified. Both install `area_degradation`, a module global
read when their main() runs, as losses.degradation; it is replaced here by `arealin_degradation` before main() runs, so their
own records (physsr.json / calibsr.json) name this function. physlin.json in --out records the operator and source hashes.
The decode and encode are continued linearly below their thresholds, so renders outside [0, 1] keep finite gradients.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CALIB_DIR = ROOT / "experiments/dynamic_sr_modules_20261010/geoprobe_20261017"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def srgb_decode(x):
    import torch
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def srgb_encode(y):
    import torch
    return torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def arealin_degradation(rgb, size):
    import torch.nn.functional as F
    h, w = rgb.shape[-2:]
    fy, fx = h // int(size[0]), w // int(size[1])
    if fy != fx or fy * int(size[0]) != h or fx * int(size[1]) != w:
        raise ValueError(f"area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}")
    return srgb_encode(F.avg_pool2d(srgb_decode(rgb)[None], fy)[0]).clamp(0, 1)


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] not in ("physsr", "calib") or argv[1:3] != ["--lr-operator", "area"]:
        raise SystemExit("usage: physlin.py {physsr,calib} --lr-operator area <wrapped script arguments>")
    which, rest = argv[0], argv[1:]
    if which == "physsr":
        sys.path.insert(0, str(HERE))
        import physsr as target
    else:
        sys.path.insert(0, str(CALIB_DIR))
        import calibsr_v2 as target
    target.area_degradation = arealin_degradation
    sys.argv = [target.__file__] + rest
    target.main()
    if "--out" in rest:
        out = Path(rest[rest.index("--out") + 1])
        out = out if out.is_absolute() else ROOT / out
        losses = sys.modules["losses"]
        if losses.degradation is not arealin_degradation:
            raise SystemExit("the linear-light operator was not installed")
        record = dict(lr_operator="arealin", wrapped=which, argv=rest,
                      sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, target.__file__, losses.__file__)})
        (out / "physlin.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physlin=which)), flush=True)


if __name__ == "__main__":
    main()
