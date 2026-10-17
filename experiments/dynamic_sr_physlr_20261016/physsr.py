"""SR stage with a selectable LR forward operator (Claude, 4dsr-7c, 2026-10-16). Wraps srfix.py; nothing is modified.

  physsr.py --lr-operator {bicaa,area} <srfix.py arguments ...>

bicaa  the registered operator (losses.degradation: torch bicubic, antialias=True) -- unchanged behaviour.
area   losses.degradation replaced by an exact integer-factor box average (F.avg_pool2d, clamp [0,1]): the forward model of
       a camera whose LR pixel integrates the HR pixels it covers (pixel binning), matching physlr.py area4 scenes.
The swap is a module-attribute patch of the `losses` module that cullretrain.py imports (lr_loss and the D-SSIM LR branch
both call losses.degradation). physsr.json in --out records the operator and source hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOD = ROOT / "experiments/dynamic_sr_modules_20261010"
if str(MOD) not in sys.path:
    sys.path.insert(0, str(MOD))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def area_degradation(rgb, size):
    import torch.nn.functional as F
    h, w = rgb.shape[-2:]
    fy, fx = h // int(size[0]), w // int(size[1])
    if fy != fx or fy * int(size[0]) != h or fx * int(size[1]) != w:
        raise ValueError(f"area operator needs an exact integer factor: {(h, w)} -> {tuple(size)}")
    return F.avg_pool2d(rgb[None], fy)[0].clamp(0, 1)


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "--lr-operator" or argv[1] not in ("bicaa", "area"):
        raise SystemExit("usage: physsr.py --lr-operator {bicaa,area} <srfix.py arguments>")
    op, rest = argv[1], argv[2:]
    import srfix  # noqa: E402  (imports cullretrain, which imports losses)
    losses = sys.modules["losses"]
    original = losses.degradation
    if op == "area":
        losses.degradation = area_degradation
    sys.argv = [str(MOD / "srfix.py")] + rest
    srfix.main()
    if "--out" in rest:
        out = Path(rest[rest.index("--out") + 1])
        out = out if out.is_absolute() else ROOT / out
        record = dict(lr_operator=op, degradation_function=f"{losses.degradation.__module__}.{losses.degradation.__name__}",
                      original_function=f"{original.__module__}.{original.__name__}", argv=rest,
                      sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, losses.__file__)})
        (out / "physsr.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physsr=op)), flush=True)


if __name__ == "__main__":
    main()
