"""SR stage with the linear-light area operator through peer 4dsr-e9's modkit6.py entry (Claude, 4dsr-7c, 2026-10-18). physlin.py is
byte-frozen (the wave-12 arms M1 / M2 record its sha256); this file imports its operator and adds the modkit6 entry, so the
unified spatial correction of the main method (4x4 per-camera warp field, `--warp-field 4`) can run on area4lin scenes:

  physlin2.py modkit6 --lr-operator area <modkit6.py own switches> -- <srfix arguments ...>

The modkit chain (modkit6 -> modkit5 -> modkit4 -> modkit2 -> modkit) installs `calibsr_v2.area_degradation` as losses.degradation
when modkit.main() runs; that module global is replaced here by physlin.arealin_degradation before modkit6.main() runs, so the
chain's own records (modkit.json) name this function. Nothing in the chain is modified. physlin2.json in --out records the
operator and source hashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import physlin  # noqa: E402

ROOT = physlin.ROOT


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "modkit6" or argv[1:3] != ["--lr-operator", "area"]:
        raise SystemExit("usage: physlin2.py modkit6 --lr-operator area <modkit6.py arguments>")
    rest = argv[1:]
    sys.path.insert(0, str(physlin.CALIB_DIR))
    import modkit6 as target
    import calibsr_v2
    if target.modkit5.modkit4.modkit2.modkit.calibsr_v2 is not calibsr_v2:
        raise SystemExit("the modkit chain does not use the shared calibsr_v2 module")
    calibsr_v2.area_degradation = physlin.arealin_degradation
    sys.argv = [target.__file__] + rest
    target.main()
    if "--out" in rest:
        out = Path(rest[rest.index("--out") + 1])
        out = out if out.is_absolute() else ROOT / out
        losses = sys.modules["losses"]
        if losses.degradation is not physlin.arealin_degradation:
            raise SystemExit("the linear-light operator was not installed")
        files = (__file__, physlin.__file__, target.__file__, calibsr_v2.__file__, losses.__file__)
        record = dict(lr_operator="arealin", wrapped="modkit6", argv=rest,
                      sources={str(Path(f).resolve().relative_to(ROOT)): physlin.sha(f) for f in files})
        (out / "physlin2.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(physlin2="modkit6")), flush=True)


if __name__ == "__main__":
    main()
