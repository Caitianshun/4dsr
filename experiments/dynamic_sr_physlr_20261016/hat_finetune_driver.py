"""Wave-14 HAT-L area-matched fine-tune with the iteration count fixed by the pre-registered budget rule (Claude 4dsr-7c, 2026-10-18).

  hat_finetune_driver.py --bench BENCH.json --out FT_DIR

Rule (README "第十四波", written before the bench ran): plain training is used when it fits (peak <= 16 GiB, leaving room for the
concurrent LR-parent worker on the same 24 GiB card; bench and fine-tune run under a 16 GiB per-process cap), otherwise gradient
checkpointing; with s = the bench's seconds per iteration of the chosen setting, iterations
N = clamp(floor(7200 / s / 500) * 500, 2000, 6000) (fine-tune budget 2 GPU hours). Everything else
equals the existing area-matched SwinIR fine-tune: same external data and split, L1, Adam 2e-5 with cosine to 1/100 over N, batch 6,
seed 20261017, 4x4 sRGB box LR. driver.json in FT_DIR records the bench values and the decision; finetune_swinir_area.py runs
in-process with --arch hat_l.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def decide(bench):
    plain, ck = bench["train"].get("plain", {}), bench["train"].get("checkpoint", {})
    if "s_per_iter" in plain and plain["peak_GB"] <= 16.0:
        mode, s = "plain", plain["s_per_iter"]
    elif "s_per_iter" in ck:
        mode, s = "checkpoint", ck["s_per_iter"]
    else:
        raise SystemExit("neither training setting ran in the bench")
    n = int(min(6000, max(2000, math.floor(7200 / s / 500) * 500)))
    return dict(mode=mode, s_per_iter=s, iters=n, est_hours=n * s / 3600)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bench_path = Path(a.bench) if Path(a.bench).is_absolute() else ROOT / a.bench
    bench = json.loads(bench_path.read_text())
    d = decide(bench)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "driver.json").write_text(json.dumps(dict(bench=str(bench_path.relative_to(ROOT)), bench_train=bench["train"],
                                                     decision=d), indent=1))
    print(json.dumps(dict(hat_finetune=d)), flush=True)
    import torch
    # same 16 GiB per-process cap as the bench, so the fine-tune cannot starve the concurrent LR-parent worker
    torch.cuda.set_per_process_memory_fraction(min(1.0, 16 * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    sys.path.insert(0, str(HERE))
    import finetune_swinir_area as ft
    argv = ["--out", a.out, "--iters", str(d["iters"]), "--batch", "6", "--lr", "2e-5", "--val-every", "500", "--arch", "hat_l"]
    if d["mode"] == "checkpoint":
        argv.append("--grad-checkpoint")
    sys.argv = [ft.__file__] + argv
    ft.main()


if __name__ == "__main__":
    main()
