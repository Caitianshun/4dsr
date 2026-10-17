"""Diagnostic teachers of controlled detail quality (Claude, 4dsr-7c, 2026-10-17). CPU, mml_bw interpreter (cv2, numpy, PIL).
Pre-registration: README.md "第十波：教师强度剂量探针" (15:11 PDT).

T_alpha = T_m + alpha * HP(HR - T_m),  HP(x) = x - GaussianBlur(x, sigma 2 HR px),  rounded to 8 bit.
T_m = the matched (area-fine-tuned SwinIR) teacher of a training view. READS TRAINING-VIEW HR: a diagnostic of how much a
better single-image prior could give, never a method. alpha is calibrated on a fixed sample of training images so that the
teacher's detail-band PSNR (same HP, against HR) improves by the requested dB.

  teacher_dose.py calibrate --scene cut_roasted_beef --gains 1 2
  teacher_dose.py write --scene cut_roasted_beef --alpha A --tag diag_tmhp_d1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
SEED = 20261007
BASE_TAG = "sr_swinir_x4_areaft"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rd(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def hp(x):
    return x - np.stack([cv2.GaussianBlur(x[..., c], (0, 0), 2.0) for c in range(3)], -1)


def mix(tm, hr, alpha):
    return np.clip(np.round((tm + alpha * hp(hr - tm)) * 255), 0, 255) / 255


def psnr(mse):
    return float(-10 * np.log10(mse))


def scene_paths(scene):
    root = PREP / f"{scene}_area4"
    index = json.loads((root / f"teacher_index_{BASE_TAG}_seed{SEED}.json").read_text())
    man = json.loads((root / f"manifest_train_ready_seed{SEED}.json").read_text())
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in man["observations"]}
    return root, index, obs


def calibrate(scene, gains):
    root, index, obs = scene_paths(scene)
    entries = index["entries"][::max(1, len(index["entries"]) // 60)][:60]
    pairs = [(rd(root / e["relative_path"]), rd(root / obs[(e["camera"], int(e["frame"]))]["hr_path"])) for e in entries]
    grid = np.round(np.arange(0.0, 1.001, 0.02), 3)
    rows = []
    for a in grid:
        d = f = lo = 0.0
        for tm, hr in pairs:
            t = mix(tm, hr, a) if a else tm
            e = t - hr
            d += float((hp(e) ** 2).mean())
            f += float((e ** 2).mean())
            lo += float(((e - hp(e)) ** 2).mean())
        n = len(pairs)
        rows.append(dict(alpha=float(a), detail_psnr=psnr(d / n), full_psnr=psnr(f / n), low_psnr=psnr(lo / n)))
    base = rows[0]
    out = dict(scene=scene, samples=len(pairs), base=base, curve=rows, choice={})
    for g in gains:
        target = base["detail_psnr"] + g
        ok = [r for r in rows if r["detail_psnr"] >= target]
        if not ok:
            raise SystemExit(f"gain {g} dB not reachable")
        lo_r = max((r for r in rows if r["detail_psnr"] < target), key=lambda r: r["alpha"])
        hi_r = min(ok, key=lambda r: r["alpha"])
        t = (target - lo_r["detail_psnr"]) / (hi_r["detail_psnr"] - lo_r["detail_psnr"])
        alpha = lo_r["alpha"] + t * (hi_r["alpha"] - lo_r["alpha"])
        out["choice"][str(g)] = dict(alpha=round(float(alpha), 4))
    # verify the chosen alphas on the sample (after 8-bit rounding)
    for g, c in out["choice"].items():
        d = f = lo = 0.0
        for tm, hr in pairs:
            e = mix(tm, hr, c["alpha"]) - hr
            d += float((hp(e) ** 2).mean()); f += float((e ** 2).mean()); lo += float(((e - hp(e)) ** 2).mean())
        n = len(pairs)
        c.update(detail_gain=psnr(d / n) - base["detail_psnr"], full_gain=psnr(f / n) - base["full_psnr"], low_gain=psnr(lo / n) - base["low_psnr"])
    return out


def _write(args):
    src_t, src_h, dst, alpha = args
    t = mix(rd(src_t), rd(src_h), alpha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    Image.fromarray((t * 255).astype(np.uint8)).save(tmp, compress_level=1)
    tmp.replace(dst)
    return sha(dst)


def write(scene, alpha, tag, workers):
    root, index, obs = scene_paths(scene)
    out_index = root / f"teacher_index_{tag}_seed{SEED}.json"
    if out_index.exists():
        raise FileExistsError(f"completed teacher preserved: {out_index}")
    jobs, entries = [], []
    for e in index["entries"]:
        key = (e["camera"], int(e["frame"]))
        rel = f"{tag}/{e['camera']}/{int(e['frame']):04d}.png"
        jobs.append((root / e["relative_path"], root / obs[key]["hr_path"], root / rel, alpha))
        entries.append(dict(e, relative_path=rel))
    with cf.ProcessPoolExecutor(workers) as ex:
        shas = list(ex.map(_write, jobs, chunksize=16))
    for e, s in zip(entries, shas):
        e["sha256"] = s
    new = dict(index, entries=entries, oracle=True, HR_quality_evaluation=False,
               input_policy=f"DIAGNOSTIC: matched teacher {BASE_TAG} + {alpha} * HP(HR - teacher), HP = x - Gaussian(sigma 2); "
                            "reads training-view HR",
               teacher_config=dict(kind="diagnostic_teacher_dose", tag=tag, base_tag=BASE_TAG, alpha=alpha, hp_sigma=2.0,
                                   base_index=dict(path=str((root / f"teacher_index_{BASE_TAG}_seed{SEED}.json").relative_to(ROOT)),
                                                   sha256=sha(root / f"teacher_index_{BASE_TAG}_seed{SEED}.json")),
                                   code=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(Path(__file__)))))
    out_index.write_text(json.dumps(new, indent=1))
    return dict(index=str(out_index.relative_to(ROOT)), images=len(entries))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("calibrate", "write"))
    p.add_argument("--scene", required=True)
    p.add_argument("--gains", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--alpha", type=float)
    p.add_argument("--tag")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    if a.mode == "calibrate":
        r = calibrate(a.scene, a.gains)
        print(json.dumps(dict(base=r["base"], choice=r["choice"], samples=r["samples"]), indent=1))
        out = ROOT / "output/dynamic_sr_physlr_20261016/teacher_dose" / f"{a.scene}_calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=1))
    else:
        if a.alpha is None or not a.tag:
            raise SystemExit("write needs --alpha and --tag")
        print(json.dumps(write(a.scene, a.alpha, a.tag, a.workers)))


if __name__ == "__main__":
    main()
