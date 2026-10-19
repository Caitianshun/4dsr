"""Blind-review material for the area4 x4 operating point (AGENTS.md 0.3 acceptability gate; Claude 4dsr-7c, 2026-10-18).

  br_build.py sisr  --positions POS.json [--positions POS2.json] --out-dir DIR
                                                               CPU: matched SwinIR teacher applied to the cam00 LR of every
                                                               registered cam00 frame (single-image SR reference, diagnostic only)
  br_build.py build --positions POS.json [--positions POS2.json] --renders DIR --sisr DIR --review-dir DIR --key KEY.json
                    [--seed-base 20261018]

Several positions files are read in the order given; window numbers continue across them (positions_v1.json windows 1-24, then
the MeetRoom Discussion addendum positions_v1_disc.json windows 25-30).

Windows come from the registered positions file (HR-only, fixed before any method crop existed). Per window one composite PNG:
top = the clean HR crop ("HR"), bottom = the candidates in a shuffled order labelled A, B, C, ...; each crop shown at 3x with
nearest-neighbour zoom, no other text. Candidates:
  cam00 windows (novel view): LR bicubic x4 (anchor), matched teacher on the cam00 LR (single-image SR reference), and the naive /
      areaTm / w4 renders (render_dump.py npz, cam00);
  training-camera windows (teacher review): LR bicubic x4, off-the-shelf SwinIR cache (sr_swinir_x4), matched SwinIR cache
      (sr_swinir_x4_areaft).
The order of window i (1-based, file traversal order) is numpy default_rng(seed_base + i).permutation(n). The label -> candidate key
and the input sha256 go to KEY.json, which must stay outside the review material. Composites are named w01.png, w02.png, ... and
carry no scene, frame or method name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
FT = ROOT / "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1"
SHORT = dict(cook_spinach="Cook07", cut_roasted_beef="Cut07", meetroom_vrheadset="VRHeadset07", meetroom_discussion="Discussion07")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
Z = 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def windows(paths):
    """[(index, scene, camera, frame, name, box)] in file traversal order, numbering continued across the positions files."""
    out = []
    for path in paths:
        pos = json.loads(Path(path).read_text())
        for scene, cams in pos.items():
            if scene.startswith("_"):
                continue
            for cam, frames in cams.items():
                for f, boxes in frames.items():
                    for name, box in boxes.items():
                        out.append((len(out) + 1, scene, cam, int(f), name, tuple(int(v) for v in box)))
    return out


def obs_of(scene):
    m = json.loads((PREP / f"{scene}_area4/manifest.json").read_text())
    return {(o["camera_id"], int(o["frame_index"])): o for o in m["observations"]}


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def bicubic_up(lr):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    y = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    return (y[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def cmd_sisr(a):
    import sys
    import torch
    sys.path.insert(0, str(HERE))
    import hat_screen as HS
    torch.set_num_threads(16)
    s = json.loads((FT / "summary.json").read_text())
    net = HS.swinir_net("cpu", weights=(ROOT / s["weights"]["path"], s["weights"]["sha256"]))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    rec = {}
    for i, scene, cam, f, name, box in windows(a.positions):
        if cam != "cam00" or (scene, f) in done:
            continue
        done.add((scene, f))
        o = obs_of(scene)[("cam00", f)]
        lp = PREP / f"{scene}_area4" / o["lr_path"]
        if sha(lp) != o["lr_sha256"]:
            raise SystemExit(f"LR changed: {lp}")
        p = out / f"{scene}_cam00_{f:04d}.png"
        if not p.exists():
            x = torch.from_numpy(rgb(lp).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            with torch.no_grad():
                y = HS.run_swinir(net, x).clamp(0, 1)
            Image.fromarray((y[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)).save(p)
        rec[p.name] = dict(lr=str(lp.relative_to(ROOT)), lr_sha256=o["lr_sha256"], sha256=sha(p))
        print(p, flush=True)
    stem = "_".join(Path(q).stem for q in a.positions)
    (out / f"sisr_index_{stem}.json").write_text(json.dumps(dict(teacher=dict(path=s["weights"]["path"], sha256=s["weights"]["sha256"]),
                                                          code=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),
                                                                    sha256=sha(__file__)), frames=rec), indent=1))


def render_frame(rdir, f):
    d = np.load(rdir / f"f{f:04d}.npz")
    cams = [str(c) for c in d["cameras"]]
    return np.ascontiguousarray(d["rgb"][cams.index("cam00")].transpose(1, 2, 0))


def cmd_build(a):
    rdir, sdir = Path(a.renders), Path(a.sisr)
    rev = Path(a.review_dir)
    rev.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(FONT, 26)
    key = dict(positions=[dict(path=str(Path(q).resolve().relative_to(ROOT)), sha256=sha(q)) for q in a.positions], seed_base=a.seed_base,
               code=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(__file__)), windows=[])
    obs_cache = {}
    for i, scene, cam, f, name, (x0, y0, x1, y1) in windows(a.positions):
        obs = obs_cache.setdefault(scene, obs_of(scene))
        o = obs[(cam, f)]
        root = PREP / f"{scene}_area4"
        hp, lp = root / o["hr_path"], root / o["lr_path"]
        if sha(hp) != o["hr_sha256"] or sha(lp) != o["lr_sha256"]:
            raise SystemExit(f"HR/LR changed: {scene} {cam} {f}")
        hr = rgb(hp)
        cands = [("bicubic", bicubic_up(rgb(lp)), str(lp.relative_to(ROOT)))]
        if cam == "cam00":
            sp = sdir / f"{scene}_cam00_{f:04d}.png"
            cands.append(("sisr_matched_on_cam00_LR", rgb(sp), str(sp.resolve().relative_to(ROOT))))
            for arm in ("naive", "areaTm", "w4"):
                rp = rdir / f"{SHORT[scene]}_area4_{arm}"
                cands.append((f"render_{arm}", render_frame(rp, f), str((rp / f"f{f:04d}.npz").resolve().relative_to(ROOT))))
        else:
            for tag, label in (("sr_swinir_x4", "teacher_offtheshelf"), ("sr_swinir_x4_areaft", "teacher_matched")):
                tp = root / tag / cam / f"{f:04d}.png"
                cands.append((label, rgb(tp), str(tp.relative_to(ROOT))))
        for _, img, src in cands:
            if img.shape != hr.shape:
                raise SystemExit(f"size mismatch {src}: {img.shape} vs {hr.shape}")
        perm = np.random.default_rng(a.seed_base + i).permutation(len(cands))
        cw, gap, lab = 128 * Z, 16, 40
        n = len(cands)
        W = n * cw + (n + 1) * gap
        H = lab + cw + gap + lab + cw + gap
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        crop = lambda img: Image.fromarray(img[y0:y1, x0:x1]).resize((cw, cw), Image.NEAREST)  # noqa: E731
        hx = (W - cw) // 2
        d.text((hx, 4), "HR", fill=(0, 0, 0), font=font)
        sheet.paste(crop(hr), (hx, lab))
        rows = []
        for j, k in enumerate(perm):
            letter = chr(ord("A") + j)
            x = gap + j * (cw + gap)
            d.text((x, lab + cw + gap + 4), letter, fill=(0, 0, 0), font=font)
            sheet.paste(crop(cands[k][1]), (x, lab + cw + gap + lab))
            rows.append(dict(label=letter, candidate=cands[k][0], source=cands[k][2]))
        out = rev / f"w{i:02d}.png"
        sheet.save(out)
        key["windows"].append(dict(file=out.name, sha256=sha(out), scene=scene, camera=cam, frame=f, window=name,
                                   box=[x0, y0, x1, y1], kind="novel_view" if cam == "cam00" else "teacher", labels=rows))
        print(out.name, scene, cam, f, name, n, flush=True)
    kp = Path(a.key)
    if kp.exists():
        raise SystemExit(f"key exists: {kp}")
    kp.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("sisr", "build"))
    p.add_argument("--positions", action="append", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--renders")
    p.add_argument("--sisr")
    p.add_argument("--review-dir")
    p.add_argument("--key")
    p.add_argument("--seed-base", type=int, default=20261018)
    a = p.parse_args()
    dict(sisr=cmd_sisr, build=cmd_build)[a.mode](a)


if __name__ == "__main__":
    main()
