"""HR frames for fine-tuning a pixel-area-matched SwinIR teacher (Claude, 4dsr-7c, 2026-10-17). CPU only.

Sources: local multi-view videos that are NOT evaluation domains (no N3DV kitchen scenes, no MeetRoom): Immersive
(Welder, Flames, Alexa Meade Exhibit), DeskGames (cube, dominos, uno), SelfCap (corgi, hair), Long360, ImViD opera,
VRU GZ / DG. Per source: 5 cameras x 6 times, decoded with ffmpeg, halved with cv2.INTER_AREA (the half rung of the
2026-10-16 trait screening, sharp to the pixel grid on these sources), cropped to a multiple of 8, written as PNG to
data/dynamic_sr/area_teacher_20261017/hr/<source>/<cam>_<t>.png with an index (sha256) for the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/cai_tianshun/Project/4dsr")
D = ROOT / "data/dynamic_sr"
HX = D / "highres_expansion_20261010"
SC = D / "selfcap_native_x4_20261010/raw"
FFMPEG = "/home/cai_tianshun/miniconda3/envs/ani/bin/ffmpeg"
OUT = D / "area_teacher_20261017"
SOURCES = {  # name: (directory, pattern, cameras, t0 seconds)
    "imm_welder": (HX / "raw/immersive/01_Welder/01_Welder", "camera_{:04d}.mp4", [1, 10, 20, 30, 40], 0.5),
    "imm_flames": (HX / "raw/immersive/02_Flames/02_Flames", "camera_{:04d}.mp4", [1, 10, 20, 30, 40], 0.5),
    "imm_exhibit": (HX / "raw/immersive/09_Alexa_Meade_Exhibit/09_Alexa_Meade_Exhibit", "camera_{:04d}.mp4", [1, 10, 20, 30, 40], 0.5),
    "desk_cube": (HX / "raw/deskgames/cube/videos", "cam{:02d}.mp4", [1, 5, 9, 13, 17], 30.0),
    "desk_dominos": (HX / "raw/deskgames/dominos/videos", "cam{:02d}.mp4", [1, 5, 9, 13, 17], 30.0),
    "desk_uno": (HX / "raw/deskgames/uno/videos", "cam{:02d}.mp4", [1, 5, 9, 13, 17], 30.0),
    "selfcap_corgi": (SC / "corgi-release/corgi-release/videos", "{:04d}.mp4", [1, 6, 11, 16, 21], 3.0),
    "selfcap_hair": (SC / "hair-release/hair-release/videos", "{:04d}.mp4", [1, 6, 11, 16, 21], 3.0),
    "long360": (HX / "long360/raw/long360", "view_{:03d}.mp4", [1, 8, 15, 22, 29], 10.0),
    "imvid_opera": (HX / "imvid/raw/scene1_opera", "cam{:02d}.mp4", [1, 10, 20, 30, 38], 0.5),
    "vru_gz": (D / "vru_gz_dg_20261016/raw/GZ/videos/train", "{}.mp4", [1, 8, 15, 22, 29], 0.5),
    "vru_dg": (D / "vru_gz_dg_20261016/raw/DG/Basketball_dg", "{}.mp4", [1, 8, 15, 22, 29], 0.5),
}
OFFSETS = [0.0, 1.3, 2.6, 3.9, 5.2, 6.5]


def probe(path):
    r = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def decode(path, t, w, h):
    cmd = [FFMPEG, "-v", "error", "-threads", "4", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(buf, np.uint8)
    if a.size != w * h * 3:
        return None
    return a.reshape(h, w, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="*", default=list(SOURCES))
    a = p.parse_args()
    index = {}
    for name in a.sources:
        root, pat, cams, t0 = SOURCES[name]
        for cam in cams:
            path = root / pat.format(cam)
            if not path.exists():
                print("missing", path, flush=True)
                continue
            w, h = probe(path)
            for k, dt in enumerate(OFFSETS):
                fr = decode(path, t0 + dt, w, h)
                if fr is None:
                    continue
                half = cv2.resize(fr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                hh, ww = (half.shape[0] // 8) * 8, (half.shape[1] // 8) * 8
                half = half[:hh, :ww]
                out = OUT / "hr" / name / f"{cam}_{k}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), cv2.cvtColor(half, cv2.COLOR_RGB2BGR))
                index[str(out.relative_to(OUT))] = dict(source=name, video=str(path.relative_to(ROOT)), t=t0 + dt,
                                                         size=[ww, hh], sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        print(name, "frames", sum(1 for v in index.values() if v["source"] == name), flush=True)
    (OUT / "index.json").write_text(json.dumps(dict(purpose="area-matched SwinIR fine-tuning (non-evaluation sources)",
                                                    generator=str(Path(__file__).relative_to(ROOT)),
                                                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                    frames=index), indent=1))
    print("frames", len(index))


if __name__ == "__main__":
    main()
