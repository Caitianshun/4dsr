"""Exploratory compressed-LR track (2026-10-13): are per-view SR teachers still sufficient when the LR camera streams are
H.264-compressed, and does the multi-view 4D fit remove codec error? Local controller steps only; training runs through the
unchanged cullretrain.py lr / pipeline.py (remote or local).

  derive    data/dynamic_sr/full_time_20261007/prepared/<scene>_h264qp<QP>/: every camera's LR frames 0..299 encoded as one
            30 fps libx264 stream (constant QP, GOP 30, no B-frames, preset medium, yuv420p; ffmpeg 7.1 binary of the
            registered `ani` conda env used only as a codec tool) and decoded back to RGB PNGs; hr/ hard-linked; manifest =
            original with every observation's lr_sha256 replaced and a `compression_derivation` block. Splits unchanged
            (12 training cameras, cam00 test). /usr/bin/python3 -I.
  init      unmodified full_initialization.generate on the derived manifest (pure LR, train cameras x frames 0/150/299).
            Interpreter /home/cai_tianshun/Project/4dgs/.venv/bin/python with CUDA hidden (matches the recorded cv2 4.10).
  register  LR plan = the original 12-camera prefix plan of the same scene/seed with only `data` rebuilt by the
            validate_manifest formulas (train-LR digest, camera centre, 1.1 x max radius) and a `compression` note;
            `configuration` byte-identical (checked). /usr/bin/python3 -I.
  tasks     pipeline.py task table: author coarse replay (--coarse-only) then the gateaff fine stage, as in the holdout flow.
No original file is modified; derived directories refuse to overwrite different content.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
SCENES = ("meetroom_discussion", "cook_spinach", "meetroom_vrheadset", "cut_roasted_beef")  # last two added 2026-10-14
SEEDS = (20261007, 20261008)
QPS = (27, 37)
FFMPEG = Path("/home/cai_tianshun/miniconda3/envs/ani/bin/ffmpeg")
GENERATOR = ROOT / "experiments/dynamic_sr_multiview_footprint_20261007/full_initialization.py"
ORIGINAL_PREFIX = ROOT / "output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes"
LR_NS = ROOT / "output/dynamic_sr_cull_retrain_20261009/wave1/lr/compress_20261013"
SHORT = {"meetroom_discussion": "Discussion", "cook_spinach": "Cook", "meetroom_vrheadset": "VRHeadset", "cut_roasted_beef": "Cut"}
FRAMES = (0, 150, 299)
CODEC = dict(encoder="libx264", preset="medium", rate_control="constant QP", gop=30, b_frames=0, pix_fmt="yuv420p", fps=30)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def entry(path: Path) -> dict:
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha(path)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def derived_directory(scene: str, qp: int) -> Path:
    if scene not in SCENES or qp not in QPS:
        raise ValueError(f"unregistered scene/QP {scene} {qp}")
    return PREPARED / f"{scene}_h264qp{qp}"


def write_new(path: Path, value: dict) -> Path:
    text = json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"refusing to change an existing file: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


def codec_roundtrip(frames: list[Path], qp: int, mp4: Path):
    import numpy as np
    from PIL import Image
    images = [np.asarray(Image.open(p).convert("RGB")) for p in frames]
    h, w = images[0].shape[:2]
    enc = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(CODEC["fps"]), "-i", "-", "-c:v", CODEC["encoder"], "-preset", CODEC["preset"], "-qp", str(qp),
           "-g", str(CODEC["gop"]), "-bf", str(CODEC["b_frames"]), "-pix_fmt", CODEC["pix_fmt"], str(mp4)]
    subprocess.run(enc, input=np.stack(images).tobytes(), check=True)
    dec = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = np.frombuffer(subprocess.run(dec, check=True, capture_output=True).stdout, np.uint8).reshape(-1, h, w, 3)
    if len(out) != len(images):
        raise ValueError("decoded frame count differs")
    clean = np.stack(images).astype(np.float64)
    psnr = float(np.mean([10 * math.log10(255.0 ** 2 / max(((o.astype(np.float64) - c) ** 2).mean(), 1e-12)) for o, c in zip(out, clean)]))
    return out, psnr


def derive(scene: str, qp: int) -> dict:
    from PIL import Image
    source = PREPARED / scene / "manifest.json"
    original = read(source)
    dest = derived_directory(scene, qp)
    if dest.is_symlink():
        raise ValueError("derived directory must not be a symlink")
    manifest = copy.deepcopy(original)
    rows = {}
    for row in manifest["observations"]:
        rows.setdefault(row["camera_id"], []).append(row)
    streams = {}
    for camera in sorted(rows):
        cam_rows = sorted(rows[camera], key=lambda r: int(r["frame_index"]))
        if [int(r["frame_index"]) for r in cam_rows] != list(range(300)):
            raise ValueError(f"{camera}: expected frames 0..299")
        stream_dir = dest / "streams"
        stream_dir.mkdir(parents=True, exist_ok=True)
        mp4 = stream_dir / f"{camera}.mp4"
        decoded, psnr = codec_roundtrip([PREPARED / scene / r["lr_path"] for r in cam_rows], qp, mp4)
        for r, img in zip(cam_rows, decoded):
            path = dest / r["lr_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.png")
            Image.fromarray(img).save(tmp)
            new = sha(tmp)
            if path.exists() and sha(path) != new:
                raise ValueError(f"refusing to replace different LR: {path}")
            tmp.replace(path)
            r["lr_sha256"] = new
        streams[camera] = dict(stream=entry(mp4), kbps=mp4.stat().st_size * 8 / (300 / CODEC["fps"]) / 1000, lr_psnr_vs_clean_dB=psnr)
        print(scene, qp, camera, f"{streams[camera]['kbps']:.0f} kbps", f"LR PSNR {psnr:.2f}", flush=True)
    # hard-link HR (same inode, registered HR SHA stays valid)
    links = 0
    for row in manifest["observations"]:
        src, dst = PREPARED / scene / row["hr_path"], dest / row["hr_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src, dst):
                raise ValueError(f"different HR present: {dst}")
            continue
        os.link(src, dst)
        links += 1
    version = subprocess.run([str(FFMPEG), "-hide_banner", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    manifest["compression_derivation"] = dict(
        schema="h264_lr_stream_compression_v1", qp=qp, codec=CODEC, ffmpeg=str(FFMPEG), ffmpeg_version=version,
        original_manifest=entry(source), generation_source=entry(Path(__file__)), streams=streams,
        modified_original_fields=["observations[*].lr_sha256 (LR pixels replaced by the decoded stream)"],
        hr_layout="hard links to the original HR images", splits_unchanged=True,
        purpose="exploratory: realistic transmission degradation of the LR camera streams; not the registered benchmark protocol")
    write_new(dest / "manifest.json", manifest)
    receipt = dict(scene=scene, qp=qp, manifest=entry(dest / "manifest.json"), hr_links_created=links,
                   mean_kbps=sum(s["kbps"] for s in streams.values()) / len(streams),
                   mean_lr_psnr_vs_clean_dB=sum(s["lr_psnr_vs_clean_dB"] for s in streams.values()) / len(streams))
    write_new(dest / "derivation_receipt.json", receipt)
    return receipt


def init(scene: str, qp: int, seed: int) -> dict:
    sys.path.insert(0, str(GENERATOR.parent))
    spec = importlib.util.spec_from_file_location("compress_unmodified_full_initialization", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = derived_directory(scene, qp) / "manifest.json"
    receipt = module.generate(manifest, seed, 1)
    ready = read(manifest.parent / f"manifest_train_ready_seed{seed}.json")
    report = read(manifest.parent / ready["initialization"]["report_path"])
    if report["used_HR"] or report["used_heldout_pixels"] or report["CUDA_calls"]:
        raise ValueError("initializer boundary violated")
    out = dict(status=receipt["status"], seed=seed, point_count=report["point_count"], generator=entry(GENERATOR),
               interpreter=sys.executable, ready_manifest=entry(manifest.parent / f"manifest_train_ready_seed{seed}.json"))
    write_new(manifest.parent / f"compression_initialization_seed{seed}.json", out)
    return out


def tag(scene: str, qp: int, seed: int) -> str:
    return f"{SHORT[scene]}{str(seed)[-2:]}_qp{qp}"


def plan_dir(scene: str, qp: int, seed: int) -> Path:
    return LR_NS / "registry" / f"{scene}_h264qp{qp}" / f"seed_{seed}"


def register(scene: str, qp: int, seed: int) -> dict:
    root = derived_directory(scene, qp)
    original_path = ORIGINAL_PREFIX / scene / f"seed_{seed}" / "config.json"
    original = read(original_path)
    m = read(root / f"manifest_train_ready_seed{seed}.json")
    train = m["splits"]["train"]
    if train != original["data"]["train_cameras"] or m["splits"]["test"] != ["cam00"]:
        raise ValueError("training split differs from the original plan")
    init_ = m["initialization"]
    if init_["seed"] != seed or not init_["train_only_lr"] or sha(root / init_["npz_path"]) != init_["npz_sha256"]:
        raise ValueError("seed-specific pure-LR initialization required")
    rows = [r for r in m["observations"] if r["split"] == "train"]
    inputs = []
    for r in rows:
        p = root / r["lr_path"]
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p) != r["lr_sha256"]:
            raise ValueError(f"illegal or changed LR input {p}")
        inputs.append(dict(camera=r["camera_id"], frame=int(r["frame_index"]), path=str(p.relative_to(ROOT)), sha256=r["lr_sha256"]))
    centers = [[float(m["cameras"][c]["c2w"][i][3]) for i in range(3)] for c in train]
    center = [sum(x[i] for x in centers) / len(centers) for i in range(3)]
    extent = 1.1 * max(math.sqrt(sum((x[i] - center[i]) ** 2 for i in range(3))) for x in centers)
    if max(abs(a - b) for a, b in zip(center, original["data"]["train_camera_center"])) > 1e-9 or \
            abs(extent - original["data"]["cameras_extent"]) > 1e-9:
        raise ValueError("camera geometry differs from the original plan")
    center, extent = original["data"]["train_camera_center"], original["data"]["cameras_extent"]  # same cameras: keep bytes
    data = copy.deepcopy(original["data"])
    data.update(manifest=entry(root / f"manifest_train_ready_seed{seed}.json"), original_manifest=entry(root / "manifest.json"),
                initialization_receipt=entry(root / f"initialization_seed_{seed}/complete.json"), initial_points=entry(root / init_["npz_path"]),
                train_cameras=train, training_observations=len(rows), training_LR_sources_sha256=digest(inputs),
                train_camera_center=center, cameras_extent=extent)
    if "training_LR_bytes_SHA_verified" in data:  # absent from e.g. the N3DV dev Cook prefix plan; do not add new keys
        data["training_LR_bytes_SHA_verified"] = len(inputs)
    degradation = f"LR streams H.264 constant QP {qp}, GOP 30 (exploratory compressed-LR track)"
    if "domain_protocol" in data:
        data["domain_protocol"] = dict(data["domain_protocol"], seed=seed, degradation=degradation)
    else:  # e.g. the N3DV dev Cook prefix plan has no domain_protocol block
        data["degradation"] = degradation
    plan = copy.deepcopy(original)
    plan["status"] = f"registered_h264qp{qp}_LR_parent_pending_GPU"
    plan["data"] = data
    plan["author_adaptations"] = original["author_adaptations"] + [
        f"Exploratory compressed-LR track (2026-10-13): training LR replaced by H.264 QP {qp} decoded frames; configuration unchanged."]
    plan["compression"] = dict(qp=qp, derivation=entry(root / "derivation_receipt.json"), original_plan=entry(original_path),
                               registry_source=entry(Path(__file__)))
    if plan["configuration"] != original["configuration"]:
        raise ValueError("configuration changed")
    write_new(plan_dir(scene, qp, seed) / "config.json", plan)
    return dict(plan=str(plan_dir(scene, qp, seed).relative_to(ROOT)), train_LR=len(inputs))


def tasks(pairs, out: Path) -> list:
    table = []
    for scene, qp, seed in pairs:
        t = tag(scene, qp, seed)
        plan = str(plan_dir(scene, qp, seed).relative_to(ROOT))
        coarse = LR_NS / "runs" / f"{t}_coarse"
        fine = LR_NS / "runs" / f"{t}_gateaff"
        table.append(dict(name=f"lr_coarse_{t}", deps=[], done=str((coarse / "complete.json").relative_to(ROOT)),
                          argv=["lr", "--parent", plan, "--coarse-only", "--arm", f"h264qp{qp}_author_coarse",
                                "--out", str(coarse.relative_to(ROOT))]))
        table.append(dict(name=f"lr_fine_{t}", deps=[f"lr_coarse_{t}"], done=str((fine / "complete.json").relative_to(ROOT)),
                          argv=["lr", "--parent", plan, "--fine0", str((coarse / "checkpoint_fine_00000.pt").relative_to(ROOT)),
                                "--arm", "gateaff", "--prune-gate", "none", "--affine", "--out", str(fine.relative_to(ROOT))]))
    if out.exists() and read(out) != table:
        raise ValueError(f"refusing to change an existing task table {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + "\n")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("derive", "init", "register", "tasks"))
    p.add_argument("--scene", choices=SCENES, default="meetroom_discussion")
    p.add_argument("--qps", type=int, nargs="+", default=[27])
    p.add_argument("--seed", type=int, choices=SEEDS, default=20261007)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.mode == "tasks":
        out = (ROOT / a.out).resolve()
        if not out.is_relative_to(LR_NS):
            raise SystemExit("task tables must sit inside the synced LR namespace")
        print(json.dumps([t["name"] for t in tasks([(a.scene, q, a.seed) for q in a.qps], out)]))
        return
    for qp in a.qps:
        fn = dict(derive=lambda: derive(a.scene, qp), init=lambda: init(a.scene, qp, a.seed),
                  register=lambda: register(a.scene, qp, a.seed))[a.mode]
        print(json.dumps(fn(), ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main()
