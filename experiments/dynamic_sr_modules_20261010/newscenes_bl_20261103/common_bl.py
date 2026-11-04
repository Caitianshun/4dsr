"""CPU utilities for the two confirmation scenes; no GPU or job submission."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / "output/dynamic_sr_modules_20261010/newscenes_bl_20261103"
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
REG = ROOT / "output/dynamic_sr_modules_20261010/newscenes_reg_20261103"
OLD = ROOT / "experiments/dynamic_sr_modules_20261010/area4_baselines_20261024"
STATIC = ROOT / "experiments/dynamic_sr_modules_20261010/static_sr_v2_20261026"
SR4D = ROOT / "experiments/dynamic_sr_modules_20261010/sr4d_area4_20261028"
SCENES = {"sear_steak": "sear_steak", "flame_salmon": "flame_salmon_1"}
SEED = 20261007
METHOD_NAMES = {"cem": "cem_area4", "sr4d": "sr4d_author_area4",
                "splatsure": "splatsure_wu_v2", "iesrgs": "iesrgs_wu_v2"}
REMOTE = {"a100": "/home/ubuntu/3DGS/4dsr", "cts": "/home/cts/Project/4DSR"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path):
    return local(path).relative_to(ROOT).as_posix()


def sha(path):
    h = hashlib.sha256()
    with local(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": rel(path), "sha256": sha(path)}


def read(path):
    return json.loads(local(path).read_text())


def bounded(path):
    p = local(path)
    require(p.resolve().is_relative_to(OUT.resolve()) or p.resolve().is_relative_to(CODE.resolve()),
            "Output must be in newscenes_bl_20261103")
    require(not any(x.is_symlink() for x in (p, *p.parents)), "Symlink output refused")
    return p


def text_new(path, text):
    p = bounded(path)
    if p.exists():
        require(p.read_text() == text, "Preserve existing file: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        stream.write(text)
    return p


def write_new(path, value):
    return text_new(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, local(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_path(scene):
    require(scene in SCENES, "Unsupported scene")
    return PREP / (scene + "_area4") / f"manifest_train_ready_seed{SEED}.json"


def geometry(m):
    """Recompute from training poses, retaining their real camera identifiers."""
    train = sorted(m["splits"]["train"])
    require("cam00" not in train and train, "Training cameras required")
    centers = []
    for c in train:
        pose = np.asarray(m["cameras"][c]["w2c"], np.float64)
        require(pose.shape == (4, 4) and np.isfinite(pose).all(), "Invalid pose")
        center = np.linalg.inv(pose)[:3, 3]
        require(np.allclose(center, np.asarray(m["cameras"][c]["c2w"])[:3, 3],
                            rtol=0, atol=1e-10), "w2c/c2w disagree")
        centers.append(center.tolist())
    # Explicit sequential sums also match the historical Python 3.10 producer.
    mean = []
    for axis in range(3):
        total = 0.0
        for center in centers:
            total += center[axis]
        mean.append(total / len(centers))
    radius = max(math.sqrt(sum((v - c) ** 2 for v, c in zip(row, mean))) for row in centers)
    require(math.isfinite(radius) and radius > 0, "Degenerate scene")
    return {"train_cameras": train, "train_camera_center": mean, "cameras_extent": 1.1 * radius,
            "rule": "1.1 * max distance of actual training camera centers from their mean",
            "source": "inverse(manifest.cameras[train].w2c); held-out camera excluded"}


def validate_metadata(scene):
    mp = manifest_path(scene)
    m = read(mp)
    require(m["scene"] == scene and m["scale"] == 4, "Scene/scale mismatch")
    require(m["physical_lr_derivation"]["formation"] == "area4", "Area4 required")
    require(m["resolutions"] == {"hr": [1344, 1008], "lr": [336, 252]}, "Resolution mismatch")
    require(m["frame_indices"] == list(range(300)), "All 300 frames required")
    require(m["splits"]["test"] == ["cam00"] and not m["splits"].get("dev"), "Split mismatch")
    require(set(m["cameras"]) == set(m["splits"]["train"]) | {"cam00"}, "Camera set mismatch")
    if scene == "flame_salmon":
        require("cam03" not in m["cameras"] and "cam17" not in m["cameras"], "Missing IDs must stay absent")
    expected = {(c, f) for c in m["cameras"] for f in range(300)}
    keys = [(r["camera_id"], r["frame_index"]) for r in m["observations"]]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Observation set incomplete/duplicate")
    for r in m["observations"]:
        require(r["camera_id"] in m["splits"][r["split"]], "Observation split mismatch")
        require(abs(r["time"] - r["frame_index"] / 300.0) < 1e-12, "Time mismatch")
    require(m["initialization"]["train_only_lr"], "Shared initializer must be training LR only")
    rp = REG / "lr/registry" / (scene + "_area4") / f"seed_{SEED}/config.json"
    reg = read(rp)
    require(reg["scene"] == scene and reg["seed"] == SEED, "Registry identity mismatch")
    require(reg["data"]["manifest"] == identity(mp), "Registry manifest SHA mismatch")
    require(reg["data"]["initial_points"]["sha256"] == m["initialization"]["npz_sha256"],
            "Registry initialization identity mismatch")
    g = geometry(m)
    require(abs(reg["data"]["cameras_extent"] - g["cameras_extent"]) < 1e-10
            and np.max(np.abs(np.asarray(reg["data"]["train_camera_center"])
                              - g["train_camera_center"])) < 1e-10, "Registry geometry mismatch")
    require(reg["configuration"]["OptimizationParams"]["batch_size"] ==
            (2 if scene == "sear_steak" else 4), "Author scene batch mismatch")
    return mp, m, rp, reg, g


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                rows.append({"field": path + "/" + key, "old": a.get(key), "new": b.get(key)})
            else:
                rows.extend(differences(a[key], b[key], path + "/" + key))
        return rows
    return [] if a == b else [{"field": path or "/", "old": a, "new": b}]
