#!/usr/bin/env python3
"""Build isolated upstream runtimes; disclose only camera/runtime adaptations."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    source, target = args.source.resolve(), args.target.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    expected = {"yang": "3fe460f1f2342526bccf6c01b63da125faf0fa94", "4c4d": "ed6a3cb69782c4147151b3898944bc38132bae00"}[args.method]
    if commit != expected or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Requires clean pinned author source")
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.so", "build", "*.egg-info"))
    changes, reasons = [], {}
    path = target / "simple-knn/simple_knn.cu"
    before = path.read_text()
    after = "#include <cfloat>\n" + before
    path.write_text(after)
    changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/simple-knn/simple_knn.cu", tofile="b/simple-knn/simple_knn.cu"))
    reasons["simple-knn/simple_knn.cu"] = "Declare standard FLT_MAX explicitly for CUDA12.8 compilation; the numerical constant and kernel are unchanged."
    path = target / "utils/camera_utils.py"
    before = path.read_text()
    needle = "FoVx=cam_info.FovX, FoVy=cam_info.FovY,"
    replacement = ("FoVx=(2 * np.arctan(orig_w / (2 * cam_info.fl_x)) if cam_info.fl_x > 0 else cam_info.FovX), "
                   "FoVy=(2 * np.arctan(orig_h / (2 * cam_info.fl_y)) if cam_info.fl_y > 0 else cam_info.FovY),")
    if before.count(needle) != 1:
        raise ValueError("Unexpected upstream camera interface")
    after = before.replace(needle, replacement)
    path.write_text(after)
    changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/utils/camera_utils.py", tofile="b/utils/camera_utils.py"))
    reasons["utils/camera_utils.py"] = "Native Blender reader sets FoV=-1 when explicit intrinsics exist. Supply FoV derived from the same focal lengths so CUDA covariance projection and projection matrix use consistent intrinsics; no learned correction or changed rays."
    if args.method == "yang":
        for relative in ["utils/data_utils.py", "scene/dataset_readers.py"]:
            path = target / relative
            before = path.read_text()
            if "dtype=np.byte" not in before:
                raise ValueError("Unexpected native image dtype interface")
            after = before.replace("dtype=np.byte", "dtype=np.uint8")
            path.write_text(after)
            changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/"+relative, tofile="b/"+relative))
            reasons[relative] = "Pillow compatibility: signed int8 RGB arrays are rejected by the installed Pillow. uint8 stores exactly the same raw 8-bit RGB bytes; no resampling or color operation changed. See dtype_byte_equivalence.json."
    for relative in ["train.py", "render.py"]:
        path = target / relative
        if not path.exists():
            continue
        before = path.read_text()
        after = before.replace("torch.load(checkpoint)", "torch.load(checkpoint, weights_only=False)")
        if after != before:
            path.write_text(after)
            changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/"+relative, tofile="b/"+relative))
            reasons[relative] = "Explicit trusted local native checkpoint loading for PyTorch >=2.6 default change; no training objective/optimizer/schedule change."
    (target / "adapter.patch").write_text("".join(changes))
    record = {"method": args.method, "upstream_commit": commit, "upstream_path": str(source), "runtime_path": str(target),
              "adaptation_reasons": reasons, "original_sha256": {r: sha(source/r) for r in reasons},
              "generated_sha256": {r: sha(target/r) for r in reasons}, "patch_sha256": sha(target/"adapter.patch"),
              "core_model_loss_optimizer_schedule_modified": False, "GPU_calls": 0,
              "native_training_verified": False}
    (target / "adapter_identity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["yang", "4c4d"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    build(p.parse_args())
