#!/usr/bin/env python3
"""Prepare reviewable MeetRoom I/O patches, or explicitly build a new runtime.

Default writes small patches/identities only. --materialize copies source code
to the new deployment directory; no data, GPU, native loader or training runs.
Existing runtime/source directories are never changed. CEM already has exact
manifest CameraInfo support; its patch adds an initialization-provenance guard.
"""
import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil

from prepare_local_cem import COMMITS, HERE, ROOT, UPSTREAM, identity, sha


def changed_reader(method, text):
    if method == "localdygs":
        anchor = "def readColmapSceneInfo(path, images, eval, lod, frames_start_end=[0,300],llffhold=8):\n"
        add = ('    if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n'
               '        from meetroom_adapter import local_colmap_scene\n'
               '        return local_colmap_scene(path, frames_start_end, globals())\n')
    else:
        anchor = "def readColmapSceneInfoNeural3DVideo(path, images, eval, args):\n"
        add = ('    if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n'
               '        from meetroom_adapter import initialization_contract\n'
               '        initialization_contract(path, "cem")\n')
    if text.count(anchor) != 1:
        raise ValueError("Pinned native loader entry changed")
    return text.replace(anchor, anchor + add, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["localdygs", "cem", "both"], default="both")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    rows = []
    for method in (["localdygs", "cem"] if args.method == "both" else [args.method]):
        source = ROOT / ("third_party/localdygs_calibrated_runtime_v3_20261020" if method == "localdygs" else "third_party/cem4dgs_calibrated_runtime_v3_20261020")
        reader = source / "scene/dataset_readers.py"
        prior = json.loads((source / "protocol_interface_receipt.json").read_text())
        if prior.get("upstream_commit") != COMMITS[method] or not prior.get("core_losses_sampling_optimization_schedule_unchanged"):
            raise ValueError("Unrecognized or scientifically modified base runtime")
        if method == "localdygs":
            expected = sha(UPSTREAM[method] / "scene/dataset_readers.py")
        else:
            expected = next(x["runtime_sha256"] for x in prior["changes"] if x["path"] == "scene/dataset_readers.py")
        if sha(reader) != expected:
            raise ValueError("Pinned base runtime reader changed since its identity receipt")
        before = reader.read_text()
        after = changed_reader(method, before)
        ast.parse(after)
        changes = {"scene/dataset_readers.py": (before, after)}
        if method == "localdygs":
            scene_file = source / "scene/__init__.py"
            if sha(scene_file) != sha(UPSTREAM[method] / "scene/__init__.py"):
                raise ValueError("Pinned LocalDyGS Scene dispatch source identity changed")
            scene_before = scene_file.read_text()
            anchor = '        if os.path.exists(os.path.join(args.source_path, "sparse")):'
            if scene_before.count(anchor) != 1:
                raise ValueError("Pinned LocalDyGS Scene dispatch changed")
            replacement = ('        if os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST"):\n'
                           '            # This isolated runtime is explicitly for MeetRoom.\n'
                           '            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod, frames_start_end)\n'
                           '            dataset_type = "colmap"\n'
                           '        elif os.path.exists(os.path.join(args.source_path, "sparse")):')
            scene_after = scene_before.replace(anchor, replacement, 1)
            ast.parse(scene_after)
            changes["scene/__init__.py"] = (scene_before, scene_after)
        patch = HERE / f"patches/{method}_meetroom_manifest.patch"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text("".join(line for name, (old, new) in changes.items() for line in
                        difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="a/" + name, tofile="b/" + name)))
        destination = HERE / f"runtime/{method}_meetroom"
        if args.materialize:
            if destination.exists():
                raise FileExistsError("Refusing existing isolated runtime: " + str(destination))
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "build", "*.egg-info"))
            for name, (old, new) in changes.items():
                (destination / name).write_text(new)
            shutil.copy2(HERE / "meetroom_adapter.py", destination / "meetroom_adapter.py")
            receipt = json.loads((destination / "protocol_interface_receipt.json").read_text())
            receipt.update(meetroom_adapter_source=identity(HERE / "meetroom_adapter.py"),
                           meetroom_patch=identity(patch), source_runtime_reader=identity(reader),
                           native_GPU_acceptance="pending", core_losses_sampling_optimization_schedule_unchanged=True)
            (destination / "protocol_interface_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        rows.append(dict(method=method, patch=identity(patch), base_runtime=str(source), source_reader=identity(reader),
                         adapter=identity(HERE / "meetroom_adapter.py"), target_runtime=str(destination),
                         patched_paths=list(changes),
                         materialized=args.materialize, scientific_core_changes=False,
                         LocalDyGS_initialization="pcds/downsample_0_300.ply, or author's sparse/0/points3D.bin fallback with verified LR-only provenance",
                         CEM_initialization="colmap_0/sparse/0/points3D.ply frame0 training-LR sparse only",
                         native_GPU_acceptance="pending; no CUDA/train/remote action performed"))
    out = HERE / "notes/meetroom_adapter_patch_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema="meetroom_CPU_reviewable_patch_v1", entries=rows), indent=2) + "\n")
    print(json.dumps(dict(status="source_patches_ready_GPU_HOLD", patches=len(rows), materialized=args.materialize)))


if __name__ == "__main__":
    main()
