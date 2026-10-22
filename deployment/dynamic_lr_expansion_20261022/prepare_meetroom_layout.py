#!/usr/bin/env python3
"""Plan the author's LocalDyGS MeetRoom sparse layout; never run COLMAP/train.

Default plan only, with no asset copies. A later --execute copies the existing
verified frame-zero sparse files byte for byte into a new source directory and
freezes provenance using the author sparse-binary fallback. It never relabels
the N3DV dense union as a MeetRoom initialization.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from meetroom_adapter import read_manifest
from prepare_local_cem import HERE, identity, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-init", type=Path, required=True)
    parser.add_argument("--native-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Later CPU asset layout only; no GPU/training")
    args = parser.parse_args()
    if args.plan_only and args.execute:
        raise ValueError("Choose plan-only or execute")
    unused, value = read_manifest(args.manifest)
    sparse = args.native_init / "frame_0000/triangulated"
    paths = [sparse / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    commands = [sys.executable, str(HERE / "freeze_meetroom_initialization.py"), "--method", "localdygs",
                "--manifest", str(args.manifest.resolve()), "--source", str(args.output.resolve()),
                "--asset", "sparse/0/points3D.bin", "--input-receipt", str((args.native_inputs / "input_receipt.json").resolve()),
                "--native-receipt", str((args.native_init / "complete.json").resolve()),
                "--calibration-acceptance", str((args.native_init / "camera_initialization_acceptance.json").resolve()),
                "--native-ply", str((args.native_init / "cem_data/colmap_0/sparse/0/points3D.ply").resolve())]
    plan = dict(schema="LocalDyGS_MeetRoom_author_sparse_layout_v1", status="CPU_LAYOUT_PLAN_ONLY_NOT_COPIED",
                scene=value["scene"], manifest_sha256=sha(args.manifest),
                source_files=[identity(p) for p in paths], output=str(args.output.resolve()),
                selected_author_branch="sparse/0/points3D.bin fallback; pcds conversion preserves same points/colors",
                prohibited_asset="localdygs_data/downsampled_points_50.ply N3DV dense union",
                future_provenance_argv=commands, training_or_GPU_actions=0)
    if args.execute:
        if args.output.exists():
            raise FileExistsError("Refusing existing native source layout")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        calibration = json.loads((args.native_init / "camera_initialization_acceptance.json").read_text())
        if calibration.get("status") != "passed" or calibration.get("manifest_sha256") != sha(args.manifest):
            raise ValueError("Native calibration acceptance is missing/mismatched")
        destination = args.output / "sparse/0"
        destination.mkdir(parents=True)
        for path in paths:
            shutil.copy2(path, destination / path.name)
            if sha(path) != sha(destination / path.name):
                raise ValueError("Native sparse copy identity changed")
        subprocess.run(commands, check=True)
        plan["status"] = "CPU_LAYOUT_AND_PROVENANCE_PASSED_GPU_HOLD"
    output = HERE / f"notes/meetroom_layout_{value['scene']}.json"
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(status=plan["status"], plan=str(output))))


if __name__ == "__main__":
    main()
