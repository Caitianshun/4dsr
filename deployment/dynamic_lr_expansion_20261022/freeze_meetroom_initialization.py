#!/usr/bin/env python3
"""Freeze an existing native initialization contract, on CPU, after preparation.

No point cloud is made or copied, and no COLMAP, GPU or training task starts.
Requires matching frozen COLMAP inputs, completion, and calibrated-camera
acceptance receipts. It cannot bless a point cloud from another input protocol.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from meetroom_adapter import read_manifest
from prepare_local_cem import ROOT, identity, sha


def verify_sparse_binary_correspondence(asset, reference, expected_sha256):
    import numpy as np
    from plyfile import PlyData
    if not reference.is_file() or sha(reference) != expected_sha256:
        raise ValueError("Sparse-binary fallback requires the exact native frame0 PLY identity")
    loader_path = ROOT / "third_party/localdygs_20261020/scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("verify_author_sparse_binary", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    xyz, rgb, unused = loader.read_points3D_binary(str(asset))
    points = PlyData.read(reference)["vertex"]
    ply_xyz = np.stack([points[k] for k in ("x", "y", "z")], axis=1)
    ply_rgb = np.stack([points[k] for k in ("red", "green", "blue")], axis=1)
    original = np.concatenate([xyz.astype(np.float32).astype(np.float64), rgb.astype(np.float64)], axis=1)
    converted = np.concatenate([ply_xyz.astype(np.float32).astype(np.float64), ply_rgb.astype(np.float64)], axis=1)
    # model_converter may reorder IDs. Preserve xyz/RGB association and repeats.
    original = original[np.lexsort(original[:, ::-1].T)]
    converted = converted[np.lexsort(converted[:, ::-1].T)]
    if original.shape != converted.shape or not np.array_equal(original, converted):
        raise ValueError("Sparse binary points/colors differ from the receipt-bound native PLY")
    return dict(native_ply=identity(reference), author_binary_reader=identity(loader_path), points=len(xyz),
                point_and_RGB_joint_multiset_equal_after_native_PLY_float32_conversion=True, point_order_assumed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["localdygs", "cem"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset", required=True, help="Local: sparse/0/points3D.bin author fallback or pcds/downsample_0_300.ply; CEM: colmap_0/sparse/0/points3D.ply")
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-ply", type=Path, help="For Local sparse-binary fallback: the native frame0 PLY already identified by the native receipt")
    parser.add_argument("--calibration-acceptance", type=Path, required=True)
    args = parser.parse_args()
    unused, value = read_manifest(args.manifest)
    allowed = {"sparse/0/points3D.bin", "pcds/downsample_0_300.ply"} if args.method == "localdygs" else {"colmap_0/sparse/0/points3D.ply"}
    if args.asset not in allowed:
        raise ValueError("Asset does not follow the pinned author's initialization contract")
    inputs, native, calibration = [json.loads(p.read_text()) for p in (args.input_receipt, args.native_receipt, args.calibration_acceptance)]
    if inputs.get("manifest_sha256") != sha(args.manifest) or inputs.get("train_cameras") != sorted(value["splits"]["train"]):
        raise ValueError("Initialization inputs do not match this manifest and training split")
    if inputs.get("HR_images_read") is not False or inputs.get("heldout_images_read") is not False:
        raise ValueError("Initialization input receipt must exclude HR and heldout pixels")
    if (native.get("status") != "completed" or native.get("source_receipt_sha256") != sha(args.input_receipt)
            or native.get("used_HR") is not False or native.get("used_test_pixels") is not False):
        raise ValueError("Initialization did not complete under the matched LR-only input identity")
    if (calibration.get("status") != "passed" or calibration.get("manifest_sha256") != sha(args.manifest)
            or calibration.get("COLMAP_poses_intrinsics_unchanged") is not True
            or calibration.get("initialization_receipt_sha256") != sha(args.native_receipt)):
        raise ValueError("Known camera/initialization calibration acceptance is missing")
    asset = args.source / args.asset
    if not asset.is_file() or asset.stat().st_size == 0:
        raise FileNotFoundError(asset)
    # Match the actual point asset to a file hashed by the native process.
    expected = native.get("localdygs_initialization" if args.asset.startswith("pcds/") else "cem_initialization", {})
    binary_proof = None
    if args.asset.endswith(".bin"):
        reference = args.native_ply or Path(expected.get("path", "missing-native-frame0-ply"))
        binary_proof = verify_sparse_binary_correspondence(asset, reference, expected.get("sha256"))
    elif expected.get("sha256") != sha(asset):
        raise ValueError("Native receipt does not identify this exact point asset; do not relabel a different point cloud")
    receipt = dict(schema="meetroom_author_initialization_contract_v1", status="passed", method=args.method,
                   manifest_sha256=sha(args.manifest), train_cameras=sorted(value["splits"]["train"]),
                   HR_pixels_used=False, heldout_pixels_used=False, selected_asset=args.asset, asset_sha256=sha(asset),
                   source_receipts=[identity(p) for p in (args.input_receipt, args.native_receipt, args.calibration_acceptance)],
                   sparse_binary_correspondence=binary_proof,
                   native_GPU_model_acceptance="not implied by this CPU provenance check")
    output = args.source / "initialization_contract.json"
    text = json.dumps(receipt, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise FileExistsError("Refusing an existing different initialization contract")
    output.write_text(text)
    print(json.dumps(dict(status="passed_CPU_provenance_only", contract=str(output))))


if __name__ == "__main__":
    main()
