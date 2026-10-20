#!/usr/bin/env python3
"""Fail closed before GPU stages unless CPU environment/camera QA passed."""
import argparse
import json
from pathlib import Path
import hashlib

p=argparse.ArgumentParser()
p.add_argument("--dataset",type=Path,required=True)
p.add_argument("--manifest",type=Path,required=True)
p.add_argument("--environment-receipt",type=Path,required=True)
p.add_argument("--projection-audit",type=Path,required=True)
p.add_argument("--initialization-receipt",type=Path)
p.add_argument("--loader-audit",type=Path)
p.add_argument("--scratch-tempdir",type=Path)
p.add_argument("--deployment-contract",type=Path)
a=p.parse_args()
stage=json.loads((a.dataset/"stage_receipt.json").read_text())
env=json.loads(a.environment_receipt.read_text())
qa=json.loads(a.projection_audit.read_text())
identity=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
if env["returncode"]!=0 or qa["status"]!="passed" or qa["manifest_sha256"]!=identity or stage["manifest_sha256"]!=identity:
    raise ValueError("Native environment/camera/manifest gate failed")
if stage["used_HR_pixels"] or stage["used_heldout_pixels"] or json.loads((a.dataset/"transforms_test.json").read_text())["frames"]:
    raise ValueError("Training pixel boundary is not LR-only")
if qa["max_projection_error_pixels"]>=.002:
    raise ValueError("Native projection is inaccurate")
if a.loader_audit:
    loader=json.loads(a.loader_audit.read_text())
    if loader["status"]!="passed" or loader["manifest_sha256"]!=identity or not all(
            r["dtype_bytes_equal"] and r["native_loader_tensor_equal"] for r in loader["checked"]):
        raise ValueError("Native Pillow compatibility byte equivalence failed")
if a.initialization_receipt:
    initialization=json.loads(a.initialization_receipt.read_text())
    if (initialization["status"]!="complete" or initialization["manifest_sha256"]!=identity
            or initialization["used_HR_pixels"] or initialization["used_heldout_pixels"]):
        raise ValueError("Initialization boundary/identity failed")
    if hashlib.sha256((a.dataset/"points3d.ply").read_bytes()).hexdigest()!=initialization["point_cloud_sha256"]:
        raise ValueError("Registered native initialization point bytes changed")
if a.deployment_contract:
    contract=json.loads(a.deployment_contract.read_text())
    if contract["status"]!="passed" or contract["manifest_sha256"]!=identity:
        raise ValueError("Native CPU deployment contract failed")
    if contract["training_wrapper_sha256"]!=hashlib.sha256((Path(__file__).parent/"train_exec.py").read_bytes()).hexdigest():
        raise ValueError("Training environment wrapper differs from the CPU deployment contract")
    for row in contract["receipt_files"]:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("Native CPU deployment contract receipt changed: "+row["kind"])
if a.scratch_tempdir:
    if a.scratch_tempdir.parent!=Path("/tmp") or not a.scratch_tempdir.name.startswith("4dsr-s4d-") or len(str(a.scratch_tempdir))>55:
        raise ValueError("Requires own short job-specific /tmp directory for native DataLoader IPC")
    a.scratch_tempdir.mkdir(mode=0o700,exist_ok=False)
print("Native environment, full-projection camera and original-LR gates passed")
