#!/usr/bin/env python3
"""CPU exercise of actual native camera construction/full projection matrices."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def main(a):
    import numpy as np
    import torch
    from prepare_native import normalized_poses
    runtime = a.runtime.resolve()
    sys.path.insert(0, str(runtime))
    # Avoid scene/__init__.py, which imports the native CUDA model. This loads
    # the unmodified camera class plus generated native camera data adapter.
    package = ModuleType("scene")
    package.__path__ = [str(runtime/"scene")]
    sys.modules["scene"] = package
    # loadCam imports PILtoTorch from a file that also imports unrelated CUDA
    # point-operations. Extract that exact unused image helper so CPU camera QA
    # does not depend on a compiled GPU-only extension.
    general = ModuleType("utils.general_utils")
    nodes = [n for n in ast.parse((runtime/"utils/general_utils.py").read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "PILtoTorch"]
    helper_scope = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runtime/"utils/general_utils.py"), "exec"), helper_scope)
    general.PILtoTorch = helper_scope["PILtoTorch"]
    sys.modules["utils.general_utils"] = general
    cameras = importlib.import_module("scene.cameras")
    loader = importlib.import_module("utils.camera_utils")
    m = json.loads(a.manifest.read_text())
    receipt = json.loads((a.dataset/"stage_receipt.json").read_text())
    poses, similarity = normalized_poses(m)
    if not np.allclose(similarity, receipt["native_world_from_manifest_world"]):
        raise ValueError("Normalization changed")
    errors, tested = [], []
    for camera_id in sorted(m["cameras"]):
        for kind in (["hr"] if camera_id == "cam00" else ["lr"]):
            width, height = m["resolutions"][kind]
            k = np.asarray(m["cameras"][camera_id]["K_"+kind])
            native_c2w = poses[camera_id] @ np.diag([1., -1., -1., 1.])
            native_w2c = np.linalg.inv(native_c2w)
            info = SimpleNamespace(uid=0, width=width, height=height, R=native_w2c[:3,:3].T,
                                   T=native_w2c[:3,3], FovX=-1., FovY=-1., image=np.empty(0),
                                   image_name=camera_id, timestamp=299/30., depth=None,
                                   cx=k[0,2]+.5, cy=k[1,2]+.5, fl_x=k[0,0], fl_y=k[1,1], image_path="unread")
            cam = loader.loadCam(SimpleNamespace(resolution=1, dataloader=True, data_device="cpu"), 0, info, 1.)
            # Points well inside each camera's field of view. Build them in the
            # original world and explicitly apply the saved similarity.
            xyz_cam = np.array([[-.25,-.1,2.], [0.,0.,5.], [.7,.2,8.], [-.4,.3,10.]])
            original_w2c = np.asarray(m["cameras"][camera_id]["w2c"])
            original_world = np.c_[xyz_cam,np.ones(4)] @ np.linalg.inv(original_w2c).T
            normalized_world = original_world @ similarity.T
            clip = normalized_world @ cam.full_proj_transform.double().cpu().numpy()
            actual = ((clip[:,:2]/clip[:,3:4]+1)*[width,height]-1)/2
            expected = xyz_cam @ k.T
            expected = expected[:,:2]/expected[:,2:]
            error = float(np.max(np.abs(actual-expected)))
            errors.append(error)
            tested.append({"camera": camera_id, "resolution": kind, "max_error_pixels": error,
                           "native_time": cam.timestamp, "manifest_time": 299/300.,
                           "FoV_valid": bool(cam.FoVx > 0 and cam.FoVy > 0)})
    if max(errors) >= .002 or not all(r["FoV_valid"] for r in tested):
        raise ValueError(f"Native projection mismatch: {max(errors)}")
    result = {"status": "passed", "GPU_calls": 0, "parameter_updates": 0,
              "used_image_pixels": False, "actual_native_camera_module": str(runtime/"scene/cameras.py"),
              "actual_native_loader_module": str(runtime/"utils/camera_utils.py"),
              "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              "max_projection_error_pixels": max(errors), "checked": tested,
              "time_mapping": "manifest frame/300 -> native frame/30; train and HR heldout both map once to [0,10]",
              "principal_point": "Preserved K by +0.5 corner-coordinate conversion, not centered approximation"}
    result["dependency_isolation"] = "Native PILtoTorch helper AST extracted unchanged; unrelated CUDA-only pointops import avoided; actual native loadCam and Camera executed"
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "max_projection_error_pixels": max(errors)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
