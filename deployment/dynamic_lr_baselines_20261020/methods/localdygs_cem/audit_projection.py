"""CPU-only comparison of native camera projection against manifest K/w2c.

CEM's CUDA tensor moves are removed in an AST copy for this CPU numeric check.
No renderer, optimizer, CUDA allocation, or image file is used by this audit.
"""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from protocol_interface import apply_localdygs_projection

ROOT = Path(__file__).resolve().parents[4]


def graphics(path):
    spec = importlib.util.spec_from_file_location("audit_graphics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_class(path, name, helpers):
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    class CpuMoves(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                return node.func.value
            return node
    node = CpuMoves().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    context = {"torch": torch, "nn": nn, "np": np, "math": math,
               "getWorld2View2": helpers.getWorld2View2,
               "getProjectionMatrix": helpers.getProjectionMatrix}
    if hasattr(helpers, "getProjectionMatrixCV"):
        context["getProjectionMatrixCV"] = helpers.getProjectionMatrixCV
    exec(compile(module, str(path), "exec"), context)
    return context[name]


def main():
    local_root = ROOT / "third_party/localdygs_20261020"
    cem_root = ROOT / "third_party/cem4dgs_20261020"
    local = native_class(local_root / "scene/cameras.py", "Camera", graphics(local_root / "utils/graphics_utils.py"))
    cem = native_class(cem_root / "scene/cameras.py", "Cameravideo", graphics(cem_root / "utils/graphics_utils.py"))
    records = []
    for scene in ["cook_spinach", "cut_roasted_beef"]:
        manifest_path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed20261007.json"
        value = json.loads(manifest_path.read_text())
        for camera, spec in value["cameras"].items():
            w2c = np.asarray(spec["w2c"])
            xyz_camera = np.array([[x, y, z, 1] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [4, 12]], dtype=np.float64)
            xyz_world = xyz_camera @ np.linalg.inv(w2c).T
            for level in ["lr", "hr"]:
                width, height = value["resolutions"][level]
                k = np.asarray(spec[f"K_{level}"])
                expected = xyz_camera[:, :3] @ k.T
                expected = expected[:, :2] / expected[:, 2:3]
                args = dict(colmap_id=0, R=w2c[:3, :3].T, T=w2c[:3, 3],
                            FoVx=2 * math.atan(width / (2 * k[0, 0])),
                            FoVy=2 * math.atan(height / (2 * k[1, 1])),
                            image_name="CPU only", uid=0, data_device="cpu", gt_alpha_mask=None)
                loc = local(image=torch.zeros(3, height, width), **args)
                corrected = apply_localdygs_projection(loc, k)
                cxr = (k[0, 2] + 0.5) / width - 0.5
                cyr = (k[1, 2] + 0.5) / height - 0.5
                cm = cem(image=None, image_path="not_read", resolution=(width, height), cxr=cxr, cyr=cyr, **args)
                for name, native in [("localdygs", corrected), ("cem", cm)]:
                    clip = xyz_world @ native.full_proj_transform.cpu().numpy().astype(np.float64)
                    ndc = clip[:, :2] / clip[:, 3:4]
                    actual = (ndc + 1) * np.array([width, height]) / 2 - 0.5
                    error = float(abs(expected - actual).max())
                    if error >= 0.005:
                        raise RuntimeError((scene, camera, level, name, error))
                    records.append(dict(scene=scene, camera=camera, resolution=level, method=name, max_pixel_error=error, points=len(xyz_world)))
    result = dict(status="passed", max_pixel_error=max(x["max_pixel_error"] for x in records),
                  native_full_projection_cases=len(records), tolerance_pixels=0.005,
                  CUDA_allocations=0, images_read=0,
                  CEM_check="native class AST copy with cuda tensor moves removed for CPU only; matrix calculations unchanged",
                  records=records)
    out = ROOT / "output/dynamic_lr_baselines_20261020/localdygs_cem/camera_projection_cpu_acceptance.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
