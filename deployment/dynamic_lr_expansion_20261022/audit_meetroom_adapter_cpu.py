#!/usr/bin/env python3
"""Audit MeetRoom manifest adapters and original camera math on CPU only.

No data is copied. --native-projection additionally instantiates CPU copies of
author Camera classes after removing CUDA tensor moves from an AST copy. It
checks the unchanged matrix formulas, not a native CUDA renderer/model load.
"""
import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from meetroom_adapter import ManifestColmapDataset, camera_records, read_manifest
from prepare_local_cem import HERE, ROOT, identity, sha


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_author_camera(path, name, torch, graphics):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    class OnlyCpuMoves(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            return node.func.value if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda" else node
    node = OnlyCpuMoves().visit(node)
    scope = dict(torch=torch, nn=torch.nn, np=np, math=math, getWorld2View2=graphics.getWorld2View2,
                 getProjectionMatrix=graphics.getProjectionMatrix)
    if hasattr(graphics, "getProjectionMatrixCV"):
        scope["getProjectionMatrixCV"] = graphics.getProjectionMatrixCV
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    return scope[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-projection", action="store_true")
    args = parser.parse_args()
    cem_root = ROOT / "third_party/cem4dgs_calibrated_runtime_v3_20261020"
    local_root = ROOT / "third_party/localdygs_calibrated_runtime_v3_20261020"
    protocol = load_module("meetroom_cpu_protocol", local_root / "protocol_interface.py")
    native = {}
    torch = None
    if args.native_projection:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(2)
        def forbidden_cuda(*args, **kwargs):
            raise RuntimeError("CUDA allocation attempted during CPU-only acceptance")
        torch.cuda._lazy_init = forbidden_cuda
        native["localdygs"] = cpu_author_camera(local_root / "scene/cameras.py", "Camera", torch,
                                               load_module("local_cpu_graphics", local_root / "utils/graphics_utils.py"))
        native["cem"] = cpu_author_camera(cem_root / "scene/cameras.py", "Cameravideo", torch,
                                          load_module("cem_cpu_graphics", cem_root / "utils/graphics_utils.py"))
    rows, projection = [], []
    all_metadata = 0
    LR_images_read = zero_images_checked = 0
    for scene in ("meetroom_discussion", "meetroom_vrheadset"):
        path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed20261007.json"
        unused, value = read_manifest(path)
        os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"] = str(path)
        for split in ("train", "test"):
            dataset = ManifestColmapDataset(path, split)
            assert len(dataset) == len(value["splits"][split]) * 300
            assert dataset.image_times == [r["frame"] / 300.0 for r in dataset.records]
            if split == "test":
                assert all(x.startswith("protocol-zero://cam00/") for x in dataset.image_paths)
            else:
                assert all("/hr/" not in x and "/lr/cam00/" not in x for x in dataset.image_paths)
            if torch is not None:
                from PIL import Image
                original_open = Image.open
                def only_training_LR(path, *args, **kwargs):
                    if "/hr/" in str(path) or "/lr/cam00/" in str(path):
                        raise RuntimeError("Heldout/HR pixel access attempted by adapter")
                    return original_open(path, *args, **kwargs)
                Image.open = only_training_LR
                try:
                    if split == "test":
                        for index in range(300):
                            image, pose, time = dataset[index]
                            assert tuple(image.shape) == (3, 180, 320) and not bool(torch.any(image))
                            assert time == index / 300.0
                            zero_images_checked += 1
                    else:
                        for camera in range(dataset.cameras):
                            for frame in (0, 150, 299):
                                index = camera * 300 + frame
                                image, pose, time = dataset[index]
                                assert tuple(image.shape) == (3, 180, 320) and image.device.type == "cpu"
                                assert bool(torch.isfinite(image).all()) and time == frame / 300.0
                                LR_images_read += 1
                finally:
                    Image.open = original_open
            for level in ("lr", "hr"):
                metadata = camera_records(value, split, level)
                all_metadata += len(metadata)
                assert all((r["width"], r["height"]) == tuple(value["resolutions"][level]) for r in metadata)
                assert all(r["time"] == r["frame"] / 300.0 and r["timestamp"] == r["frame"] for r in metadata)
            rows.append(dict(scene=scene, split=split, count=len(dataset), zero_test_paths=split == "test",
                             manifest_sha256=sha(path), LR_shape=value["resolutions"]["lr"], HR_shape=value["resolutions"]["hr"]))
        # Exercise the current CEM manifest CameraInfo factory, not a rephrased
        # path sorter. Test HR is metadata-only; no reference pixels are read.
        for hr_mode in (False, True):
            if hr_mode:
                os.environ["DYNAMIC_LR_RENDER_HR"] = "1"
            else:
                os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
            infos = protocol.cem_camera_infos(lambda **kw: SimpleNamespace(**kw))
            assert len(infos) == 3900
            assert sum(x.image_path.startswith("protocol-zero://") for x in infos) == 300
            for info in infos:
                camera = "cam00" if info.image_path.startswith("protocol-zero://") else Path(info.image_path).parent.name
                level = "hr" if camera == "cam00" and hr_mode else "lr"
                w, h = value["resolutions"][level]
                k = np.asarray(value["cameras"][camera][f"K_{level}"])
                assert (info.width, info.height) == (w, h)
                assert abs((info.cxr + .5) * w - .5 - k[0, 2]) < 1e-9
                assert abs((info.cyr + .5) * h - .5 - k[1, 2]) < 1e-9
                assert info.timestamp in range(300)
        os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
        for camera, spec in value["cameras"].items():
            w2c = np.asarray(spec["w2c"])
            xyz = np.asarray([[x, y, z, 1] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (4, 12)], dtype=float)
            world = xyz @ np.linalg.inv(w2c).T
            for level in ("lr", "hr"):
                w, h = value["resolutions"][level]
                k = np.asarray(spec[f"K_{level}"])
                direct = xyz[:, :3] @ k.T
                direct = direct[:, :2] / direct[:, 2:3]
                p = protocol.projection_numpy(k, w, h)
                clip = xyz @ p.T
                numeric = (clip[:, :2] / clip[:, 3:4] + 1) * np.asarray([w, h]) / 2 - .5
                assert abs(numeric - direct).max() < 1e-8
                if native:
                    shared = dict(colmap_id=0, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                        FoVx=2 * math.atan(w / (2 * k[0, 0])), FoVy=2 * math.atan(h / (2 * k[1, 1])),
                        image_name="CPU metadata", uid=0, data_device="cpu", gt_alpha_mask=None)
                    loc = native["localdygs"](image=torch.zeros(3, h, w), **shared)
                    loc = protocol.apply_localdygs_projection(loc, k)
                    cm = native["cem"](image=None, image_path="not_read", resolution=(w, h),
                        cxr=(k[0, 2] + .5) / w - .5, cyr=(k[1, 2] + .5) / h - .5, **shared)
                    for method, view in (("localdygs", loc), ("cem", cm)):
                        clip = world @ view.full_proj_transform.numpy().astype(float)
                        actual = (clip[:, :2] / clip[:, 3:4] + 1) * np.asarray([w, h]) / 2 - .5
                        error = float(abs(actual - direct).max())
                        assert error < .005, (scene, camera, level, method, error)
                        projection.append(dict(scene=scene, camera=camera, level=level, method=method, max_pixel_error=error))
    result = dict(schema="meetroom_adapter_CPU_acceptance_v1", status="passed_CPU_metadata_and_projection",
                  metadata_records_checked=all_metadata, datasets=rows,
                  native_Scene_constructor_and_GPU_model_load="not tested; requires prepared author initialization and later admitted GPU",
                  CUDA_allocations=0, training_LR_images_read=LR_images_read,
                  heldout_images_read=0, HR_images_read=0, test_zero_images_checked=zero_images_checked,
                  data_copies=0, train_or_remote_actions=0,
                  native_projection_cases=len(projection),
                  native_projection_definition="author classes AST copy: CUDA tensor moves removed; unchanged camera matrix equations",
                  max_native_pixel_error=max((r["max_pixel_error"] for r in projection), default=None),
                  native_projection=projection, initialization_assets="missing; pcds or author sparse fallback and provenance contract required",
                  source=identity(Path(__file__)), adapter=identity(HERE / "meetroom_adapter.py"))
    (HERE / "notes/meetroom_adapter_cpu_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("datasets", "native_projection")}))


if __name__ == "__main__":
    main()
