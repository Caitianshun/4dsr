"""Exercise native LR dataset and metadata-only heldout camera contracts on CPU.

LocalDyGS runs its actual Neural3D_NDC_Dataset constructor/getitem. CEM runs
its actual CameraInfo2 factory, native split/order expressions, and im_reader.
No model/Scene constructor, optimizer, CUDA tensor, test LR or test HR is used.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import resource
from pathlib import Path
import sys
import types
import numpy as np
import torch
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["localdygs", "cem"], required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native-worker-epoch", action="store_true")
    a = p.parse_args()
    runtime = a.runtime.resolve()
    path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{a.scene}/manifest_train_ready_seed20261007.json"
    value = json.loads(path.read_text())
    os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"] = str(path)
    os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
    sys.path.insert(0, str(runtime))
    package = types.ModuleType("scene")
    package.__path__ = [str(runtime / "scene")]
    sys.modules["scene"] = package
    interface = load("protocol_interface", runtime / "protocol_interface.py")
    expected = {(x["camera_id"], x["frame_index"]): x for x in value["observations"]}
    rows, pixels, max_matrix, worker_receipt = [], [], 0., None
    if a.method == "localdygs":
        module = load("scene.neural_3D_dataset_NDC", runtime / "scene/neural_3D_dataset_NDC.py")
        for split, level in [("train", "lr"), ("test", "lr"), ("test", "hr")]:
            os.environ["DYNAMIC_LR_RENDER_HR"] = "1" if level == "hr" else "0"
            dataset = module.Neural3D_NDC_Dataset(str(path.parent), split=split)
            for index, image_path in enumerate(dataset.image_paths):
                frame = round(dataset.image_times[index] * 300)
                camera = Path(image_path).parent.name
                assert camera in value["splits"][split]
                spec = value["cameras"][camera]
                r, t = dataset.image_poses[index]
                max_matrix = max(max_matrix, float(np.abs(r.T - np.asarray(spec["w2c"])[:3, :3]).max()), float(np.abs(t - np.asarray(spec["w2c"])[:3, 3]).max()))
                assert np.array_equal(dataset.protocol_intrinsics[index], spec[f"K_{level}"])
                if split == "train":
                    assert Path(image_path).resolve() == (path.parent / expected[camera, frame]["lr_path"]).resolve()
                    assert Path(image_path).is_file()
                else:
                    assert image_path.startswith("protocol-zero://cam00/")
                if split == "test" or frame in [0, 150, 299]:
                    image, _, returned_time = dataset[index]
                    assert image.shape == (3, *value["resolutions"][level][::-1])
                    assert returned_time == frame / 300.
                    if split == "test":
                        assert torch.count_nonzero(image).item() == 0
                    else:
                        with Image.open(image_path) as original:
                            assert np.array_equal(np.asarray(original.convert("RGB")), np.rint(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    pixels.append(dict(split=split, level=level, camera=camera, frame=frame))
            rows.append(dict(split=split, level=level, observations=len(dataset), cameras=sorted({Path(x).parent.name for x in dataset.image_paths})))
            if split == "train" and a.native_worker_epoch:
                from scene.spacetime_dataset import FourDGSdataset
                from torch.utils.data import DataLoader
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                new_soft = soft if soft == resource.RLIM_INFINITY else max(soft, min(65536, hard) if hard != resource.RLIM_INFINITY else 65536)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                os.environ.update(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
                native = FourDGSdataset(dataset, None, "dynerf")
                loader = DataLoader(native, batch_size=1, shuffle=True, num_workers=16, collate_fn=list)
                seen = set()
                for batch in loader:
                    camera = batch[0]
                    assert camera.colmap_id not in seen
                    seen.add(camera.colmap_id)
                    assert tuple(camera.original_image.shape) == (3, 252, 336)
                    assert not camera.original_image.is_cuda
                assert len(seen) == len(dataset)
                worker_receipt = dict(loader="actual FourDGSdataset + native DataLoader arguments", workers=16, batch_size=1,
                                      shuffle=True, full_epoch_observations=len(seen), original_soft_nofile=soft,
                                      soft_nofile=new_soft, hard_nofile=hard, TMPDIR="/tmp", CUDA_allocations=0)
    else:
        readers = load("scene.dataset_readers", runtime / "scene/dataset_readers.py")
        tree = ast.parse((runtime / "scene/__init__.py").read_text())
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "im_reader")
        context = dict(torch=torch, Image=Image, ImageFile=ImageFile)
        # Train image conversion is checked against the author's implementation.
        utils = load("utils.general_utils", runtime / "utils/general_utils.py")
        context["PILtoTorch"] = utils.PILtoTorch
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(runtime / "scene/__init__.py"), "exec"), context)
        for level in ["lr", "hr"]:
            os.environ["DYNAMIC_LR_RENDER_HR"] = "1" if level == "hr" else "0"
            infos = sorted(interface.cem_camera_infos(readers.CameraInfo2), key=lambda x: x.image_name)
            splits = dict(train=[x for x in infos if "cam00" not in x.image_path], test=[x for x in infos if "cam00" in x.image_path])
            for split, cameras in splits.items():
                if level == "hr" and split == "train":
                    continue
                names = set()
                for item in cameras:
                    camera, frame = Path(item.image_path).parent.name, item.timestamp
                    names.add(camera)
                    assert camera in value["splits"][split]
                    spec = value["cameras"][camera]
                    max_matrix = max(max_matrix, float(np.abs(item.R.T - np.asarray(spec["w2c"])[:3, :3]).max()), float(np.abs(item.T - np.asarray(spec["w2c"])[:3, 3]).max()))
                    width, height = value["resolutions"][level if split == "test" else "lr"]
                    k = np.asarray(spec[f"K_{level if split == 'test' else 'lr'}"])
                    assert (item.width, item.height) == (width, height)
                    assert abs((item.cxr + .5) * width - .5 - k[0, 2]) < 1e-10
                    assert abs((item.cyr + .5) * height - .5 - k[1, 2]) < 1e-10
                    if split == "train":
                        assert Path(item.image_path).resolve() == (path.parent / expected[camera, frame]["lr_path"]).resolve()
                        assert Path(item.image_path).is_file()
                    else:
                        assert item.image_path.startswith("protocol-zero://cam00/")
                    if split == "test" or frame in [0, 150, 299]:
                        image = context["im_reader"](item.image_path, (width, height), 1.)
                        assert image.shape == (3, height, width)
                        if split == "test":
                            assert torch.count_nonzero(image).item() == 0
                        pixels.append(dict(split=split, level=level, camera=camera, frame=frame))
                rows.append(dict(split=split, level=level, observations=len(cameras), cameras=sorted(names)))
                if split == "train" and a.native_worker_epoch:
                    import random
                    from collections import defaultdict
                    from itertools import compress
                    from joblib import Parallel, delayed
                    from types import SimpleNamespace
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    new_soft = soft if soft == resource.RLIM_INFINITY else max(soft, min(65536, hard) if hard != resource.RLIM_INFINITY else 65536)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                    os.environ.update(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
                    scene_class = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Scene")
                    methods = [x for x in scene_class.body if isinstance(x, ast.FunctionDef) and x.name in ["getTCams", "getTrainCameras"]]
                    copied = ast.ClassDef(name="CpuSceneLoader", bases=[], keywords=[], body=methods, decorator_list=[])
                    worker_context = dict(__name__="__main__", compress=compress, Parallel=Parallel, delayed=delayed,
                                          random=random, defaultdict=defaultdict, im_reader=context["im_reader"])
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[copied], type_ignores=[])), str(runtime / "scene/__init__.py"), "exec"), worker_context)
                    cpu_scene = worker_context["CpuSceneLoader"]()
                    cpu_scene.lazy_loader = True
                    cpu_scene.train_cameras = {1.0: [SimpleNamespace(**vars(c), resolution=(c.width, c.height), im_scale=1.) for c in cameras]}
                    cpu_scene.samplelist = [True] * len(cameras)
                    ordered, images = cpu_scene.getTrainCameras(return_as="generator", shuffle=True, job_batch_size=1)
                    seen = set()
                    for camera, image in zip(ordered, images):
                        assert tuple(image.shape) == (3, 252, 336)
                        assert not image.is_cuda and camera.image_path not in seen
                        seen.add(camera.image_path)
                    assert len(seen) == len(cameras)
                    worker_receipt = dict(loader="actual Scene.getTCams/getTrainCameras AST bodies and native im_reader; CPU metadata objects replace GPU Camera storage only",
                                          workers=2, job_batch_size=1, shuffle=True, full_epoch_observations=len(seen),
                                          original_soft_nofile=soft, soft_nofile=new_soft, hard_nofile=hard, TMPDIR="/tmp", CUDA_allocations=0)
    for item in rows:
        assert item["observations"] == len(value["splits"][item["split"]]) * 300
        assert item["cameras"] == sorted(value["splits"][item["split"]])
    assert max_matrix == 0.
    assert not torch.cuda.is_initialized()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="passed", method=a.method, scene=a.scene,
                  manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  runtime_interface_sha256=hashlib.sha256((runtime / "protocol_interface.py").read_bytes()).hexdigest(),
                  actual_loader_scope=__doc__, camera_identity_from="actual loader image_path parent name, never enumeration index",
                  max_pose_matrix_error=max_matrix, train_LR_mapping_checked_all=True,
                  test_LR_read=False, test_HR_read=False, heldout_pixels="zeros; metadata only",
                  CUDA_allocations=0, observations=rows, pixel_contract_samples=len(pixels))
    result["native_worker_epoch"] = worker_receipt
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
