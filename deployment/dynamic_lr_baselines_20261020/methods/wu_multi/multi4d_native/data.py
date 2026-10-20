"""Full-time area-LR inputs for the pinned Multi4D adapter.

The camera/three-branch initialization follows the audited 2026-09-24
multi_data.py adapter. Splits and normalized time now come from the registered
manifest. This module never reads HR pixels. CUDA is used only by the native
Gaussian constructors inside ManifestScene, not by the preparation helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Input escapes manifest directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("physical_lr_derivation"):
        raise ValueError("Original LR is required; area4 re-formation is forbidden")
    if int(manifest["scale"]) != 4:
        raise ValueError("This adapter requires registered x4 inputs")
    width, height = manifest["resolutions"]["hr"]
    if manifest["resolutions"]["lr"] != [width // 4, height // 4] or width % 4 or height % 4:
        raise ValueError("HR and LR dimensions must have an exact factor of four")
    splits = manifest["splits"]
    train = splits["train"]
    if len(train) != len(set(train)) or not train or "cam00" in train:
        raise ValueError("Invalid training-camera split")
    if splits.get("test") != ["cam00"] or splits.get("dev", []):
        raise ValueError("Current protocol requires cam00-only holdout and no development cameras")
    if set(train) & set(splits["test"]):
        raise ValueError("Training and heldout cameras overlap")
    rows = manifest["observations"]
    keys = [(r["camera_id"], int(r["frame_index"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate camera/frame observation")
    expected = {(c, f) for c in [*train, "cam00"] for f in range(300)}
    if set(keys) != expected:
        raise ValueError("Current protocol requires all 300 frames of each registered camera")
    for row in rows:
        expected_split = "train" if row["camera_id"] in train else "test"
        if row["split"] != expected_split:
            raise ValueError(f"Observation split disagrees with camera split: {row}")
        time = float(row["time"])
        if not np.isfinite(time) or abs(time - int(row["frame_index"]) / 300.0) > 1e-12:
            raise ValueError("Current full-time manifest must preserve frame_index / 300 timestamps")
    return manifest


def inventory(manifest_path, teacher_index=None, verify_files=True):
    """Validate all training LR/teacher identities without opening any HR/test image."""
    from PIL import Image

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    records = [r for r in manifest["observations"] if r["split"] == "train"]
    inputs = []
    for row in records:
        path = checked_path(root, row["lr_path"])
        if verify_files:
            if sha(path) != row["lr_sha256"]:
                raise ValueError(f"Training LR identity changed: {path}")
            with Image.open(path) as image:
                if list(image.size) != manifest["resolutions"]["lr"] or image.mode != "RGB":
                    raise ValueError(f"Invalid LR dimensions/mode: {path}")
        inputs.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]),
                           time=float(row["time"]), lr_path=row["lr_path"], sha256=row["lr_sha256"]))
    result = dict(schema="multi4d_area_inputs_v1", scene=manifest["scene"], manifest=str(manifest_path.resolve()),
                  manifest_sha256=sha(manifest_path), train_cameras=manifest["splits"]["train"],
                  test_cameras=["cam00"], frame_indices=list(range(300)), training_observations=len(records),
                  resolutions=manifest["resolutions"], inputs=inputs, training_files_verified=bool(verify_files),
                  used_HR_pixels=False, used_heldout_pixels=False,
                  time_parameterization="registered observation.time; frame_index / 300; native TR maps once to [0,10]")
    if teacher_index is not None:
        teacher_index = Path(teacher_index)
        index = json.loads(teacher_index.read_text())
        if index["manifest_sha256"] != sha(manifest_path) or index.get("oracle", False):
            raise ValueError("Teacher must be a legal frozen prior bound to this exact manifest")
        entries = index["entries"]
        teacher_keys = [(e["camera"], int(e["frame"])) for e in entries]
        train_keys = {(r["camera_id"], int(r["frame_index"])) for r in records}
        if len(teacher_keys) != len(set(teacher_keys)) or set(teacher_keys) != train_keys:
            raise ValueError("Teacher entries must cover exactly the registered training observations")
        lookup = {(r["camera_id"], int(r["frame_index"])): r for r in records}
        for entry in entries:
            row = lookup[entry["camera"], int(entry["frame"])]
            if entry.get("lr_sha256") != row["lr_sha256"]:
                raise ValueError("Teacher source LR identity differs from the manifest")
            path = checked_path(root, entry["relative_path"])
            if verify_files:
                if sha(path) != entry["sha256"]:
                    raise ValueError(f"Teacher image identity changed: {path}")
                with Image.open(path) as image:
                    if list(image.size) != manifest["resolutions"]["hr"] or image.mode != "RGB":
                        raise ValueError(f"Invalid teacher dimensions/mode: {path}")
        result["teacher"] = dict(index=str(teacher_index.resolve()), sha256=sha(teacher_index),
                                 entries=len(entries), oracle=False, config=index.get("teacher_config"))
    return result


def camera(manifest, record, image):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    width, height = manifest["resolutions"]["lr"]
    calibration = manifest["cameras"][record["camera_id"]]
    intrinsics, w2c = np.asarray(calibration["K_lr"]), np.asarray(calibration["w2c"])
    cam = Camera(R=w2c[:3, :3].T, T=w2c[:3, 3],
                 FoVx=focal2fov(intrinsics[0, 0], width), FoVy=focal2fov(intrinsics[1, 1], height),
                 image=image, image_name=f"{record['camera_id']}/{int(record['frame_index']):04d}",
                 time=float(record["time"]))
    cam.image_width, cam.image_height = width, height
    cam.projection_matrix[2, 0] = (2 * intrinsics[0, 2] + 1 - width) / width
    cam.projection_matrix[2, 1] = (2 * intrinsics[1, 2] + 1 - height) / height
    cam.full_proj_transform = cam.world_view_transform @ cam.projection_matrix
    cam.record_key = (record["camera_id"], int(record["frame_index"]))
    return cam


def projection_check(manifest, cameras):
    errors = []
    # Check every camera at the start and end of the sequence, including off-axis points.
    subset = [cam for cam in cameras if cam.record_key[1] in (0, 299)]
    for cam in subset:
        calibration = manifest["cameras"][cam.record_key[0]]
        intrinsics, w2c = np.asarray(calibration["K_lr"]), np.asarray(calibration["w2c"])
        xyz_camera = np.array([[-1., -.5, 5.], [0., 0., 10.], [2., 1., 15.]])
        xyz_world = np.c_[xyz_camera, np.ones(3)] @ np.linalg.inv(w2c).T
        clip = xyz_world @ cam.full_proj_transform.double().cpu().numpy()
        ndc = clip[:, :2] / clip[:, 3:]
        pixel = ((ndc + 1) * np.array(manifest["resolutions"]["lr"]) - 1) / 2
        expected = xyz_camera @ intrinsics.T
        expected = expected[:, :2] / expected[:, 2:]
        errors.append(float(np.abs(pixel - expected).max()))
    if not errors or max(errors) >= .001:
        raise ValueError(f"Camera projection check failed: {errors}")
    return dict(max_projection_error_hr_pixels=max(errors), checked_camera_frames=len(errors),
                coordinates="OpenCV pixel centers; transposed row-vector renderer",
                time="registered observation.time; frame_index / 300", last_time=299 / 300)


class ManifestScene:
    def __init__(self, manifest, init, foreground, background, transient, out):
        import torch
        from PIL import Image
        from utils.graphics_utils import BasicPointCloud
        from utils.sh_utils import SH2RGB
        from scene.dataset_readers import getNerfppNorm

        self.manifest_path = Path(manifest)
        self.m = load_manifest(self.manifest_path)
        root = self.manifest_path.parent
        self.records = [r for r in self.m["observations"] if r["split"] == "train"]
        self.train_camera, self.input_identity = [], []
        for row in self.records:
            path = checked_path(root, row["lr_path"])
            if sha(path) != row["lr_sha256"]:
                raise ValueError(f"Training LR changed: {path}")
            with Image.open(path) as image:
                if list(image.size) != self.m["resolutions"]["lr"]:
                    raise ValueError(f"LR dimensions differ: {path}")
                rgb = np.asarray(image.convert("RGB")).copy()
            # Match the old adapter: uint8 -> [0,1] exactly once, all images remain on CPU.
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.
            self.train_camera.append(camera(self.m, row, image))
            self.input_identity.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]),
                                            time=float(row["time"]), lr_path=str(path), sha256=row["lr_sha256"]))
        self.projection = projection_check(self.m, self.train_camera)
        init = Path(init)
        report = json.loads((init.parent / "report.json").read_text())
        if report["source_frames"] != [0] or report["used_hr_images"] or report["used_development_or_test_images"]:
            raise ValueError("Multi4D static initialization requires frame-zero training LR only")
        if report["manifest_sha256"] != sha(self.manifest_path) or report["npz_sha256"] != sha(init):
            raise ValueError("Initialization is not bound to this manifest or its bytes changed")
        for source in report["source_images"]:
            if source["frame_index"] != 0 or source["camera_id"] not in self.m["splits"]["train"]:
                raise ValueError("Illegal initialization source")
            if sha(checked_path(root, source["path"])) != source["sha256"]:
                raise ValueError("Initialization source image changed")
        with np.load(init) as values:
            xyz, colors = values["points"], values["colors"]
            if not np.all(values["source_frame_index"] == 0) or not np.all(np.isfinite(xyz)):
                raise ValueError("Invalid frame-zero initialization points")
        if colors.min() < 0 or colors.max() > 1:
            raise ValueError("Initialization colors must already be in [0,1]")
        norm = getNerfppNorm(self.train_camera)
        self.cameras_extent = norm["radius"]
        centers = np.hstack(norm["cam_centers"]).T
        hi = np.vstack((centers.max(0) + 5 * self.cameras_extent, xyz.max(0))).min(0)
        lo = np.vstack((centers.min(0) - 5 * self.cameras_extent, xyz.min(0))).max(0)
        dynamic = np.hstack([np.random.random((10000, 1)) * (hi[i] - lo[i]) + lo[i] for i in range(3)])
        dynamic_colors = SH2RGB(np.random.random((10000, 3)))
        foreground._deformation.deformation_net.set_aabb(dynamic.max(0), dynamic.min(0))
        foreground.create_from_pcd(BasicPointCloud(dynamic, dynamic_colors, np.zeros_like(dynamic)), self.cameras_extent)
        background.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), self.cameras_extent)
        tr_xyz = foreground.get_xyz.detach().cpu().numpy()[:1]
        tr_rgb = SH2RGB(foreground._features_dc.squeeze(1).detach().cpu()[:1]).numpy()
        transient.create_from_pcd(BasicPointCloud(tr_xyz, tr_rgb, np.zeros_like(tr_xyz)), self.cameras_extent)
        if len(tr_xyz) != 1 or bool(torch.isfinite(transient._scaling).all()):
            raise ValueError("Pinned singleton-KNN bug no longer matches the audited implementation")
        transient._scaling = torch.nn.Parameter(foreground._scaling[:1].detach().clone())
        self.singleton_scale_fix = "One TR seed inherits its identical FG donor scale; native singleton three-neighbor KNN is undefined"
        self.gaussians, self.gaussians_second, self.gaussians_transient = foreground, background, transient
        self.model_path = str(out)

    def getTrainCameras(self):
        return self.train_camera

    def getTestCameras(self):
        return []
