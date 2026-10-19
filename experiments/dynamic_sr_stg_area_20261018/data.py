"""Registered area4 inputs and OpenCV-calibrated STG cameras.

make_camera reads calibration only. It never opens LR, teacher or HR pixels.
The training Scene reads only its declared training LR and initialization.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing input or path outside manifest root: {relative}")
    return path


def make_camera(manifest, row, start=None, duration=None, device=None):
    """STG rendering fields; no pixels read, per-camera rays shared within manifest.

    Set manifest['_segment_start'], ['_segment_length'] and optionally ['_device'].
    A segment time is (global frame - segment start) / segment length.
    """
    import torch
    if start is not None:
        manifest["_segment_start"] = int(start)
    if duration is not None:
        manifest["_segment_length"] = int(duration)
    if device is not None:
        manifest["_device"] = str(device)
    width, height = map(int, manifest["resolutions"]["hr"])
    start, length = int(manifest.get("_segment_start", 0)), int(manifest.get("_segment_length", 50))
    frame = int(row["frame_index"])
    if length != 50 or not start <= frame < start + length:
        raise ValueError("Camera must belong to a declared native 50-frame STG segment")
    camera_id = row["camera_id"]
    device = str(manifest.get("_device", "cuda"))
    cache = manifest.setdefault("_stg_camera_cache", {})
    key = (camera_id, device, width, height)
    if key not in cache:
        calibration = manifest["cameras"][camera_id]
        K = np.asarray(calibration["K_hr"], dtype=np.float64)
        w2c = np.asarray(calibration["w2c"], dtype=np.float64)
        if K.shape != (3, 3) or w2c.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(w2c).all():
            raise ValueError("Invalid calibration")
        if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8:
            raise ValueError("Only the registered pinhole intrinsics with zero skew are supported")
        c2w = np.linalg.inv(w2c)
        near, far = .01, 100.
        projection = np.zeros((4, 4), np.float32)
        projection[0, 0], projection[1, 1] = 2 * K[0, 0] / width, 2 * K[1, 1] / height
        projection[0, 2], projection[1, 2] = (2 * K[0, 2] + 1 - width) / width, (2 * K[1, 2] + 1 - height) / height
        projection[2, 2], projection[2, 3] = (far + near) / (far - near), -(far * near) / (far - near)
        projection[3, 2] = 1
        world = torch.tensor(w2c.T.copy(), dtype=torch.float32, device=device)
        proj = torch.tensor(projection.T.copy(), dtype=torch.float32, device=device)
        center = torch.tensor(c2w[:3, 3].copy(), dtype=torch.float32, device=device)
        # Integer OpenCV pixel centers; consistent with the +1 term above and native pix2ndc.
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                              torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        directions = torch.stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], torch.ones_like(x)), -1)
        rotation = torch.tensor(c2w[:3, :3].T.copy(), dtype=torch.float32, device=device)
        directions = torch.nn.functional.normalize(directions @ rotation, dim=-1)
        ray_o = center.expand(height, width, 3).permute(2, 0, 1).unsqueeze(0)
        ray_d = directions.permute(2, 0, 1).unsqueeze(0)
        cache[key] = dict(R=w2c[:3, :3].T, T=w2c[:3, 3],
                          FoVx=2 * math.atan(width / (2 * K[0, 0])), FoVy=2 * math.atan(height / (2 * K[1, 1])),
                          image_width=width, image_height=height, znear=near, zfar=far,
                          world_view_transform=world, projection_matrix=proj, full_proj_transform=world @ proj,
                          camera_center=center, rays=torch.cat((ray_o, ray_d), dim=1), rayo=None, rayd=None)
    return SimpleNamespace(**cache[key], uid=frame, colmap_id=camera_id, image_name=camera_id,
                           timestamp=(frame - start) / length, record_key=(camera_id, frame),
                           original_image=None, global_frame=frame, segment_start=start, segment_length=length)


def projection_check(manifest, cameras):
    errors = []
    seen = set()
    for cam in cameras:
        if cam.image_name in seen:
            continue
        seen.add(cam.image_name)
        calibration = manifest["cameras"][cam.image_name]
        K, w2c = np.asarray(calibration["K_hr"]), np.asarray(calibration["w2c"])
        points = np.array([[-1., -.5, 5.], [0., 0., 10.], [2., 1., 15.]])
        world = np.c_[points, np.ones(len(points))] @ np.linalg.inv(w2c).T
        clip = world @ cam.full_proj_transform.detach().double().cpu().numpy()
        pixels = ((clip[:, :2] / clip[:, 3:] + 1) * np.array(manifest["resolutions"]["hr"]) - 1) / 2
        expected = points @ K.T
        errors.append(float(np.abs(pixels - expected[:, :2] / expected[:, 2:]).max()))
    if not errors or max(errors) >= .001:
        raise ValueError(f"STG camera projection check failed: {errors}")
    return dict(max_error_hr_pixels=max(errors), checked_cameras=len(errors),
                coordinates="registered OpenCV pixel centers including principal point")


class ManifestScene:
    def __init__(self, manifest_path, init, start, gaussians, out, seed=0):
        import random
        import torch
        from PIL import Image
        from utils.graphics_utils import BasicPointCloud
        self.manifest_path = Path(manifest_path).resolve()
        self.m = json.loads(self.manifest_path.read_text())
        self.m.update(_segment_start=int(start), _segment_length=50)
        root = self.manifest_path.parent
        if start not in range(0, 300, 50) or self.m["splits"]["test"] != ["cam00"]:
            raise ValueError("Expected cam00 holdout and one of six native 50-frame segments")
        self.records = [r for r in self.m["observations"] if r["split"] == "train" and start <= int(r["frame_index"]) < start + 50]
        keys = {(r["camera_id"], int(r["frame_index"])) for r in self.records}
        expected = {(c, f) for c in self.m["splits"]["train"] for f in range(start, start + 50)}
        if keys != expected or len(keys) != len(self.records) or "cam00" in self.m["splits"]["train"]:
            raise ValueError("Training segment does not exactly cover declared train cameras/frames")
        self.train_camera, self.input_identity = [], []
        for row in self.records:
            path = checked_path(root, row["lr_path"])
            if sha(path) != row["lr_sha256"]:
                raise ValueError(f"LR bytes changed: {path}")
            with Image.open(path) as image:
                if list(image.size) != self.m["resolutions"]["lr"] or image.mode != "RGB":
                    raise ValueError("Unexpected LR shape/mode")
                rgb = np.asarray(image).copy()
            cam = make_camera(self.m, row)
            cam.original_image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            self.train_camera.append(cam)
            self.input_identity.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]), sha256=row["lr_sha256"]))
        self.projection = projection_check(self.m, self.train_camera)
        # Match native Scene's one-time shuffled camera list. The fixed seed is reset by the runner.
        random.shuffle(self.train_camera)
        init = Path(init).resolve()
        report = json.loads((init.parent / "report.json").read_text())
        if report["source_frames"] != [0] or report["used_hr_images"] or report["used_development_or_test_images"]:
            raise ValueError("Initializer must use frame-zero training LR only")
        if report["manifest_sha256"] != sha(self.manifest_path) or report["npz_sha256"] != sha(init):
            raise ValueError("Initializer manifest/hash mismatch")
        with np.load(init) as values:
            xyz, colors = values["points"].copy(), values["colors"].copy()
            if not np.all(values["source_frame_index"] == 0) or not np.isfinite(xyz).all() or not np.isfinite(colors).all():
                raise ValueError("Invalid initializer")
        if xyz.shape != colors.shape or len(xyz) < 4 or colors.min() < 0 or colors.max() > 1:
            raise ValueError("Invalid LR point/color arrays")
        centers = np.array([np.linalg.inv(np.asarray(self.m["cameras"][c]["w2c"]))[:3, 3] for c in self.m["splits"]["train"]])
        self.cameras_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        # This is a declared representation seed, not per-frame triangulation.
        times = np.zeros((len(xyz), 1), np.float32)
        gaussians.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz), times), self.cameras_extent)
        self.gaussians, self.model_path = gaussians, str(out)
        self.initialization = dict(source_frame=0, source_kind="registered training LR triangulation shared with Multi4D",
                                   local_temporal_center_seed=0., native_log_temporal_width_seed=0.,
                                   per_frame_SfM=False, reused_frame0_geometry_for_each_segment=True,
                                   points=len(xyz), init_sha256=sha(init), report_sha256=sha(init.parent / "report.json"))

    def getTrainCameras(self):
        return self.train_camera

    def getTestCameras(self):
        return []

    def save(self, iteration):
        path = Path(self.model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        self.gaussians.save_ply(str(path))

    def recordpoints(self, iteration, message):
        with (Path(self.model_path) / "points.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(iteration=int(iteration), points=len(self.gaussians._xyz), message=message)) + "\n")
