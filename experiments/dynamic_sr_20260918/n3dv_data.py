"""Explicit-camera loader for the N3DV sampling/coupling pilot.

The stock 4DGaussians N3DV loader shares decoded-image caches across resolution
settings. This loader consumes a prepared manifest and never infers a split or
silently resizes an image. Image coordinates use the standard pinhole convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "n3dv_dynamic_sr_pilot_v1":
        raise ValueError(f"Unsupported manifest: {path}")
    manifest["_manifest_path"] = str(path)
    manifest["_root"] = str(path.parent)
    return manifest


def parse_n3dv_cameras(raw_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Map sorted existing cam*.mp4 files to rows, exactly as the official loader.

    LLFF's input axes are [down, right, backward]. Conversion below equals the
    official 4DGaussians pose conversion, with an explicit OpenCV world-to-camera
    matrix: +x right, +y down, +z forward.
    """
    raw_dir = Path(raw_dir)
    videos = sorted(raw_dir.glob("cam*.mp4"))
    bounds = np.load(raw_dir / "poses_bounds.npy", allow_pickle=False)
    if len(videos) != len(bounds) or not videos:
        raise ValueError(f"Video/pose mismatch: {len(videos)} / {len(bounds)}")
    result: dict[str, dict[str, Any]] = {}
    for row, video in zip(bounds, videos):
        pose = row[:-2].reshape(3, 5)
        height, width, focal = pose[:, 4]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.stack((pose[:, 1], pose[:, 0], -pose[:, 2]), axis=1)
        c2w[:3, 3] = pose[:, 3]
        rotation = c2w[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError(f"Non-orthogonal rotation in {video.name}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
            raise ValueError(f"Invalid rotation determinant in {video.name}")
        w2c = np.linalg.inv(c2w)
        k_raw = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
        # Original package: 2704x2028 -> 1352x1014 -> center crop 1344x1008.
        # Coordinates are zero-based pixel centers. cv2.resize and align_corners
        # False use u'=(u+0.5)*s-0.5; principal points require the same offset.
        sx, sy = 1352.0 / width, 1014.0 / height
        resize = np.array([[sx, 0, (sx - 1) / 2],
                           [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)
        crop = np.array([[1, 0, -4], [0, 1, -3], [0, 0, 1]], dtype=np.float64)
        k_hr = crop @ resize @ k_raw
        k_lr = np.array([[0.25, 0, -0.375], [0, 0.25, -0.375], [0, 0, 1]]) @ k_hr
        result[video.stem] = {
            "camera_id": video.stem,
            "video": str(video.resolve()),
            "raw_width": int(round(width)), "raw_height": int(round(height)),
            "K_raw": k_raw.tolist(), "K_hr": k_hr.tolist(), "K_lr": k_lr.tolist(),
            "w2c": w2c.tolist(), "c2w": c2w.tolist(),
            "near_far_metadata": row[-2:].tolist(),
        }
    return result


class N3DVPreparedDataset(Dataset):
    def __init__(self, manifest: str | Path | dict[str, Any], split: str = "train",
                 resolution: str = "lr", device: str = "cpu", cache: bool = False):
        if split not in ("train", "dev", "test") or resolution not in ("lr", "hr"):
            raise ValueError((split, resolution))
        self.manifest = load_manifest(manifest) if not isinstance(manifest, dict) else manifest
        self.root = Path(self.manifest["_root"])
        self.split, self.resolution, self.device = split, resolution, device
        self.observations = [x for x in self.manifest["observations"] if x["split"] == split]
        self.cameras = self.manifest["cameras"]
        self._cache: dict[int, torch.Tensor] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        obs = self.observations[index]
        camera = self.cameras[obs["camera_id"]]
        path = self.root / obs[self.resolution + "_path"]
        width, height = self.manifest["resolutions"][self.resolution]
        if self._cache is not None and index in self._cache:
            image = self._cache[index]
        else:
            with Image.open(path) as im:
                if im.size != (width, height):
                    raise ValueError(f"Unexpected image size {im.size}: {path}")
                image = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float().div_(255)
            image = image.to(self.device)
            if self._cache is not None:
                self._cache[index] = image
        return {"image": image, "image_path": str(path), "camera_id": obs["camera_id"],
                "frame_index": obs["frame_index"], "time": obs["time"],
                "K": np.asarray(camera["K_" + self.resolution], dtype=np.float64),
                "w2c": np.asarray(camera["w2c"], dtype=np.float64),
                "c2w": np.asarray(camera["c2w"], dtype=np.float64),
                "width": width, "height": height, "split": self.split}


def observation_to_4dgs_camera(observation: dict[str, Any], uid: int = 0):
    """Requires the independent official 4DGaussians checkout on sys.path."""
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    k, w2c = observation["K"], observation["w2c"]
    width, height = observation["width"], observation["height"]
    camera = Camera(colmap_id=uid, R=w2c[:3, :3].T, T=w2c[:3, 3],
                    FoVx=focal2fov(k[0, 0], width), FoVy=focal2fov(k[1, 1], height),
                    image=observation["image"], gt_alpha_mask=None,
                    image_name=f"{observation['camera_id']}/{observation['frame_index']:04d}",
                    uid=uid, data_device=str(observation["image"].device),
                    time=observation["time"], mask=None)
    # Original rasterizer maps NDC to zero-based centers with
    # ((ndc+1)*size-1)/2. Account for the supplied noncentral principal point.
    camera.projection_matrix[2, 0] = (2 * k[0, 2] + 1 - width) / width
    camera.projection_matrix[2, 1] = (2 * k[1, 2] + 1 - height) / height
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    camera.intrinsics = np.asarray(k).copy()
    return camera


def load_initial_points(manifest: str | Path | dict[str, Any]) -> dict[str, np.ndarray]:
    m = load_manifest(manifest) if not isinstance(manifest, dict) else manifest
    with np.load(Path(m["_root"]) / m["initialization"]["npz_path"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}
