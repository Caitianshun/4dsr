"""CEM area4 observation interface; no learned parameters or test pixels.

The legacy environment key selects the manifest so that the already audited
native reader and read-only test-logger bypass remain byte-identical. The
training level is LR unless an independent, explicitly labelled HR reference
sets CEM_PROTOCOL_TRAIN_LEVEL=hr. This module never supplies teacher images.
"""
import json
import math
import os
from pathlib import Path

import numpy as np


def manifest():
    path = Path(os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"])
    value = json.loads(path.read_text())
    assert value["scene"] in {"cook_spinach", "cut_roasted_beef"}
    assert value["physical_lr_derivation"]["formation"] == "area4"
    assert value["physical_lr_derivation"]["scale"] == value["scale"] == 4
    assert value["resolutions"] == {"hr": [1344, 1008], "lr": [336, 252]}
    assert value["frame_indices"] == list(range(300))
    assert value["splits"]["test"] == ["cam00"]
    assert value["splits"]["dev"] == []
    assert set(value["splits"]["train"]) == set(value["cameras"]) - {"cam00"}
    train_level = os.environ.get("CEM_PROTOCOL_TRAIN_LEVEL", "lr")
    assert train_level in {"lr", "hr"}
    return path, value


def observations(split):
    assert split in {"train", "test"}
    path, value = manifest()
    cameras = value["splits"][split]
    records = [item for item in value["observations"] if item["camera_id"] in cameras]
    records.sort(key=lambda item: (item["camera_id"], item["frame_index"]))
    expected = {(camera, frame) for camera in cameras for frame in range(300)}
    actual = [(item["camera_id"], item["frame_index"]) for item in records]
    assert len(actual) == len(expected) and set(actual) == expected
    assert all(item["split"] == split and item["time"] == item["frame_index"] / 300.0 for item in records)
    level = os.environ.get("CEM_PROTOCOL_TRAIN_LEVEL", "lr") if split == "train" else (
        "hr" if os.environ.get("DYNAMIC_LR_RENDER_HR") == "1" else "lr")
    return path, value, records, level


def projection_numpy(k, width, height, near=0.01, far=100.0):
    """Exact K including crop/half-pixel principal points, native ndc2pix."""
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0], p[1, 1] = 2 * k[0, 0] / width, 2 * k[1, 1] / height
    p[0, 2] = 2 * (k[0, 2] + 0.5) / width - 1
    p[1, 2] = 2 * (k[1, 2] + 0.5) / height - 1
    p[3, 2], p[2, 2], p[2, 3] = 1, far / (far - near), -far * near / (far - near)
    return p


def cem_camera_infos(factory):
    path, value = manifest()
    result = []
    # Physical suffixes are preserved despite Cut's absent cam04 stream.
    camera_ids = {camera: index + 1 for index, camera in enumerate(sorted(value["cameras"]))}
    for split in ["train", "test"]:
        _, _, records, level = observations(split)
        width, height = value["resolutions"][level]
        for item in records:
            spec = value["cameras"][item["camera_id"]]
            k, w2c = np.asarray(spec[f"K_{level}"]), np.asarray(spec["w2c"])
            camera = factory(
                uid=camera_ids[item["camera_id"]], R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                FovY=2 * math.atan(height / (2 * k[1, 1])), FovX=2 * math.atan(width / (2 * k[0, 0])),
                image_path=str(path.parent / item[f"{level}_path"]) if split == "train" else f"protocol-zero://cam00/{item['frame_index']:04d}.png",
                image_name=f"{item['frame_index']:04d}.png", width=width, height=height,
                near=0.01, far=300.0, timestamp=item["frame_index"], pose=None, hpdirecitons=None,
                cxr=(float(k[0, 2]) + 0.5) / width - 0.5, cyr=(float(k[1, 2]) + 0.5) / height - 0.5)
            result.append(camera)
    return result
