"""Deterministic calibrated data/projection interface; no trainable parameters.

Only LR image paths are exposed. HR mode changes the held-out pixel grid and K;
ground truth for the final metric must come from the independent HR evaluator.
"""
import json
import math
import os
from pathlib import Path
import numpy as np


def manifest():
    path = Path(os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"])
    value = json.loads(path.read_text())
    assert "_area4" not in str(path)
    assert value["frame_indices"] == list(range(300))
    assert value["splits"]["test"] == ["cam00"]
    assert "cam00" not in value["splits"]["train"]
    return path, value


def observations(split):
    path, value = manifest()
    cameras = value["splits"]["train"] if split == "train" else value["splits"]["test"]
    records = [x for x in value["observations"] if x["camera_id"] in cameras]
    records.sort(key=lambda x: (x["camera_id"], x["frame_index"]))
    hr = split == "test" and os.environ.get("DYNAMIC_LR_RENDER_HR") == "1"
    level = "hr" if hr else "lr"
    return path, value, records, level


def init_localdygs_dataset(dataset):
    path, value, records, level = observations(dataset.split)
    dataset.img_wh = tuple(value["resolutions"][level])
    dataset.image_paths = [str(path.parent / x["lr_path"]) if dataset.split == "train"
                           else f"protocol-zero://cam00/{x['frame_index']:04d}.png" for x in records]
    dataset.image_poses = []
    dataset.image_times = []
    dataset.protocol_intrinsics = []
    for item in records:
        spec = value["cameras"][item["camera_id"]]
        w2c = np.asarray(spec["w2c"], dtype=np.float64)
        dataset.image_poses.append((w2c[:3, :3].T.copy(), w2c[:3, 3].copy()))
        dataset.image_times.append(item["frame_index"] / 300.0)
        dataset.protocol_intrinsics.append(np.asarray(spec[f"K_{level}"], dtype=np.float64))
    first = dataset.protocol_intrinsics[0]
    dataset.focal = [float(first[0, 0]), float(first[1, 1])]
    dataset.cam_number = len(set(x["camera_id"] for x in records))
    dataset.time_number = 300
    dataset.std_frames = []
    all_cameras = sorted(value["cameras"])
    dataset.near_fars = np.asarray([value["cameras"][c]["near_far_metadata"] for c in all_cameras])
    dataset.poses_all = np.asarray([value["cameras"][c]["c2w"][:3] for c in all_cameras])
    dataset.poses = np.asarray([value["cameras"][c]["c2w"][:3] for c in value["splits"]["train"]])
    # Preserve an author-style spiral preview. It never supplies training views.
    from scene.neural_3D_dataset_NDC import get_spiral
    dataset.val_poses = get_spiral(dataset.poses_all, dataset.near_fars, N_views=300)


def projection_numpy(k, width, height, near=0.01, far=100.0):
    """Graphdeco rasterizer ndc2pix(v,S)=((v+1)*S-1)/2 convention."""
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = 2 * k[0, 0] / width
    p[1, 1] = 2 * k[1, 1] / height
    p[0, 2] = 2 * (k[0, 2] + 0.5) / width - 1
    p[1, 2] = 2 * (k[1, 2] + 0.5) / height - 1
    p[3, 2] = 1
    p[2, 2] = far / (far - near)
    p[2, 3] = -far * near / (far - near)
    return p


def apply_localdygs_projection(camera, k):
    import torch
    camera.FoVx = 2 * math.atan(camera.image_width / (2 * k[0, 0]))
    camera.FoVy = 2 * math.atan(camera.image_height / (2 * k[1, 1]))
    p = projection_numpy(k, camera.image_width, camera.image_height, camera.znear, camera.zfar)
    camera.projection_matrix = torch.as_tensor(p, dtype=camera.world_view_transform.dtype, device=camera.world_view_transform.device).T
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    return camera


def cem_camera_infos(factory):
    path, value = manifest()
    result = []
    camera_ids = {c: index + 1 for index, c in enumerate(sorted(value["cameras"]))}
    for split in ["train", "test"]:
        _, _, records, level = observations(split)
        width, height = value["resolutions"][level]
        for item in records:
            spec = value["cameras"][item["camera_id"]]
            k = np.asarray(spec[f"K_{level}"])
            w2c = np.asarray(spec["w2c"])
            # getProjectionMatrixCV places P[0,2]=2*cxr, P[1,2]=2*cyr.
            cxr = (float(k[0, 2]) + 0.5) / width - 0.5
            cyr = (float(k[1, 2]) + 0.5) / height - 0.5
            result.append(factory(
                uid=camera_ids[item["camera_id"]], R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                FovY=2 * math.atan(height / (2 * k[1, 1])), FovX=2 * math.atan(width / (2 * k[0, 0])),
                image_path=str(path.parent / item["lr_path"]) if split == "train" else f"protocol-zero://cam00/{item['frame_index']:04d}.png",
                image_name=f"{item['frame_index']:04d}.png",
                width=width, height=height, near=0.01, far=300.0,
                timestamp=item["frame_index"], pose=None, hpdirecitons=None, cxr=cxr, cyr=cyr))
    return result
