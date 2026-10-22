"""Exact-manifest MeetRoom I/O for pinned LocalDyGS/CEM runtimes.

The module imports no Torch at module scope. It preserves the LocalDyGS
Colmap scene/point-cloud contract and CEM's frame-zero sparse initialization.
It changes data access only: registered LR pixels, exact K, and cam00 zero
metadata. No reconstruction model, loss, optimizer or schedule is changed.
"""
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path=None):
    path = Path(path or os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"])
    value = json.loads(path.read_text())
    if not value["scene"].startswith("meetroom_") or value.get("physical_lr_derivation"):
        raise ValueError("Requires a registered original-bicubic MeetRoom manifest")
    if value["frame_indices"] != list(range(300)) or value["splits"]["test"] != ["cam00"] or "cam00" in value["splits"]["train"]:
        raise ValueError("Unexpected frame/camera split")
    return path, value


def camera_records(value, split, level="lr"):
    """Metadata only. These are the same rays at LR and HR pixel grids."""
    import numpy as np
    expected = {(c, f) for c in value["splits"][split] for f in range(300)}
    observations = sorted((o for o in value["observations"] if o["split"] == split),
                          key=lambda o: (o["camera_id"], o["frame_index"]))
    if len(observations) != len(expected) or {(o["camera_id"], o["frame_index"]) for o in observations} != expected:
        raise ValueError("Incomplete/duplicate observations")
    result = []
    for item in observations:
        spec = value["cameras"][item["camera_id"]]
        k, w2c = np.asarray(spec[f"K_{level}"], dtype=np.float64), np.asarray(spec["w2c"], dtype=np.float64)
        if abs(item["time"] - item["frame_index"] / 300.0) > 1e-10:
            raise ValueError("LocalDyGS time is frame/300; CEM timestamp is the integer frame")
        result.append(dict(camera_id=item["camera_id"], frame=item["frame_index"], time=item["frame_index"] / 300.0,
                           timestamp=item["frame_index"], width=value["resolutions"][level][0], height=value["resolutions"][level][1],
                           K=k, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                           path=item["lr_path"] if split == "train" else f"protocol-zero://cam00/{item['frame_index']:04d}.png"))
    return result


class ManifestColmapDataset:
    """Author Colmap_Dataset output interface, without resize or heldout reads."""
    def __init__(self, manifest_path, split):
        self.manifest_path, value = read_manifest(manifest_path)
        self.split = split
        self.records = camera_records(value, split, "lr")
        self.N_frames, self.first_frame, self.final_frame = 300, 0, 300
        self.cam_number = self.cameras = len(value["splits"][split])
        self.image_paths = [str(self.manifest_path.parent / r["path"]) if split == "train" else r["path"] for r in self.records]
        self.image_poses = [(r["R"], r["T"]) for r in self.records]
        self.image_times = [r["time"] for r in self.records]
        self.protocol_intrinsics = [r["K"] for r in self.records]
        self.img_wh = tuple(value["resolutions"]["lr"])
        self.img_whs = [self.img_wh for _ in value["splits"][split]]
        self.focal = [[value["cameras"][c]["K_lr"][0][0], value["cameras"][c]["K_lr"][1][1]] for c in sorted(value["splits"][split])]

    def __len__(self):
        return len(self.records)

    def load_pose(self, index):
        return self.image_poses[index]

    def __getitem__(self, index):
        # Torch is used only by an actual author Dataset invocation, on CPU.
        import numpy as np
        import torch
        from PIL import Image
        if self.split == "test":
            image = torch.zeros((3, self.img_wh[1], self.img_wh[0]), dtype=torch.float32)
        else:
            with Image.open(self.image_paths[index]) as source:
                if source.size != self.img_wh or source.mode != "RGB":
                    raise ValueError("Training LR shape/mode mismatch; resizing is forbidden")
                image = torch.from_numpy(np.asarray(source, dtype=np.float32).copy().transpose(2, 0, 1) / 255.0)
        return image, self.image_poses[index], self.image_times[index]


def initialization_contract(source, method, manifest_path=None):
    """Require traceable train-LR point assets; never invent/fallback across methods."""
    source = Path(source)
    path, value = read_manifest(manifest_path)
    receipt_path = source / "initialization_contract.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Missing verified training-LR initialization contract: " + str(receipt_path))
    receipt = json.loads(receipt_path.read_text())
    if not (receipt.get("status") == "passed" and receipt.get("method") == method and
            receipt.get("manifest_sha256") == sha(path) and receipt.get("train_cameras") == sorted(value["splits"]["train"]) and
            receipt.get("HR_pixels_used") is False and receipt.get("heldout_pixels_used") is False):
        raise ValueError("Initialization provenance does not match the training-only protocol")
    allowed = {"pcds/downsample_0_300.ply", "sparse/0/points3D.bin"} if method == "localdygs" else {"colmap_0/sparse/0/points3D.ply"}
    if receipt.get("selected_asset") not in allowed:
        raise ValueError("Non-author initialization asset contract")
    asset = source / receipt["selected_asset"]
    if not asset.is_file() or sha(asset) != receipt.get("asset_sha256"):
        raise ValueError("Initialization asset missing or changed")
    return asset, receipt


def local_colmap_scene(source, frames_start_end, namespace):
    """Called at the pinned author's Colmap loader entry, before legacy paths."""
    if frames_start_end != [0, 300]:
        raise ValueError("The frozen MeetRoom comparison uses the full 300-frame clip")
    path, value = read_manifest()
    asset, receipt = initialization_contract(source, "localdygs", path)
    train, test = ManifestColmapDataset(path, "train"), ManifestColmapDataset(path, "test")
    image = train[0][0]  # Same single training-LR image access as native colmap_format_infos.
    infos = []
    for index, row in enumerate(train.records):
        infos.append(namespace["CameraInfo"](uid=index, R=row["R"], T=row["T"],
            FovX=2 * math.atan(row["width"] / (2 * row["K"][0, 0])), FovY=2 * math.atan(row["height"] / (2 * row["K"][1, 1])),
            image=image, image_path=None, image_name=str(index), width=row["width"], height=row["height"], time=row["time"], mask=None))
    # Preserve the author's pcds branch and its explicit sparse-binary fallback.
    # The fallback point conversion is local to this new source directory.
    if asset.suffix == ".bin":
        ply = Path(source) / "pcds/downsample_0_300.ply"
        converted = Path(source) / "initialization_conversion.json"
        if ply.exists():
            previous = json.loads(converted.read_text()) if converted.is_file() else {}
            if previous.get("source_sha256") != sha(asset) or previous.get("output_sha256") != sha(ply):
                raise ValueError("Sparse fallback conversion exists without matching identities")
        else:
            xyz, rgb, unused = namespace["read_points3D_binary"](str(asset))
            ply.parent.mkdir(parents=True, exist_ok=True)
            namespace["storePly"](str(ply), xyz, rgb)
            converted.write_text(json.dumps(dict(source_sha256=sha(asset), output_sha256=sha(ply),
                                                source="author sparse binary fallback; no new points"), indent=2) + "\n")
    else:
        ply = asset
    pcd = namespace["fetchPly"](str(ply))
    return namespace["SceneInfo"](point_cloud=pcd, train_cameras=train, test_cameras=test,
        video_cameras=namespace["get_render_poses"](infos), nerf_normalization=namespace["getNerfppNorm"](infos),
        ply_path=str(ply), maxtime=300)
