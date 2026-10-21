"""Controlled CEM post-parent SR protocol; no native model assumptions from Wu.

The image correction equations match the current project recipe. Native CEM
geometry, SH appearance, motion knots, hierarchical parents and lifetimes are
rendered and optimized through the author's own model and rasterizer.
"""
import hashlib
import json
from pathlib import Path

ARMS = {
    "areaL0": dict(lr_operator="area", teacher=None, teacher_weight=0.0, w4=False),
    "naive": dict(lr_operator="bicaa", teacher="original", teacher_weight=0.2, w4=False),
    "areaT": dict(lr_operator="area", teacher="original", teacher_weight=0.2, w4=False),
    "areaTm": dict(lr_operator="area", teacher="matched", teacher_weight=0.2, w4=False),
    "areaTm_w4": dict(lr_operator="area", teacher="matched", teacher_weight=0.2, w4=True),
    "areaOracle": dict(lr_operator="area", teacher="train_HR", teacher_weight=0.2, w4=False),
}
GROUPS = ["xyz", "f_dc", "f_rest", "opacity", "opacity_center", "opacity_var",
          "scaling", "rotation", "xyz_disp", "motion_xyz", "motion_rotation"]
SCHEDULES = {
    "cook_spinach": ("Cook_seed20261007_actual_revisionA_1791420669221585809", "dcba36237c390755d33ef8fc5397474188dd6f65be892745f116c9aec64c5b5f"),
    "cut_roasted_beef": ("Cut_seed20261007_actual_revisionA_1791421283013422721", "b633ec33da0023e3846ce8d374491c8bb2974aa62234fc152ce248475c19b0b7"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_child(root, path):
    root, target = Path(root).resolve(), (Path(root) / path).resolve()
    assert target.is_relative_to(root), "Input path escapes manifest directory"
    return target


def load_inputs(manifest, schedule, arm, teacher_index=None):
    manifest, schedule = Path(manifest).resolve(), Path(schedule).resolve()
    value, table = json.loads(manifest.read_text()), json.loads(schedule.read_text())
    assert value["scene"] in SCHEDULES and value["physical_lr_derivation"]["formation"] == "area4"
    assert value["frame_indices"] == list(range(300)) and value["splits"]["test"] == ["cam00"]
    assert table["schema"] == "full_native_SR_schedule_v1" and sha(schedule) == SCHEDULES[value["scene"]][1]
    assert arm in ARMS
    controls = dict(ARMS[arm])
    keys = [tuple(key) for key in table["record_keys"]]
    assert len(keys) == len(set(keys))
    expected = {(camera, frame) for camera in value["splits"]["train"] for frame in range(300)}
    assert set(keys) == expected and "cam00" not in value["splits"]["train"]
    observations = {(item["camera_id"], item["frame_index"]): item for item in value["observations"] if item["split"] == "train"}
    assert set(observations) == expected
    rows = table["rows"]
    assert len(rows) == 6000
    for row in rows:
        assert len(row) == 3 and all(type(index) is int and 0 <= index < len(keys) for index in row)
        selected = [keys[index] for index in row]
        assert len({key[0] for key in selected}) == 3 and len({key[1] for key in selected}) == 1
    targets = {}
    index_identity = None
    if controls["teacher"] in {"original", "matched"}:
        assert teacher_index is not None
        teacher_index = Path(teacher_index).resolve()
        teacher = json.loads(teacher_index.read_text())
        assert teacher["status"] == "completed_teacher_inventory" and not teacher["oracle"]
        assert teacher["manifest_sha256"] == sha(manifest)
        assert set(teacher["train_cameras"]) == set(value["splits"]["train"])
        required_kind = "swinir_classical_x4" if controls["teacher"] == "original" else "swinir_area_finetuned_x4"
        assert teacher["teacher_config"]["teacher"] == required_kind
        for entry in teacher["entries"]:
            key = (entry["camera"], entry["frame"])
            assert key in observations and key not in targets
            assert entry["lr_sha256"] == observations[key]["lr_sha256"]
            targets[key] = dict(path=str(safe_child(manifest.parent, entry["relative_path"])), sha256=entry["sha256"])
        assert set(targets) == expected
        index_identity = dict(path=str(teacher_index), sha256=sha(teacher_index), pretrained_teacher=teacher["teacher_config"]["weights"])
    elif controls["teacher"] == "train_HR":
        assert teacher_index is None
        targets = {key: dict(path=str(safe_child(manifest.parent, item["hr_path"])), sha256=item["hr_sha256"]) for key, item in observations.items()}
    else:
        assert teacher_index is None
    return dict(manifest_path=manifest, manifest=value, observations=observations,
                schedule_path=schedule, record_keys=keys, rows=rows, controls=controls,
                targets=targets, teacher_index=index_identity)


def degradation(rgb, size, operator):
    import torch.nn.functional as F
    if operator == "area":
        assert tuple(rgb.shape[-2:]) == (int(size[0]) * 4, int(size[1]) * 4)
        return F.avg_pool2d(rgb[None], 4)[0].clamp(0, 1)
    assert operator == "bicaa"
    return F.interpolate(rgb[None], size=tuple(size), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)


def image_loss(prediction, target, native_ssim):
    return 0.8 * (prediction - target).abs().mean() + 0.2 * (1 - native_ssim(prediction[None], target[None]))


def native_regularization(gaussians, opt, iteration):
    import torch
    # Exact native train.py expressions, evaluated once per optimizer step.
    value = gaussians._xyz.sum() * 0
    active = iteration > opt.progressive_growing_steps + opt.make_dynamic_interval
    if opt.static_reg > 0 and active:
        value = value + opt.static_reg * torch.log(gaussians._xyz_disp.norm(dim=-1) + 0.001).mean()
    if opt.motion_reg > 0 and active and gaussians._xyz_motion.shape[0] > 0:
        diff1 = gaussians._xyz_motion[:, :1] - gaussians._xyz_motion[:, 1:]
        value = value + opt.motion_reg * diff1.norm(dim=-1).mean()
    if opt.rot_reg > 0 and active and gaussians._xyz_motion.shape[0] > 0:
        r1, r2 = gaussians._rotation_motion[:, 1:], gaussians._rotation_motion[:, :-1]
        r_i = 1 - (r1 * r2).sum(dim=-1) / r1.norm(dim=-1).clamp_min(1e-6) / r2.norm(dim=-1).clamp_min(1e-6)
        value = value + opt.rot_reg * r_i.mean()
    return value


class CameraCorrection:
    """Mean-identity 3x4 colour affine; optional mean-zero 4x4 xy warp.

    All SR arms start the common colour affine at zero. Native CEM parent has
    no extra affine to inherit. No time offset or five-parameter calibration.
    Held-out cameras use identity for both corrections.
    """
    def __init__(self, cameras, w4, device="cuda"):
        import torch
        self.names = list(cameras)
        assert "cam00" not in self.names
        self.index = {camera: index for index, camera in enumerate(self.names)}
        self.delta = torch.zeros(len(cameras), 3, 4, device=device, requires_grad=True)
        self.field = torch.zeros(len(cameras), 2, 4, 4, device=device, requires_grad=w4)
        self.use_w4 = w4
        self.affine_optimizer = torch.optim.Adam([self.delta], lr=0.001)
        self.field_optimizer = torch.optim.Adam([self.field], lr=0.002) if w4 else None

    def apply(self, image, camera):
        import torch
        import torch.nn.functional as F
        if camera not in self.index:
            return image
        index = self.index[camera]
        if self.use_w4:
            height, width = image.shape[-2:]
            py, px = torch.meshgrid(torch.arange(height, dtype=image.dtype, device=image.device) + 0.5,
                                    torch.arange(width, dtype=image.dtype, device=image.device) + 0.5, indexing="ij")
            displacement = self.field[index] - self.field.mean(0)
            up = F.interpolate(displacement[None], size=(height, width), mode="bilinear", align_corners=True)[0]
            grid = torch.stack([(px + up[0]) * (2 / width) - 1, (py + up[1]) * (2 / height) - 1], -1)
            image = F.grid_sample(image[None], grid[None], mode="bicubic", padding_mode="border", align_corners=False)[0]
        delta = self.delta[index] - self.delta.mean(0)
        return torch.einsum("ij,jhw->ihw", torch.eye(3, dtype=image.dtype, device=image.device) + delta[:, :3], image) + delta[:, 3, None, None]

    def step(self, cursor):
        self.affine_optimizer.step()
        self.affine_optimizer.zero_grad(set_to_none=True)
        if self.field_optimizer is not None:
            for group in self.field_optimizer.param_groups:
                group["lr"] = 0.002 * (0.1 ** min(max(cursor / 6000, 0), 1))
            self.field_optimizer.step()
            self.field_optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(names=self.names, delta=self.delta.detach().clone(), field=self.field.detach().clone(), w4=self.use_w4,
                    affine_optimizer=self.affine_optimizer.state_dict(),
                    field_optimizer=self.field_optimizer.state_dict() if self.field_optimizer else None)

    def load_for_render(self, value):
        import torch
        assert value["names"] == self.names and value["w4"] == self.use_w4
        assert value["delta"].shape == self.delta.shape and value["field"].shape == self.field.shape
        with torch.no_grad():
            self.delta.copy_(value["delta"])
            self.field.copy_(value["field"])
