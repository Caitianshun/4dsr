"""CPU-only matching of SR observations, equations and differentiable paths.

Uses the actual project CameraAffine/modkit2 equations and native CEM SSIM.
Does not construct a native CEM GPU model or update any model parameters.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sr_protocol import ARMS, SCHEDULES, CameraCorrection, degradation, image_loss, load_inputs
from check_native_contract import load_nodes
from sr_stage import native_camera
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    torch.set_num_threads(1)
    torch.manual_seed(20261007)
    metadata = []
    for scene, (registration, _) in SCHEDULES.items():
        manifest = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed20261007.json"
        schedule = ROOT / f"output/dynamic_sr_multiview_footprint_20261007/full_materialized/{registration}/{scene}/seed_20261007/schedule.json"
        for arm, controls in ARMS.items():
            teacher = manifest.parent / ("teacher_index_sr_swinir_x4_seed20261007.json" if controls["teacher"] == "original" else "teacher_index_sr_swinir_x4_areaft_seed20261007.json") if controls["teacher"] in {"original", "matched"} else None
            data = load_inputs(manifest, schedule, arm, teacher)
            metadata.append(dict(scene=scene, arm=arm, manifest_sha256=sha(manifest), schedule_sha256=sha(schedule),
                teacher_index_sha256=sha(teacher) if teacher else None, rows=len(data["rows"]), keys=len(data["record_keys"]),
                targets=len(data["targets"]), train_HR_used=arm == "areaOracle", heldout_targets=0))
    affine_source = ROOT / "experiments/dynamic_sr_cull_retrain_20261009/cullretrain.py"
    warp_source = ROOT / "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/modkit2.py"
    area_source = ROOT / "experiments/dynamic_sr_modules_20261010/geoprobe_20261017/calibsr_v2.py"
    bicaa_source = ROOT / "experiments/dynamic_sr_multiview_footprint_20261007/losses.py"
    ns = dict(torch=torch, math=math, REG={})
    load_nodes(affine_source, ["CameraAffine"], ns)
    load_nodes(warp_source, ["make_ext_affine"], ns)
    load_nodes(area_source, ["area_degradation"], ns)
    load_nodes(bicaa_source, ["degradation"], ns)
    camera_ids = ["cam01", "cam03", "cam05"]
    original = ns["make_ext_affine"](ns["CameraAffine"], 0, 4, 0.002, 0)(camera_ids, device="cpu", lr_init=0.001, lr_final=0.001, max_steps=6000)
    new = CameraCorrection(camera_ids, True, device="cpu")
    with torch.no_grad():
        delta = torch.randn_like(new.delta) * 0.01
        field = torch.randn_like(new.field) * 0.1
        new.delta.copy_(delta)
        original.delta.copy_(delta)
        new.field.copy_(field)
        original.wfield.copy_(field)
    first = torch.rand(3, 24, 32, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    actual, expected = new.apply(first, "cam03"), original.apply(second, "cam03")
    assert torch.equal(actual, expected)
    gradients_actual = torch.autograd.grad(actual.square().mean(), [first, new.delta, new.field])
    gradients_expected = torch.autograd.grad(expected.square().mean(), [second, original.delta, original.wfield])
    gradient_errors = [float((x - y).abs().max()) for x, y in zip(gradients_actual, gradients_expected)]
    assert max(gradient_errors) == 0
    assert torch.equal(new.apply(first, "cam00"), first)
    assert torch.equal(degradation(first, (6, 8), "area"), ns["area_degradation"](first, (6, 8)))
    assert torch.equal(degradation(first, (6, 8), "bicaa"), ns["degradation"](first, (6, 8)))
    # Execute author's actual 11x11 Gaussian-window SSIM in CPU-only namespace.
    ssim_source = args.runtime / "utils/loss_utils.py"
    loss_ns = dict(torch=torch, F=F, Variable=Variable, exp=math.exp)
    load_nodes(ssim_source, ["gaussian", "create_window", "ssim", "_ssim"], loss_ns)
    target = torch.rand_like(first)
    loss = image_loss(new.apply(first, "cam03"), target, loss_ns["ssim"])
    gradients = torch.autograd.grad(loss, [first, new.delta, new.field])
    assert all(bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any()) for gradient in gradients)
    # Exercise the actual new SR/render camera helper with CPU metadata, then
    # run the author's projection algebra. No CUDA camera constructor is used.
    import numpy as np
    projection_ns = dict(np=np, math=math, torch=SimpleNamespace(zeros=lambda *shape: np.zeros(shape)))
    load_nodes(args.runtime / "utils/graphics_utils.py", ["getProjectionMatrixCV", "getProjectionMatrix"], projection_ns)
    max_camera_error = 0.0
    for scene in SCHEDULES:
        manifest = json.loads((ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed20261007.json").read_text())
        for camera in sorted(manifest["cameras"]):
            for frame in [0, 150, 299]:
                view = native_camera(manifest, camera, frame, SimpleNamespace)
                width, height = view.resolution
                k = np.asarray(manifest["cameras"][camera]["K_hr"])
                p = projection_ns["getProjectionMatrixCV"](view.near, view.far, view.FoVx, view.FoVy, view.cxr, view.cyr) if view.cyr != 0 else projection_ns["getProjectionMatrix"](view.near, view.far, view.FoVx, view.FoVy)
                points = np.array([[0, 0, 10, 1], [.5, -.2, 20, 1]])
                clip = points @ p.T
                uv = ((clip[:, :2] / clip[:, 3:4] + 1) * [width, height] - 1) / 2
                projected = points[:, :3] @ k.T
                max_camera_error = max(max_camera_error, float(np.abs(uv - projected[:, :2] / projected[:, 2:3]).max()))
                assert view.timestamp == frame and np.array_equal(view.R.T, np.asarray(manifest["cameras"][camera]["w2c"])[:3, :3])
    assert max_camera_error < 1e-9
    # Compare regularizers to expressions extracted from actual native train.py.
    training = next(node for node in ast.parse((args.runtime / "train.py").read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "training")
    loop = next(node for node in training.body if isinstance(node, ast.While))
    native_blocks = [node for node in loop.body if isinstance(node, ast.If) and any(name in ast.unparse(node.test) for name in ["opt.static_reg", "opt.motion_reg", "opt.rot_reg"])]
    assert len(native_blocks) == 3
    code = "def native_regularizer_reference(gaussians, opt, iteration):\n    loss = gaussians._xyz.sum() * 0\n"
    code += "\n".join("    " + line for block in native_blocks for line in ast.unparse(block).splitlines()) + "\n    return loss\n"
    regularizer_ns = dict(torch=torch)
    exec(compile(code, str(args.runtime / "train.py") + ":regularizers", "exec"), regularizer_ns)
    from sr_protocol import native_regularization
    model = SimpleNamespace(_xyz=torch.randn(4, 3), _xyz_disp=torch.randn(4, 3, requires_grad=True),
        _xyz_motion=torch.randn(2, 5, 3, requires_grad=True), _rotation_motion=torch.randn(2, 5, 4, requires_grad=True))
    opt = SimpleNamespace(static_reg=0.0001, motion_reg=0.0001, rot_reg=0.001, progressive_growing_steps=400, make_dynamic_interval=100)
    reg_actual = native_regularization(model, opt, 50001)
    reg_expected = regularizer_ns["native_regularizer_reference"](model, opt, 50001)
    assert torch.equal(reg_actual, reg_expected)
    reg_gradients_actual = torch.autograd.grad(reg_actual, [model._xyz_disp, model._xyz_motion, model._rotation_motion])
    reg_gradients_expected = torch.autograd.grad(reg_expected, [model._xyz_disp, model._xyz_motion, model._rotation_motion])
    assert all(torch.equal(x, y) for x, y in zip(reg_gradients_actual, reg_gradients_expected))
    # Assert the migration cannot silently enter author CEC/topology or Wu code.
    tree = ast.parse((HERE / "sr_stage.py").read_text())
    called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    forbidden = {"densify", "densify_and_prune", "prune_points", "prune_invisible", "prune_nan_points", "expand_duration", "extract_dynamic_points_from_static"}
    assert not (called & forbidden)
    assert "correct_gaussians" not in (HERE / "sr_stage.py").read_text()
    assert not torch.cuda.is_initialized()
    result = dict(status="passed_CPU_SR_contract", native_GPU_fixture_pending=True, CUDA_initialized=False,
        native_CEM_model_constructed=False, native_training_executed=False, parameter_updates=0,
        metadata_cases=metadata, equation_value_bitwise_equal=True, equation_gradient_max_abs_errors=gradient_errors,
        area_bicaa_match_current_source=True, native_SSIM_finite_backward=True,
        SR_render_camera_max_pixel_error=max_camera_error, native_regularizers_value_and_gradient_bitwise_equal=True,
        heldout_correction_exact_identity=True, no_topology_call_in_SR_AST=True,
        semantic_limits="CPU equations/metadata contract only; does not establish native CUDA compatibility, runtime cost, or reconstruction quality",
        sources=[dict(path=str(path), sha256=sha(path)) for path in [Path(__file__), HERE / "sr_protocol.py", HERE / "sr_stage.py", affine_source, warp_source, area_source, bicaa_source, ssim_source]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", metadata_cases=len(metadata), source_equations_match=True, CUDA_initialized=False, receipt=str(args.output), sha256=sha(args.output))))


if __name__ == "__main__":
    main()
