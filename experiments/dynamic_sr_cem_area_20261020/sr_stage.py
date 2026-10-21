"""CEM controlled six-arm SR continuation, absolute 50001..56000.

This is a new supervised continuation experiment, not an author's CEM SR
method. The accepted native parent is immutable. All eleven native RAdam
parameter groups and their state continue; topology/CEC remain finished.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback

from sr_protocol import ARMS, GROUPS, CameraCorrection, degradation, image_loss, load_inputs, native_regularization, safe_child, sha


def native_camera(manifest, camera, frame, constructor):
    import numpy as np
    spec = manifest["cameras"][camera]
    width, height = manifest["resolutions"]["hr"]
    k, w2c = np.asarray(spec["K_hr"]), np.asarray(spec["w2c"])
    physical_ids = {name: index + 1 for index, name in enumerate(sorted(manifest["cameras"]))}
    return constructor(colmap_id=physical_ids[camera], R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
        FoVx=2 * math.atan(width / (2 * k[0, 0])), FoVy=2 * math.atan(height / (2 * k[1, 1])),
        gt_alpha_mask=None, image=None, image_name=f"{frame:04d}.png", image_path=f"protocol-metadata://{camera}/{frame:04d}.png",
        uid=physical_ids[camera], timestamp=int(frame), data_device="cuda", near=0.01, far=300.0,
        cxr=(float(k[0, 2]) + 0.5) / width - 0.5, cyr=(float(k[1, 2]) + 0.5) / height - 0.5,
        resolution=(width, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-stage2-receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--wiring-fixture", action="store_true")
    args = parser.parse_args()
    assert args.gpu.isdecimal() and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu, "Broker physical GPU mask mismatch before imports"
    assert 0 < args.steps <= 6000
    assert (args.steps == 6000 and not args.wiring_fixture) or (args.steps < 6000 and args.wiring_fixture)
    assert not args.output.exists(), "Never overwrite a branch or replay it automatically"
    inputs = load_inputs(args.manifest, args.schedule, args.arm, args.teacher_index)
    runtime, parent, output = args.runtime.resolve(), args.parent_checkpoint.resolve(), args.output.resolve()
    protocol = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    assert protocol["version"] == "native_area4_v1"
    assert sha(runtime / "protocol_interface.py") == protocol["interface_sha256"]
    for item in protocol["core_source_files"] + protocol["configurations"]:
        assert sha(runtime / item["path"]) == item["sha256"]
    parent_receipt = json.loads(args.parent_stage2_receipt.read_text())
    assert parent_receipt["status"] == "completed" and parent_receipt["stage"] == 2
    assert parent_receipt["train_level"] == "lr" and parent_receipt["endpoint"] == 50000
    assert parent_receipt["manifest"]["sha256"] == sha(args.manifest)
    assert parent_receipt["final_checkpoint"]["sha256"] == sha(parent)
    os.environ.update(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = soft if soft == resource.RLIM_INFINITY else max(soft, 65536 if hard == resource.RLIM_INFINITY else min(65536, hard))
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    sys.path.insert(0, str(runtime))
    # Imports occur only after the separately admitted physical-device gate.
    import numpy as np
    import torch
    from PIL import Image
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene import init_model
    from scene.cameras import Cameravideo
    from utils.loss_utils import ssim
    output.mkdir(parents=True)
    began = time.monotonic()
    state = dict(status="running", protocol_type="CEM_area4_backbone_control_v1", arm=args.arm,
        result_scope="wiring_fixture_only" if args.wiring_fixture else "controlled_fixed_topology_SR_continuation",
        started_utc=datetime.now(timezone.utc).isoformat(), steps_requested=args.steps,
        manifest=dict(path=str(args.manifest.resolve()), sha256=sha(args.manifest)),
        schedule=dict(path=str(args.schedule.resolve()), sha256=sha(args.schedule), source="exact current Wu same-time rows; no new sampling"),
        parent_checkpoint=dict(path=str(parent), sha256=sha(parent)),
        parent_stage2_receipt=dict(path=str(args.parent_stage2_receipt.resolve()), sha256=sha(args.parent_stage2_receipt)),
        runtime_receipt=dict(path=str(runtime / "protocol_interface_receipt.json"), sha256=sha(runtime / "protocol_interface_receipt.json")),
        source_files=[dict(path=str(Path(__file__).resolve().parent / name), sha256=sha(Path(__file__).resolve().parent / name)) for name in ["sr_stage.py", "sr_protocol.py"]],
        controls=inputs["controls"], teacher_index=inputs["teacher_index"], train_HR_used=args.arm == "areaOracle",
        heldout_pixels_used=False, native_scheduler_repaired=False, optimizer_state_inherited=True,
        parent_affine=False, common_SR_affine="zero initialized; lr0.001; mean identity; cam00 identity",
        topology_events=0, CEC_events=0, densification_events=0, pruning_events=0,
        gradient_error_statistics_updated=False, native_background="author random_background setting, shared across three views per step",
        native_regularizers="exact static/motion/rot expressions once per step",
        steps_completed=0, parameter_updates=0, absolute_start=50001,
        checkpoint_order="SR endpoint saved after all requested optimizer updates; native parent endpoint untouched",
        device_guard=dict(physical_mask=args.gpu, native_logical_device=0),
        RLIMIT_NOFILE=dict(previous_soft=soft, new_soft=new_soft, hard=hard))
    try:
        p = argparse.ArgumentParser()
        model_group, optimizer_group, pipeline_group = ModelParams(p), OptimizationParams(p), PipelineParams(p)
        native_args = p.parse_args([])
        native_args.__dict__.update(json.loads((runtime / f"configs/N3V_s2/{inputs['manifest']['scene']}_prepared_lr.json").read_text()))
        native_args.duration = 300
        native_args.source_path, native_args.model_path = str(parent.parent), str(output)
        dataset, opt, pipe = model_group.extract(native_args), optimizer_group.extract(native_args), pipeline_group.extract(native_args)
        # Same native placement as author stage2: logical CUDA0 model tensors
        # and CPU RAdam step scalars stay on their original device types.
        payload, iteration = torch.load(parent, weights_only=False)
        assert iteration == 50000 and len(payload) == 29
        gaussians = init_model(dataset.model, dataset)
        gaussians.restore(payload, opt)
        del payload
        assert [group["name"] for group in gaussians.optimizer.param_groups] == GROUPS
        assert type(gaussians.optimizer).__name__ == "RAdam"
        assert all(parameter.requires_grad for group in gaussians.optimizer.param_groups for parameter in group["params"])
        correction = CameraCorrection(inputs["manifest"]["splits"]["train"], inputs["controls"]["w4"])
        # Native checkpoint format does not save RNG. Every arm starts this same
        # explicitly recorded SR RNG; the frozen schedule is already identical.
        random.seed(20261007)
        np.random.seed(20261007)
        torch.manual_seed(20261007)
        torch.cuda.manual_seed_all(20261007)
        state["SR_RNG"] = dict(seed=20261007, inherited_from_parent=False, reason="native capture has no RNG payload")
        topology_before = dict(parent=gaussians._parent.detach().clone(), level=gaussians._level.detach().clone(),
                               static=int(gaussians._xyz.shape[0]), motion=int(gaussians._xyz_motion.shape[0]),
                               duration=gaussians.duration, keyframe_num=gaussians.keyframe_num)
        state["initial_point_counts"] = dict(rendered_gaussians=topology_before["static"], virtual_motion_centers=topology_before["motion"])
        state["native_optimizer_groups"] = [dict(name=group["name"], initial_lr=group["lr"], parameter_shapes=[list(p.shape) for p in group["params"]]) for group in gaussians.optimizer.param_groups]
        checked = set()

        def read_rgb(path, expected_sha, expected_size):
            path = Path(path)
            if path not in checked:
                assert sha(path) == expected_sha
                checked.add(path)
            with Image.open(path) as image:
                assert image.size == tuple(expected_size)
                array = np.array(image.convert("RGB"), copy=True)
            return torch.from_numpy(array).permute(2, 0, 1).float().cuda().div_(255)

        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        log = (output / "trajectory.jsonl").open("x", buffering=1)
        nonfinite_var_corrected = 0
        gradient_counts = {name: dict(steps_with_grad=0, sampled_first_and_final_steps_with_nonzero_grad=0) for name in GROUPS}
        def optimizer_steps():
            return {group["name"]: [int(gaussians.optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
                    for group in gaussians.optimizer.param_groups}
        initial_optimizer_steps = optimizer_steps()
        for cursor in range(1, args.steps + 1):
            iteration = 50000 + cursor
            gaussians.update_learning_rate(iteration)  # Preserve author's early return and absolute LR clock.
            gaussians.optimizer.zero_grad(set_to_none=True)
            keys = [inputs["record_keys"][index] for index in inputs["rows"][cursor - 1]]
            bg = torch.rand(3, device="cuda") if opt.random_background else background
            images = []
            for camera, frame in keys:
                view = native_camera(inputs["manifest"], camera, frame, Cameravideo)
                image = render(view, gaussians, pipe, bg, near=dataset.near, far=dataset.far)["render"]
                assert tuple(image.shape) == (3, 1008, 1344) and bool(torch.isfinite(image).all())
                images.append(correction.apply(image, camera))
            item = inputs["observations"][keys[0]]
            lr = read_rgb(safe_child(inputs["manifest_path"].parent, item["lr_path"]), item["lr_sha256"], [336, 252])
            lr_term = image_loss(degradation(images[0], lr.shape[-2:], inputs["controls"]["lr_operator"]), lr, ssim)
            teacher_term = images[1].sum() * 0 + images[2].sum() * 0
            if inputs["controls"]["teacher_weight"]:
                for position in [1, 2]:
                    target = inputs["targets"][keys[position]]
                    rgb = read_rgb(target["path"], target["sha256"], [1344, 1008])
                    teacher_term = teacher_term + 0.2 * image_loss(images[position], rgb, ssim)
            regularizer = native_regularization(gaussians, opt, iteration)
            loss = lr_term + teacher_term + regularizer
            assert bool(torch.isfinite(loss)), "Nonfinite SR objective; no automatic retry"
            loss.backward()
            with torch.no_grad():
                var = gaussians._opacity_duration_var
                if var.numel() and var.grad is not None:
                    nonfinite_var_corrected += int((~torch.isfinite(var.grad)).sum())
                    var.grad = var.grad.nan_to_num()  # Original author's dynamic opacity safeguard.
                finite_grad = [torch.isfinite(p.grad).all() for group in gaussians.optimizer.param_groups for p in group["params"] if p.grad is not None]
                assert bool(torch.stack(finite_grad).all()), "Nonfinite native gradient; fixed topology forbids silently pruning it"
                for group in gaussians.optimizer.param_groups:
                    has_grad = any(p.grad is not None for p in group["params"])
                    gradient_counts[group["name"]]["steps_with_grad"] += int(has_grad)
                    # Record nonzero evidence only at first/final step to avoid
                    # adding eleven device synchronizations on every iteration.
                    if cursor in {1, args.steps}:
                        nonzero = any(p.grad is not None and bool((p.grad != 0).any()) for p in group["params"])
                        gradient_counts[group["name"]]["sampled_first_and_final_steps_with_nonzero_grad"] += int(nonzero)
                gaussians.optimizer.step()
                correction.step(cursor)
                state["parameter_updates"] += 1
                assert (int(gaussians._xyz.shape[0]), int(gaussians._xyz_motion.shape[0])) == (topology_before["static"], topology_before["motion"])
            state["steps_completed"] = cursor
            log.write(json.dumps(dict(cursor=cursor, absolute_iteration=iteration, keys=keys, LR_term=float(lr_term.detach()),
                teacher_term=float(teacher_term.detach()), regularizer=float(regularizer.detach()), loss=float(loss.detach()),
                topology_events=0, native_LRs={group["name"]: group["lr"] for group in gaussians.optimizer.param_groups})) + "\n")
            del images, loss, lr_term, teacher_term, regularizer
        log.close()
        assert torch.equal(gaussians._parent, topology_before["parent"]) and torch.equal(gaussians._level, topology_before["level"])
        assert gaussians.duration == topology_before["duration"] and gaussians.keyframe_num == topology_before["keyframe_num"]
        assert all(bool(torch.isfinite(p).all()) for group in gaussians.optimizer.param_groups for p in group["params"])
        checkpoint = output / f"checkpoint_{args.steps:05d}.pt"
        torch.save(dict(schema="CEM_area4_SR_fixed_topology_v1", model=gaussians.capture(), iteration=50000 + args.steps,
                        SR_updates=args.steps, correction=correction.state_dict(), native_arguments=vars(native_args),
                        protocol=state, rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), checkpoint)
        state.update(status="completed", checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
                     final_point_counts=dict(rendered_gaussians=int(gaussians._xyz.shape[0]), virtual_motion_centers=int(gaussians._xyz_motion.shape[0])),
                     topology_parent_and_level_unchanged=True, native_gradient_participation=gradient_counts,
                     nonfinite_dynamic_opacity_gradients_corrected_like_author=nonfinite_var_corrected,
                     unique_input_images_SHA_checked=len(checked), final_iteration=50000 + args.steps)
        final_optimizer_steps = optimizer_steps()
        state["native_RAdam_step_counters"] = {name: dict(initial=initial_optimizer_steps[name], final=final_optimizer_steps[name],
            updates=[end - start for start, end in zip(initial_optimizer_steps[name], final_optimizer_steps[name])]) for name in GROUPS}
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic() - began, finished_utc=datetime.now(timezone.utc).isoformat(),
                     torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     peak_memory_scope="Torch allocator only; external Faiss allocations excluded", gpu_name=torch.cuda.get_device_name(0))
        (output / "training_receipt.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
