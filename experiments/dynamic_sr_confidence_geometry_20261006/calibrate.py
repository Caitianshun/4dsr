"""Fixed train32 coefficient calibration and finite complete-Adam diagnostics.

No source model optimizer step, HR pixels, HR-derived supervision or quality
selection is allowed. Five independent shadow updates per registered probe and
two additional separate-backward checks plus four unchanged-path repeats are
explicitly charged to diagnostics. A rescue restores frozen coefficient statistics
and recollects only the missing parameter-gradient bank.
The evaluation-GPU lock prevents local endpoint evaluation from sharing its GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import shutil
import sys
import time
import traceback

try:
    from .cg_common import ROOT, HERE, OUT, local, read, write, sha, bound, module, setup
except ImportError:
    from cg_common import ROOT, HERE, OUT, local, read, write, sha, bound, module, setup


GROUPS = ("position", "shared_deformation", "SH", "scale_rotation", "opacity", "children")
SHADOW_ARMS = ("LR_only", "C1", "R", "G", "RG")
BACKWARD_TOLERANCE = dict(relative_l2=2e-5, absolute_max=1e-9)
NATIVE_REPEAT_ENVELOPE_MULTIPLIER = 4.
ADAM_TOLERANCE = dict(relative_update_l2=2e-4, absolute_parameter_max=1e-6)


def group_name(name):
    if name.startswith("children."):
        return "children"
    if name.startswith("deformation."):
        return "shared_deformation"
    if name == "_xyz":
        return "position"
    if name in ("_features_dc", "_features_rest"):
        return "SH"
    if name in ("_scaling", "_rotation"):
        return "scale_rotation"
    if name == "_opacity":
        return "opacity"
    raise ValueError(f"Unregistered parameter group: {name}")


def gpu_snapshot():
    """Read the explicitly bound physical GPU without initializing CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Bind exactly one reserved GPU with CUDA_VISIBLE_DEVICES")
    output = subprocess.check_output([
        "nvidia-smi", "-i", visible,
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    parts = [part.strip() for part in output.split(",")]
    if len(parts) != 5:
        raise RuntimeError("Unexpected nvidia-smi GPU query")
    applications = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader"], text=True).strip()
    busy = [line for line in applications.splitlines()
            if line.split(",")[0].strip() == parts[2]
            and line.split(",")[1].strip() != str(os.getpid())]
    if busy:
        raise RuntimeError(f"Reserved GPU has existing compute processes: {busy}")
    return dict(physical_binding=visible, index=parts[0], name=parts[1], uuid=parts[2],
                memory_used=parts[3], utilization=parts[4], applications=applications,
                checked_unix=time.time())


def cpu_grads(names, gradients):
    return {name: None if value is None else value.detach().cpu().clone()
            for name, value in zip(names, gradients)}


def combine(*terms):
    result = {}
    for gradients, weight in terms:
        for name, gradient in gradients.items():
            if gradient is None:
                result.setdefault(name, None)
                continue
            value = gradient * float(weight)
            if result.get(name) is None:
                result[name] = value.clone()
            else:
                result[name].add_(value)
    return result


def compare_gradients(a, b, parameter_sizes):
    rows = {}
    for group in GROUPS:
        square_a = square_b = delta_square = dot = 0.
        maximum = 0.
        elements = active = conflict = 0
        for name in a:
            if group_name(name) != group:
                continue
            x, y = a[name], b[name]
            elements += parameter_sizes[name]
            if x is not None:
                square_a += float(x.double().square().sum())
            if y is not None:
                square_b += float(y.double().square().sum())
            if x is not None and y is not None:
                dot += float((x.double() * y.double()).sum())
                delta = x.double() - y.double()
                active += int(((x != 0) & (y != 0)).sum())
                conflict += int((x.double() * y.double() < 0).sum())
            else:
                delta = x.double() if x is not None else y.double() if y is not None else None
            if delta is not None:
                delta_square += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.)
        rows[group] = dict(a_l2=square_a ** .5, b_l2=square_b ** .5,
            a_rms=(square_a / max(elements, 1)) ** .5, b_rms=(square_b / max(elements, 1)) ** .5,
            cosine=dot / (square_a * square_b) ** .5 if square_a and square_b else None,
            difference_l2=delta_square ** .5,
            difference_relative_l2=(delta_square / max(square_a, square_b, 1e-300)) ** .5,
            difference_maximum_absolute=maximum,
            sign_conflict_fraction=conflict / active if active else None, elements=elements)
    return rows


def equivalence_pass(rows, tolerance):
    return all(row["difference_relative_l2"] <= tolerance["relative_l2"]
               or row["difference_maximum_absolute"] <= tolerance["absolute_max"]
               for row in rows.values())


def native_equivalence(rows, joint_repeat, separate_repeat, tolerance=BACKWARD_TOLERANCE,
                       registered_minimum=None):
    """Compare coupled backprop difference with unchanged native repeated calls.

    The fixed factor is registered after preserving the initial failed absolute
    tolerance and its same-objective CUDA-repeat diagnostic. This changes only
    engineering acceptance, never a model coefficient or quality selection.
    """
    envelope = {}
    for group, row in rows.items():
        relative = max(tolerance["relative_l2"], NATIVE_REPEAT_ENVELOPE_MULTIPLIER *
                       max(joint_repeat[group]["difference_relative_l2"],
                           separate_repeat[group]["difference_relative_l2"]))
        absolute = max(tolerance["absolute_max"], NATIVE_REPEAT_ENVELOPE_MULTIPLIER *
                       max(joint_repeat[group]["difference_maximum_absolute"],
                           separate_repeat[group]["difference_maximum_absolute"]))
        if registered_minimum is not None:
            relative = max(relative, registered_minimum[group]["relative_l2_tolerance"])
            absolute = max(absolute, registered_minimum[group]["absolute_max_tolerance"])
        passed = (math.isfinite(row["difference_relative_l2"])
                  and math.isfinite(row["difference_maximum_absolute"])
                  and (row["difference_relative_l2"] <= relative
                       or row["difference_maximum_absolute"] <= absolute))
        envelope[group] = dict(relative_l2_tolerance=relative, absolute_max_tolerance=absolute,
                               passed=passed)
    return all(row["passed"] for row in envelope.values()), envelope


def register_Adam_native_envelope():
    """Bind prior unchanged-path repeats before any rescue CUDA operation."""
    paths = [OUT / "Adam_equivalence_failure_diagnostic.json",
             OUT / "calibration_failure_3/Adam_equivalence_acceptance.json"]
    paths = [path for path in paths if path.exists()]
    repeats = []
    for path in paths:
        evidence = read(path)
        if "joint_repeat" in evidence:
            repeats.extend([evidence["joint_repeat"], evidence["separate_repeat"]])
        for row in evidence.get("rows", []):
            for key in ("same_joint_Adam_repeat", "same_separate_Adam_repeat"):
                if key in row:
                    repeats.append(row[key])
    minimum = {}
    for group in GROUPS:
        relative_noise = max((row[group]["difference_relative_l2"] for row in repeats), default=0.)
        absolute_noise = max((row[group]["difference_maximum_absolute"] for row in repeats), default=0.)
        minimum[group] = dict(
            observed_unchanged_path_relative_noise=relative_noise,
            observed_unchanged_path_absolute_noise=absolute_noise,
            relative_l2_tolerance=max(ADAM_TOLERANCE["relative_update_l2"], 4 * relative_noise),
            absolute_max_tolerance=max(ADAM_TOLERANCE["absolute_parameter_max"], 4 * absolute_noise))
    registration = dict(status="registered_before_rescue", evidence=[dict(path=str(path.relative_to(ROOT)),
        sha256=sha(path)) for path in paths], minimum_envelope=minimum, multiplier=4,
        final_rule="max(registered previous same-path repeated noise, current same-path repeated noise)*4; original floors retained",
        coupled_joint_vs_separate_difference_used_to_set_tolerance=False, quality_used=False)
    path = OUT / "Adam_native_noise_registration.json"
    if path.exists() and read(path) != registration:
        raise ValueError("Registered native-noise evidence changed")
    write(path, registration)
    return registration


def norm_summary(gradients, parameter_sizes):
    return {group: {key: value for key, value in row.items() if key in ("a_l2", "a_rms", "elements")}
            for group, row in compare_gradients(gradients, gradients, parameter_sizes).items()}


def main_locked(args, resources):
    # Environment-dependent imports happen after claiming the evaluation GPU.
    shared, policy = setup()
    import numpy as np
    import torch
    from common import image_tensor, resized_camera, downsample
    from n3dv_data import load_manifest, observation_to_4dgs_camera
    from motion_model import load_model, capacity_summary
    from training_support import initial_identity, digest_state
    resume = module("cg_calibration_resume", ROOT / "experiments/dynamic_sr_20260920/resume_control.py")
    gm = module("cg_calibration_depth", HERE / "depth_prior.py")
    cm = module("cg_calibration_confidence", HERE / "confidence_cache.py")
    opmod = module("cg_calibration_operator", HERE / "degradation_operator.py")
    geometry_checks = read(OUT / "geometry_cuda_checks_v2.json")
    if geometry_checks.get("status") != "passed" or geometry_checks["depth_prior_sha256"] != sha(HERE / "depth_prior.py"):
        raise ValueError("G must pass its CUDA fixture check at this exact source version")
    torch.set_num_threads(4)
    protocol = read(OUT / "protocol.json")
    protocol_hash = sha(OUT / "protocol.json")
    geometry_path = args.geometry_index
    confidence_path = args.confidence_index
    geometry_index, confidence_index = read(geometry_path), read(confidence_path)
    if geometry_index.get("privileged_train_hr", True) is not False:
        raise ValueError("Geometry must explicitly exclude HR-derived targets")
    if confidence_index.get("hr_derived_training_items", True) is not False:
        raise ValueError("Confidence must explicitly exclude HR-derived targets")
    if geometry_index["parent_sha256"] != protocol["parent"]["sha256"]:
        raise ValueError("Geometry does not originate from registered U6000")
    for index in (geometry_index, confidence_index):
        if index["manifest_sha256"] != protocol["manifest"]["sha256"]:
            raise ValueError("Training cache manifest mismatch")
    manifest = load_manifest(bound(protocol["manifest"]))
    parent_path = bound(protocol["parent"])
    teachers = read(bound(protocol["teacher"]))
    teacher_by = {(e["camera"], int(e["frame"])): e for e in teachers["entries"]}
    legal = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"] if o["split"] == "train"}
    train76 = sorted((key for key in legal if key[1] in (0, 40, 80, 118)))
    if len(legal) != 1140 or len(train76) != 76:
        raise ValueError("Registered 1140/train76 boundary differs")
    probes = train76[:32]
    backup_probes = train76[32:64]
    plan = dict(observations=[dict(camera=c, frame=f) for c, f in probes],
        backup_G_observations=[dict(camera=c, frame=f) for c, f in backup_probes],
        selection="camera_id then original frame; first 32 legal train76",
        backup_G_selection="next 32 legal train76; only if G denominator is numerically zero",
        quality_selection=False, k_R_confidence_used=False, HR_used=False,
        shadow_rescue_only=(OUT / "calibration_coefficients_pending_shadow.json").exists(),
        coefficient_recalculation_on_rescue=False, giant_gradient_bank_from_disk=False,
        backward_equivalence_observations=[dict(camera=c, frame=f) for c, f in probes[:2]],
        shadow_arms=list(SHADOW_ARMS), shadow_updates=166,
        backward_tolerance=BACKWARD_TOLERANCE, adam_tolerance=ADAM_TOLERANCE,
        native_repeat_envelope_multiplier=NATIVE_REPEAT_ENVELOPE_MULTIPLIER,
        tolerance_revision_evidence=["backward_equivalence_failure_diagnostic.json", "Adam_equivalence_failure_diagnostic.json",
                                    "preserved calibration_failure_1 and calibration_failure_2"],
        native_repeat_comparison="same graph, same objective, same parameters; repeated joint and repeated separate")
    Adam_noise_registration = register_Adam_native_envelope()
    plan["Adam_noise_evidence"] = Adam_noise_registration
    plan_path = OUT / "calibration_plan.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Calibration plan changed")
    write(plan_path, plan)
    fixed_coefficients_path = OUT / "calibration_coefficients_pending_shadow.json"
    fixed_coefficients_sha = sha(fixed_coefficients_path) if fixed_coefficients_path.exists() else None
    fixed_coefficients = read(fixed_coefficients_path) if fixed_coefficients_sha else None
    allowed = {str(local(e["path"]).resolve()): e["sha256"] for e in protocol["training_files"]}
    for path, index in ((geometry_path, geometry_index), (confidence_path, confidence_index)):
        for e in index["entries"]:
            allowed[str((path.parent / e["path"]).resolve())] = e["sha256"]
    actual_reads = Counter()
    verified = set()

    def guard(event, arguments):
        if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".npz", ".npy"):
            return
        if path.suffix.lower() in (".png", ".jpg", ".jpeg") and "hr" in path.parts:
            raise RuntimeError(f"Calibration may not read HR pixels: {path}")
        if str(path) not in allowed:
            raise RuntimeError(f"Calibration attempted to read unregistered pixel/cache input: {path}")
        actual_reads[str(path)] += 1
    sys.addaudithook(guard)

    def verify(path):
        path = Path(path).resolve()
        if path not in verified:
            if allowed.get(str(path)) != sha(path):
                raise ValueError(f"Calibration input checksum differs: {path}")
            verified.add(path)

    # CPU identity verification supplies a spaced second idle reading.
    for camera, frame in probes:
        obs, teacher = legal[camera, frame], teacher_by[camera, frame]
        for rel in (obs["lr_path"], teacher["relative_path"]):
            verify(Path(manifest["_root"]) / rel)
    resources.append(gpu_snapshot())
    started = time.monotonic()
    cost = Counter(rgb_forwards=0, moment_forwards=0, source_param_gradients=0,
                   effective_xyz_gradients=0, pixel_gradient_calls=0,
                   backward_equivalence_calls=0, shadow_optimizer_rounds=0,
                   shadow_adam_calls=0, formal_training_updates=0,
                   disconnected_G_route_gradient_calls=0,
                   native_RGB_rasterizer_backwards=0, native_moment_rasterizer_backwards=0)
    model = load_model(parent_path, manifest)
    if model.branch != "ordinary_split" or capacity_summary(model)["active_gaussians"] != 132972:
        raise ValueError("Calibration mother differs from fixed U6000")
    resume.assert_state_equal(model.checkpoint["model"][12], model.g.optimizer.state_dict())
    resume.assert_state_equal(model.checkpoint["motion_refinement"]["child_optimizer"], model.child_optimizer.state_dict())
    resume.restore_global_rng(model.checkpoint["rng"])

    def rng():
        return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                    numpy=np.random.get_state(), python=random.getstate())
    rng_before = rng()
    before = initial_identity(model)
    trainable = {name: value for name, value in shared.all_named(model).items() if value.requires_grad}
    names, parameters = list(trainable), list(trainable.values())
    parameter_sizes = {name: value.numel() for name, value in trainable.items()}
    parameter_before = {name: value.detach().cpu().clone() for name, value in trainable.items()}
    position = parameter_before["_xyz"]
    if position.dtype != torch.float32:
        raise ValueError("Registered position precision audit assumes float32 parameters")
    ulp_up = torch.nextafter(position, torch.full_like(position, float("inf"))) - position
    ulp_down = position - torch.nextafter(position, torch.full_like(position, -float("inf")))
    position_ulp = torch.maximum(ulp_up, ulp_down)
    precision_dir = OUT / "calibration_precision"
    precision_dir.mkdir(parents=True, exist_ok=True)
    precision_path = precision_dir / "position_float32_ulp.pt"
    torch.save(dict(initial_position=position, ulp_towards_positive=ulp_up,
                    ulp_towards_negative=ulp_down, conservative_ulp=position_ulp), precision_path)
    position_precision = dict(path=str(precision_path.relative_to(ROOT)), sha256=sha(precision_path),
        units="one float32 spacing at each original canonical xyz coordinate",
        shape=list(position.shape), ulp_min=float(position_ulp.min()), ulp_max=float(position_ulp.max()))
    rng_identity = digest_state(rng_before)
    source_paths = [HERE / name for name in ("calibrate.py", "degradation_operator.py", "depth_prior.py", "confidence_cache.py")]
    source_paths += [ROOT / "experiments/dynamic_sr_prior_guidance_20260927" / name for name in ("gradient_policy.py", "shared.py")]
    source_paths += [ROOT / "experiments/dynamic_sr_motion_bound_20260923/motion_model.py",
                     ROOT / "experiments/dynamic_sr_20260918/common.py",
                     ROOT / "experiments/dynamic_sr_20260918/n3dv_data.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    for path in source_paths:
        snapshot = OUT / "calibration_sources" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    op = opmod.ActualDegradation((1008, 1344), (252, 336), device="cuda")
    checks = read(OUT / "operator_checks_cuda3090_full.json")
    if not checks["passed"]:
        raise ValueError("Actual-degradation operator must pass before calibration")
    exact_case = next(c for c in checks["cases"] if c["operator"]["hr_size"] == [1008, 1344]
                      and c["operator"]["runtime_dtype"] == "torch.float32")
    if exact_case["operator"]["axes"] != op.metadata()["axes"]:
        raise ValueError("Operator differs from accepted impulse axes")
    geometry = gm.GeometryCache(geometry_path)
    confidence = cm.ConfidenceCache(confidence_path)
    geometry_by = {(e["camera"], int(e["frame"])): e for e in geometry_index["entries"]}
    confidence_by = {(e["camera"], int(e["frame"])): e for e in confidence_index["entries"]}
    camera_cache, lr_cache, baseline_cache = {}, {}, {}
    centres = {c: np.asarray(manifest["cameras"][c]["c2w"])[:3, 3] for c in manifest["splits"]["train"]}

    def camera_and_lr(key):
        if key not in lr_cache:
            observation = legal[key]
            path = Path(manifest["_root"]) / observation["lr_path"]
            verify(path)
            lr_cache[key] = image_tensor(path, device="cpu")
            calibration = manifest["cameras"][key[0]]
            item = dict(image=lr_cache[key], camera_id=key[0], frame_index=key[1], time=observation["time"],
                        width=336, height=252, K=np.asarray(calibration["K_lr"], np.float64),
                        w2c=np.asarray(calibration["w2c"], np.float64))
            camera_cache[key] = resized_camera(observation_to_4dgs_camera(item, len(camera_cache)), 1008, 1344)
        return camera_cache[key], lr_cache[key].cuda()

    def reg_loss(target_model):
        g, h = target_model.g, target_model.h
        return g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)

    def parent_baseline(key):
        if key not in baseline_cache:
            camera, lr = camera_and_lr(key)
            with torch.no_grad():
                raw = policy.render_model(model, camera)["render"]
                cost["rgb_forwards"] += 1
                baseline_cache[key] = downsample(raw, (252, 336)).cpu()
        projected = baseline_cache[key]
        return projected, float((projected - lr_cache[key]).abs().mean())

    gradient_banks, calibration_rows, equivalence_rows, recollection_rows = [], [], [], []
    full_square = q_square = lr_xyz_square = g_xyz_square = 0.
    pixel_elements = xyz_elements = 0
    torch.cuda.reset_peak_memory_stats()
    for number, key in enumerate(probes):
        camera, lr = camera_and_lr(key)
        teacher_path = Path(manifest["_root"]) / teacher_by[key]["relative_path"]
        verify(teacher_path)
        teacher = image_tensor(teacher_path)
        verify(geometry_path.parent / geometry_by[key]["path"])
        verify(confidence_path.parent / confidence_by[key]["path"])
        target = geometry.get(*key, device="cuda")
        weight = confidence.get(*key, device="cuda")
        packet = policy.render_model(model, camera, trace=True)
        cost["rgb_forwards"] += 1
        raw, state = packet["render"], packet["effective_state"]
        xyz = state["xyz"]
        moments = gm.render_moments(camera, xyz, state["cov"], state["opacity"], lr_size=(252, 336))
        cost["moment_forwards"] += 1
        if moments["effective_xyz"] is not xyz:
            raise ValueError("G must use the very same RGB effective_xyz tensor")
        loss_g, g_diagnostic = gm.geometry_loss(moments, target)
        loss_lr = (downsample(raw, (252, 336)) - lr).abs().mean()
        loss_sr = (raw - teacher).abs().mean()
        loss_r = op.detail_loss(raw, teacher, weight, k_r=1.)
        loss_reg = reg_loss(model)
        if fixed_coefficients is None:
            loss_q_uniform = op.Q(raw - teacher).abs().mean()
            full_pixel, = torch.autograd.grad(loss_sr, raw, retain_graph=True)
            q_pixel, = torch.autograd.grad(loss_q_uniform, raw, retain_graph=True)
            lr_xyz, = torch.autograd.grad(loss_lr, xyz, retain_graph=True)
            g_xyz, = torch.autograd.grad(loss_g, xyz, retain_graph=True)
            cost["pixel_gradient_calls"] += 2
            cost["effective_xyz_gradients"] += 2
            cost["native_RGB_rasterizer_backwards"] += 1
            cost["native_moment_rasterizer_backwards"] += 1
        forbidden = torch.autograd.grad(loss_g, [state["cov"], state["opacity"], state["sh"]],
                                        allow_unused=True, retain_graph=True)
        cost["disconnected_G_route_gradient_calls"] += 1
        if any(value is not None and bool(value.abs().max() != 0) for value in forbidden):
            raise ValueError("G direct covariance/opacity/SH gradient route is active")
        losses = dict(LR=loss_lr, full_SR=loss_sr, R_unscaled=loss_r, G_unscaled=loss_g, reg=loss_reg)
        gradients = {label: cpu_grads(names, torch.autograd.grad(value, parameters,
                     allow_unused=True, retain_graph=True)) for label, value in losses.items()}
        cost["source_param_gradients"] += len(losses)
        cost["native_RGB_rasterizer_backwards"] += 3
        cost["native_moment_rasterizer_backwards"] += 1
        separated = joint = joint_again = separated_again = None
        if number < 2:
            model.g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            (loss_lr + .1 * loss_sr + loss_reg).backward(retain_graph=True)
            joint = cpu_grads(names, [parameter.grad for parameter in parameters])
            model.g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            loss_lr.backward(retain_graph=True)
            (.1 * loss_sr).backward(retain_graph=True)
            loss_reg.backward(retain_graph=True)
            separated = cpu_grads(names, [parameter.grad for parameter in parameters])
            model.g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            (loss_lr + .1 * loss_sr + loss_reg).backward(retain_graph=True)
            joint_again = cpu_grads(names, [parameter.grad for parameter in parameters])
            model.g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            loss_lr.backward(retain_graph=True)
            (.1 * loss_sr).backward(retain_graph=True)
            loss_reg.backward(retain_graph=True)
            separated_again = cpu_grads(names, [parameter.grad for parameter in parameters])
            model.g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            cost["backward_equivalence_calls"] += 8
            cost["native_RGB_rasterizer_backwards"] += 6
            comparison = compare_gradients(joint, separated, parameter_sizes)
            joint_repeat = compare_gradients(joint, joint_again, parameter_sizes)
            separate_repeat = compare_gradients(separated, separated_again, parameter_sizes)
            passed, envelope = native_equivalence(comparison, joint_repeat, separate_repeat)
            equivalence_rows.append(dict(camera=key[0], frame=key[1], gradient_comparison=comparison,
                same_joint_repeat=joint_repeat, same_separate_repeat=separate_repeat,
                registered_native_noise_envelope=envelope, gradient_equivalent=passed))
            write(OUT / "backward_equivalence_acceptance.json", dict(rows=equivalence_rows,
                    plan_sha256=sha(plan_path), cost=dict(cost)))
            if not passed:
                raise ValueError("Joint/separate backward implementation equivalence failed")
        # Rescue recreates the lost parameter bank, never a candidate coefficient.
        route_row = dict(camera=key[0], frame=key[1],
            G_effective_pairs=int(g_diagnostic["effective_pairs"]),
            G_active_pair_fraction=float(g_diagnostic["active_pair_fraction"]),
            same_RGB_and_G_effective_xyz=True, G_direct_covariance_opacity_SH_disconnected=True,
            confidence_mean=float(weight.mean()), loss={label: float(value.detach()) for label, value in losses.items()})
        if fixed_coefficients is None:
            # Aggregate sum squared / all elements, never average frame ratios.
            fs = float(full_pixel.double().square().sum())
            qs = float(q_pixel.double().square().sum())
            ls = float(lr_xyz.double().square().sum())
            gs = float(g_xyz.double().square().sum())
            full_square += fs; q_square += qs; pixel_elements += raw.numel()
            lr_xyz_square += ls; g_xyz_square += gs; xyz_elements += xyz.numel()
            calibration_rows.append(dict(**route_row, pixel_elements=raw.numel(),
                xyz_elements=xyz.numel(), full_sr_pixel_square_sum=fs, q_c1_pixel_square_sum=qs,
                LR_effective_xyz_square_sum=ls, G_effective_xyz_square_sum=gs))
            del full_pixel, q_pixel, lr_xyz, g_xyz, loss_q_uniform
        else:
            recollection_rows.append(route_row)
        baseline_cache[key] = downsample(raw.detach(), (252, 336)).cpu()
        gradient_banks.append(dict(key=key, gradients=gradients, joint=joint, separated=separated,
                                  joint_repeat=joint_again, separate_repeat=separated_again))
        stage = "shadow_gradient_recollection" if fixed_coefficients else "fixed32_calibration"
        write(OUT / "calibration_progress.json", dict(stage=stage, completed=number+1,
                cost=dict(cost), seconds=time.monotonic()-started))
        print(json.dumps(dict(stage=stage, completed=number+1)), flush=True)
        del packet, raw, state, xyz, moments, losses, loss_g, loss_lr, loss_sr, loss_r, loss_reg
        del forbidden, teacher, target, weight

    epsilon = torch.finfo(torch.float32).eps
    repeat_calibration_statistics = None
    if fixed_coefficients is not None:
        k_r, lambda_g = fixed_coefficients["k_R"], fixed_coefficients["lambda_G"]
        full_rms = fixed_coefficients["full_sr_pixel_gradient_rms"]
        q_rms = fixed_coefficients["Q_c1_pixel_gradient_rms"]
        lr_xyz_rms = fixed_coefficients["LR_effective_xyz_gradient_rms"]
        g_xyz_rms = fixed_coefficients["G_effective_xyz_gradient_rms"]
        calibration_rows = fixed_coefficients["primary_rows"]
        backup_rows = fixed_coefficients["backup_G_rows"]
        q_floor, g_floor = epsilon * full_rms, epsilon * lr_xyz_rms
        q_fallback = q_rms <= q_floor
        if k_r != (1. if q_fallback else full_rms / q_rms) or lambda_g != .1 * lr_xyz_rms / g_xyz_rms:
            raise ValueError("Frozen coefficient values do not match their original aggregate RMS")
        if [(row["camera"], row["frame"]) for row in calibration_rows] != probes:
            raise ValueError("Frozen primary observation identities differ")
        repeat_calibration_statistics = dict(
            purpose="recollect lost 32 parameter-gradient banks for shadow rescue",
            observation_rows=recollection_rows, pixel_coefficient_gradient_calls=0,
            effective_xyz_coefficient_gradient_calls=0, coefficient_recalculated=False,
            candidate_coefficients_generated=False, frozen_origin_sha256=fixed_coefficients_sha)
    else:
        full_rms = (full_square / pixel_elements) ** .5
        q_rms = (q_square / pixel_elements) ** .5
        lr_xyz_rms = (lr_xyz_square / xyz_elements) ** .5
        g_xyz_rms = (g_xyz_square / xyz_elements) ** .5
        epsilon = torch.finfo(torch.float32).eps
        q_floor = epsilon * full_rms
        q_fallback = q_rms <= q_floor
        k_r = 1. if q_fallback else full_rms / q_rms
        g_floor = epsilon * lr_xyz_rms
        # A zero G denominator is a broken/uninformative gradient, never lambda_G=0.
        backup_rows = []
        if g_xyz_rms <= g_floor:
            for key in backup_probes:
                camera, lr = camera_and_lr(key)
                verify(geometry_path.parent / geometry_by[key]["path"])
                target = geometry.get(*key, device="cuda")
                packet = policy.render_model(model, camera, trace=True)
                state = packet["effective_state"]
                moments = gm.render_moments(camera, state["xyz"], state["cov"], state["opacity"], lr_size=(252, 336))
                ll = (downsample(packet["render"], (252, 336)) - lr).abs().mean()
                lg, diagnostic = gm.geometry_loss(moments, target)
                gl, = torch.autograd.grad(ll, state["xyz"], retain_graph=True)
                gg, = torch.autograd.grad(lg, state["xyz"])
                ls, gs = float(gl.double().square().sum()), float(gg.double().square().sum())
                backup_rows.append(dict(camera=key[0], frame=key[1], LR_effective_xyz_square_sum=ls,
                                       G_effective_xyz_square_sum=gs, xyz_elements=gl.numel(),
                                       G_effective_pairs=int(diagnostic["effective_pairs"])))
                cost["rgb_forwards"] += 1; cost["moment_forwards"] += 1; cost["effective_xyz_gradients"] += 2
                cost["native_RGB_rasterizer_backwards"] += 1; cost["native_moment_rasterizer_backwards"] += 1
                del packet, state, moments, target, ll, lg, gl, gg
            total = sum(row["xyz_elements"] for row in backup_rows)
            lr_xyz_rms = (sum(row["LR_effective_xyz_square_sum"] for row in backup_rows) / total) ** .5
            g_xyz_rms = (sum(row["G_effective_xyz_square_sum"] for row in backup_rows) / total) ** .5
            g_floor = epsilon * lr_xyz_rms
        if not all(math.isfinite(value) for value in (full_rms, q_rms, lr_xyz_rms, g_xyz_rms, k_r)):
            raise FloatingPointError("Nonfinite coefficient calibration")
        if g_xyz_rms <= g_floor or lr_xyz_rms <= 0:
            write(OUT / "calibration.json", dict(status="G_uncalibratable", primary_rows=calibration_rows,
                backup_rows=backup_rows, LR_effective_xyz_rms=lr_xyz_rms, G_effective_xyz_rms=g_xyz_rms,
                cause="G effective_xyz denominator remains below registered precision floor; no G/RG execution permitted",
                cost=dict(cost)))
            raise FloatingPointError("G coefficient cannot be calibrated from fixed/backup train32")
        lambda_g = .1 * lr_xyz_rms / g_xyz_rms
        if not math.isfinite(lambda_g) or lambda_g <= 0:
            raise FloatingPointError("Invalid weak-geometry coefficient")
        write(fixed_coefficients_path, dict(
            status="fixed_coefficients_pending_shadow_acceptance", k_R=k_r, lambda_G=lambda_g,
            full_sr_pixel_gradient_rms=full_rms, Q_c1_pixel_gradient_rms=q_rms,
            LR_effective_xyz_gradient_rms=lr_xyz_rms, G_effective_xyz_gradient_rms=g_xyz_rms,
            confidence_used_for_k_R=False, primary_rows=calibration_rows, backup_G_rows=backup_rows,
            source_hashes=source_hashes, cost=dict(cost)))
        fixed_coefficients_sha = sha(fixed_coefficients_path)

    curve = read(bound(protocol["lr_curve"]))["rows"][6000]
    if curve[0] != 6001:
        raise ValueError("First suffix LR curve has wrong optimizer age")
    shadow_rows = []
    for number, bank in enumerate(gradient_banks):
        key, gradients = bank["key"], bank["gradients"]
        camera, lr = camera_and_lr(key)
        neighbour_camera = min((c for c in centres if c != key[0]),
                               key=lambda c: (float(np.linalg.norm(centres[c]-centres[key[0]])), c))
        neighbours = dict(self=key, same_time_view=(neighbour_camera, key[1]),
                          same_view_time=(key[0], key[1]+2 if key[1] < 118 else key[1]-2))
        baselines = {label: parent_baseline(probe_key) for label, probe_key in neighbours.items()}
        weighted = dict(LR=gradients["LR"], full_SR=combine((gradients["full_SR"], .1)),
                        R=combine((gradients["R_unscaled"], .1*k_r)),
                        G=combine((gradients["G_unscaled"], lambda_g)))
        gradient_report = dict(
            norms={name: norm_summary(value, parameter_sizes) for name, value in weighted.items()},
            pairwise={a+"_vs_"+b: compare_gradients(weighted[a], weighted[b], parameter_sizes)
                      for a, b in (("LR", "full_SR"), ("LR", "R"), ("LR", "G"), ("R", "G"))},
            composite={"C1": norm_summary(combine((weighted["LR"], 1), (weighted["full_SR"], 1)), parameter_sizes),
                       "R": norm_summary(combine((weighted["LR"], 1), (weighted["R"], 1)), parameter_sizes),
                       "G": norm_summary(combine((weighted["LR"], 1), (weighted["full_SR"], 1), (weighted["G"], 1)), parameter_sizes),
                       "RG": norm_summary(combine((weighted["LR"], 1), (weighted["R"], 1), (weighted["G"], 1)), parameter_sizes)},
            regularizer_norms=norm_summary(gradients["reg"], parameter_sizes),
            coefficients=dict(full_SR=.1, R=.1*k_r, G=lambda_g),
            G_shadow_uses_full_fixed_coefficient=True, formal_G_first_step_ramp=1/500)
        arm_gradients = dict(
            LR_only=combine((weighted["LR"], 1), (gradients["reg"], 1)),
            C1=bank["joint"] if bank["joint"] is not None else combine((weighted["LR"], 1), (weighted["full_SR"], 1), (gradients["reg"], 1)),
            R=combine((weighted["LR"], 1), (weighted["R"], 1), (gradients["reg"], 1)),
            G=combine((weighted["LR"], 1), (weighted["full_SR"], 1), (weighted["G"], 1), (gradients["reg"], 1)),
            RG=combine((weighted["LR"], 1), (weighted["R"], 1), (weighted["G"], 1), (gradients["reg"], 1)))
        lr_update = joint_update = separate_update = joint_repeat_update = None
        arm_rows = {}
        for arm in SHADOW_ARMS + (("C1_separate_backward", "C1_joint_repeat", "C1_separate_repeat") if number < 2 else ()):
            clone = load_model(parent_path, manifest)
            # A fresh independent exact parent state, including both Adam histories.
            clone_identity = initial_identity(clone)
            if clone_identity != before:
                raise ValueError("Shadow clone does not include complete parent model/two Adam histories")
            clone.g.update_learning_rate(13201)
            for optimizer, rates in ((clone.g.optimizer, curve[1]), (clone.child_optimizer, curve[2])):
                for param_group in optimizer.param_groups:
                    if not math.isclose(param_group["lr"], rates[param_group["name"]], rel_tol=1e-14, abs_tol=0):
                        raise ValueError("Shadow optimizer LR differs from first formal suffix step")
                    param_group["lr"] = rates[param_group["name"]]
                optimizer.zero_grad(set_to_none=True)
            resume.restore_global_rng(rng_before)
            if digest_state(rng()) != rng_identity:
                raise ValueError("Shadow clone RNG differs from exact mother RNG")
            clone_lr_identity = digest_state(([group["lr"] for group in clone.g.optimizer.param_groups],
                                             [group["lr"] for group in clone.child_optimizer.param_groups]))
            named_clone = {name: value for name, value in shared.all_named(clone).items() if value.requires_grad}
            selected_gradients = {"C1_separate_backward": bank["separated"],
                                  "C1_joint_repeat": bank["joint_repeat"],
                                  "C1_separate_repeat": bank["separate_repeat"]}.get(arm)
            if selected_gradients is None:
                selected_gradients = arm_gradients[arm]
            for name, parameter in named_clone.items():
                gradient = selected_gradients[name]
                parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
            clone.g.optimizer.step(); clone.child_optimizer.step()
            cost["shadow_optimizer_rounds"] += 1; cost["shadow_adam_calls"] += 2
            update = {name: parameter.detach().cpu() - parameter_before[name] for name, parameter in named_clone.items()}
            if arm == "LR_only":
                lr_update = update
            if arm == "C1":
                joint_update = update
            if arm == "C1_separate_backward":
                separate_update = update
            if arm == "C1_joint_repeat":
                joint_repeat_update = update
            probes_after = {}
            with torch.no_grad():
                for label, probe_key in ({} if arm in ("C1_joint_repeat", "C1_separate_repeat") else neighbours).items():
                    cam, truth = camera_and_lr(probe_key)
                    prediction = policy.render_model(clone, cam)["render"]
                    cost["rgb_forwards"] += 1
                    degraded = downsample(prediction, (252, 336)).cpu()
                    base, baseline_loss = baselines[label]
                    loss_after = float((degraded - lr_cache[probe_key]).abs().mean())
                    delta = degraded - base
                    probes_after[label] = dict(camera=probe_key[0], frame=probe_key[1], LR_L1_before=baseline_loss,
                        LR_L1_after=loss_after, LR_L1_delta=loss_after-baseline_loss,
                        LR_output_change_MAE=float(delta.abs().mean()),
                        LR_output_change_RMS=float(delta.double().square().mean().sqrt()))
            row = dict(complete_parent_Adam_identity=True, exact_parent_RNG_restored=True,
                       parent_model_and_Adam_sha256=clone_identity["model_and_optimizer_sha256"],
                       clone_LR_sha256=clone_lr_identity, clone_RNG_sha256=rng_identity, observations=probes_after,
                       parameter_update_norms=norm_summary(update, parameter_sizes),
                       update_vs_LR_only=compare_gradients(update, lr_update, parameter_sizes))
            if arm == "C1_separate_repeat":
                comparison = compare_gradients(joint_update, separate_update, parameter_sizes)
                native_joint = compare_gradients(joint_update, joint_repeat_update, parameter_sizes)
                native_separate = compare_gradients(separate_update, update, parameter_sizes)
                passed, envelope = native_equivalence(comparison, native_joint, native_separate,
                    dict(relative_l2=ADAM_TOLERANCE["relative_update_l2"],
                         absolute_max=ADAM_TOLERANCE["absolute_parameter_max"]),
                    registered_minimum=Adam_noise_registration["minimum_envelope"])
                position_deltas = dict(joint_vs_separate=joint_update["_xyz"]-separate_update["_xyz"],
                    same_joint_repeat=joint_update["_xyz"]-joint_repeat_update["_xyz"],
                    same_separate_repeat=separate_update["_xyz"]-update["_xyz"])
                precision_summary = {}
                precision_arrays = {}
                for label, delta_position in position_deltas.items():
                    ratio = delta_position.double().abs() / position_ulp.double()
                    precision_summary[label] = dict(delta_max=float(delta_position.abs().max()),
                        delta_in_ULP_max=float(ratio.max()),
                        nonzero_coordinates=int((delta_position != 0).sum()),
                        nonzero_points=int((delta_position != 0).any(dim=1).sum()),
                        coordinates=int(delta_position.numel()))
                    precision_arrays[label] = dict(delta=delta_position,
                        absolute_delta_over_per_coordinate_ULP=ratio,
                        max_absolute_delta_per_point=delta_position.abs().max(dim=1).values,
                        max_ULP_per_point=ratio.max(dim=1).values)
                delta_path = precision_dir / (key[0] + "_" + str(key[1]) + "_Adam_position_deltas.pt")
                torch.save(precision_arrays, delta_path)
                precision_summary["arrays"] = dict(path=str(delta_path.relative_to(ROOT)), sha256=sha(delta_path))
                equivalence_rows[number].update(Adam_update_comparison=comparison, Adam_update_equivalent=passed,
                    Adam_full_state_clones=True, exact_clone_RNG_and_LR=True,
                    position_float32_precision=precision_summary, same_joint_Adam_repeat=native_joint,
                    same_separate_Adam_repeat=native_separate, registered_Adam_noise_envelope=envelope)
                write(OUT / "Adam_equivalence_acceptance.json", dict(rows=equivalence_rows,
                       cost=dict(cost), coefficient_origin_sha256=fixed_coefficients_sha,
                       registered_noise_evidence_sha256=sha(OUT / "Adam_native_noise_registration.json"),
                       position_precision=position_precision))
                if not passed:
                    raise ValueError("Joint/separate backward complete-Adam one-step equivalence failed")
            arm_rows[arm] = row
            del clone, named_clone
            for parameter in parameters:
                parameter.grad = None
        shadow_rows.append(dict(camera=key[0], frame=key[1], gradients=gradient_report, shadow=arm_rows))
        gradient_banks[number] = None  # discard all giant probe gradients after scalar reporting
        write(OUT / "calibration_progress.json", dict(stage="finite_Adam_shadow", completed=number+1,
                cost=dict(cost), seconds=time.monotonic()-started))
        print(json.dumps(dict(stage="finite_Adam_shadow", completed=number+1)), flush=True)
    for parameter in parameters:
        parameter.grad = None
    after = initial_identity(model)
    if after != before:
        raise ValueError("Calibration source mother model or two Adam states changed")
    resume.restore_global_rng(rng_before)
    if digest_state(rng()) != digest_state(rng_before):
        raise ValueError("Calibration altered parent RNG state")
    if source_hashes != {str(path.relative_to(ROOT)): sha(path) for path in source_paths}:
        raise ValueError("Calibration code changed while running")
    if protocol_hash != sha(OUT / "protocol.json"):
        raise ValueError("Registered protocol changed during calibration")
    if sha(fixed_coefficients_path) != fixed_coefficients_sha:
        raise ValueError("Frozen coefficient origin changed during shadow acceptance")
    torch.cuda.synchronize()
    final_cost = dict(**dict(cost), seconds=time.monotonic()-started,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                     hardware=torch.cuda.get_device_name(), torch=str(torch.__version__),
                     physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), resource_preflight=resources,
                     giant_parameter_gradients_saved=False)
    identities = dict(protocol_sha256=protocol_hash, parent=protocol["parent"], manifest=protocol["manifest"],
        teacher=protocol["teacher"], geometry_index=dict(path=str(geometry_path.relative_to(ROOT)), sha256=sha(geometry_path)),
        confidence_index=dict(path=str(confidence_path.relative_to(ROOT)), sha256=sha(confidence_path)),
        source_hashes=source_hashes, fixed_observation_plan_sha256=sha(plan_path),
        coefficient_origin_sha256=fixed_coefficients_sha,
        initial_identity=before, mother_model_two_Adam_unchanged=after == before,
        parent_RNG_restored_exactly=True, parent_RNG_sha256=rng_identity,
        position_precision=position_precision, Adam_native_noise_registration=dict(
            path=str((OUT / "Adam_native_noise_registration.json").relative_to(ROOT)),
            sha256=sha(OUT / "Adam_native_noise_registration.json")), HR_derived_training=False)
    report = dict(status="passed", k_R=k_r, lambda_G=lambda_g,
        full_sr_pixel_gradient_rms=full_rms, Q_c1_pixel_gradient_rms=q_rms,
        LR_effective_xyz_gradient_rms=lr_xyz_rms, G_effective_xyz_gradient_rms=g_xyz_rms,
        calibrated_G_fraction=.1, G_ramp_steps=500, confidence_used_for_k_R=False,
        aggregation="sqrt(sum squared across 32 observations / total corresponding elements); ratio of aggregate RMS",
        k_R_precision_floor=q_floor, k_R_1_numerical_fallback=q_fallback,
        G_precision_floor=g_floor, primary_rows=calibration_rows, backup_G_rows=backup_rows,
        repeated_gradients_for_shadow_only=repeat_calibration_statistics,
        identities=identities, cost=final_cost)
    write(OUT / "gradient_probe.json", dict(status="passed", observations=shadow_rows,
        backward_equivalence=equivalence_rows, registration=plan, identities=identities, cost=final_cost,
        interpretation="Finite training-only causal diagnostics. Euclidean gradient cosine is not Adam update direction. "
        "R image-domain auxiliary gradient is in D0 nullspace, but actual Adam/renderer updates may change LR and neighbours.",
        actual_image_reads=dict(actual_reads), all_reads_within_legal_training_whitelist=True))
    # Publish the passed calibration only after all acceptance/identity checks.
    write(OUT / "calibration.json", report)
    print(json.dumps(dict(status="passed", k_R=k_r, lambda_G=lambda_g, cost=final_cost)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-index", type=Path, default=OUT / "cache/geometry/index.json")
    parser.add_argument("--confidence-index", type=Path, default=OUT / "cache/confidence_view/cache_manifest.json")
    parser.add_argument("--inherited-gpu-lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("geometry_index", "confidence_index"):
        setattr(args, key, local(getattr(args, key)).resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "calibration.json").exists():
        raise RuntimeError("Calibration is fixed once; inspect the existing receipt before any rerun")
    if args.inherited_gpu_lock_fd is None:
        registration = register_Adam_native_envelope()
        print(json.dumps(dict(stage="registered_shadow_rescue_noise_envelope",
            evidence_sha256=sha(OUT / "Adam_native_noise_registration.json"),
            minimum_envelope=registration["minimum_envelope"])), flush=True)
        # This outer process never imports torch or owns a CUDA context. Keep its
        # lock until the CUDA child has fully exited, not merely until a Python
        # function returns and starts interpreter/CUDA cleanup.
        with (OUT / "evaluation_gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            resources = gpu_snapshot()
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--geometry-index", str(args.geometry_index),
                       "--confidence-index", str(args.confidence_index),
                       "--inherited-gpu-lock-fd", str(lock.fileno())]
            child = subprocess.Popen(command, pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL)
            record = dict(status="running", supervisor_pid=os.getpid(), cuda_child_pid=child.pid,
                          supervisor_owns_CUDA=False, lock_held_until_child_exit=True,
                          source_sha256=sha(__file__), preflight=resources)
            write(OUT / "calibration_supervisor.json", record)
            returncode = child.wait()
            record.update(status="completed" if returncode == 0 else "failed", returncode=returncode,
                          cuda_child_fully_exited=True)
            write(OUT / "calibration_supervisor.json", record)
        raise SystemExit(returncode)
    try:
        fd = args.inherited_gpu_lock_fd
        if os.fstat(fd).st_ino != (OUT / "evaluation_gpu.lock").stat().st_ino:
            raise ValueError("Inherited evaluation lock refers to another file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first_resource = gpu_snapshot()
        main_locked(args, [first_resource])
    except BaseException:
        write(OUT / "calibration_failed.json", dict(status="failed", traceback=traceback.format_exc(),
                formal_training_updates=0, coefficient_search=False))
        raise


if __name__ == "__main__":
    main()
