"""Continue a fixed-topology SR control with reconstructed independent samplers.

This deliberately supports only the recorded 6k SR-weight-0.1 fork. The
appearance-only intervention freezes spatial support, opacity and shared
deformation features; it is not an intervention on motion alone.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/dynamic_sr_20260918"
CONTROL = ROOT / "experiments/dynamic_sr_20260919"
FORK_STEP = 6000


def exposure_key(record):
    return f'{record["camera_id"]}/{record["frame_index"]}'


def rebuild_samplers(seed, records, prior_ids, prefix_steps, full_steps,
                     recorded_full_exposure):
    """Reconstruct old two-generator draws and verify the saved full exposure.

    The historical checkpoint did not store the independent RNG states. The
    prefix is reconstructed, not an independently observed prefix count. Its
    source is checked against the actual complete run's saved SR counter.
    """
    if not records or not prior_ids or not 0 <= prefix_steps <= full_steps:
        raise ValueError("Invalid sampler reconstruction inputs")
    lr_rng = random.Random(seed + 177)
    sr_rng = random.Random(seed + 211)
    prefix = collections.Counter()
    digest = hashlib.sha256()
    for _ in range(prefix_steps):
        li, si = lr_rng.randrange(len(records)), sr_rng.choice(prior_ids)
        prefix[exposure_key(records[si])] += 1
        digest.update(f"{li},{si}\n".encode())
    prefix_states = (lr_rng.getstate(), sr_rng.getstate())
    total = prefix.copy()
    for _ in range(prefix_steps, full_steps):
        lr_rng.randrange(len(records))
        si = sr_rng.choice(prior_ids)
        total[exposure_key(records[si])] += 1
    if total != collections.Counter(recorded_full_exposure):
        raise ValueError("Reconstructed SR exposure does not match source exposure.json")
    lr_rng.setstate(prefix_states[0])
    sr_rng.setstate(prefix_states[1])
    return lr_rng, sr_rng, prefix, digest.hexdigest()


def apply_parameter_mode(parameters, appearance_params, mode):
    if mode not in ("joint", "appearance_only"):
        raise ValueError(mode)
    appearance_ids = {id(p) for p in appearance_params}
    if not appearance_ids or not appearance_ids <= {id(p) for p in parameters.values()}:
        raise ValueError("Appearance parameters are missing from named parameters")
    roles = {}
    for name, p in parameters.items():
        original_trainable = p.requires_grad
        # Preserve upstream constants such as HexPlaneField.aabb. Being listed
        # in Adam's parameter groups does not mean a parameter was trainable.
        selected = original_trainable and (mode == "joint" or id(p) in appearance_ids)
        p.requires_grad_(selected)
        # Adam skips parameters with grad=None, including their old momentum.
        # Zero gradients alone would still allow momentum to move frozen values.
        p.grad = None
        roles[name] = dict(trainable=selected, original_trainable=original_trainable,
                           numel=p.numel(), shape=list(p.shape))
    return roles


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_state_equal(left, right, label="optimizer"):
    """Check the restored Adam state, including moments and each parameter step."""
    import torch
    if torch.is_tensor(left):
        if (not torch.is_tensor(right) or left.dtype != right.dtype
                or not torch.equal(left.detach().cpu(), right.detach().cpu())):
            raise AssertionError(f"State mismatch: {label}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"State keys mismatch: {label}")
        for key in left:
            assert_state_equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"State length mismatch: {label}")
        for i, (a, b) in enumerate(zip(left, right)):
            assert_state_equal(a, b, f"{label}[{i}]")
    elif left != right:
        raise AssertionError(f"State mismatch: {label}")


def support_snapshot(g, times):
    """Hash all points' deformed xyz/scale/rotation/opacity at fixed times."""
    import torch
    result = {}
    with torch.no_grad():
        for timestamp in times:
            t = torch.full((len(g._xyz), 1), float(timestamp), device=g._xyz.device)
            outputs = g._deformation(g._xyz, g._scaling, g._rotation,
                                     g._opacity, g.get_features, t)
            row = {}
            for name, output in zip(("xyz", "scale_raw", "rotation_raw", "opacity_raw"), outputs[:4]):
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Nonfinite support: {timestamp}/{name}")
                row[name] = tensor_digest(output)
            result[str(timestamp)] = row
        result["deformation_table"] = tensor_digest(g._deformation_table)
    return result


def restore_global_rng(rng):
    import numpy as np
    import torch
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Run with the same visible CUDA-device count as the source checkpoint")
    torch.set_rng_state(rng["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def checkpoint_with_samplers(path, g, h, o, metadata, lr_rng, sr_rng,
                             prefix_exposure, additional_exposure):
    import numpy as np
    import torch
    from controlled_fit import persist
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(
        model=g.capture(), hidden=vars(h), optim=vars(o), metadata=metadata,
        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                 numpy=np.random.get_state(), python=random.getstate()),
        samplers=dict(lr=lr_rng.getstate(), sr=sr_rng.getstate(),
                      source_prefix_exposure=dict(prefix_exposure),
                      additional_exposure=dict(additional_exposure))), temp)
    temp.replace(path)
    persist(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--mode", choices=("joint", "appearance_only"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=12000, help="Additional steps after source intervention step 6000")
    parser.add_argument("--milestones", default="6000,12000", help="Additional-step checkpoints; endpoint is always included")
    parser.add_argument("--skip-fit", action="store_true", help="Skip the fixed training-view HR/prior evaluation for smoke runs")
    args = parser.parse_args()
    if not 1 <= args.steps <= 12000:
        parser.error("--steps must be in [1, 12000]")
    milestones = {int(x) for x in args.milestones.split(",") if x}
    milestones = {s for s in milestones if 0 < s <= args.steps} | {args.steps}
    out, source = Path(args.out).resolve(), Path(args.source_run).resolve()
    if out.exists():
        raise FileExistsError(out)
    source_config = json.loads((source / "config.json").read_text())
    source_complete = json.loads((source / "complete.json").read_text())
    if (source_config["teacher"] != "sr" or source_config["weight"] != .1
            or source_config["dense"] or source_config["steps"] != 18000
            or source_complete["intervention_step"] != 18000):
        raise ValueError("Expected a complete 18k fixed-topology sr_w01 source run")
    source_checkpoint = source / f"checkpoint_{FORK_STEP}.pt"
    manifest_path = Path(source_config["manifest"])

    sys.path.insert(0, str(OLD))
    sys.path.insert(0, str(CONTROL))
    import numpy as np
    import torch
    from common import (UPSTREAM, appearance_parameters, downsample, image_tensor,
                        load_checkpoint, named_parameters, render_image, resized_camera,
                        seed_all, sha256)
    from controlled_fit import fit_diagnostic, write_json, persist
    from n3dv_data import load_manifest
    from run_experiment import load_training

    torch.set_num_threads(4)
    seed_all(source_config["seed"])
    manifest_sha = sha256(manifest_path)
    if manifest_sha != source_config["manifest_sha256"]:
        raise ValueError("Source manifest hash changed")
    m = load_manifest(manifest_path)
    g, h, o, ck = load_checkpoint(source_checkpoint)
    metadata = ck["metadata"]
    if (metadata["intervention_step"] != FORK_STEP
            or metadata["step"] != source_config["scheduler_offset"] + FORK_STEP
            or metadata["manifest_sha"] != manifest_sha
            or metadata["parent_sha"] != source_config["parent_sha256"]):
        raise ValueError("Source checkpoint provenance disagrees with source config")
    for key in ("teacher", "weight", "dense", "seed", "prior_cameras"):
        if metadata["args"][key] != source_config[key]:
            raise ValueError(f"Source checkpoint argument mismatch: {key}")
    assert_state_equal(ck["model"][-2], g.optimizer.state_dict())
    records, cameras = load_training(m)
    hr_width, hr_height = m["resolutions"]["hr"]
    hr_cameras = [resized_camera(c, hr_height, hr_width) for c in cameras]
    prior_cameras = set(source_config["prior_cameras"].split(","))
    ids = [i for i, r in enumerate(records) if r["camera_id"] in prior_cameras]
    recorded_exposure = json.loads((source / "exposure.json").read_text())
    lr_rng, sr_rng, prefix_exposure, prefix_draw_sha = rebuild_samplers(
        source_config["seed"], records, ids, FORK_STEP, source_config["steps"], recorded_exposure)
    observations = {(r["camera_id"], r["frame_index"]): r for r in m["observations"]}
    paths = {}
    for i in ids:
        record = records[i]
        obs = observations[(record["camera_id"], record["frame_index"])]
        path = Path(m["_root"]) / "sr_swinir_x4" / record["camera_id"] / Path(obs["lr_path"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[i] = path
    params = named_parameters(g)
    roles = apply_parameter_mode(params, appearance_parameters(g), args.mode)
    frozen_initial = {name: tensor_digest(p) for name, p in params.items() if not p.requires_grad}
    frame_times = sorted({float(c.time) for c in cameras})
    probe_times = [frame_times[0], frame_times[len(frame_times)//2], frame_times[-1]]
    initial_support = support_snapshot(g, probe_times)
    initial_points, initial_sh = len(g._xyz), g.active_sh_degree
    metric = None
    if not args.skip_fit:
        import lpips
        metric = lpips.LPIPS(net="alex").cuda().eval().requires_grad_(False)

    out.mkdir(parents=True)
    (out / "sources").mkdir()
    config = dict(
        **(vars(args) | {"source_run": str(source)}), source_config_sha256=sha256(source / "config.json"),
        source_checkpoint=str(source_checkpoint), parent_sha256=sha256(source_checkpoint),
        manifest=str(manifest_path), manifest_sha256=manifest_sha,
        source_metadata=metadata, seed=source_config["seed"], teacher="sr", weight=.1,
        prior_cameras=source_config["prior_cameras"], fork_intervention_step=FORK_STEP,
        endpoint_intervention_step=FORK_STEP + args.steps,
        scheduler_offset=metadata["step"], scheduler_max_steps=o.position_lr_max_steps,
        initial_points=initial_points, sh_degree=initial_sh, optimizer_preserved_exact=True,
        parameter_roles=roles, trainable_parameter_count=sum(p.numel() for p in params.values() if p.requires_grad),
        frozen_parameter_hashes=frozen_initial, initial_support_hashes=initial_support,
        support_probe_times=probe_times, source_prefix_sr_exposure=dict(prefix_exposure),
        source_prefix_draw_sha256=prefix_draw_sha,
        reconstructed_full_exposure_matches_source=True,
        sampler_verification="Both RNGs fast-forwarded 6000 draws; full 18000-step SR Counter exactly matches source. Prefix was reconstructed, not separately logged by the source.",
        information_boundary="Only training LR and fixed SwinIR SR targets enter gradients. HR used only for fixed diagnostics. appearance_only freezes all spatial-support/opacity parameters and shared deformation features, not merely motion.",
        topology_changes=False, source_snapshot_records=source_config["sources"],
        device=torch.cuda.get_device_name(), visible_cuda_devices=torch.cuda.device_count(),
        torch=torch.__version__, sources=[])
    source_files = [Path(__file__), CONTROL / "controlled_fit.py", OLD / "common.py",
                    OLD / "run_experiment.py", OLD / "n3dv_data.py", OLD / "evaluate.py",
                    UPSTREAM / "scene/gaussian_model.py", UPSTREAM / "scene/deformation.py",
                    UPSTREAM / "gaussian_renderer/__init__.py", UPSTREAM / "scene/hexplane.py"]
    for i, path in enumerate(source_files):
        destination = out / "sources" / f"{i}_{path.name}"
        shutil.copy2(path, destination)
        config["sources"].append(dict(path=str(path), sha256=sha256(path), copy=str(destination)))
    write_json(out / "config.json", config)
    restore_global_rng(ck["rng"])
    write_json(out / "rng_restored.json", dict(restored_after_model_data_metric_initialization=True,
                                              cuda_states=len(ck["rng"]["cuda"])))
    cache, exposure = {}, collections.Counter()
    draw_digest = hashlib.sha256()
    started = time.monotonic()
    train_seconds, diagnostic_seconds = 0., 0.
    frozen_audits = []

    with (out / "training.jsonl").open("w", buffering=1) as log:
        for additional_step in range(1, args.steps + 1):
            tick = time.monotonic()
            intervention_step = FORK_STEP + additional_step
            total_step = metadata["step"] + additional_step
            g.update_learning_rate(total_step)
            li, si = lr_rng.randrange(len(records)), sr_rng.choice(ids)
            draw_digest.update(f"{li},{si}\n".encode())
            target = records[li]["image"].cuda()
            g.optimizer.zero_grad(set_to_none=True)
            pkg = render_image(g, hr_cameras[li])
            prediction = downsample(pkg["render"], target.shape[-2:])
            lr_loss = (prediction - target).abs().mean()
            reg = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
            (lr_loss + reg).backward()
            if si not in cache:
                cache[si] = image_tensor(paths[si], device="cpu")
            teacher_package = render_image(g, hr_cameras[si])
            teacher_loss = (teacher_package["render"] - cache[si].cuda()).abs().mean()
            (teacher_loss * source_config["weight"]).backward()
            exposure[exposure_key(records[si])] += 1
            if not torch.isfinite(lr_loss + reg + teacher_loss):
                raise FloatingPointError(intervention_step)
            for name, parameter in params.items():
                if not roles[name]["trainable"] and parameter.grad is not None:
                    raise AssertionError(f"Frozen parameter received a gradient: {name}")
            g.optimizer.step()
            torch.cuda.synchronize()
            train_seconds += time.monotonic() - tick
            if additional_step == 1 or intervention_step % 100 == 0:
                row = dict(step=intervention_step, intervention_step=intervention_step,
                           additional_step=additional_step, total_step=total_step,
                           lr_l1=float(lr_loss), teacher_l1=float(teacher_loss), reg=float(reg),
                           points=len(g._xyz), train_s=train_seconds,
                           wall_s=time.monotonic() - started,
                           peak_gb=torch.cuda.max_memory_allocated()/1e9,
                           lr_observation=exposure_key(records[li]), sr_observation=exposure_key(records[si]),
                           additional_draw_sha256=draw_digest.hexdigest(),
                           learning_rates={group["name"]: group["lr"] for group in g.optimizer.param_groups})
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if additional_step in milestones:
                tick = time.monotonic()
                if len(g._xyz) != initial_points or g.active_sh_degree != initial_sh:
                    raise AssertionError("Topology or SH degree changed")
                for name, parameter in params.items():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(f"Nonfinite parameter at endpoint: {name}")
                current_frozen = {name: tensor_digest(params[name]) for name in frozen_initial}
                if current_frozen != frozen_initial:
                    raise AssertionError("Frozen parameters changed bitwise")
                support = support_snapshot(g, probe_times)
                support_unchanged = support == initial_support
                if args.mode == "appearance_only" and not support_unchanged:
                    raise AssertionError("Frozen spatial support/opacity changed at probe times")
                audit = dict(intervention_step=intervention_step,
                             frozen_parameters_bitwise_unchanged=current_frozen == frozen_initial,
                             frozen_parameter_count=len(frozen_initial),
                             support_bitwise_unchanged=support_unchanged, support_hashes=support)
                frozen_audits.append(audit)
                write_json(out / f"freeze_audit_{intervention_step}.json", audit)
                meta = dict(scene=m["scene"], stage="resume_control", mode=args.mode,
                            step=total_step, intervention_step=intervention_step,
                            additional_step=additional_step, fork_intervention_step=FORK_STEP,
                            extent=metadata["extent"], args=vars(args),
                            parent_sha=config["parent_sha256"], manifest_sha=manifest_sha,
                            elapsed_s=time.monotonic()-started, train_s=train_seconds,
                            points=len(g._xyz), additional_draw_sha256=draw_digest.hexdigest())
                checkpoint_with_samplers(out / f"checkpoint_{intervention_step}.pt", g, h, o,
                                         meta, lr_rng, sr_rng, prefix_exposure, exposure)
                if not args.skip_fit:
                    aggregate = fit_diagnostic(g, m, prior_cameras, out, intervention_step, metric)
                    print(json.dumps(dict(event="fit", step=intervention_step, aggregate=aggregate)), flush=True)
                diagnostic_seconds += time.monotonic()-tick
        log.flush()
        os.fsync(log.fileno())
    (out / "checkpoint_final.pt").symlink_to(f"checkpoint_{FORK_STEP + args.steps}.pt")
    write_json(out / "exposure.json", dict(exposure))
    write_json(out / "full_exposure.json", dict(prefix_exposure + exposure))
    write_json(out / "complete.json", dict(**meta, diagnostic_s=diagnostic_seconds,
                                           wall_s=time.monotonic()-started,
                                           additional_steps=args.steps,
                                           frozen_verification=dict(
                                               required=args.mode == "appearance_only",
                                               passed=all(a["frozen_parameters_bitwise_unchanged"] and
                                                          a["support_bitwise_unchanged"] for a in frozen_audits)
                                               if args.mode == "appearance_only" else None,
                                               parameter_count=len(frozen_initial),
                                               probe_times=probe_times),
                                           freeze_audits=frozen_audits,
                                           full_endpoint=args.steps == 12000))
    persist(out / "checkpoint_final.pt")


if __name__ == "__main__":
    main()
