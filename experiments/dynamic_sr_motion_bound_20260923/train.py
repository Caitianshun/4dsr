"""Ordinary and parent-motion-bound spatial splits from one pure-LR parent.

Both branches use one saved selection and the same canonical child initial
values. Full-RGB LR closure, 0.1 SR supervision and original base regularizers
update the base and children jointly. Split once before training; keep the
active rendering topology fixed afterward. No masks or HR target reads.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from motion_model import (BRANCHES, OLD, ROOT, capacity_summary, make_model,
                          refinement_state, render_model)

sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_20260920"))
from common import (UPSTREAM, downsample, image_tensor, load_checkpoint,
                    named_parameters, resized_camera, sha256, write_json)
from n3dv_data import load_manifest
from run_experiment import load_training
from resume_control import assert_state_equal, restore_global_rng


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, model, metadata, lr_rng, sr_rng, exposure):
    """Base capture plus child values, optimizer and motion-binding metadata."""
    payload = dict(model=model.g.capture(), hidden=vars(model.h), optim=vars(model.o),
                   metadata=metadata,
                   rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                            numpy=np.random.get_state(), python=random.getstate()),
                   samplers=dict(lr=lr_rng.getstate(), sr=sr_rng.getstate(),
                                 source_prefix_exposure={}, additional_exposure=dict(exposure)),
                   motion_refinement=refinement_state(model))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def teacher_observations(m, records, args):
    """Use explicit SR paths or existing frozen teacher files; no fixed count."""
    observations = {(r["camera_id"], int(r["frame_index"])): r
                    for r in m["observations"] if r["split"] == "train"}
    selected = set(args.prior_cameras.split(",")) if args.prior_cameras else None
    paths = {}
    for i, record in enumerate(records):
        if selected is not None and record["camera_id"] not in selected:
            continue
        obs = observations[(record["camera_id"], int(record["frame_index"]))]
        relative = obs.get("sr_path")
        path = (Path(m["_root"]) / relative if relative is not None else
                Path(m["_root"]) / args.prior_subdir / record["camera_id"] / Path(obs["lr_path"]).name)
        if path.is_file():
            paths[i] = path
        elif selected is not None or relative is not None:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("No frozen training SR targets found; prepare teachers first")
    return list(paths), paths


def finite_gradients(parameters):
    return all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum())
                         for p in parameters if p.grad is not None))


def run(args, out):
    m = load_manifest(args.manifest)
    manifest_hash, parent_hash = sha256(args.manifest), sha256(args.checkpoint)
    g, h, o, ck = load_checkpoint(args.checkpoint)
    if ck.get("surface_residual", {}).get("branch", "joint") != "joint":
        raise ValueError("Start all forks from the same pure-LR base, not a residual checkpoint")
    if "motion_refinement" in ck:
        raise ValueError("Start from the common unsplit parent checkpoint")
    if ck["metadata"].get("manifest_sha") != manifest_hash:
        raise ValueError("Parent checkpoint/manifest hash mismatch")
    assert_state_equal(ck["model"][12], g.optimizer.state_dict())
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    if selection.get("parent_sha256") != parent_hash:
        raise ValueError("Selection/common parent SHA-256 mismatch")
    if selection.get("manifest_sha256") != manifest_hash:
        raise ValueError("Selection/manifest SHA-256 mismatch")
    selection_hash = sha256(args.selection)
    model = make_model(g, h, o, ck, m, args.branch, selection)
    initial_capacity = capacity_summary(model)
    records, cameras = load_training(m)
    if not records:
        raise ValueError("Empty training split")
    width, height = map(int, m["resolutions"]["hr"])
    cameras = [resized_camera(c, height, width) for c in cameras]
    ids, paths = teacher_observations(m, records, args)
    for record in records:
        if record["split"] != "train":
            raise AssertionError("Nontraining observation in loader")
    # Restore after all construction. These camera streams intentionally begin
    # a new fork; they are identical across branches, not a legacy continuation.
    restore_global_rng(ck["rng"])
    lr_rng, sr_rng = random.Random(args.seed + 177), random.Random(args.seed + 211)
    source_files = [Path(__file__), Path(__file__).with_name("motion_model.py"),
                    OLD / "common.py", OLD / "n3dv_data.py", OLD / "run_experiment.py",
                    ROOT / "experiments/dynamic_sr_20260920/resume_control.py",
                    UPSTREAM / "scene/gaussian_model.py", UPSTREAM / "scene/deformation.py",
                    UPSTREAM / "scene/hexplane.py", UPSTREAM / "gaussian_renderer/__init__.py"]
    (out / "sources").mkdir()
    sources = []
    for index, path in enumerate(source_files):
        copy = out / "sources" / f"{index}_{path.name}"
        shutil.copyfile(path, copy)
        sources.append(dict(path=str(path), sha256=sha256(path), snapshot=str(copy)))
    config = dict(**vars(args), manifest_sha256=manifest_hash, parent_sha256=parent_hash,
                  selection_sha256=selection_hash,
                  parent_metadata=ck["metadata"], initial_points=model.original_count,
                  sh_degree=g.active_sh_degree,
                  time_range=[min(r["time"] for r in records), max(r["time"] for r in records)],
                  lr_size=list(reversed(m["resolutions"]["lr"])), teacher_count=len(ids),
                  teacher_cameras=sorted({records[i]["camera_id"] for i in ids}),
                  teacher_inputs=[dict(camera=records[i]["camera_id"], frame=records[i]["frame_index"],
                                       path=str(paths[i]), sha256=sha256(paths[i])) for i in ids],
                  topology_changes="one shared selection split at step 0, fixed afterward",
                  capacity=initial_capacity, initialization=model.initialization,
                  child_learning_rates={group["name"]: group["lr"] for group in model.child_optimizer.param_groups},
                  scheduler_offset=ck["metadata"]["step"],
                  information_boundary="complete train LR RGB and original frozen SR only; no mask or HR pixel reads; no p targets",
                  photometric_loss="full-image mean RGB L1 for LR and SR, including foreground and background equally per pixel",
                  sr_route="joint base and child parameters", lr_route="joint base and child parameters",
                  extra_child_regularization=False,
                  global_rng="restored parent after construction", samplers="new identical fork streams",
                  gpu=torch.cuda.get_device_name(), visible_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=str(torch.__version__), sources=sources)
    write_json(out / "config.json", config)
    write_json(out / "status.json", dict(status="running", pid=os.getpid(), started_utc=stamp()))
    cache, exposure, draw = {}, collections.Counter(), hashlib.sha256()
    milestones = {int(v) for v in args.milestones.split(",") if 0 < int(v) <= args.steps} | {args.steps}
    base_parameters = list(named_parameters(g).values())
    child_parameters = list(model.children.parameters())
    all_parameters = base_parameters + child_parameters
    initial_points, initial_degree = initial_capacity["active_gaussians"], g.active_sh_degree
    started, train_seconds = time.monotonic(), 0.0
    torch.cuda.reset_peak_memory_stats()
    with (out / "training.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            tick = time.monotonic()
            g.update_learning_rate(ck["metadata"]["step"] + step)
            li, si = lr_rng.randrange(len(records)), sr_rng.choice(ids)
            draw.update(f"{li},{si}\n".encode())
            g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            target_lr = records[li]["image"].cuda()
            rendered_lr = render_model(model, cameras[li])["render"]
            lr_loss = (downsample(rendered_lr, target_lr.shape[-2:]) - target_lr).abs().mean()
            regulation = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
            (lr_loss + regulation).backward()
            audit = step == 1 or step in milestones
            if si not in cache:
                target = image_tensor(paths[si], device="cpu")
                if tuple(target.shape) != (3, height, width):
                    raise ValueError(f"Teacher shape mismatch: {paths[si]}")
                cache[si] = target
            prediction = render_model(model, cameras[si])["render"]
            sr_loss = (prediction - cache[si].cuda()).abs().mean()
            loss = lr_loss + regulation + args.sr_weight * sr_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (args.sr_weight * sr_loss).backward()
            if audit and not finite_gradients(all_parameters):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            norms = ({"base": gradient_norm(base_parameters), "children": gradient_norm(child_parameters)}
                     if audit else None)
            g.optimizer.step()
            model.child_optimizer.step()
            torch.cuda.synchronize()
            train_seconds += time.monotonic() - tick
            exposure[f'{records[si]["camera_id"]}/{records[si]["frame_index"]}'] += 1
            if step == 1 or step % 100 == 0 or step in milestones:
                row = dict(step=step, lr_l1=float(lr_loss), teacher_l1=float(sr_loss),
                           reg=float(regulation), sr_weight=args.sr_weight,
                           points=capacity_summary(model)["active_gaussians"],
                           train_s=train_seconds, wall_s=time.monotonic() - started,
                           peak_gb=torch.cuda.max_memory_allocated() / 1e9, draw_sha256=draw.hexdigest())
                if norms is not None:
                    row.update(gradient_norms=norms, base_joint_updates=True,
                               parent_motion_binding=args.branch == "bound_split")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if step in milestones:
                current_capacity = capacity_summary(model)
                if current_capacity["active_gaussians"] != initial_points or g.active_sh_degree != initial_degree:
                    raise AssertionError("Fixed topology or SH degree changed")
                if not all(bool(torch.isfinite(p).all()) for p in all_parameters):
                    raise FloatingPointError(f"Nonfinite parameter at step {step}")
                metadata = dict(scene=m["scene"], stage="motion_bound", branch=args.branch,
                                step=ck["metadata"]["step"] + step, intervention_step=step,
                                extent=ck["metadata"]["extent"], args=vars(args),
                                manifest=str(Path(args.manifest).resolve()), manifest_sha=manifest_hash,
                                parent_sha=parent_hash, selection_sha256=selection_hash,
                                points=current_capacity["active_gaussians"], capacity=current_capacity,
                                elapsed_s=time.monotonic() - started, train_s=train_seconds,
                                draw_sha256=draw.hexdigest())
                save(out / f"checkpoint_{step}.pt", model, metadata, lr_rng, sr_rng, exposure)
        log.flush()
        os.fsync(log.fileno())
    for source in sources:
        if sha256(source["path"]) != source["sha256"]:
            raise RuntimeError(f"Source changed during this run: {source['path']}")
    if sha256(args.selection) != selection_hash:
        raise RuntimeError("Selection file changed during this run")
    (out / "checkpoint_final.pt").symlink_to(f"checkpoint_{args.steps}.pt")
    write_json(out / "exposure.json", dict(exposure))
    complete = dict(**metadata, status="completed", parameter_updates=args.steps,
                    full_sampler_states_saved=True, source_unchanged=True,
                    wall_s=time.monotonic() - started, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                    finished_utc=stamp(), smoke=args.smoke)
    write_json(out / "complete.json", complete)
    write_json(out / "status.json", complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True, help="Shared saved parent-point selection")
    parser.add_argument("--out", required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--milestones", default="1200,6000")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--sr-weight", type=float, default=0.1)
    parser.add_argument("--prior-cameras", help="Optional camera list; otherwise discover available training teachers")
    parser.add_argument("--prior-subdir", default="sr_swinir_x4")
    parser.add_argument("--smoke", action="store_true", help="Exactly 20 updates, checkpoint20; not a scientific result")
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.milestones = 20, "20"
    if args.steps <= 0 or args.sr_weight <= 0 or not math.isfinite(args.sr_weight):
        parser.error("steps and finite sr-weight must be positive")
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, out)
    except BaseException:
        failure = dict(status="failed", branch=args.branch, pid=os.getpid(),
                       failed_utc=stamp(), traceback=traceback.format_exc())
        write_json(out / "failed.json", failure)
        write_json(out / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
