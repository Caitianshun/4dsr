"""U/W/F: identical ordinary split, full-camera teachers, fixed losses."""
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

ROOT = Path(__file__).resolve().parents[2]
MOTION = ROOT / 'experiments/dynamic_sr_motion_bound_20260923'
sys.path.insert(0, str(MOTION))
from motion_model import OLD, capacity_summary, make_model, refinement_state, render_model
from detail_loss import OPERATOR, teacher_loss
from training_support import read, load_teacher_index, TeacherCache, initial_identity, fixed_gradient_audit

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
                   motion_refinement=refinement_state(model),
                   detail_supervision=metadata["detail_supervision"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def finite_gradients(parameters):
    return all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum())
                         for p in parameters if p.grad is not None))


def run(args, out):
    m = load_manifest(args.manifest)
    args.branch = "ordinary_split"
    args.sr_weight = .2 if args.method == "W" else .1
    manifest_hash, parent_hash = sha256(args.manifest), sha256(args.checkpoint)
    g, h, o, ck = load_checkpoint(args.checkpoint)
    if ck.get("surface_residual", {}).get("branch", "joint") != "joint":
        raise ValueError("Start all forks from the same pure-LR base, not a residual checkpoint")
    if "motion_refinement" in ck:
        raise ValueError("Start from the common unsplit parent checkpoint")
    if ck["metadata"].get("manifest_sha") != manifest_hash:
        raise ValueError("Parent checkpoint/manifest hash mismatch")
    assert ck["metadata"].get("mode") == "lr_integrated" and ck["metadata"]["step"] == 7200
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
    paths, teacher_inputs = load_teacher_index(args.teacher_index, m, records)
    ids = list(paths)
    schedule = read(args.schedule)
    assert schedule['manifest_sha256'] == manifest_hash and schedule['steps'] == 6000 and schedule['seed'] == args.seed
    assert schedule['record_keys'] == [[r['camera_id'], int(r['frame_index'])] for r in records]
    args.prior_cameras = ','.join(schedule['train_cameras'])
    args.prior_subdir = 'sr_swinir_x4'
    old_ids = [i for i,r in enumerate(records) if r['camera_id'] in schedule['original_teacher_cameras']]
    alpha, calibration = 0., None
    if args.method == 'F':
        if not args.calibration: raise ValueError('F requires fixed calibration')
        calibration = read(args.calibration)
        expected = dict(manifest_sha256=manifest_hash,parent_sha256=parent_hash,selection_sha256=selection_hash,
                        teacher_index_sha256=sha256(args.teacher_index),schedule_sha256=sha256(args.schedule),
                        detail_loss_sha256=sha256(Path(__file__).with_name('detail_loss.py')))
        assert calibration['status'] == 'calibrated'
        for k,v in expected.items():
            if calibration[k] != v: raise ValueError(f'Calibration identity mismatch: {k}')
        alpha = calibration['alpha']
        assert math.isfinite(alpha) and 0 < alpha < 1000
    for record in records:
        if record["split"] != "train":
            raise AssertionError("Nontraining observation in loader")
    # Restore after all construction. These camera streams intentionally begin
    # a new fork; they are identical across branches, not a legacy continuation.
    restore_global_rng(ck["rng"])
    lr_rng, sr_rng = random.Random(args.seed + 177), random.Random(args.seed + 211)
    identity = initial_identity(model)
    if calibration is not None and identity != calibration['initial_identity']:
        raise ValueError('Calibrated initial model/optimizer differs from formal F')
    source_files = [Path(__file__), MOTION / 'motion_model.py',
                    Path(__file__).with_name('detail_loss.py'),Path(__file__).with_name('training_support.py'),
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
                  teacher_inputs=teacher_inputs, teacher_index_sha256=sha256(args.teacher_index),
                  schedule_sha256=sha256(args.schedule), calibration_sha256=sha256(args.calibration) if args.calibration else None,
                  original_prior_cameras=','.join(schedule['original_teacher_cameras']),
                  alpha=alpha, initial_identity=identity, operator=OPERATOR,
                  fixed_lr_draw_sha256=schedule['lr_draw_sha256'], fixed_sr_frame_sha256=schedule['sr_frame_sha256'],
                  fixed_draw_sha256=schedule['draw_sha256'],
                  calibration_record=calibration, teacher_cache='64-image float32 CPU LRU, original image_tensor conversion; H(T) online no_grad',
                  inference='unchanged ordinary_split B renderer and parameters; no teacher or residual network', 
                  topology_changes="one shared selection split at step 0, fixed afterward",
                  capacity=initial_capacity, initialization=model.initialization,
                  child_learning_rates={group["name"]: group["lr"] for group in model.child_optimizer.param_groups},
                  scheduler_offset=ck["metadata"]["step"],
                  information_boundary="complete train LR RGB and original frozen SR only; no mask or HR pixel reads; no p targets",
                  photometric_loss="original LR mean L1 + fixed full RGB teacher mean L1 + F-only signed H residual mean L1",
                  sr_route="joint base and child parameters", lr_route="joint base and child parameters",
                  extra_child_regularization=False,
                  global_rng="restored parent after construction", samplers="new identical fork streams",
                  gpu=torch.cuda.get_device_name(), visible_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=str(torch.__version__), sources=sources)
    write_json(out / "config.json", config)
    write_json(out / "status.json", dict(status="running", pid=os.getpid(), started_utc=stamp()))
    cache = TeacherCache(paths, (3,height,width), maximum=64)
    exposure, draw = collections.Counter(), hashlib.sha256()
    lr_draw, sr_times = hashlib.sha256(), hashlib.sha256()
    audit_seconds = 0.0
    detail_metadata = dict(method=args.method, alpha=alpha,sr_weight=args.sr_weight,operator=OPERATOR,
                           teacher_index_sha256=sha256(args.teacher_index),schedule_sha256=sha256(args.schedule),
                           calibration_sha256=sha256(args.calibration) if args.calibration else None,
                           calibration_seconds=calibration.get('seconds') if calibration else None,
                           detail_loss_sha256=sha256(Path(__file__).with_name('detail_loss.py')))
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
            li, old_si = lr_rng.randrange(len(records)), sr_rng.choice(old_ids)
            expected_li, si, expected_old_si = schedule['rows'][step-1]
            assert li == expected_li and old_si == expected_old_si
            assert records[si]['frame_index'] == records[old_si]['frame_index']
            draw.update(f"{li},{si}\n".encode())
            lr_draw.update(f"{li}\n".encode())
            sr_times.update(f"{records[si]['frame_index']}\n".encode())
            g.optimizer.zero_grad(set_to_none=True)
            model.child_optimizer.zero_grad(set_to_none=True)
            target_lr = records[li]["image"].cuda()
            rendered_lr = render_model(model, cameras[li])["render"]
            lr_loss = (downsample(rendered_lr, target_lr.shape[-2:]) - target_lr).abs().mean()
            regulation = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
            (lr_loss + regulation).backward()
            audit = step == 1 or step in milestones
            target = cache.get(si)
            prediction = render_model(model, cameras[si])["render"]
            teacher_total, sr_loss, detail_loss = teacher_loss(prediction,target,args.method,alpha,target_lr.shape[-2:])
            loss = lr_loss + regulation + teacher_total
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            teacher_total.backward()
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
                           reg=float(regulation), sr_weight=args.sr_weight,method=args.method,alpha=alpha,
                           teacher_total=float(teacher_total), detail_l1=float(detail_loss) if detail_loss is not None else None,
                           weighted_detail=alpha*float(detail_loss) if detail_loss is not None else 0.,
                           points=capacity_summary(model)["active_gaussians"],
                           train_s=train_seconds, wall_s=time.monotonic() - started,
                           peak_gb=torch.cuda.max_memory_allocated() / 1e9, draw_sha256=draw.hexdigest())
                if norms is not None:
                    row.update(gradient_norms=norms, base_joint_updates=True,
                               parent_motion_binding=args.branch == "bound_split")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if step in milestones:
                ga = fixed_gradient_audit(model,records,cameras,cache,args.method,alpha,schedule['original_teacher_cameras'])
                ga['step'] = step
                audit_seconds += ga['seconds']
                with (out/'fixed_gradient_audit.jsonl').open('a') as ag: ag.write(json.dumps(ga,allow_nan=False)+'\n')
                current_capacity = capacity_summary(model)
                if current_capacity["active_gaussians"] != initial_points or g.active_sh_degree != initial_degree:
                    raise AssertionError("Fixed topology or SH degree changed")
                if not all(bool(torch.isfinite(p).all()) for p in all_parameters):
                    raise FloatingPointError(f"Nonfinite parameter at step {step}")
                metadata = dict(scene=m["scene"], stage="detail_supervision", branch=args.branch, method=args.method, detail_supervision=detail_metadata,
                                step=ck["metadata"]["step"] + step, intervention_step=step,
                                extent=ck["metadata"]["extent"], args=vars(args),
                                manifest=str(Path(args.manifest).resolve()), manifest_sha=manifest_hash,
                                parent_sha=parent_hash, selection_sha256=selection_hash,
                                points=current_capacity["active_gaussians"], capacity=current_capacity,
                                elapsed_s=time.monotonic() - started, train_s=train_seconds,
                                draw_sha256=draw.hexdigest(),lr_draw_sha256=lr_draw.hexdigest(),sr_frame_sha256=sr_times.hexdigest(),
                                initial_identity=identity,gradient_audit_seconds=audit_seconds)
                save(out / f"checkpoint_{step}.pt", model, metadata, lr_rng, sr_rng, exposure)
        log.flush()
        os.fsync(log.fileno())
    if args.steps == 6000:
        assert draw.hexdigest() == schedule['draw_sha256']
        assert lr_draw.hexdigest() == schedule['lr_draw_sha256'] and sr_times.hexdigest() == schedule['sr_frame_sha256']
    for source in sources:
        if sha256(source["path"]) != source["sha256"]:
            raise RuntimeError(f"Source changed during this run: {source['path']}")
    if sha256(args.selection) != selection_hash:
        raise RuntimeError("Selection file changed during this run")
    assert sha256(args.teacher_index) == config['teacher_index_sha256'] and sha256(args.schedule) == config['schedule_sha256']
    (out / "checkpoint_final.pt").symlink_to(f"checkpoint_{args.steps}.pt")
    write_json(out / "exposure.json", dict(exposure))
    complete = dict(**metadata, status="completed", parameter_updates=args.steps,
                    full_sampler_states_saved=True, source_unchanged=True,
                    wall_s=time.monotonic() - started, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                    teacher_cache_hits=cache.hits,teacher_cache_misses=cache.misses,
                    finished_utc=stamp(), smoke=args.smoke)
    write_json(out / "complete.json", complete)
    write_json(out / "status.json", complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True, help="Shared saved parent-point selection")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=['U','W','F'], required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--milestones", default="1200,6000")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--smoke", action="store_true", help="Exactly 20 updates, checkpoint20; not a scientific result")
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.milestones = 20, "20"
    if args.steps != 6000 and not args.smoke:
        parser.error('Formal protocol is exactly6000 updates')
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, out)
    except BaseException:
        failure = dict(status="failed", method=args.method, pid=os.getpid(),
                       failed_utc=stamp(), traceback=traceback.format_exc())
        write_json(out / "failed.json", failure)
        write_json(out / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
