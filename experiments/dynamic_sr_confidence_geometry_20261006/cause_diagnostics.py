"""Finite read-only cause diagnostics; all oracle outputs stay diagnostic-only.

Modes deliberately separate CPU cache inspection from frozen GPU rendering and
from evaluation of the three registered 500-update LR-only interventions. This
file never changes model parameters, Adam state, camera calibration or training
weights, and never turns a held-out observation into a legitimate model input.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_prior import ROOT, sha256, write_json

FRAMES = [0, 40, 80, 118]
FIT_FRAMES = [0, 40]
VALIDATION_FRAMES = [80, 118]
OLD = ROOT / "output/dynamic_sr_prior_diagnosis_20260929"
DEFAULT_ROI = OLD / "spectrum/roi_protocol.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def backend():
    sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_prior_guidance_20260927"))
    from gradient_policy import effective_state
    from shared import evaluator, load_manifest, load_model, rasterize
    return effective_state, evaluator(), load_manifest, load_model, rasterize


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image(manifest, observation, kind):
    source = Path(manifest.get("_root", manifest["_path"].parent)) / observation[kind + "_path"]
    if sha256(source) != observation[kind + "_sha256"]:
        raise ValueError(f"{kind} checksum mismatch: {source}")
    array = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"Unreadable image: {source}")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


def degrade_chw(raw, shape):
    value = torch.as_tensor(raw, dtype=torch.float32)
    return F.interpolate(value[None], size=shape, mode="bicubic", antialias=True,
                         align_corners=False)[0].clamp(0, 1)


def cached_raw(directory, camera, frame, checkpoint_sha256=None):
    path = Path(directory) / f"{camera}_{frame:04d}.npz"
    receipt = path.with_suffix(".json")
    if not path.exists():
        return None, None
    if not receipt.exists():
        raise ValueError(f"Unregistered float cache: {path}")
    metadata = read(receipt)
    digest = sha256(path)
    if metadata.get("npz_sha256") != digest:
        raise ValueError("Float-cache artifact checksum mismatch")
    if checkpoint_sha256 is not None and metadata["checkpoint"]["sha256"] != checkpoint_sha256:
        raise ValueError("Float-cache checkpoint mismatch")
    if metadata.get("camera") != camera or int(metadata.get("frame")) != frame:
        raise ValueError("Float-cache camera/time mismatch")
    with np.load(path, allow_pickle=False) as raw:
        array = raw["rgb_raw"].copy()
    return array, dict(path=str(path), sha256=digest, receipt_sha256=sha256(receipt),
                       checkpoint=metadata["checkpoint"])


def metric_lr(prediction, truth, mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    if mask is not None:
        error = error[np.asarray(mask).astype(bool)]
    if not error.size:
        return dict(mse=None, l1=None, psnr=None)
    mse = float(np.square(error).mean())
    return dict(mse=mse, l1=float(np.abs(error).mean()),
                psnr=-10. * math.log10(max(mse, 1e-12)))


def frequency(raw, truth):
    fft = module("cause_frozen_fft", ROOT / "experiments/dynamic_sr_same_observation_20260930/frequency.py")
    return fft.decompose(np.clip(np.asarray(raw).transpose(1, 2, 0), 0, 1).astype(np.float32),
                         truth.astype(np.float32))


def roi_mask(regions, camera, shape, hr_shape):
    """Each camera keeps its own pre-existing diagnostic coordinates."""
    h, w = shape
    hh, ww = hr_shape
    mask = np.zeros((h, w), bool)
    name = "lamp_wall_corner_reference" if camera == "cam01" else "background_books_reference"
    if name not in regions.get(camera, {}):
        return mask
    x0, y0, x1, y1 = regions[camera][name]
    mask[int(y0 * h / hh):int(math.ceil(y1 * h / hh)),
         int(x0 * w / ww):int(math.ceil(x1 * w / ww))] = True
    return mask


def fit_gain_bias(predictions, truths, mask):
    outputs = []
    for channel in range(3):
        x = np.concatenate([prediction[..., channel][mask] for prediction in predictions])
        y = np.concatenate([truth[..., channel][mask] for truth in truths])
        if not len(x):
            return None
        design = np.stack((x, np.ones_like(x)), 1).astype(np.float64)
        solution, _, rank, _ = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)
        outputs.append(dict(gain=float(solution[0]), bias=float(solution[1]), rank=int(rank)))
    return outputs


def apply_gain_bias(prediction, coefficients):
    gain = np.array([value["gain"] for value in coefficients])
    bias = np.array([value["bias"] for value in coefficients])
    return np.clip(prediction * gain + bias, 0, 1)


def shift_image(prediction, shift):
    height, width = prediction.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    # Positive shifts displace content right/down: output(x,y)=input(x-dx,y-dy).
    return cv2.remap(prediction.astype(np.float32), x - shift[0], y - shift[1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def fit_shift(predictions, truths, mask):
    if not mask.any():
        return None
    # Preregistered bounded diagnostic oracle, not a training hyperparameter grid.
    best = None
    for dy in np.arange(-2., 2.001, .25):
        for dx in np.arange(-2., 2.001, .25):
            objective = np.mean([metric_lr(shift_image(p, (dx, dy)), y, mask)["mse"]
                                 for p, y in zip(predictions, truths)])
            candidate = (float(objective), float(dx), float(dy))
            if best is None or candidate < best:
                best = candidate
    return dict(dx_lr=best[1], dy_lr=best[2], fit_mse=best[0],
                grid="[-2,+2] LRpx in0.25 steps, fixed deterministic tie break")


def calibration_audit(manifest):
    width_hr, height_hr = manifest["resolutions"]["hr"]
    width_lr, height_lr = manifest["resolutions"]["lr"]
    rows = []
    for camera, calibration in manifest["cameras"].items():
        source = np.asarray(calibration["K_hr"], np.float64)
        expected = source.copy()
        expected[0, :] *= width_lr / width_hr
        expected[1, :] *= height_lr / height_hr
        expected[0, 2] = (source[0, 2] + .5) * width_lr / width_hr - .5
        expected[1, 2] = (source[1, 2] + .5) * height_lr / height_hr - .5
        w2c = np.asarray(calibration["w2c"], np.float64)
        c2w = np.asarray(calibration["c2w"], np.float64)
        rows.append(dict(camera=camera,
            intrinsics_half_pixel_maxabs=float(np.abs(expected - np.asarray(calibration["K_lr"])).max()),
            pose_inverse_maxabs=float(np.abs(w2c @ c2w - np.eye(4)).max()),
            rotation_orthogonality_maxabs=float(np.abs(w2c[:3, :3] @ w2c[:3, :3].T - np.eye(3)).max()),
            rotation_determinant=float(np.linalg.det(w2c[:3, :3]))))
    observations = manifest["observations"]
    keys = [(o["camera_id"], int(o["frame_index"])) for o in observations]
    duplicate_keys = len(keys) - len(set(keys))
    time_error = max(abs(float(o["time"]) - int(o["frame_index"]) / 300.) for o in observations)
    return dict(rows=rows, duplicate_camera_frame_keys=duplicate_keys, time_frame_over300_maxabs=time_error,
                limitation="Internal mapping consistency only; no claim of independently correct physical calibration.")


def independent_reprojection(manifest):
    directory = OLD / "image_priors/geometry"
    rows = []
    for path in sorted(directory.glob("cam*.npz")):
        stem = path.stem.split("_")
        cameras = stem[:3]
        with np.load(path, allow_pickle=False) as arrays:
            xyz = arrays["xyz"]
            observed = [arrays["xy" + str(i)] for i in range(3)]
            third_valid = arrays["third_valid"].astype(bool)
            positive = arrays["pair_valid"].astype(bool)
            errors = []
            for camera, target in zip(cameras, observed):
                prediction, z = project(xyz, manifest["cameras"][camera])
                errors.append(np.linalg.norm(prediction - target, axis=1))
                positive &= z > 0
            stacked = np.stack(errors, 1) if len(xyz) else np.empty((0, 3))
            ok = third_valid & positive
            rows.append(dict(path=str(path), sha256=sha256(path), points=len(xyz),
                accepted_third_view_points=int(ok.sum()),
                third_view_reprojection_lr_px_mean=float(stacked[ok, 2].mean()) if ok.any() else None,
                all_view_reprojection_lr_px_max=float(stacked[ok].max()) if ok.any() else None,
                geometry_truth=False))
    return dict(status="completed" if rows else "unknown_no_existing_tracks", rows=rows,
                limitation="Sparse LR textured correspondence support; third view is excluded from initial triangulation. Unmatched wall pixels remain unknown.")


def project(xyz, calibration):
    world = np.c_[xyz, np.ones(len(xyz))]
    camera = world @ np.asarray(calibration["w2c"], np.float64)[:3, :].T
    pixels = camera @ np.asarray(calibration["K_lr"], np.float64).T
    denominator = np.where(np.abs(camera[:, 2]) > 1e-9, camera[:, 2], np.nan)
    return pixels[:, :2] / denominator[:, None], camera[:, 2]


def sample(array, pixels):
    return cv2.remap(np.asarray(array, np.float32), pixels[:, 0].astype(np.float32)[:, None],
                     pixels[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))[:, 0]


def cached_diagnostics(args, manifest, regions):
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    h, w = reversed(manifest["resolutions"]["lr"])
    hh, ww = reversed(manifest["resolutions"]["hr"])
    rows = []
    oracles = []
    provenance = []
    # P1's legacy source directory differs; absence is explicit, never fabricated.
    directories = dict(U6000=OLD / "observables/U6000", J1=OLD / "observables/r1_J1",
                       Async2=OLD / "observables/r1_Async2")
    for model, directory in directories.items():
        for camera in ["cam00", "cam01"]:
            predictions, truths = {}, {}
            mask = roi_mask(regions, camera, (h, w), (hh, ww))
            for frame in FRAMES:
                raw, receipt = cached_raw(directory, camera, frame)
                if raw is None:
                    rows.append(dict(model=model, camera=camera, frame=frame, status="missing_cache", lr_mse=None, lr_l1=None, hr_low_mse=None, hr_mse=None, region_lr_mse=None))
                    continue
                provenance.append(receipt)
                prediction = degrade_chw(raw, (h, w)).permute(1, 2, 0).numpy()
                lr = image(manifest, observations[camera, frame], "lr")
                hr = image(manifest, observations[camera, frame], "hr")
                quality = metric_lr(prediction, lr)
                fft = frequency(raw, hr)
                regional = metric_lr(prediction, lr, mask)
                rows.append(dict(model=model, camera=camera, frame=frame, status="completed",
                    lr_mse=quality["mse"], lr_l1=quality["l1"], hr_low_mse=fft["low_mse"], hr_mse=fft["mse"],
                    region_lr_mse=regional["mse"]))
                predictions[frame], truths[frame] = prediction, lr
            if set(predictions) != set(FRAMES):
                continue
            # Static proxy comes exclusively from the fitting LR frames, never 80/118.
            smooth = {frame: cv2.blur(truths[frame], (3, 3)) for frame in FIT_FRAMES}
            stable = (np.abs(smooth[0] - smooth[40]).mean(-1) <= 2 / 255.) & mask
            stable[:3] = stable[-3:] = False
            stable[:, :3] = stable[:, -3:] = False
            coefficients = fit_gain_bias([predictions[f] for f in FIT_FRAMES], [truths[f] for f in FIT_FRAMES], stable)
            translation = fit_shift([predictions[f] for f in FIT_FRAMES], [truths[f] for f in FIT_FRAMES], stable)
            oracle = dict(model=model, camera=camera, fit_frames=FIT_FRAMES, validation_frames=VALIDATION_FRAMES,
                region_pixels=int(mask.sum()), fitting_static_proxy_pixels=int(stable.sum()),
                gain_bias=coefficients, translation=translation, evaluations=[], legal_main_result=False,
                limitation="Development LR oracle; may compensate erroneous geometry. Static proxy means 0/40 LR patch color stability, not semantic truth.")
            for frame in FRAMES:
                variants = dict(original=predictions[frame])
                if coefficients is not None:
                    variants["gain_bias"] = apply_gain_bias(predictions[frame], coefficients)
                if translation is not None:
                    variants["translation"] = shift_image(predictions[frame], (translation["dx_lr"], translation["dy_lr"]))
                for variant, prediction in variants.items():
                    oracle["evaluations"].append(dict(frame=frame, split="fit" if frame in FIT_FRAMES else "validation",
                        variant=variant, full=metric_lr(prediction, truths[frame]), fixed_region=metric_lr(prediction, truths[frame], mask),
                        fitting_static_proxy=metric_lr(prediction, truths[frame], stable)))
            oracles.append(oracle)
    report = dict(status="completed", mode="cached", rows=rows, oracles=oracles,
        calibration=calibration_audit(manifest), independent_lr_tracks=independent_reprojection(manifest),
        source_artifacts=provenance, unavailable=dict(P1="No P1 observables under this cache schema; existing same_observation frequency report must remain separately cited."),
        parameter_updates=0, information_boundary="All development LR and HR reads only in diagnostic process; no camera/oracle outputs are consumed by training.")
    csv_write(args.out / "cached_quality.csv", rows)
    write_json(args.out / "cached_diagnostics.json", report)
    return report


def contribution_moments(camera, state, region_mask=None):
    """Actual native compositing weights via precomputed-color derivatives."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from depth_prior import packed_covariance, normalize_moments
    xyz = state["xyz"].detach()
    z = (torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1) @ camera.world_view_transform.to(xyz))[:, 2]
    colors = torch.stack((torch.ones_like(z), z, z.square()), 1).detach().contiguous().requires_grad_(True)
    settings = GaussianRasterizationSettings(image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2), bg=torch.zeros(3, device=xyz.device),
        scale_modifier=1., viewmatrix=camera.world_view_transform.to(xyz), projmatrix=camera.full_proj_transform.to(xyz),
        sh_degree=0, campos=camera.camera_center.to(xyz), prefiltered=False, debug=False)
    raw, radii, _ = GaussianRasterizer(raster_settings=settings)(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
        colors_precomp=colors, opacities=state["opacity"].detach().contiguous(), scales=None, rotations=None,
        cov3D_precomp=packed_covariance(state["cov"].detach()))
    objective = raw[0].sum() if region_mask is None else (raw[0] * torch.as_tensor(region_mask, device=xyz.device)).sum()
    weights = torch.autograd.grad(objective, colors)[0][:, 0].detach()
    normalized = normalize_moments(raw.detach(), (int(camera.image_height) // 4, int(camera.image_width) // 4))
    return normalized, radii.detach(), weights


def weighted_mean(values, weights):
    values, weights = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float((values[good] * weights[good]).sum() / weights[good].sum()) if good.any() else None


def render_raw(model, camera, state, rasterize, dc_only=False):
    sh = state["sh"]
    if dc_only:
        sh = sh.clone()
        sh[:, 1:] = 0
    return rasterize(model.g, camera, state["xyz"], state["cov"], state["opacity"], sh)["render"]


def render_diagnostics(args, manifest, regions):
    effective_state, evaluator, load_manifest, load_model, rasterize = backend()
    manifest = load_manifest(args.manifest)
    manifest["_path"] = args.manifest
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    model = load_model(args.resume, manifest)
    parent_sha = sha256(args.resume)
    model.g._deformation.eval()
    h, w = reversed(manifest["resolutions"]["lr"])
    hh, ww = reversed(manifest["resolutions"]["hr"])
    states = {}
    with torch.no_grad():
        for frame in FRAMES:
            states[frame] = effective_state(model, float(observations["cam02", frame]["time"]))
    camera = evaluator.render_camera(manifest, observations["cam01", 40], 0)
    mask_hr = roi_mask(regions, "cam01", (hh, ww), (hh, ww))
    moments, _, target_weights = contribution_moments(camera, states[40], mask_hr)
    weights = target_weights.cpu().numpy()
    selected = weights > 1e-6
    point_ids = np.flatnonzero(selected)
    points = states[40]["xyz"].cpu().numpy()[selected]
    source_rows = []
    source_support = []
    angles = []
    target_center = np.asarray(manifest["cameras"]["cam01"]["c2w"])[:3, 3]
    target_direction = points - target_center
    stable_votes = np.zeros(len(points), int)
    for source_camera in manifest["splits"]["train"]:
        cal = manifest["cameras"][source_camera]
        source_cam = evaluator.render_camera(manifest, observations[source_camera, 40], 0)
        sm, radii, contribution = contribution_moments(source_cam, states[40])
        actual = contribution.cpu().numpy()[selected]
        xy, z = project(points, cal)
        center = np.asarray(cal["c2w"])[:3, 3]
        direction = points - center
        cos = (direction * target_direction).sum(1) / np.maximum(np.linalg.norm(direction, axis=1) * np.linalg.norm(target_direction, axis=1), 1e-12)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        inside = (z > 0) & np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        actual_visible = actual > 1e-6
        usable_center = inside & actual_visible
        source_support.append(actual_visible)
        angles.append(angle)
        raw, provenance = cached_raw(OLD / "observables/U6000", source_camera, 40, parent_sha)
        if raw is None:
            with torch.no_grad():
                raw = render_raw(model, source_cam, states[40], rasterize).cpu().numpy()
        prediction = degrade_chw(raw, (h, w)).permute(1, 2, 0).numpy()
        truth = image(manifest, observations[source_camera, 40], "lr")
        observed = sample(truth, xy)
        predicted = sample(prediction, xy)
        error = np.abs(observed - predicted).mean(-1)
        sampled_alpha = sample(sm["alpha"].cpu().numpy(), xy)
        depth_mean = sample(sm["expected_z"].cpu().numpy(), xy)
        depth_variance = sample(sm["variance_z_raw"].cpu().numpy(), xy)
        # Fixed source-pixel LR temporal stability is a diagnostic proxy only.
        temporal = [sample(cv2.blur(image(manifest, observations[source_camera, f], "lr"), (3, 3)), xy) for f in FRAMES]
        stack = np.stack(temporal)
        stability = np.max(np.abs(stack - stack[1:2]).mean(-1), axis=0)
        stable_votes += (usable_center & np.isfinite(stability) & (stability <= 2 / 255.)).astype(int)
        source_rows.append(dict(camera=source_camera, frame=40, point_count=len(points),
            actual_visible_points=int(actual_visible.sum()), actual_source_alpha_contribution_sum=float(actual.sum()),
            contributing_points_with_center_inside=int(usable_center.sum()),
            target_mass_with_actual_source_contribution=float(weights[selected][actual_visible].sum() / max(weights.sum(), 1e-12)),
            target_weighted_angle_degrees=weighted_mean(angle, weights[selected] * actual_visible),
            target_weighted_source_l1=weighted_mean(error, weights[selected] * actual_visible),
            target_weighted_footprint_radius_lr=weighted_mean(radii.cpu().numpy()[selected] / 4., weights[selected] * actual_visible),
            target_weighted_alpha=weighted_mean(sampled_alpha, weights[selected] * actual_visible),
            target_weighted_abs_z_minus_parentmean=weighted_mean(np.abs(z - depth_mean), weights[selected] * actual_visible),
            target_weighted_parent_variance=weighted_mean(depth_variance, weights[selected] * actual_visible),
            coverage_source="exact native d(sum image alpha)/d per-point unit color, not opacity or nearest-camera substitute",
            physical_visibility_truth=False))
        print(json.dumps(dict(mode="cause_source_contribution", camera=source_camera, done=len(source_rows))), flush=True)
    support_count = np.stack(source_support).sum(0)
    static_selected = stable_votes >= 2
    static_ids = point_ids[static_selected]
    np.savez_compressed(args.out / "surface_support.npz", point_ids=point_ids, target_contribution=weights[selected],
        source_visible=np.stack(source_support), source_angles=np.stack(angles), actual_support_count=support_count,
        static_lr_votes=stable_votes, static_point_ids=static_ids, reference_xyz=points)
    dc_rows = []
    freeze_rows = []
    displacement_rows = []
    for frame in FRAMES:
        state = states[frame]
        delta = state["xyz"][static_ids] - states[40]["xyz"][static_ids]
        displacement_rows.append(dict(frame=frame, point_count=len(static_ids),
            displacement_world_mean=float(delta.norm(dim=1).mean()) if len(static_ids) else None,
            displacement_world_p95=float(torch.quantile(delta.norm(dim=1), .95)) if len(static_ids) else None,
            opacity_abs_change_mean=float((state["opacity"][static_ids] - states[40]["opacity"][static_ids]).abs().mean()) if len(static_ids) else None,
            sh_abs_change_mean=float((state["sh"][static_ids] - states[40]["sh"][static_ids]).abs().mean()) if len(static_ids) else None))
        for camera_id in ["cam00", "cam01", "cam02", "cam06", "cam12", "cam18"]:
            obs = observations[camera_id, frame]
            cam = evaluator.render_camera(manifest, obs, 0)
            with torch.no_grad():
                original = render_raw(model, cam, state, rasterize).cpu().numpy()
                dc_only = render_raw(model, cam, state, rasterize, True).cpu().numpy()
                truth_lr = image(manifest, obs, "lr")
                baseline = metric_lr(degrade_chw(original, (h, w)).permute(1, 2, 0), truth_lr)
                dc_metric = metric_lr(degrade_chw(dc_only, (h, w)).permute(1, 2, 0), truth_lr)
                dc_rows.append(dict(camera=camera_id, frame=frame, split=obs["split"], baseline_lr=baseline, dc_only_lr=dc_metric,
                    delta_lr_mse=dc_metric["mse"] - baseline["mse"], geometry_fixed=True))
                if camera_id in ["cam00", "cam01"]:
                    truth_hr = image(manifest, obs, "hr")
                    dc_rows[-1].update(baseline_frequency=frequency(original, truth_hr), dc_frequency=frequency(dc_only, truth_hr))
                if len(static_ids) and camera_id in ["cam00", "cam01"]:
                    modified = {name: tensor.clone() for name, tensor in state.items()}
                    for name in modified:
                        modified[name][static_ids] = states[40][name][static_ids]
                    fixed = render_raw(model, cam, modified, rasterize).cpu().numpy()
                    truth_hr = image(manifest, obs, "hr")
                    freeze_rows.append(dict(camera=camera_id, frame=frame, reference_frame=40,
                        static_support_points=len(static_ids), baseline_lr=baseline,
                        frozen_lr=metric_lr(degrade_chw(fixed, (h, w)).permute(1, 2, 0), truth_lr),
                        baseline_frequency=frequency(original, truth_hr), frozen_frequency=frequency(fixed, truth_hr),
                        static_support_from="Same frozen point projects into >=2 real training LR sources with 3x3 patch color change <=2/255 across registered frames; no held-out/HR masks",
                        semantic_static_truth=False))
    report = dict(status="completed", mode="render", parent_sha256=parent_sha,
        target=dict(camera="cam01", frame=40, region="lamp_wall_corner_reference", contributor_points=len(point_ids),
                    actual_target_alpha_mass=float(weights.sum()), alpha_lr_mean=float(moments["alpha"].mean()),
                    mean_raw_variance=float(moments["variance_z_raw"].mean())),
        sources=source_rows, actual_support_histogram={str(i): int((support_count == i).sum()) for i in range(len(source_rows) + 1)},
        dc_only=dc_rows, background_motion=displacement_rows, background_frozen=freeze_rows,
        static_support_points=len(static_ids), parameter_updates=0,
        limitations=["Contribution/occlusion derived from U6000 can inherit erroneous geometry; physical support remains unknown without independent LR tracks.",
                     "Projected same-point center and pixel footprint are model-inferred, not uniquely identified surfaces.",
                     "DC-only is a diagnostic intervention; it can improve held-out colors while harming training fit without proving geometry.",
                     "LR color stability is a conservative proxy, can include constant-texture moving surfaces; background freezing is evidence with this limitation."])
    write_json(args.out / "render_diagnostics.json", report)
    csv_write(args.out / "surface_sources.csv", source_rows)
    return report


def probe_evaluation(args, manifest, regions):
    effective_state, evaluator, load_manifest, load_model, rasterize = backend()
    manifest = load_manifest(args.manifest)
    manifest["_path"] = args.manifest
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    model = load_model(args.resume, manifest)
    model.g._deformation.eval()
    h, w = reversed(manifest["resolutions"]["lr"])
    rows = []
    surface = None
    if args.surface_support is not None:
        with np.load(args.surface_support, allow_pickle=False) as arrays:
            surface = {name: arrays[name].copy() for name in ["point_ids", "target_contribution", "reference_xyz"]}
    for frame in FRAMES:
        with torch.no_grad():
            state = effective_state(model, float(observations["cam02", frame]["time"]))
        for camera_id in ["cam00", "cam01", *manifest["splits"]["train"]]:
            observation = observations[camera_id, frame]
            camera = evaluator.render_camera(manifest, observation, 0)
            with torch.no_grad():
                raw = render_raw(model, camera, state, rasterize).cpu().numpy()
            lr = image(manifest, observation, "lr")
            prediction = degrade_chw(raw, (h, w)).permute(1, 2, 0).numpy()
            lr_metrics = metric_lr(prediction, lr)
            row = dict(probe=args.label, step=args.probe_step, camera=camera_id, frame=frame,
                       split=observation["split"], lr_mse=lr_metrics["mse"], lr_l1=lr_metrics["l1"],
                       surface_lr_l1=None, surface_position_delta_mean=None, hr_low_mse=None, hr_total_mse=None)
            if surface is not None:
                point_ids = surface["point_ids"]
                if int(point_ids.max()) >= len(state["xyz"]):
                    raise ValueError("Probe topology differs from frozen surface support")
                xyz = state["xyz"].cpu().numpy()[point_ids]
                # Fixed reference 3D locations, projected into each source; do not
                # let a moving current position chase a more favorable residual.
                xy, z = project(surface["reference_xyz"], manifest["cameras"][camera_id])
                pred = sample(prediction, xy)
                true = sample(lr, xy)
                error = np.abs(pred - true).mean(-1)
                weights = surface["target_contribution"] * (z > 0)
                row["surface_lr_l1"] = weighted_mean(error, weights)
                row["surface_position_delta_mean"] = weighted_mean(np.linalg.norm(xyz - surface["reference_xyz"], axis=1), weights)
            if camera_id in ["cam00", "cam01"]:
                fft = frequency(raw, image(manifest, observation, "hr"))
                row["hr_low_mse"], row["hr_total_mse"] = fft["low_mse"], fft["mse"]
            rows.append(row)
    child = model.children
    raw_offsets = child.offset_raw.detach()
    bounded = raw_offsets.tanh().abs()
    report = dict(status="completed", mode="probe", label=args.label, probe_step=args.probe_step,
        checkpoint_sha256=sha256(args.resume), checkpoint_metadata=model.checkpoint["metadata"], rows=rows,
        children_offset=dict(abs_tanh_mean=float(bounded.mean()), abs_tanh_p95=float(torch.quantile(bounded.flatten(), .95)),
            fraction_abs_tanh_ge095=float((bounded >= .95).float().mean()),
            mean_tanh_derivative=float((1 - bounded.square()).mean()),
            note="Saturation diagnostic; actual active gradients/Adam displacements belong to trainer logs."),
        surface_reference=str(args.surface_support) if args.surface_support is not None else None,
        parameter_updates=0, legal_main_result=False,
        limitation="Fixed train76 and dev8 diagnostics, not full benchmark evaluation; surface correspondence is inferred by frozen U6000.")
    csv_write(args.out / "probe_quality.csv", rows)
    write_json(args.out / "probe_evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cached", "render", "probe"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--roi", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--surface-support", type=Path)
    parser.add_argument("--label", default="U6000")
    parser.add_argument("--probe-step", type=int, choices=[0, 100, 500], default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode in ["render", "probe"] and args.resume is None:
        parser.error("GPU diagnostic modes require --resume")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    manifest = read(args.manifest)
    manifest["_path"] = args.manifest
    regions = read(args.roi)["regions_by_camera_xyxy_exclusive"]
    try:
        report = (cached_diagnostics if args.mode == "cached" else render_diagnostics if args.mode == "render" else probe_evaluation)(args, manifest, regions)
        write_json(args.out / "complete.json", dict(status="completed", mode=args.mode,
            diagnostic_only=True, parameter_updates=0, manifest_sha256=sha256(args.manifest),
            roi_sha256=sha256(args.roi), source_sha256=sha256(__file__),
            seconds=time.monotonic() - started, torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name() if args.mode != "cached" else None,
            visible_cuda=os.environ.get("CUDA_VISIBLE_DEVICES") if args.mode != "cached" else None,
            output_status=report["status"]))
    except BaseException:
        write_json(args.out / "failed.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
