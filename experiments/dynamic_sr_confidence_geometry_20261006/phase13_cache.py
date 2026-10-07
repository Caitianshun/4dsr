"""Frozen phase13 time/ST/mass evidence, with no RAFT inference or training.

Uses exactly the existing view descriptor H(T), LR closure and s_view. Candidate
counts are bounded by two. Temporal validity comes from legal training-LR flow,
FB/in-bounds/alpha/Census/photo support; detail reliability comes separately from
aligned frozen teacher H. Missing evidence stays unknown. Full training caches
require all 19x59 registered bidirectional temporal pairs; partial diagnostic
flow coverage cannot silently masquerade as a completed formal cache.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import confidence_cache as view

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOW_INDEX = ROOT / "output/dynamic_sr_prior_diagnosis_20260929/image_priors/flow_index.json"
TEMPORAL_POLICY = dict(schema=1, maximum_candidate_edges=2, flow_branch="A",
    flow_input="native registered training LR only; frozen RAFT-large C_T_SKHT_V2",
    flow_FB_maximum_lr_pixels_offset=.5, flow_FB_maximum_relative_magnitude=.05,
    alpha_minimum=view.POLICY["alpha_minimum"], census_radius=view.POLICY["census_radius"],
    census_maximum_fraction=view.POLICY["census_maximum_fraction"],
    photometric_maximum_rgb_mae=view.POLICY["photometric_maximum_rgb_mae"],
    texture_minimum_local_std=view.POLICY["texture_minimum_local_std"],
    unknown_confidence=view.POLICY["unknown_confidence"],
    detail_descriptor="exact existing view features: H(T)=T-Up(D(T)), AA reduced to2xcoarseLR",
    detail_error="AA source-footprint H warp, channel MAE,3x3 average; errors averaged BEFORE exponentiation",
    scale="copy frozen view s_lr and s_view exactly; no time/ST/mass recalibration",
    ST_time_direction="camera-major/frame-major preregistered observation index alternates next/previous; missing boundary edge remains missing",
    mass="mean of final HR c_LR*c_ST per observation, broadcast to every pixel",
    occlusion="FB inconsistency and in-bounds are LR flow occlusion/failure proxies, not true occlusion labels",
    cross_time_depth_comparison=False, no_new_gain_calibration=True)
RAFT_WEIGHT_SHA256 = "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"


def expected_pairs():
    return {(camera, first, first + 2) for camera in view.CAMERAS for first in view.FRAMES[:-1]}


class LegalFlowCache:
    """Read only hash-bound native-LR bidirectional flow; never infer missing flow."""
    def __init__(self, index_path, inputs, maximum=32):
        self.index_path = Path(index_path).resolve()
        self.index = view.read(self.index_path)
        if not self.index.get("status", "").startswith("completed"):
            raise ValueError("Only completed frozen flow indices accepted")
        if self.index["manifest_sha256"] != view.sha(inputs.manifest_path):
            raise ValueError("Frozen flow belongs to another manifest")
        if self.index.get("weight_sha256") != RAFT_WEIGHT_SHA256 or self.index.get("num_flow_updates") != 12:
            raise ValueError("Temporal evidence must reuse the registered frozen RAFT family and12 updates")
        self.inputs = inputs
        self.maximum = int(maximum)
        self.entries = {}
        self.cache = OrderedDict()
        for row in self.index.get("rows", self.index.get("entries", [])):
            if row.get("branch", "A") != "A":
                continue
            if row.get("information") in ["HR", "privileged", "diagnostic_HR_only"]:
                raise ValueError("Privileged flow forbidden")
            camera, first, second = row["camera"], int(row["first"]), int(row["second"])
            key = (camera, first, second)
            if key not in expected_pairs() or key in self.entries:
                raise ValueError("Illegal/duplicate native-LR temporal pair")
            sources = row.get("sources", [])
            if len(sources) != 2:
                raise ValueError("Both legal flow input identities are required")
            for source, frame in zip(sources, [first, second]):
                observation = inputs.records[camera, frame]
                expected_path = (inputs.root / observation["lr_path"]).resolve()
                # Permit an old machine's absolute root while retaining its
                # original record; compare the registry suffix and exact bytes.
                recorded = str(source["path"]).replace("\\", "/")
                if (source["camera"] != camera or int(source["frame"]) != frame or
                    source["sha256"] != observation["lr_sha256"] or
                    not recorded.endswith(str(Path(observation["lr_path"])).replace("\\", "/")) or
                    "/hr/" in recorded):
                    raise ValueError("Flow sources are not the exact legal training LR pair")
                if view.sha(expected_path) != observation["lr_sha256"]:
                    raise ValueError("Original legal LR identity changed")
            self.entries[key] = row
        self.missing = expected_pairs() - set(self.entries)
        self.opened = {}

    def report(self):
        return dict(status="complete_legal_temporal_flow" if not self.missing else "incomplete_legal_temporal_flow",
            index_sha256=view.sha(self.index_path), selected_branch="A_native_LR",
            available_undirected_pairs=len(self.entries), required_undirected_pairs=len(expected_pairs()),
            available_directed_edges=2 * len(self.entries), required_directed_edges=2 * len(expected_pairs()),
            missing_pairs=[dict(camera=c, first=f, second=g) for c, f, g in sorted(self.missing)],
            formal_training_ready=not self.missing, flow_inference_run=False, HR_pixels_read=False)

    def get(self, camera, first, second, shape):
        canonical = (camera, min(first, second), max(first, second))
        row = self.entries.get(canonical)
        if row is None:
            return None
        if canonical in self.cache:
            arrays = self.cache.pop(canonical)
        else:
            path = Path(row["path"])
            if not path.is_absolute():
                root_relative = ROOT / path
                path = root_relative if root_relative.exists() else self.index_path.parent / path
            if view.sha(path) != row["sha256"]:
                raise ValueError("Frozen temporal flow checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in ["forward", "backward"]}
            for array in arrays.values():
                if array.shape != (*self.inputs.shape_lr, 2) or not np.isfinite(array).all():
                    raise ValueError("Registered native LR flow must be finite HxWx2 in LR pixel units")
            self.opened[str(path)] = row["sha256"]
        self.cache[canonical] = arrays
        while len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        forward = arrays["forward"] if first < second else arrays["backward"]
        backward = arrays["backward"] if first < second else arrays["forward"]
        return resize_flow(forward, shape), resize_flow(backward, shape), row["sha256"]


def resize_flow(flow, shape):
    h, w = shape
    old_h, old_w = flow.shape[:2]
    resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= w / old_w
    resized[..., 1] *= h / old_h
    return resized


def temporal_evidence(target, source, forward, backward):
    """Flow support decides correspondence, teacher H error decides reliability."""
    h, w = target["depth"].shape
    if forward.shape != (h, w, 2) or backward.shape != forward.shape:
        raise ValueError("Flow must be scaled onto the same coarse feature grid")
    yy, xx = np.indices((h, w), dtype=np.float32)
    pixel = np.stack((xx, yy), -1)
    xy = pixel + forward
    magnitude = np.linalg.norm(forward, axis=-1)
    reverse = view.sample(backward, xy)
    fb_error = np.linalg.norm(forward + reverse, axis=-1)
    # Original registration is in LR pixels. Divide its constant component by
    # coarse_factor; the relative flow term scales with the vector automatically.
    bound = TEMPORAL_POLICY["flow_FB_maximum_lr_pixels_offset"] / view.POLICY["coarse_factor"] + TEMPORAL_POLICY["flow_FB_maximum_relative_magnitude"] * magnitude
    inside = ((xy[..., 0] >= 2) & (xy[..., 0] <= w - 3) &
              (xy[..., 1] >= 2) & (xy[..., 1] <= h - 3))
    finite = np.isfinite(forward).all(-1) & np.isfinite(reverse).all(-1)
    fb = fb_error <= bound
    alpha = ((target["alpha"] >= TEMPORAL_POLICY["alpha_minimum"]) &
             (view.sample(source["alpha"], xy) >= TEMPORAL_POLICY["alpha_minimum"]))
    # Dynamic camera-z can change along a true temporal correspondence. Do not
    # impose a static depth-agreement test across different scene times.
    geometric = inside & finite & fb & alpha
    footprint = view.projection_footprint(xy)
    aligned_lr = view.mip_sample(source["lr"], xy, footprint)
    gray, aligned_gray = target["lr"].mean(-1), aligned_lr.mean(-1)
    textured = ((view.local_std(gray) >= TEMPORAL_POLICY["texture_minimum_local_std"]) &
                (view.local_std(aligned_gray) >= TEMPORAL_POLICY["texture_minimum_local_std"]))
    photo = np.abs(target["lr"] - aligned_lr).mean(-1)
    census = view.census_error(gray, aligned_gray)
    observed = ((photo <= TEMPORAL_POLICY["photometric_maximum_rgb_mae"]) &
                (census <= TEMPORAL_POLICY["census_maximum_fraction"]))
    radius = TEMPORAL_POLICY["census_radius"]
    patch = cv2.erode(geometric.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    valid = patch & textured & observed
    # Flow is in coarse pixels here: LR vectors multiply by4 at the teacher
    # canvas or divide by2 at this descriptor grid. This states coordinate units,
    # not a claim that arbitrary warp and descriptor reduction commute exactly.
    # Use the same prefiltered descriptor/footprint comparison as frozen view.
    aligned_teacher_detail = view.mip_sample(source["detail"], xy, footprint)
    error = np.abs(target["detail"] - aligned_teacher_detail).mean(-1)
    error = cv2.boxFilter(error.astype(np.float32), -1, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    valid &= np.isfinite(error)
    return error, valid, dict(valid_fraction=float(valid.mean()), unknown_fraction=float((~valid).mean()),
        fov_fraction=float(inside.mean()), fb_supported_fraction=float(fb.mean()),
        alpha_supported_fraction=float(alpha.mean()), photometric_supported_fraction=float((patch & textured & observed).mean()),
        mean_FB_coarse_pixels=float(fb_error[patch].mean()) if patch.any() else None,
        mean_footprint=float(footprint[geometric].mean()) if geometric.any() else None,
        role="FB/alpha/photo/Census only validate LR correspondence; aligned teacher H decides detail reliability")


def aggregate_evidence(edges, shape):
    if len(edges) > 2:
        raise ValueError("At most two registered candidate evidence edges")
    total = np.zeros(shape, np.float32)
    count = np.zeros(shape, np.uint8)
    for error, valid in edges:
        total += np.where(valid, error, 0)
        count += valid.astype(np.uint8)
    return total / np.maximum(count, 1), count


def edge_table(inputs):
    nearest = view.nearest_cameras(inputs.manifest)
    table = {}
    for index, (camera, frame) in enumerate(inputs.keys):
        previous = frame - 2 if frame - 2 in view.FRAMES else None
        following = frame + 2 if frame + 2 in view.FRAMES else None
        selected = following if index % 2 == 0 else previous
        table[camera, frame] = dict(camera=camera, frame=frame, observation_index=index,
            time_candidates=[value for value in [previous, following] if value is not None],
            ST_spatial_camera=nearest[camera][0], ST_temporal_frame=selected,
            ST_temporal_direction="next" if index % 2 == 0 else "previous",
            maximum_time_candidates=2, maximum_ST_candidates=2)
    return table


class FeatureParentStore:
    def __init__(self, inputs, view_index, parent_index, maximum=64):
        self.inputs = inputs
        self.view_path = Path(view_index)
        self.view_index = view.read(self.view_path)
        if self.view_index.get("status") != "completed_frozen_train_confidence" or self.view_index.get("hr_derived_training_items") is not False:
            raise ValueError("The original completed legal view cache is required")
        if self.view_index["manifest_sha256"] != view.sha(inputs.manifest_path):
            raise ValueError("View cache manifest mismatch")
        self.feature_path = self.view_path.parent / "features/index.json"
        features = view.read(self.feature_path)
        self.features = {(e["camera"], int(e["frame"])): e for e in features["entries"]}
        self.parent_path = Path(parent_index)
        parents = view.read(self.parent_path)
        if parents["manifest_sha256"] != view.sha(inputs.manifest_path) or parents.get("privileged_train_hr", False):
            raise ValueError("Frozen parent cache is not the legal same-manifest support")
        self.parents = {(e["camera"], int(e["frame"])): e for e in parents["entries"]}
        if set(self.features) != set(inputs.keys) or set(self.parents) != set(inputs.keys):
            raise ValueError("Feature/parent caches must cover all legal observations")
        self.cache = OrderedDict()
        self.maximum = int(maximum)

    def get(self, key):
        if key in self.cache:
            result = self.cache.pop(key)
        else:
            feature = self.features[key]
            path = self.feature_path.parent / feature["path"]
            if view.sha(path) != feature["sha256"]:
                raise ValueError("Frozen view descriptor hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                result = {name: arrays[name].copy() for name in ["e_lr", "detail", "lr"]}
            parent = self.parents[key]
            path = self.parent_path.parent / parent["path"]
            if view.sha(path) != parent["sha256"]:
                raise ValueError("Frozen parent moment hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                result["depth"], result["alpha"] = view.aggregate_parent(arrays["depth_lr"], arrays["alpha_lr"])
        self.cache[key] = result
        while len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return result


def build_cache(inputs, flow_index, view_index, parent_index, output):
    flows = LegalFlowCache(flow_index, inputs)
    if flows.missing:
        raise ValueError(f"Formal time/ST cache needs complete legal flow: missing {len(flows.missing)}/{len(expected_pairs())} bidirectional pairs; use --audit-flow for zero-update missing-evidence audit")
    store = FeatureParentStore(inputs, view_index, parent_index)
    scales = store.view_index["scales"]
    if not np.isfinite([scales["s_lr"], scales["s_view"]]).all() or min(scales["s_lr"], scales["s_view"]) <= 0:
        raise ValueError("Original frozen confidence scales are invalid")
    table = edge_table(inputs)
    output = Path(output)
    # Permit the existing core cache root, but never replace an existing branch
    # or prior phase13 registration. The view/depth caches remain untouched.
    output.mkdir(parents=True, exist_ok=True)
    paths = {mode: output / f"confidence_{mode}" for mode in ["time", "ST", "mass"]}
    registration_dir = output / "phase13_registration"
    if any(path.exists() for path in paths.values()) or registration_dir.exists():
        raise FileExistsError("Existing phase13 cache/registration; choose a new output root")
    registration_dir.mkdir()
    for path in paths.values():
        path.mkdir()
    view.write(registration_dir / "edge_table.json", dict(rows=list(table.values()), policy=TEMPORAL_POLICY,
        input_order="Exact LegalInputs camera-major/frame-major keys; local deterministic registration, no training RNG"))
    cals = {c: view.coarse_calibration(inputs.manifest["cameras"][c], view.POLICY["coarse_factor"]) for c in view.CAMERAS}
    entries = {mode: [] for mode in paths}
    coverages = {mode: [] for mode in paths}
    started = time.monotonic()
    for n, key in enumerate(inputs.keys):
        target = store.get(key)
        shape = target["depth"].shape
        registration = table[key]
        temporal = {}
        edge_stats = []
        for other_frame in registration["time_candidates"]:
            flow = flows.get(key[0], key[1], other_frame, shape)
            if flow is None:
                raise AssertionError("Validated full flow index unexpectedly lost a registered pair")
            error, valid, stats = temporal_evidence(target, store.get((key[0], other_frame)), flow[0], flow[1])
            temporal[other_frame] = (error, valid)
            edge_stats.append(dict(axis="time", frame=other_frame, flow_sha256=flow[2], **stats))
        time_error, time_count = aggregate_evidence(list(temporal.values()), shape)
        spatial_key = (registration["ST_spatial_camera"], key[1])
        space_error, space_valid, spatial_stats = view.neighbour_evidence(target, store.get(spatial_key), cals[key[0]], cals[spatial_key[0]])
        st_edges = [(space_error, space_valid)]
        if registration["ST_temporal_frame"] is not None:
            st_edges.append(temporal[registration["ST_temporal_frame"]])
        st_error, st_count = aggregate_evidence(st_edges, shape)
        final_st = None
        for mode, error, count in [("time", time_error, time_count), ("ST", st_error, st_count)]:
            cl, ce, confidence = view.confidence_from_errors(target["e_lr"], error, count, scales, inputs.shape_hr)
            if mode == "ST":
                final_st = confidence
            path = paths[mode] / f"{key[0]}_{key[1]:04d}.npz"
            np.savez_compressed(path, c_lr=cl, c_view=ce, c_evidence=ce, valid_neighbours=count,
                                mean_detail_error=error)
            entries[mode].append(dict(camera=key[0], frame=key[1], path=path.name, sha256=view.sha(path), frozen=True,
                                     c_lr_shape=list(cl.shape), c_view_shape=list(ce.shape)))
            coverages[mode].append(dict(camera=key[0], frame=key[1], **view.statistics(confidence),
                valid_fraction=float((count > 0).mean()), unknown_fraction=float((count == 0).mean()),
                two_valid_edges_fraction=float((count == 2).mean()), candidate_edges=len(temporal) if mode == "time" else len(st_edges)))
        mass = float(final_st.astype(np.float64).mean())
        # Base ConfidenceCache can consume the same field schema. c_lr is a
        # constant mass and c_view is1 here; their HR product is exactly mass.
        path = paths["mass"] / f"{key[0]}_{key[1]:04d}.npz"
        cl = np.full(inputs.shape_lr, mass, np.float32)
        ce = np.ones(shape, np.float32)
        np.savez_compressed(path, c_lr=cl, c_view=ce, valid_neighbours=st_count, observation_mass=np.float64(mass))
        entries["mass"].append(dict(camera=key[0], frame=key[1], path=path.name, sha256=view.sha(path), frozen=True,
            c_lr_shape=list(cl.shape), c_view_shape=list(ce.shape), ST_observation_mean=mass))
        coverages["mass"].append(dict(camera=key[0], frame=key[1], mean_weight=mass, kish_effective_fraction=1.,
            valid_fraction=float((st_count > 0).mean()), unknown_fraction=float((st_count == 0).mean()),
            two_valid_edges_fraction=float((st_count == 2).mean()), candidate_edges=len(st_edges)))
        view.write(registration_dir / f"edges_{key[0]}_{key[1]:04d}.json", dict(registration=registration,
            time=edge_stats, ST_spatial=dict(neighbour=spatial_key[0], **spatial_stats), scale_reused=scales["s_view"]))
        if (n + 1) % 60 == 0:
            print(json.dumps(dict(stage="phase13_evidence", entries=n + 1, seconds=time.monotonic() - started)), flush=True)
    identity = dict(**inputs.identity(), policy=TEMPORAL_POLICY, source_sha256=view.sha(__file__),
        confidence_source_sha256=view.sha(Path(view.__file__)), flow_index_sha256=view.sha(flow_index),
        original_view_index_sha256=view.sha(view_index), parent_index_sha256=view.sha(parent_index),
        edge_table_sha256=view.sha(registration_dir / "edge_table.json"), scales_recalibrated=False)
    for mode in paths:
        view.write(paths[mode] / "cache_manifest.json", dict(schema=1, status="completed_frozen_train_confidence", entries=entries[mode],
            manifest_sha256=view.sha(inputs.manifest_path), identity=identity, scales=scales, mode=mode,
            hr_derived_training_items=False, parameter_updates=0, frozen=True,
            summary=dict(mean_weight=float(np.mean([r["mean_weight"] for r in coverages[mode]])),
                mean_valid_fraction=float(np.mean([r["valid_fraction"] for r in coverages[mode]])),
                mean_unknown_fraction=float(np.mean([r["unknown_fraction"] for r in coverages[mode]]))),
            coverage=coverages[mode], flow_read_log=flows.opened,
            source_index_sha256=dict(original_view=view.sha(view_index), parent=view.sha(parent_index)),
            output="Existing ConfidenceCache reads c_lr and legacy c_view(c_evidence) fields; mass mode broadcasts final ST observation mean",
            phase13_training_authorized_by_this_builder=False))
    view.write(registration_dir / "complete.json", dict(status="completed_frozen_phase13_evidence", modes=list(paths),
        identity=identity, seconds=time.monotonic() - started, gpu_used=False, RAFT_new_inference_run=False,
        parameter_updates=0, formal_training_must_wait_for_core_quality_decision=True))
    return {mode: path / "cache_manifest.json" for mode, path in paths.items()}


def factor_cache_path(method, view_index, phase13_root):
    """Two-factor consumer mapping; no new gains or losses are invented here."""
    if method in ["G_Q_uniform", "G+Q_uniform"]:
        return None  # caller uses all-one confidence, original0.1*k_R
    if method in ["G_FullSR_ST", "G+FullSR_ST", "RG_ST"]:
        return Path(phase13_root) / "confidence_ST/cache_manifest.json"
    if method == "RG_time":
        return Path(phase13_root) / "confidence_time/cache_manifest.json"
    if method == "RG_mass":
        return Path(phase13_root) / "confidence_mass/cache_manifest.json"
    if method in ["RG", "RG_view", "R"]:
        return Path(view_index)
    raise ValueError(f"Unregistered phase13 consumer method: {method}")


Phase13ConfidenceCache = view.ConfidenceCache


def self_check(path):
    yy, xx = np.indices((32, 48), dtype=np.float32)
    pattern = .4 + .1 * np.sin(xx * .7) + .1 * np.cos(yy * .5)
    lr = np.repeat(pattern[..., None], 3, -1)
    detail = np.repeat((.02 * np.sin(xx * .7))[..., None], 3, -1)
    target = dict(depth=np.ones((32, 48), np.float32), alpha=np.ones((32, 48), np.float32), lr=lr, detail=detail)
    flow = np.zeros((32, 48, 2), np.float32)
    same_error, same_valid, _ = temporal_evidence(target, target, flow, flow)
    assert same_valid.any() and float(same_error[same_valid].max()) < 1e-7
    wrong = dict(target, detail=-detail)
    wrong_error, wrong_valid, _ = temporal_evidence(target, wrong, flow, flow)
    assert np.array_equal(same_valid, wrong_valid) and float(wrong_error[wrong_valid].mean()) > .01
    missing = np.zeros_like(same_valid)
    mean_error, count = aggregate_evidence([(np.full((32, 48), .01, np.float32), same_valid),
                                            (np.full((32, 48), .09, np.float32), same_valid)], (32, 48))
    assert np.all(count[same_valid] == 2) and np.allclose(mean_error[same_valid], .05)
    _, unknown_count = aggregate_evidence([(same_error, missing)], (32, 48))
    _, ce, weight = view.confidence_from_errors(np.zeros((64, 96), np.float32), np.zeros((32, 48), np.float32),
        unknown_count, dict(s_lr=1 / 255., s_view=1 / 255.), (256, 384))
    assert np.all(ce == .5) and np.all(weight == .5)
    scaled = resize_flow(np.ones((64, 96, 2), np.float32) * 2., (32, 48))
    assert np.all(scaled == 1.)
    mass = float(weight.astype(np.float64).mean())
    assert mass == .5
    # A nonuniform product must be averaged after both maps are resized and
    # multiplied; multiplying their separate spatial means is a different mass.
    e_lr = np.linspace(0, .02, 64 * 96, dtype=np.float32).reshape(64, 96)
    e_detail = np.linspace(.02, 0, 32 * 48, dtype=np.float32).reshape(32, 48)
    _, _, final_st = view.confidence_from_errors(e_lr, e_detail, np.ones((32, 48), np.uint8),
        dict(s_lr=.02, s_view=.02), (256, 384))
    observed_mass = float(final_st.astype(np.float64).mean())
    broadcast = np.full_like(final_st, observed_mass)
    mass_gap = abs(float(broadcast.astype(np.float64).sum()) - float(final_st.astype(np.float64).sum()))
    assert mass_gap <= final_st.size * np.finfo(np.float32).eps
    view.write(path, dict(status="passed", same_teacher_error_zero=True,
        FB_perfect_but_wrong_teacher_error=float(wrong_error[wrong_valid].mean()),
        errors_mean_before_exponent=True, unknown_kept_pointfive=True,
        displacement_scaled_with_grid=True, per_observation_mass=mass,
        nonuniform_final_ST_mass=observed_mass, broadcast_total_weight_rounding_gap=mass_gap,
        gpu_used=False, flow_inference_run=False, scene_parameter_updates=0, source_sha256=view.sha(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=view.DEFAULT_MANIFEST)
    parser.add_argument("--teacher-index", type=Path, default=view.DEFAULT_TEACHERS)
    parser.add_argument("--flow-index", type=Path, default=DEFAULT_FLOW_INDEX)
    parser.add_argument("--view-index", type=Path)
    parser.add_argument("--parent-index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-flow", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if args.self_check:
        self_check(args.out)
    elif args.audit_flow:
        inputs = view.LegalInputs(args.manifest, args.teacher_index)
        report = LegalFlowCache(args.flow_index, inputs).report()
        view.write(args.out, report)
        print(json.dumps({key: value for key, value in report.items() if key != "missing_pairs"}), flush=True)
    else:
        if args.view_index is None or args.parent_index is None:
            parser.error("--view-index and --parent-index required to build frozen evidence")
        inputs = view.LegalInputs(args.manifest, args.teacher_index)
        build_cache(inputs, args.flow_index, args.view_index, args.parent_index, args.out)


if __name__ == "__main__":
    main()
