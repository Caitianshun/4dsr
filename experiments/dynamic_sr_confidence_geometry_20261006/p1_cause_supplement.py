"""Recover fixed P1 diagnostic assets without changing training or old receipts.

Audit is CPU only. Render reuses four registered legacy float arrays and renders
only the four missing 0/118 observations. Finish merges P1 into a new cause-cache
report and matrix, preserving the original missing-evidence report by hash.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import torch

import cause_diagnostics as cause
import cause_matrix as matrix
from depth_prior import ROOT, sha256, write_json

OUT = ROOT / "output/dynamic_sr_confidence_geometry_20261006"
LEGACY = ROOT / "output/dynamic_sr_same_observation_20260930"
MANIFEST = ROOT / "data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json"
FIXED = [(camera, frame) for camera in ["cam00", "cam01"] for frame in cause.FRAMES]
DISCLOSURE = (
    "P1-from-start is a historical reference: independent LR initialization, "
    "1000 coarse +19200 fine updates, native 117971 points and a different "
    "densification/Adam trajectory. It has no U6000 continuation or bounded "
    "children. It is not a capacity/budget/initialization-matched six-arm control. "
    "These fixed development observations and HR reads are diagnostic only; "
    "no weights, masks or training choices are derived from them.")


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def checked(entry):
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Registered source changed: {path}")
    return path


def float_array(path, expected_shape):
    with np.load(path, allow_pickle=False) as arrays:
        array = arrays["rgb_raw"].copy()
    if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid native float32 render: {path}")
    return array


def audit(args):
    complete_path = LEGACY / "train/complete.json"
    complete = read(complete_path)
    checkpoint = LEGACY / "train" / complete["final_checkpoint"]
    expected_cp = complete["final_sha256"]
    if sha256(checkpoint) != expected_cp:
        raise ValueError("P1 checkpoint byte checksum mismatch")
    receipt_path = checkpoint.with_suffix(".json")
    receipt = read(receipt_path)
    if receipt["sha256"] != expected_cp or receipt["metadata"]["total_updates"] != 20200:
        raise ValueError("P1 fixed endpoint changed")
    if receipt["metadata"]["points"] != 117971 or receipt["metadata"]["privileged_train_hr"]:
        raise ValueError("P1 historical parameterization/information boundary changed")
    manifest = read(args.manifest)
    manifest_sha = sha256(args.manifest)
    if manifest_sha != receipt["metadata"]["manifest_sha"]:
        raise ValueError("P1 manifest mismatch")
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    h, w = reversed(manifest["resolutions"]["hr"])
    legacy_index_path = LEGACY / "figures/index.json"
    legacy_index = read(legacy_index_path)
    integrity_path = LEGACY / "final_integrity.json"
    integrity = read(integrity_path)
    # The index is registered in the historical final integrity receipt.
    values = [v for v in integrity.values() if isinstance(v, dict)]
    registered_index_sha = next((v["figures/index.json"] for v in values if "figures/index.json" in v), None)
    if registered_index_sha != sha256(legacy_index_path):
        raise ValueError("Legacy float-cache index differs from final integrity receipt")
    sources = {(e["camera"], int(e["frame"])): e for e in legacy_index["sources"]
               if e["model"] == "P1-from-start"}
    entries = []
    for camera, frame in FIXED:
        observation = observations[camera, frame]
        for kind in ["lr", "hr"]:
            source = args.manifest.parent / observation[kind + "_path"]
            if sha256(source) != observation[kind + "_sha256"]:
                raise ValueError(f"Diagnostic image changed: {source}")
        entry = dict(camera=camera, frame=frame, time=float(observation["time"]))
        if (camera, frame) in sources:
            source = sources[camera, frame]
            path = checked(source)
            array = float_array(path, (3, h, w))
            entry.update(mode="reuse_registered_legacy_float", path=str(path),
                         sha256=source["sha256"], shape=list(array.shape), dtype=str(array.dtype),
                         registration=identity(legacy_index_path))
        else:
            entry["mode"] = "render_missing_native_float"
        entries.append(entry)
    reused = [e for e in entries if e["mode"].startswith("reuse")]
    missing = [e for e in entries if e["mode"].startswith("render")]
    if len(reused) != 4 or {(e["camera"], e["frame"]) for e in missing} != {
            (c, f) for c in ["cam00", "cam01"] for f in [0, 118]}:
        raise ValueError("Fixed P1 recovery plan is not the expected four reused/four missing observations")
    roi = read(args.roi)
    if roi["manifest_sha256"] != manifest_sha:
        raise ValueError("Frozen old ROI manifest mismatch")
    dependencies = [Path(__file__), Path(cause.__file__), Path(matrix.__file__),
        Path(cause.__file__).with_name("depth_prior.py"),
        ROOT / "experiments/dynamic_sr_20260918/common.py",
        ROOT / "experiments/dynamic_sr_20260918/n3dv_data.py",
        ROOT / "experiments/dynamic_sr_detail_supervision_20260924/evaluate.py",
        ROOT / "experiments/dynamic_sr_same_observation_20260930/frequency.py",
        ROOT / "experiments/dynamic_sr_same_observation_20260930/visualize.py"]
    upstream = Path(os.environ.get("FOURDSR_UPSTREAM", "/home/cai_tianshun/Project/4dgs")).resolve()
    dependencies += [upstream / name for name in ["gaussian_renderer/__init__.py", "scene/gaussian_model.py",
        "scene/deformation.py", "scene/hexplane.py", "arguments/__init__.py", "utils/sh_utils.py",
        "utils/graphics_utils.py", "utils/general_utils.py"]]
    plan = dict(status="registered_fixed_P1_recovery", checkpoint=identity(checkpoint),
        checkpoint_receipt=identity(receipt_path), historical_complete=identity(complete_path),
        legacy_float_index=identity(legacy_index_path), legacy_integrity=identity(integrity_path),
        historical_frequency_endpoint=identity(LEGACY / "frequency/endpoints/r1_P1-from-start.json"),
        manifest=identity(args.manifest), roi=identity(args.roi), source_files=[identity(p) for p in dependencies],
        prior_missing_evidence=identity(args.base_cached), entries=entries, expected_rgb_renders=4,
        expected_parameter_updates=0, expected_backward_calls=0, expected_adam_steps=0,
        preserved_historical_missing=read(args.base_cached).get("unavailable", {}),
        historical_parameterization=receipt["metadata"], disclosure=DISCLOSURE, upstream=str(upstream),
        audited_gpu=False, completed_finite_float_caches=4)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    print(json.dumps(dict(status=plan["status"], reused=4, missing=4, checkpoint_byte_hash_verified=True)))


def load_plan(path):
    plan = read(path)
    if plan["status"] != "registered_fixed_P1_recovery":
        raise ValueError("Unregistered P1 plan")
    for entry in plan["source_files"]:
        checked(entry)
    for key in ["manifest", "roi", "checkpoint_receipt", "historical_complete", "legacy_float_index", "legacy_integrity", "prior_missing_evidence"]:
        checked(plan[key])
    return plan


def render(args):
    plan = load_plan(args.plan)
    checkpoint = checked(plan["checkpoint"])
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Physical GPU must be bound by the external resource controller")
    sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_20260918"))
    from common import load_checkpoint, render_image, named_parameters, UPSTREAM
    if UPSTREAM.resolve() != Path(plan["upstream"]):
        raise ValueError("The actual upstream checkout differs from the registered P1 renderer")
    from n3dv_data import load_manifest
    ev = cause.module("p1_cause_frozen_evaluator", ROOT / "experiments/dynamic_sr_detail_supervision_20260924/evaluate.py")
    manifest = load_manifest(checked(plan["manifest"]))
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    h, w = reversed(manifest["resolutions"]["hr"])
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        g, _, _, checkpoint_payload = load_checkpoint(checkpoint)
        if "motion_refinement" in checkpoint_payload:
            raise ValueError("P1 must use the original native parameterization")
        g._deformation.eval()
        parameters = named_parameters(g)
        versions = {name: p._version for name, p in parameters.items()}
        entries, render_calls = [], 0
        with torch.no_grad():
            for entry in plan["entries"]:
                camera, frame = entry["camera"], entry["frame"]
                if entry["mode"].startswith("reuse"):
                    path = checked(entry)
                    float_array(path, (3, h, w))
                else:
                    raw = render_image(g, ev.render_camera(manifest, observations[camera, frame], frame))["render"]
                    path = args.out / f"{camera}_{frame:04d}.npz"
                    np.savez_compressed(path, rgb_raw=raw.detach().cpu().numpy())
                    float_array(path, (3, h, w))
                    render_calls += 1
                entries.append(dict(camera=camera, frame=frame, **identity(path), mode=entry["mode"],
                                    checkpoint=plan["checkpoint"], manifest_sha256=plan["manifest"]["sha256"]))
        if render_calls != 4 or versions != {name: p._version for name, p in parameters.items()}:
            raise ValueError("Read-only fixed rendering contract violated")
        import diff_gaussian_rasterization._C as extension
        write_json(args.out / "float_index.json", dict(status="completed_fixed_P1_dev8_float_cache", entries=entries,
            plan=identity(args.plan), rgb_renders=render_calls, reused_float_arrays=4, parameter_updates=0,
            backward_calls=0, adam_steps=0, parameter_versions_unchanged=True, source_sha256=sha256(__file__),
            gpu=torch.cuda.get_device_name(), visible_cuda=os.environ["CUDA_VISIBLE_DEVICES"],
            torch=str(torch.__version__), cuda=torch.version.cuda, extension=identity(extension.__file__),
            seconds=time.monotonic() - started, disclosure=DISCLOSURE))
        print(json.dumps(dict(status="completed_fixed_P1_dev8_float_cache", rgb_renders=4, reused=4, updates=0)))
    except BaseException:
        write_json(args.out / "failed.json", dict(status="failed", traceback=traceback.format_exc(), parameter_updates=0))
        raise


def p1_report(plan, index):
    manifest = read(checked(plan["manifest"]))
    manifest["_path"] = Path(plan["manifest"]["path"])
    regions = read(checked(plan["roi"]))["regions_by_camera_xyxy_exclusive"]
    observations = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    h, w = reversed(manifest["resolutions"]["lr"])
    hh, ww = reversed(manifest["resolutions"]["hr"])
    entries = {(e["camera"], int(e["frame"])): e for e in index["entries"]}
    if set(entries) != set(FIXED) or len(index["entries"]) != 8 or index["parameter_updates"] != 0:
        raise ValueError("P1 diagnostic cache must exactly cover fixed dev8 with zero updates")
    rows, oracles, provenance = [], [], []
    for camera in ["cam00", "cam01"]:
        predictions, truths = {}, {}
        mask = cause.roi_mask(regions, camera, (h, w), (hh, ww))
        for frame in cause.FRAMES:
            entry = entries[camera, frame]
            if entry["checkpoint"]["sha256"] != plan["checkpoint"]["sha256"] or entry["manifest_sha256"] != plan["manifest"]["sha256"]:
                raise ValueError("P1 float-cache identity mismatch")
            raw = float_array(checked(entry), (3, hh, ww))
            prediction = cause.degrade_chw(raw, (h, w)).permute(1, 2, 0).numpy()
            lr = cause.image(manifest, observations[camera, frame], "lr")
            hr = cause.image(manifest, observations[camera, frame], "hr")
            quality, fft = cause.metric_lr(prediction, lr), cause.frequency(raw, hr)
            regional = cause.metric_lr(prediction, lr, mask)
            rows.append(dict(model="P1", camera=camera, frame=frame, status="completed", lr_mse=quality["mse"],
                lr_l1=quality["l1"], hr_low_mse=fft["low_mse"], hr_mse=fft["mse"], region_lr_mse=regional["mse"]))
            predictions[frame], truths[frame] = prediction, lr
            provenance.append(entry)
        smooth = {f: cv2.blur(truths[f], (3, 3)) for f in cause.FIT_FRAMES}
        stable = (np.abs(smooth[0] - smooth[40]).mean(-1) <= 2 / 255.) & mask
        stable[:3] = stable[-3:] = False
        stable[:, :3] = stable[:, -3:] = False
        coefficients = cause.fit_gain_bias([predictions[f] for f in cause.FIT_FRAMES], [truths[f] for f in cause.FIT_FRAMES], stable)
        translation = cause.fit_shift([predictions[f] for f in cause.FIT_FRAMES], [truths[f] for f in cause.FIT_FRAMES], stable)
        oracle = dict(model="P1", camera=camera, fit_frames=cause.FIT_FRAMES, validation_frames=cause.VALIDATION_FRAMES,
            region_pixels=int(mask.sum()), fitting_static_proxy_pixels=int(stable.sum()), gain_bias=coefficients,
            translation=translation, evaluations=[], legal_main_result=False, limitation=DISCLOSURE)
        for frame in cause.FRAMES:
            variants = dict(original=predictions[frame])
            if coefficients is not None:
                variants["gain_bias"] = cause.apply_gain_bias(predictions[frame], coefficients)
            if translation is not None:
                variants["translation"] = cause.shift_image(predictions[frame], (translation["dx_lr"], translation["dy_lr"]))
            for variant, prediction in variants.items():
                oracle["evaluations"].append(dict(frame=frame, split="fit" if frame in cause.FIT_FRAMES else "validation",
                    variant=variant, full=cause.metric_lr(prediction, truths[frame]),
                    fixed_region=cause.metric_lr(prediction, truths[frame], mask),
                    fitting_static_proxy=cause.metric_lr(prediction, truths[frame], stable)))
        oracles.append(oracle)
    return dict(status="completed_fixed_P1_diagnostic_supplement", rows=rows, oracles=oracles,
                source_artifacts=provenance, parameter_updates=0, disclosure=DISCLOSURE)


def finish(args):
    plan = load_plan(args.plan)
    index = read(args.float_index)
    if index.get("plan", {}).get("sha256") != sha256(args.plan):
        raise ValueError("P1 cache belongs to another registration")
    base_path = checked(plan["prior_missing_evidence"])
    base = read(base_path)
    supplement = p1_report(plan, index)
    if any(row["model"] == "P1" for row in base["rows"]):
        raise ValueError("P1 already merged")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "p1_diagnostics.json", supplement)
    merged = dict(base)
    merged["rows"] = base["rows"] + supplement["rows"]
    merged["oracles"] = base["oracles"] + supplement["oracles"]
    merged["source_artifacts"] = base["source_artifacts"] + supplement["source_artifacts"]
    merged["historical_unavailable_preserved"] = base.get("unavailable", {})
    merged["unavailable"] = {k: v for k, v in base.get("unavailable", {}).items() if k != "P1"}
    merged["P1_recovery"] = dict(status="resolved_fixed_dev8_only", base_report=identity(base_path),
        plan=identity(args.plan), float_index=identity(args.float_index), disclosure=DISCLOSURE)
    merged_path = args.out / "cached_diagnostics.json"
    write_json(merged_path, merged)
    cause.csv_write(args.out / "cached_quality.csv", merged["rows"])
    files = [str(p) for p in [merged_path, args.render_report, *args.probes] if p is not None and p.exists()]
    probe_reports = [read(p) for p in args.probes]
    rendered = read(args.render_report) if args.render_report else None
    rows, probes = matrix.build(merged, rendered, probe_reports, files)
    with (args.out / "cause_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(args.out / "cause_matrix.json", dict(status="completed_with_explicit_unknowns", rows=rows,
        statuses=matrix.STATUSES, probe_summary=probes, sources=[identity(p) for p in files],
        source_sha256=sha256(__file__), matrix_implementation=identity(matrix.__file__),
        historical_P1=supplement, preserved_old_missing_report=identity(base_path),
        limitation="Historical P1 recovery does not change intervention criteria or establish a fair six-arm comparison. " + DISCLOSURE))
    write_json(args.out / "complete.json", dict(status="completed_P1_recovery_and_matrix_supplement", parameter_updates=0,
        plan=identity(args.plan), float_index=identity(args.float_index), source_sha256=sha256(__file__),
        cached_report=identity(merged_path), matrix=identity(args.out / "cause_matrix.json"), disclosure=DISCLOSURE))
    print(json.dumps(dict(status="completed_P1_recovery_and_matrix_supplement", P1_observations=8, updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "render", "finish"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--roi", type=Path, default=cause.DEFAULT_ROI)
    parser.add_argument("--base-cached", type=Path, default=OUT / "cause_cached_v1/cached_diagnostics.json")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--float-index", type=Path)
    parser.add_argument("--render-report", type=Path)
    parser.add_argument("--probes", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action in ["render", "finish"] and args.plan is None:
        parser.error("--plan is required")
    if args.action == "finish" and args.float_index is None:
        parser.error("--float-index is required")
    torch.set_num_threads(4); cv2.setNumThreads(4)
    {"audit": audit, "render": render, "finish": finish}[args.action](args)


if __name__ == "__main__":
    main()
