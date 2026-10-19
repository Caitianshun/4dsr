"""CPU-only, hash-checked Cut07 frames0..99 backbone screening.

No renderer, model, torch or GPU is imported. LPIPS values are read, never
recomputed. The original files and their remote paths are not rewritten.

--stg-root uses {areaL0,areaTm}/f000_049 and f050_099/eval/summary.json.
--multi4d-root is the explicit eval directory containing
{areaL0,areaTm}/attempt_01/summary.json; no latest-attempt selection is made.
--wu-evaluations names a JSON object with areaTm and complete summary entries
and legacy_csv entries for areaL0, areaTm, controlled_areaTm and complete.
Each entry is {"path": "/absolute/path", "sha256": "..."}. A path alone is
accepted for pending evaluations; missing inputs keep the result incomplete.
Legacy CSV entries also need summary={path,sha256} for the original full300
summary, linking the CSV to the input/checkpoint and verifying its full mean.
Wu summaries use model="Wu", method=areaTm/complete, frames=100,
frame_indices=0..99, status/checkpoint_training_status="completed",
manifest/checkpoint/per_observation/temporal_per_pair identities,
lpips_status="alex_v0.1", the shared metric_sources, roi_protocol and
mask_receipts.cam00.mask_pixel_sha256. Rows follow the existing evaluator
format. Wu areaL0 regional/temporal evaluation is deliberately not required.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ("areaL0", "areaTm")
WU_ARMS = ("areaTm", "complete")
LEGACY_ARMS = ("areaL0", "areaTm", "controlled_areaTm", "complete")
SPATIAL = ("psnr", "ssim", "lpips", "mse")
REGIONAL = ("psnr", "ssim", "mse", "lpips_alex_spatial_mask")
ROI_METRICS = ("psnr", "ssim", "mse")
TEMPORAL = ("frame_difference_residual_l1", "frame_difference_residual_mse")
SHARED_SOURCES = (
    "experiments/dynamic_sr_20260918/evaluate.py",
    "experiments/dynamic_sr_same_observation_20260930/frequency.py",
    "experiments/dynamic_sr_multi4d_area_20261017/evaluate.py",
)
WINDOWS = (("f000_049", 0, 49), ("f050_099", 50, 99), ("f000_099", 0, 99))


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def means(rows, keys):
    require(bool(rows), "Empty metric range")
    require(all(finite(row.get(key)) for row in rows for key in keys), "Missing/nonfinite metric")
    return {key: statistics.fmean(row[key] for row in rows) for key in keys}


class Files:
    def __init__(self, mappings=()):
        self.mappings = sorted(mappings, key=lambda pair: len(pair[0].parts), reverse=True)
        self.hashes = {}

    def resolve(self, path):
        path = Path(path)
        if not path.is_absolute():
            # Original Wu receipts use project-relative paths. Resolve them
            # without changing provenance and reject parent traversal.
            require(".." not in path.parts, "Relative artifact escapes project")
            path = Path(__file__).resolve().parents[2] / path
        for old, new in self.mappings:
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path

    def sha(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            self.hashes[key] = digest.hexdigest()
        return self.hashes[key]

    def identity(self, path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": self.sha(path)}

    def checked(self, entry, *, hash_required=True):
        entry = {"path": entry} if isinstance(entry, str) else entry
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "Invalid file identity")
        path = self.resolve(entry["path"])
        require(path.is_file(), "Missing artifact: " + str(path))
        if hash_required:
            require(isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
                    "Missing SHA256: " + str(path))
        if entry.get("sha256"):
            require(self.sha(path) == entry["sha256"], "Artifact SHA256 differs: " + str(path))
        return path


def source_key(path):
    matches = [key for key in SHARED_SOURCES if str(path).endswith("/" + key)]
    return matches[0] if len(matches) == 1 else None


def metric_identity(summary, directory, manifest, files):
    """Normalize shared code, fixed ROI and full300 mask identities."""
    sources = {}
    for entry in summary["metric_sources"]:
        key = source_key(entry["path"])
        if key:
            require(key not in sources, "Duplicate shared metric source")
            files.checked(entry)
            sources[key] = entry["sha256"]
    require(set(sources) == set(SHARED_SOURCES), "Missing shared spatial/region metric source")
    roi = summary["roi_protocol"]
    require(roi.get("status") == "fixed", "ROI protocol is not fixed")
    files.checked(roi["source"])
    require(roi.get("current_manifest_sha256") == manifest["sha256"], "ROI manifest differs")
    definition = roi["definition"]
    boxes = definition["regions_by_camera_xyxy_exclusive"]["cam00"]
    require(bool(boxes) and "face_reference" in boxes, "Fixed face ROI missing")
    mask_hash = summary.get("mask_receipts", {}).get("cam00", {}).get("mask_pixel_sha256")
    png = directory / "previews/cam00_fixed_dynamic_mask.png"
    if png.is_file():
        import numpy as np
        from PIL import Image
        with Image.open(png) as image:
            mask = np.asarray(image.convert("L")) > 0
        actual = hashlib.sha256(mask.tobytes()).hexdigest()
        require(mask_hash is None or mask_hash == actual, "Declared/actual mask pixels differ")
        mask_hash = actual
    require(isinstance(mask_hash, str) and len(mask_hash) == 64, "Full300 fixed mask identity missing")
    return {"shared_metric_sources": sources, "mask_pixel_sha256": mask_hash,
            "roi_source_sha256": roi["source"]["sha256"], "cam00_roi_boxes": boxes,
            "roi_reference_images": [(x["camera"], x["frame"], x["sha256"])
                                     for x in definition["reference_images"]]}


def rows_and_pairs(summary, directory, expected_frames, files, *, legacy_multi4d=False):
    identities = {}
    paths = {}
    for key, name in (("per_observation", "per_observation.jsonl"),
                      ("temporal_per_pair", "temporal_per_pair.json")):
        if key in summary:
            paths[key] = files.checked(summary[key])
        else:
            require(legacy_multi4d, "Summary lacks artifact identity: " + key)
            paths[key] = directory / name
            require(paths[key].is_file(), "Missing Multi4D artifact: " + str(paths[key]))
        identities[key] = {**files.identity(paths[key]),
                           "hash_previously_declared": key in summary}
    rows = [json.loads(line) for line in paths["per_observation"].read_text().splitlines() if line]
    tests = [row for row in rows if row.get("scope") == "test"]
    require([(row.get("camera"), row.get("frame")) for row in tests] ==
            [("cam00", frame) for frame in expected_frames], "Duplicate/missing/out-of-range test frame")
    means(tests, SPATIAL)
    for key in SPATIAL:
        require(abs(summary["test"][key] - statistics.fmean(row[key] for row in tests)) < 1e-10,
                "Summary does not reconstruct from declared frame rows: " + key)
    pairs = read(paths["temporal_per_pair"])
    require([(row.get("previous_frame"), row.get("frame")) for row in pairs] ==
            list(zip(expected_frames[:-1], expected_frames[1:])), "Missing/duplicate temporal pair")
    means(pairs, TEMPORAL)
    return tests, pairs, identities


def load_evaluation(entry, model, method, expected_frames, manifest, files):
    path = files.checked(entry, hash_required=False)
    summary = read(path)
    allowed = {"STG": ("STG",), "Multi4D": ("Multi4D",), "Wu": ("Wu", "Wu4DGaussians")}[model]
    require(summary.get("status") == "completed" and summary.get("model") in allowed
            and summary.get("method") == method, "Incomplete or mismatched model/method")
    require(summary.get("checkpoint_training_status") == "completed", "Checkpoint training is incomplete")
    require(summary.get("lpips_status") == "alex_v0.1", "Full AlexNet v0.1 LPIPS required")
    require(summary.get("frames") == len(expected_frames), "Evaluation frame range differs")
    if model != "Multi4D":
        require(summary.get("frame_indices") == expected_frames, "Declared frame identities differ")
    files.checked(summary["manifest"])
    require(summary["manifest"]["sha256"] == manifest["sha256"], "Input manifest identity differs")
    checkpoint = files.checked(summary["checkpoint"])
    identity = metric_identity(summary, path.parent, manifest, files)
    tests, pairs, artifacts = rows_and_pairs(summary, path.parent, expected_frames, files,
                                             legacy_multi4d=model == "Multi4D")
    boxes = identity["cam00_roi_boxes"]
    for row in tests:
        for region in ("dynamic", "static"):
            means([row["regions"][region]], REGIONAL)
        require(set(row["fixed_rois"]) == set(boxes), "Fixed ROI names differ")
        for name in boxes:
            means([row["fixed_rois"][name]], ROI_METRICS)
    record = {"summary": files.identity(path), "checkpoint": files.identity(checkpoint),
              "original_checkpoint_identity": summary["checkpoint"], "artifacts": artifacts,
              "metric_identity": identity, "gpu": summary.get("gpu"),
              "posthoc_cull": summary.get("posthoc_cull"), "frames_evaluated": len(expected_frames)}
    if model == "STG":
        start = expected_frames[0]
        require(summary.get("segment_start") == start and summary.get("duration") == 50,
                "STG segment bounds differ")
        decoder = files.checked(summary["decoder"])
        receipt_path = files.checked(summary["training_receipt"])
        receipt = read(receipt_path)
        require(receipt.get("status") == "completed" and receipt.get("method") == method
                and receipt.get("segment_start") == start and receipt.get("duration") == 50
                and receipt.get("iterations") == 25000 and receipt.get("optimizer_updates") == 24999,
                "Invalid STG training completion receipt")
        require(receipt["checkpoint"]["sha256"] == summary["checkpoint"]["sha256"]
                and receipt["decoder"]["sha256"] == summary["decoder"]["sha256"]
                and receipt["manifest"]["sha256"] == manifest["sha256"], "STG receipt identity differs")
        sampling_path = receipt_path.parent / "sampling.jsonl"
        require(files.sha(sampling_path) == receipt["sampling_sha256"], "Sampling SHA differs")
        sampling = [json.loads(line) for line in sampling_path.read_text().splitlines() if line]
        require(len(sampling) == 25000, "STG sampling batch count differs")
        for iteration, row in enumerate(sampling, 1):
            observations = row["camera_frames"]
            require(row["iteration"] == iteration and len(observations) == 2
                    and observations[0][0] != observations[1][0]
                    and observations[0][1] == observations[1][1]
                    and all(camera in summary["training_cameras"] and camera != "cam00"
                            and frame in expected_frames for camera, frame in observations),
                    "Unpaired or held-out-camera STG sampling")
        record.update(decoder=files.identity(decoder), training_receipt=files.identity(receipt_path),
                      sampling=files.identity(sampling_path), capacity_points=summary.get("capacity_points"),
                      training_seconds=receipt.get("training_seconds"),
                      peak_allocated_bytes=receipt.get("peak_allocated_bytes"))
    return {"rows": tests, "pairs": pairs, "summary": summary, "record": record, "identity": identity}


def boundary_pair(left, right, manifest_path, manifest, files):
    """Exact compare.py definition: clipped float RGB frame-difference residual."""
    import numpy as np
    from PIL import Image
    lookup = {(row["camera_id"], row["frame_index"]): row for row in manifest["observations"]}
    predictions, references = [], []
    for evaluation, frame in ((left, 49), (right, 50)):
        endpoint = evaluation["summary"]["endpoints"][str(frame)]
        row = lookup["cam00", frame]
        require(endpoint.get("frame") == frame and endpoint.get("camera") == "cam00"
                and endpoint.get("reference_hr_sha256") == row["hr_sha256"], "Boundary endpoint identity differs")
        with np.load(files.checked(endpoint), allow_pickle=False) as saved:
            raw = saved["rgb_raw"]
            require(raw.shape == (3, *manifest["resolutions"]["hr"][::-1])
                    and np.isfinite(raw).all(), "Invalid endpoint shape/finite value")
            predictions.append(np.clip(raw, 0., 1.).transpose(1, 2, 0))
        hr = (manifest_path.parent / row["hr_path"]).resolve()
        require(hr.is_relative_to(manifest_path.parent.resolve()) and files.sha(hr) == row["hr_sha256"],
                "Boundary HR reference SHA differs")
        with Image.open(hr) as image:
            references.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.)
    residual = (predictions[1] - predictions[0]) - (references[1] - references[0])
    return {"previous_frame": 49, "frame": 50, "boundary": True,
            "frame_difference_residual_l1": float(np.abs(residual).mean()),
            "frame_difference_residual_mse": float((residual ** 2).mean())}


def aggregate_window(rows, pairs, start, end):
    selected = [row for row in rows if start <= row["frame"] <= end]
    require([row["frame"] for row in selected] == list(range(start, end + 1)), "Window frame coverage differs")
    selected_pairs = [row for row in pairs if start <= row["previous_frame"] < row["frame"] <= end]
    require([(row["previous_frame"], row["frame"]) for row in selected_pairs] ==
            [(frame, frame + 1) for frame in range(start, end)], "Window lacks required temporal pair")
    boundary = [row for row in selected_pairs if (row["previous_frame"], row["frame"]) == (49, 50)]
    roi_names = sorted(selected[0]["fixed_rois"])
    return {"frame_range": [start, end], "frames": len(selected), "test": means(selected, SPATIAL),
            "dynamic_static": {name: means([row["regions"][name] for row in selected], REGIONAL)
                               for name in ("dynamic", "static")},
            "fixed_rois": {name: means([row["fixed_rois"][name] for row in selected], ROI_METRICS)
                           for name in roi_names},
            "temporal": {"pairs": len(selected_pairs), **means(selected_pairs, TEMPORAL),
                         "49_to_50": boundary[0] if boundary else None}}


def windows(evaluations, manifest_path, manifest, files, *, segmented=False):
    rows = [row for evaluation in evaluations for row in evaluation["rows"]]
    pairs = [row for evaluation in evaluations for row in evaluation["pairs"]]
    if segmented:
        pairs.append(boundary_pair(evaluations[0], evaluations[1], manifest_path, manifest, files))
    rows.sort(key=lambda row: row["frame"])
    pairs.sort(key=lambda row: row["frame"])
    return {name: aggregate_window(rows, pairs, start, end) for name, start, end in WINDOWS}


def legacy_csv(entry, files, manifest):
    path = files.checked(entry)
    with path.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("scope") == "test"]
    require([(row["camera"], int(row["frame"])) for row in rows] ==
            [("cam00", frame) for frame in range(300)], "Legacy Wu CSV lacks exact full300 test identities")
    keys = ("psnr", "ssim", "lpips")
    result = {}
    for name, start, end in WINDOWS:
        selected = [row for row in rows if start <= int(row["frame"]) <= end]
        converted = [{key: float(row[key]) for key in keys} for row in selected]
        result[name] = {"frames": len(selected), "test": means(converted, keys),
                        "dynamic_static": None, "fixed_rois": None, "temporal": None,
                        "unavailable": "Original CSV has whole-image metrics only"}
    return {"status": "completed", "source": files.identity(path), "windows": result,
            "input_manifest_sha256": manifest["sha256"],
            "manifest_verification": "CSV alone has no manifest identity; linked summary is checked separately"}


def difference(left, right):
    result = {"test": {key: left["test"][key] - right["test"][key]
                       for key in ("psnr", "ssim", "lpips")}}
    for group, keys in (("dynamic_static", REGIONAL), ("fixed_rois", ROI_METRICS)):
        if left.get(group) is None or right.get(group) is None:
            result[group] = None
        else:
            require(set(left[group]) == set(right[group]), "Compared ROI/region identities differ")
            result[group] = {name: {key: left[group][name][key] - right[group][name][key] for key in keys}
                             for name in left[group]}
    result["temporal"] = ({key: left["temporal"][key] - right["temporal"][key] for key in TEMPORAL}
                          if left.get("temporal") and right.get("temporal") else None)
    if result["temporal"] is not None:
        a, b = left["temporal"].get("49_to_50"), right["temporal"].get("49_to_50")
        result["temporal"]["49_to_50"] = ({key: a[key] - b[key] for key in TEMPORAL}
                                              if a and b else None)
    return result


def screen(manifest_path, stg_root, multi4d_root, wu_mapping, files):
    manifest_path = files.resolve(str(Path(manifest_path).absolute()))
    manifest = read(manifest_path)
    manifest_identity = files.identity(manifest_path)
    require(manifest.get("scene") == "cut_roasted_beef", "Only Cut07 screening is supported")
    require(manifest.get("splits", {}).get("test") == ["cam00"], "Requires held-out cam00 manifest")
    require([(row["camera_id"], row["frame_index"]) for row in manifest["observations"]
             if row.get("split") == "test"] == [("cam00", frame) for frame in range(300)],
            "Input manifest lacks complete cam00 reference identities")
    result = {"schema": "stg_backbone_screening_v1", "status": "incomplete", "full300_complete": False,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Cut07 cam00 frames0..99; completed_screening never means full300 benchmark",
              "manifest": manifest_identity, "models": {}, "legacy_Wu": {}, "problems": [],
              "comparisons": None, "aggregation": "Arithmetic mean of per-frame metrics in each exact window; temporal means over every consecutive pair",
              "limits": ["Two independent50-frame STG models versus one300-frame Wu/Multi4D model.",
                         "Shared metrics do not equalize capacity, LR initialization, optimization, posthoc cull or hardware.",
                         "Fixed face/text ROIs do not track the object; blinded acceptance is not performed.",
                         "Wu areaL0 regional/temporal results are unavailable and not required by this screening.",
                         "Oracle is excluded; this report cannot authorize training or Cook expansion."]}
    identities = []
    loaded = {}
    for model, methods in (("STG", ARMS), ("Multi4D", ARMS), ("Wu", WU_ARMS)):
        for method in methods:
            key = model + "/" + method
            try:
                if model == "STG":
                    evaluations = [load_evaluation(str(Path(stg_root).absolute() / method / name / "eval/summary.json"),
                                                  model, method, list(range(start, end + 1)), manifest_identity, files)
                                   for name, start, end in WINDOWS[:2]]
                else:
                    entry = (str(Path(multi4d_root).absolute() / method / "attempt_01/summary.json")
                             if model == "Multi4D" else wu_mapping[method])
                    evaluations = [load_evaluation(entry, model, method,
                                                  list(range(300 if model == "Multi4D" else 100)),
                                                  manifest_identity, files)]
                require(all(e["identity"] == evaluations[0]["identity"] for e in evaluations),
                        "Metric identity differs across segments")
                metrics = windows(evaluations, manifest_path, manifest, files, segmented=model == "STG")
                result["models"][key] = {"status": "completed", "windows": metrics,
                                           "sources": [e["record"] for e in evaluations]}
                loaded[key] = evaluations
                identities.append((key, evaluations[0]["identity"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["models"][key] = {"status": "incomplete", "windows": None}
                result["problems"].append({"model": key, "reason": str(error)})
    if identities:
        for key, identity in identities[1:]:
            if identity != identities[0][1]:
                result["problems"].append({"model": key, "reason": "Shared metric/mask/ROI identity differs across models"})
    for start in (0, 50):
        if all("STG/" + method in loaded for method in ARMS):
            hashes = [loaded["STG/" + method][start // 50]["record"]["sampling"]["sha256"] for method in ARMS]
            if len(set(hashes)) != 1:
                result["problems"].append({"segment_start": start, "reason": "STG L0/Tm sampling differs"})
    for method in LEGACY_ARMS:
        try:
            entry = wu_mapping["legacy_csv"][method]
            # Require a hash-bound original summary for CSV/input/checkpoint provenance.
            require(isinstance(entry, dict) and "summary" in entry, "Legacy CSV requires linked summary identity")
            summary_path = files.checked(entry["summary"])
            summary = read(summary_path)
            require(summary.get("status") == "completed"
                    and summary["manifest"]["sha256"] == manifest_identity["sha256"], "Legacy CSV summary manifest/status differs")
            files.checked(summary["manifest"])
            files.checked(summary["checkpoint"])
            item = legacy_csv(entry, files, manifest_identity)
            with files.checked(entry).open() as stream:
                rows = [row for row in csv.DictReader(stream) if row.get("scope") == "test"]
            for metric in ("psnr", "ssim", "lpips"):
                average = statistics.fmean(float(row[metric]) for row in rows)
                require(abs(average - summary["test"][metric]) < 1e-10, "Legacy CSV full300 mean differs from summary")
            item["summary"] = files.identity(summary_path)
            item["manifest_verification"] = "Linked summary manifest/checkpoint SHA and full300 CSV means verified"
            result["legacy_Wu"][method] = item
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["legacy_Wu"][method] = {"status": "incomplete", "windows": None}
            result["problems"].append({"legacy_Wu": method, "reason": str(error)})
    if not result["problems"]:
        comparisons = {}
        for window, _, _ in WINDOWS:
            current = {key: model["windows"][window] for key, model in result["models"].items()}
            comparisons[window] = {
                "STG_teacher_gain": difference(current["STG/areaTm"], current["STG/areaL0"]),
                "Multi4D_teacher_gain": difference(current["Multi4D/areaTm"], current["Multi4D/areaL0"]),
                "STG_L0_minus_Multi4D_L0": difference(current["STG/areaL0"], current["Multi4D/areaL0"]),
                "STG_Tm_minus_Multi4D_Tm": difference(current["STG/areaTm"], current["Multi4D/areaTm"]),
                "STG_Tm_minus_Wu_Tm": difference(current["STG/areaTm"], current["Wu/areaTm"]),
                "STG_Tm_minus_Wu_complete": difference(current["STG/areaTm"], current["Wu/complete"]),
                "STG_L0_minus_Wu_legacy_L0": difference(current["STG/areaL0"], result["legacy_Wu"]["areaL0"]["windows"][window]),
                "Wu_new_Tm_minus_legacy_Tm": difference(current["Wu/areaTm"], result["legacy_Wu"]["areaTm"]["windows"][window]),
                "Wu_new_complete_minus_legacy_complete": difference(current["Wu/complete"], result["legacy_Wu"]["complete"]["windows"][window]),
            }
        result.update(status="completed_screening", comparisons=comparisons)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stg-root", type=Path, required=True)
    parser.add_argument("--multi4d-root", type=Path, required=True)
    parser.add_argument("--wu-evaluations", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mappings = []
    for value in args.path_map:
        old, separator, new = value.partition("=")
        if not separator or not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error("--path-map requires absolute OLD=NEW prefixes")
        mappings.append((Path(old), Path(new)))
    files = Files(mappings)
    try:
        result = screen(args.manifest, args.stg_root, args.multi4d_root, read(args.wu_evaluations), files)
        result["wu_evaluations_mapping"] = files.identity(args.wu_evaluations)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"schema": "stg_backbone_screening_v1", "status": "incomplete", "full300_complete": False,
                  "comparisons": None, "problems": [{"reason": str(error)}]}
    result["source"] = files.identity(Path(__file__))
    result["path_mapping"] = [{"source_prefix": str(old), "local_prefix": str(new)} for old, new in mappings]
    require(not args.out.exists(), "Preserve previous screening snapshots; choose a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.out), "problems": result["problems"]}))


if __name__ == "__main__":
    main()
