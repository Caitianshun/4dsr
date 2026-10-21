"""CPU-only fixed-frame image diagnostics for returned CEM area4 runs.

No model/checkpoint is opened, no teacher is inferred, no CUDA/SSH is used.
Raw reconstruction panels and training-camera corrected panels are separate.
Three-frame crop scores and optional GT-relative temporal differences are
local diagnostics, never substitutes for the common 300-frame main evaluator.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "output/dynamic_sr_cem_backbone_20261020"
SCENES = ("cook_spinach", "cut_roasted_beef")
ARMS = ("areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle")
WU_ARMS = dict(zip(ARMS, ("areaL0", "naive", "areaT", "areaTm", "w4", "oracle")))
FRAMES = [0, 150, 299]
LEGACY = ROOT / "experiments/dynamic_sr_20260918/evaluate.py"
FONT_CANDIDATES = (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                   Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def child(root, relative):
    path = (Path(root) / relative).resolve()
    assert path.is_relative_to(Path(root).resolve()), "Source path escapes declared image root"
    return path


def load_protocol(path):
    protocol = read_json(path)
    assert protocol["status"] == "fixed_before_new_CEM_results"
    assert protocol["frames"] == FRAMES and protocol["full_frame"]
    assert protocol["cameras"] == dict(cam00="heldout", cam01="training_diagnostic")
    references = {(row["scene"], row["camera"], row["frame"]): row for row in protocol["references"]}
    assert len(references) == 12
    for scene in SCENES:
        for camera in ["cam00", "cam01"]:
            roi = protocol["face_xyxy"][scene][camera]
            assert len(roi) == 4 and all(type(x) is int for x in roi)
            assert 0 <= roi[0] < roi[2] <= 1344 and 0 <= roi[1] < roi[3] <= 1008
            assert min(roi[2] - roi[0], roi[3] - roi[1]) > 10
    return protocol, references


def cpu_metrics():
    import cv2
    nodes = [node for node in ast.parse(LEGACY.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {"ssim_map_rgb", "metric_psnr"}]
    assert {node.name for node in nodes} == {"ssim_map_rgb", "metric_psnr"}
    namespace = dict(np=np, cv2=cv2)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LEGACY), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in ["ssim_map_rgb", "metric_psnr"]})


def score(prediction, gt, roi, metric):
    values = {}
    for region, box in [("whole", None), ("face_crop", roi)]:
        p, g = prediction, gt
        if box:
            x0, y0, x1, y1 = box
            p, g = p[y0:y1, x0:x1], g[y0:y1, x0:x1]
        p, g = p.astype(np.float32) / 255, g.astype(np.float32) / 255
        mse = float(np.mean((p - g) ** 2, dtype=np.float64))
        values[region] = dict(psnr=metric.metric_psnr(mse), mse=mse,
                              ssim=float(metric.ssim_map_rgb(p, g)[5:-5, 5:-5].mean(dtype=np.float64)))
    return values


class Assets:
    def __init__(self):
        self.records, self.signatures = {}, {}

    def verify(self, item):
        path = Path(item["path"]).resolve()
        assert path.is_file(), "Missing image: " + str(path)
        before = path.stat()
        signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        key = str(path)
        if key in self.records:
            assert self.signatures[key] == signature, "Previously verified source changed during diagnostics"
            assert self.records[key]["sha256"] == item["sha256"]
        else:
            actual = sha(path)
            after = path.stat()
            assert signature == (after.st_ino, after.st_size, after.st_mtime_ns), "Image changed while hashing"
            assert actual == item["sha256"], "Image SHA differs: " + str(path)
            self.records[key] = dict(path=key, sha256=actual, bytes=before.st_size)
            self.signatures[key] = signature
        return path

    def image(self, item):
        path = self.verify(item)
        with Image.open(path) as image:
            assert image.mode == "RGB"
            expected = (336, 252) if item.get("operation") == "LR_bicubic_visual_reference" else (1344, 1008)
            assert image.size == expected
            if item.get("operation") == "LR_bicubic_visual_reference":
                image = image.resize((1344, 1008), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.uint8).copy()
        self.verify(item)
        return array


def make_item(key, label, path, expected, role, **extra):
    return dict(key=key, label=label, path=str(Path(path).resolve()), sha256=expected, role=role, **extra)


def manifest_data(scene):
    path = ROOT / "data/dynamic_sr/full_time_20261007/prepared" / (scene + "_area4") / "manifest_train_ready_seed20261007.json"
    value = read_json(path)
    assert value["scene"] == scene and value["physical_lr_derivation"]["formation"] == "area4"
    assert value["frame_indices"] == list(range(300)) and value["splits"]["test"] == ["cam00"]
    observations = {(row["camera_id"], row["frame_index"]): row for row in value["observations"]}
    return path, value, observations


def references(scene, camera, frame, manifest_path, observations, fixed):
    observation = observations[camera, frame]
    reference = fixed[scene, camera, frame]
    hr = child(manifest_path.parent, observation["hr_path"])
    assert hr == (ROOT / reference["HR_path"]).resolve()
    assert observation["hr_sha256"] == reference["sha256"]
    items = [make_item("GT", "GT HR", hr, observation["hr_sha256"], "clean_HR_reference"),
             make_item("LR_bicubic", "LR x4 bicubic | reference", child(manifest_path.parent, observation["lr_path"]),
                       observation["lr_sha256"], "input_visual_reference_not_a_reconstruction_method",
                       operation="LR_bicubic_visual_reference", input_camera_scope="test_LR_reference_only" if camera == "cam00" else "training_LR")]
    if camera == "cam01":
        assert observation["split"] == "train"
        for suffix, key, label in [("sr_swinir_x4", "teacher_original", "Teacher original | train only"),
                                   ("sr_swinir_x4_areaft", "teacher_matched", "Teacher matched | train only")]:
            index_path = manifest_path.parent / f"teacher_index_{suffix}_seed20261007.json"
            index = read_json(index_path)
            assert index["manifest_sha256"] == sha(manifest_path) and not index["oracle"]
            entries = [row for row in index["entries"] if row["camera"] == camera and row["frame"] == frame]
            assert len(entries) == 1 and entries[0]["lr_sha256"] == observation["lr_sha256"]
            entry = entries[0]
            items.append(make_item(key, label, child(manifest_path.parent, entry["relative_path"]), entry["sha256"],
                                   "training_camera_teacher_reference_only", teacher_index=source_identity(index_path)))
    return items


def cem_endpoint(run, key, label, kind, scene, manifest_hash):
    """Read receipts and image identities only; never open a model/checkpoint."""
    run = Path(run).resolve()
    index_path = run / "result_index.json"
    index = read_json(index_path)
    if kind == "sr":
        assert index["status"] == "SR_endpoint_and_303_renders_unified_metrics_verified" and not index["fixture_only"]
        assert index["scene"] == scene and index["arm"] == key
        assert index["manifest"]["sha256"] == manifest_hash and index["actual_parameter_updates"] == 6000
        assert index["absolute_final_iteration"] == 56000 and index["fixed_topology"]
        checkpoint_sha = index["checkpoint"]["sha256"]
    else:
        assert index["status"] == "native_endpoints_and_300_renders_verified"
        assert index["manifest_sha256"] == manifest_hash
        assert index["train_level"] == ("hr" if kind == "hrref" else "lr")
        endpoint = [entry for entry in index["stages"] if Path(entry["path"]).name == "stage2.json"]
        assert len(endpoint) == 1
        state_path = run / "stage2.json"
        state = read_json(state_path)
        assert sha(state_path) == endpoint[0]["sha256"] and state["endpoint"] == 50000 and state["status"] == "completed"
        assert state["manifest"]["sha256"] == manifest_hash and not state["heldout_pixels_used"]
        checkpoint_sha = state["final_checkpoint"]["sha256"]
        assert checkpoint_sha == endpoint[0]["checkpoint_sha256"]
    records, receipt_ids = {}, [source_identity(index_path)]
    for camera, directory in [("cam00", "renders_hr"), ("cam01", "train_cam01")]:
        path = run / directory / "render_receipt.json"
        value = read_json(path)
        assert value["status"] == "completed" and value["camera"] == camera
        expected_frames = list(range(300)) if camera == "cam00" else FRAMES
        assert value["frame_indices"] == expected_frames and value["frames"] == len(expected_frames)
        assert value["finite_before_clip_quantization_passed_frames"] == len(expected_frames)
        assert value["manifest"]["sha256"] == manifest_hash and value["checkpoint"]["sha256"] == checkpoint_sha
        assert not value["heldout_pixels_read"] and not value["training_pixels_read"]
        if camera == "cam00":
            assert value["colour_affine"] == value["warp"] == "identity"
        assert [row["frame"] for row in value["records"]] == expected_frames
        for row in value["records"]:
            frame = row["frame"]
            assert row["path"] == f"{frame:04d}.png"
            raw = kind == "sr" and camera == "cam01"
            name, expected = (row["raw_path"], row["raw_sha256"]) if raw else (row["path"], row["sha256"])
            assert name == (f"{frame:04d}_raw.png" if raw else f"{frame:04d}.png")
            records[camera, frame, "raw"] = make_item(key, label + " | raw", child(run / directory, name), expected,
                "heldout_raw_reconstruction" if camera == "cam00" else "training_camera_raw_reconstruction", model_kind=kind)
            if raw:
                records[camera, frame, "corrected"] = make_item(key, label + " | corrected", child(run / directory, row["path"]), row["sha256"],
                    "training_camera_learned_affine_and_optional_w4_diagnostic", model_kind=kind)
        receipt_ids.append(source_identity(path))
    return dict(key=key, label=label, run=str(run), kind=kind, scene=scene, checkpoint_sha256=checkpoint_sha,
                receipt_identities=receipt_ids, records=records)


def wu_endpoint(folder, arm, scene, manifest_hash):
    folder = Path(folder).resolve()
    path = folder / "render_receipt.json"
    value = read_json(path)
    assert value["status"] == "completed" and value["frames"] == 300 and value["parameter_updates"] == 0
    assert value["task"]["scene"] == scene and value["task"]["arm"] == WU_ARMS[arm]
    assert value["task"]["manifest"]["sha256"] == manifest_hash
    policy = value["inference_policy"]
    assert all(policy[key] is False for key in ["posthoc_cull", "color_affine", "spatial_warp", "test_color_fit"])
    assert value["points_loaded"] == value["points_rendered"]
    assert [row["frame"] for row in value["PNGs"]] == list(range(300))
    records = {}
    for row in value["PNGs"]:
        frame = row["frame"]
        assert row["path"] == f"renders_hr/{frame:04d}.png"
        records["cam00", frame, "raw"] = make_item("Wu_" + arm, "Wu " + arm + " | raw", child(folder, row["path"]), row["sha256"],
            "historically_selected_Wu_development_reference_no_posthoc_cull")
    return dict(key="Wu_" + arm, label="Wu " + arm, scene=scene, kind="Wu_development_reference", run=str(folder),
                receipt_identities=[source_identity(path)], checkpoint_sha256=value["task"]["checkpoint"]["sha256"], records=records)


def missing_item(key, label, reason):
    return dict(key=key, label=label, missing=True, reason=reason)


def preflight(args, protocol, fixed):
    plans, missing, optional_missing = {}, [], []
    overrides = {}
    for mapping in args.hrref_run:
        scene, value = mapping.split("=", 1)
        assert scene in SCENES and scene not in overrides
        overrides[scene] = Path(value).resolve()
    for scene in args.scene:
        manifest_path, manifest, observations = manifest_data(scene)
        manifest_hash = sha(manifest_path)
        specs = [("native_parent", "CEM native LR", "parent", args.runs_root / f"{scene}_native_parent_v1")]
        specs += [(arm, "CEM " + arm, "sr", args.runs_root / f"{scene}_{arm}_v1") for arm in ARMS]
        specs += [("HRref", "CEM HRref | privileged", "hrref", overrides.get(scene, args.runs_root / f"{scene}_native_HRref_v2"))]
        endpoints = []
        for key, label, kind, run in specs:
            try:
                endpoint = cem_endpoint(run, key, label, kind, scene, manifest_hash)
                # Rehash all returned endpoint PNGs without decoding all300.
                # Default final gate cannot accept a partially returned run.
                for item in endpoint["records"].values():
                    args.assets.verify(item)
                # Decode the required fixed images before creating output.
                for camera in ["cam00", "cam01"]:
                    for frame in FRAMES:
                        item = endpoint["records"][camera, frame, "raw"]
                        args.assets.image(item)
                        if kind == "sr" and camera == "cam01":
                            args.assets.image(endpoint["records"][camera, frame, "corrected"])
                endpoints.append(endpoint)
            except (OSError, ValueError, KeyError, AssertionError) as error:
                reason = f"{type(error).__name__}: {error}"
                missing.append(dict(scene=scene, endpoint=key, run=str(Path(run).resolve()), reason=reason))
                endpoints.append(dict(key=key, label=label, kind=kind, run=str(Path(run).resolve()), missing=True, reason=reason))
        refs = {}
        for camera in ["cam00", "cam01"]:
            for frame in FRAMES:
                items = references(scene, camera, frame, manifest_path, observations, fixed)
                for item in items:
                    args.assets.image(item)
                refs[camera, frame] = items
        wu = []
        if args.wu != "off":
            label = "Cook07" if scene == "cook_spinach" else "Cut07"
            problems = []
            for arm in ARMS:
                folder = args.wu_root / label / WU_ARMS[arm]
                try:
                    endpoint = wu_endpoint(folder, arm, scene, manifest_hash)
                    for item in endpoint["records"].values():
                        args.assets.verify(item)
                    for frame in FRAMES:
                        args.assets.image(endpoint["records"]["cam00", frame, "raw"])
                    wu.append(endpoint)
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    problems.append(dict(scene=scene, endpoint="Wu_" + arm, run=str(folder), reason=f"{type(error).__name__}: {error}"))
            if problems:
                optional_missing += problems
                wu = []  # Auto includes a complete paired six-arm set only.
                if args.wu == "required":
                    missing += problems
        plans[scene] = dict(manifest_path=manifest_path, manifest=manifest, observations=observations,
                            endpoints=endpoints, references=refs, wu=wu)
    if missing and not args.allow_incomplete:
        raise ValueError("Required endpoints are incomplete; no output/final gate written. Use --allow-incomplete only for a marked preview. " + json.dumps(missing, ensure_ascii=False))
    return plans, missing, optional_missing


def selected_font(path=None):
    font_path = Path(path) if path else next((value for value in FONT_CANDIDATES if value.is_file()), None)
    assert font_path and font_path.is_file(), "Install/use DejaVu or Noto; no bitmap fallback for scientific labels"
    return font_path


def board(path, title, subtitle, tiles, region, roi, assets, font_path, columns=4):
    """Only crop/resize/layout/annotate source pixels; no image enhancement."""
    cell_width, cell_height = (384, 288) if region == "whole" else (320, 320)
    margin, gap, caption, heading = 18, 12, 60, 84
    rows = math.ceil(len(tiles) / columns)
    width = margin * 2 + columns * cell_width + (columns - 1) * gap
    height = heading + rows * (cell_height + caption + gap) + margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 18)
    small = ImageFont.truetype(str(font_path), 14)
    title_font = ImageFont.truetype(str(font_path), 22)
    draw.text((margin, 10), title, fill="#111111", font=title_font)
    draw.text((margin, 43), subtitle, fill="#333333", font=small)
    records = []
    for number, tile in enumerate(tiles):
        x = margin + number % columns * (cell_width + gap)
        y = heading + number // columns * (cell_height + caption + gap)
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill="#eeeeee", outline="#bbbbbb")
        if tile.get("missing"):
            draw.text((x + 10, y + 20), "MISSING / 未完成", fill="#9d1515", font=font)
            record = dict(key=tile["key"], label=tile["label"], missing=True, reason=tile["reason"])
        else:
            array = assets.image(tile)
            original = Image.fromarray(array)
            if region == "face":
                original = original.crop(tuple(roi))
                interpolation = Image.Resampling.NEAREST
                processing = "fixed_xyxy_crop; nearest-neighbour display resize; original source preserved"
            else:
                ImageDraw.Draw(original).rectangle(tuple(roi), outline="#ffee00", width=3)
                interpolation = Image.Resampling.LANCZOS
                processing = "whole image; fixed ROI outline; Lanczos display downsample only"
            scale = min(cell_width / original.width, cell_height / original.height)
            size = max(1, round(original.width * scale)), max(1, round(original.height * scale))
            original = original.resize(size, interpolation)
            image.paste(original, (x + (cell_width - size[0]) // 2, y + (cell_height - size[1]) // 2))
            record = dict(key=tile["key"], label=tile["label"], source=assets.records[tile["path"]], role=tile["role"],
                          source_operation=tile.get("operation", "none"), display_processing=processing,
                          display_crop_xyxy=list(roi) if region == "face" else None)
            if "teacher_index" in tile:
                record["teacher_index"] = tile["teacher_index"]
            if "metrics" in tile:
                record["diagnostic_scores"] = tile["metrics"]
        draw.text((x, y + cell_height + 4), tile["label"], fill="#111111", font=font)
        if "metrics" in tile:
            value = tile["metrics"]["whole" if region == "whole" else "face_crop"]
            draw.text((x, y + cell_height + 30), f"PSNR {value['psnr']:.2f}   SSIM {value['ssim']:.4f}  (fixed frame)", fill="#444444", font=small)
        records.append(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    image.save(path)
    return dict(path=str(path.resolve()), sha256=sha(path), width=width, height=height, title=title,
                region=region, fixed_face_xyxy=list(roi), tiles=records)


def image_panels(args, protocol, plans):
    metric, panels, scores = cpu_metrics(), [], []
    for scene, plan in plans.items():
        for camera in ["cam00", "cam01"]:
            roi = protocol["face_xyxy"][scene][camera]
            for frame in FRAMES:
                refs = plan["references"][camera, frame]
                gt = args.assets.image(refs[0])
                raw = list(refs)
                corrected = [refs[0]]
                for endpoint in plan["endpoints"]:
                    if endpoint.get("missing"):
                        raw.append(missing_item(endpoint["key"], endpoint["label"] + " | raw", endpoint["reason"]))
                        if endpoint["kind"] == "sr":
                            corrected.append(missing_item(endpoint["key"], endpoint["label"] + " | corrected", endpoint["reason"]))
                    else:
                        raw.append(endpoint["records"][camera, frame, "raw"])
                        if endpoint["kind"] == "sr" and camera == "cam01":
                            corrected.append(endpoint["records"][camera, frame, "corrected"])
                for tile in raw + (corrected if camera == "cam01" else []):
                    if not tile.get("missing"):
                        tile["metrics"] = score(args.assets.image(tile), gt, roi, metric)
                        scores.append(dict(scene=scene, camera=camera, frame=frame, key=tile["key"], role=tile["role"],
                                           source_path=tile["path"], source_sha256=tile["sha256"], metrics=tile["metrics"]))
                scope = "held-out cam00 / 测试新视角" if camera == "cam00" else "training cam01 / 训练拟合诊断"
                prefix = "PARTIAL PREVIEW / 未完成预览 | " if args.partial else ""
                for region in ["whole", "face"]:
                    name = f"{scene}_{camera}_{frame:04d}_{region}_raw.png"
                    panels.append(board(args.output / name, prefix + f"{scene} | {scope} | frame {frame:04d}",
                        "Raw native predictions. Fixed ROI. LR bicubic is an input reference; cam00 has no teacher.",
                        raw, region, roi, args.assets, args.font_path))
                    if camera == "cam01":
                        name = f"{scene}_{camera}_{frame:04d}_{region}_corrected_diagnostic.png"
                        panels.append(board(args.output / name, prefix + f"{scene} | cam01 corrected only | frame {frame:04d}",
                            "Training diagnostic: learned colour affine + optional w4. Raw versions are on the main panel.",
                            corrected, region, roi, args.assets, args.font_path))
                    if camera == "cam00" and plan["wu"]:
                        # First column is the same HR; remaining6 are matched SR
                        # arms. Rows are Wu then CEM, using the exact same ROI.
                        paired = [dict(refs[0], label="GT HR | Wu row")]
                        paired += [endpoint["records"][camera, frame, "raw"] for endpoint in plan["wu"]]
                        paired += [dict(refs[0], label="GT HR | CEM row")]
                        for endpoint in plan["endpoints"][1:7]:
                            paired.append(missing_item(endpoint["key"], endpoint["label"], endpoint["reason"]) if endpoint.get("missing") else endpoint["records"][camera, frame, "raw"])
                        for tile in paired:
                            if not tile.get("missing") and "metrics" not in tile:
                                tile["metrics"] = score(args.assets.image(tile), gt, roi, metric)
                                scores.append(dict(scene=scene, camera=camera, frame=frame, key=tile["key"], role=tile["role"], source_path=tile["path"], source_sha256=tile["sha256"], metrics=tile["metrics"]))
                        name = f"{scene}_{camera}_{frame:04d}_{region}_Wu_CEM_matched_arms.png"
                        panels.append(board(args.output / name, prefix + f"{scene} | cam00 Wu / CEM six matched arms | {frame:04d}",
                            "Wu checkpoints were historically selected development references; pipeline/budget differences remain.",
                            paired, region, roi, args.assets, args.font_path, columns=7))
    # Repeated GT and paired-panel entries are identical, not additional samples.
    unique = {(row["scene"], row["camera"], row["frame"], row["key"], row["role"]): row for row in scores}
    return panels, list(unique.values())


def temporal_diagnostics(args, protocol, plans):
    results = []
    for scene, plan in plans.items():
        candidates = [endpoint for endpoint in plan["endpoints"] + plan["wu"] if not endpoint.get("missing")]
        valid, unavailable = [], []
        for endpoint in candidates:
            absent = [frame for frame in range(300) if not Path(endpoint["records"]["cam00", frame, "raw"]["path"]).is_file()]
            if absent:
                unavailable.append(dict(method=endpoint["key"], missing_frames=absent))
            else:
                valid.append(endpoint)
        if unavailable:
            if not args.allow_incomplete:
                raise ValueError("Temporal diagnostics require all300 PNGs: " + json.dumps(unavailable))
            args.partial = True
        if not valid:
            results.append(dict(scene=scene, camera="cam00", status="no_complete_method_available", methods=[], unavailable=unavailable))
            continue
        records = {endpoint["key"]: [] for endpoint in valid}
        previous = {}
        previous_gt = previous_gt_item = None
        roi = protocol["face_xyxy"][scene]["cam00"]
        x0, y0, x1, y1 = roi
        for frame in range(300):
            observation = plan["observations"]["cam00", frame]
            gt_item = make_item("GT", "GT HR", child(plan["manifest_path"].parent, observation["hr_path"]), observation["hr_sha256"], "heldout_HR_temporal_reference")
            gt = args.assets.image(gt_item).astype(np.float32) / 255
            for endpoint in valid:
                item = endpoint["records"]["cam00", frame, "raw"]
                prediction = args.assets.image(item).astype(np.float32) / 255
                if frame:
                    error = np.abs((prediction - previous[endpoint["key"]]) - (gt - previous_gt))
                    records[endpoint["key"]].append(dict(frame_previous=frame - 1, frame=frame,
                        full_L1=float(error.mean(dtype=np.float64)), fixed_face_L1=float(error[y0:y1, x0:x1].mean(dtype=np.float64)),
                        prediction_previous_sha256=endpoint["records"]["cam00", frame - 1, "raw"]["sha256"], prediction_sha256=item["sha256"],
                        GT_previous_sha256=previous_gt_item["sha256"], GT_sha256=gt_item["sha256"]))
                previous[endpoint["key"]] = prediction
            previous_gt, previous_gt_item = gt, gt_item
        results.append(dict(scene=scene, camera="cam00", frames=300, adjacent_pairs=299,
            definition="mean(abs((P[t]-P[t-1])-(GT[t]-GT[t-1]))) on 8-bit RGB /255; t1..299",
            motion_compensated=False, interpretation="Mixes reconstruction quality and temporal error; not proof of temporal consistency or geometry; fixed face box may contain moving content.",
            unavailable=unavailable, methods=[dict(key=endpoint["key"], mean_full_L1=float(np.mean([row["full_L1"] for row in records[endpoint["key"]]])),
                mean_fixed_face_L1=float(np.mean([row["fixed_face_L1"] for row in records[endpoint["key"]]])), pairs=records[endpoint["key"]]) for endpoint in valid]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--protocol", type=Path, default=BATCH / "diagnostics/fixed_visual_protocol_v1.json")
    parser.add_argument("--runs-root", type=Path, default=BATCH / "runs")
    parser.add_argument("--hrref-run", action="append", default=[], metavar="SCENE=RUN", help="Explicit returned HRref mapping; defaults to each scene's native_HRref_v2")
    parser.add_argument("--wu-root", type=Path, default=BATCH / "wu_reference")
    parser.add_argument("--wu", choices=["auto", "off", "required"], default="auto", help="Auto includes only a complete six-arm returned Wu set per scene")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--temporal", action="store_true", help="Also stream all299 adjacent GT-relative differences")
    parser.add_argument("--temporal-only", action="store_true", help="Only temporal diagnostics; no boards/crop scores")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.scene) == len(set(args.scene)) and not args.output.exists(), "Output is immutable; choose a new directory"
    args.assets = Assets()
    args.font_path = selected_font(args.font)
    protocol, fixed = load_protocol(args.protocol)
    plans, missing, optional_missing = preflight(args, protocol, fixed)
    args.partial = bool(missing)
    args.output.mkdir(parents=True)
    panels, scores = ([], []) if args.temporal_only else image_panels(args, protocol, plans)
    temporal = temporal_diagnostics(args, protocol, plans) if args.temporal or args.temporal_only else []
    if temporal:
        write_new(args.output / "temporal_diagnostics.json", dict(status="CPU_image_diagnostic_only", results=temporal))
    endpoints = []
    for plan in plans.values():
        for endpoint in plan["endpoints"] + plan["wu"]:
            endpoints.append({key: value for key, value in endpoint.items() if key != "records"})
    groups = {}
    for row in scores:
        groups.setdefault((row["scene"], row["camera"], row["key"], row["role"]), []).append(row)
    means = []
    for (scene, camera, key, role), rows in groups.items():
        if sorted(row["frame"] for row in rows) == FRAMES:
            means.append(dict(scene=scene, camera=camera, key=key, role=role, frames=FRAMES,
                scope="arithmetic mean over3fixedframes only; not300frame mainmetrics",
                regions={region: {metric: float(np.mean([row["metrics"][region][metric] for row in rows]))
                    for metric in ["psnr", "ssim", "mse"]} for region in ["whole", "face_crop"]}))
    result = dict(status="partial_preview_missing_endpoints" if args.partial else "complete_fixed_image_diagnostics",
        final_gate=not args.partial, human_acceptance=False, temporal_acceptance=False, scientific_acceptance=False, blinded=False,
        CPU_environment=dict(python=sys.version, numpy=np.__version__, Pillow=PIL.__version__,
                             opencv=sys.modules["cv2"].__version__ if "cv2" in sys.modules else None),
        scope="fixed3frame diagnostic; main300framePSNR/SSIM/LPIPS are produced separately",
        source=source_identity(Path(__file__)), fixed_protocol=source_identity(args.protocol), legacy_metric_source=source_identity(LEGACY),
        font=source_identity(args.font_path), created_UTC=datetime.now(timezone.utc).isoformat(), GPU_calls=0, model_files_opened=0,
        missing_required_endpoints=missing, optional_Wu_unavailable=optional_missing, endpoint_receipts=endpoints,
        image_sources=list(args.assets.records.values()), panels=panels, fixed_frame_scores=scores, fixed_three_frame_means=means,
        metric_definition="PSNR uses normalized8bitRGB MSE and legacy max(MSE,1e-12); SSIM legacy RGB11x11Gaussian sigma1.5 population covariance, excludes crop's outer5px. Face SSIM computed after fixed crop; not full-image SSIM-map average.",
        LPIPS="not recomputed here; common A100300frame evaluator only", main_metric_table_generated=False,
        limitations=["Unblinded3frames do not establish human or temporal acceptance.", "Sharpness or PSNR/SSIM does not prove geometry or truthful detail.",
                     "cam00 LR bicubic is a test-input visual reference, not a legal novel-view reconstruction method; no cam00 teachers are read.",
                     "cam01 raw and corrected are separate; corrected fits learned training-camera observation correction.",
                     "Native HRref uses privileged trainingHR; Wu references were historically selected and pipelines/budgets differ."],
        temporal_requested=bool(args.temporal or args.temporal_only))
    write_new(args.output / "visual_diagnostic_receipt.json", result)
    if not args.partial:
        write_new(args.output / "complete_image_inputs_gate.json", dict(status="all_required_returned_endpoints_and_fixed_image_inputs_verified",
            diagnostic_receipt=source_identity(args.output / "visual_diagnostic_receipt.json"), human_acceptance=False, temporal_acceptance=False))
    print(json.dumps(dict(status=result["status"], panels=len(panels), final_gate=result["final_gate"], output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
