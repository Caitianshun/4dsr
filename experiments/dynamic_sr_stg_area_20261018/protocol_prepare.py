#!/usr/bin/env python3
"""Register the bounded STG area4 comparison without importing CUDA or STG.

This reads JSON provenance and hashes source files. It neither decodes training
or test images nor executes upstream code. Image-content hash checks are
optional and, when requested, apply only to registered training LR/teachers.
The protocol is a preparation artifact, never evidence of deployment/training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "output/dynamic_sr_multi4d_area_20261017"
WU = ROOT / "output/dynamic_sr_physlr_20261016/eval"
DEFAULT_OUT = ROOT / "output/dynamic_sr_stg_area_20261018"
STG_COMMIT = "427abfc58309a4a5213843dd673fb22c4529306c"
SCENES = {"Cut07": "cut_roasted_beef", "Cook07": "cook_spinach"}
WU_REFERENCES = {
    "Cut07": {
        "areaL0": "Cut07_area4_areaL0_cam00_posthoc_cull",
        "areaTm": "Cut07_area4_P7areaTm_cam00_posthoc_cull",
        "controlled_areaTm": "Cut07_area4_areaTm_cam00_posthoc_cull",
        "complete": "Cut07_area4_P7areaTm_calib_cam00_posthoc_cull",
        "areaOracle": "Cut07_area4_oracle_cam00_posthoc_cull",
    },
    "Cook07": {
        "areaL0": "Cook07_area4_areaL0_cam00_posthoc_cull",
        "areaTm": "Cook07_area4_areaTm_cam00_posthoc_cull",
        "complete": "Cook07_area4_areaTm_calib_cam00_posthoc_cull",
        "areaOracle": "Cook07_area4_oracle_cam00_posthoc_cull",
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def digest_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input path escapes registered dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_new(path, value):
    """An existing registration must not be silently replaced."""
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def baseline(path, manifest_sha):
    if not path.is_file():
        return {"path": str(path), "status": "incomplete", "metrics": None}
    result = read(path)
    valid = (result.get("status") == "completed" and result.get("frames") == 300
             and result.get("manifest", {}).get("sha256") == manifest_sha)
    metrics = result.get("test", {})
    valid = valid and all(isinstance(metrics.get(key), (int, float))
                          and math.isfinite(metrics[key]) for key in ("psnr", "ssim", "lpips"))
    return {**identity(path), "status": "completed" if valid else "incomplete",
            "metrics": {k: metrics.get(k) for k in ("psnr", "ssim", "lpips")} if valid else None,
            "evaluation_gpu": result.get("gpu"), "posthoc_cull": True,
            "manifest": result.get("manifest"), "checkpoint": result.get("checkpoint"),
            "limitation": "Full300 Wu pipeline reference; not a 50-frame or matched-training-budget control."}


def scene_protocol(tag, verify_training_files):
    spec_path = MULTI / f"spec_{tag}_v2.json"
    spec = read(spec_path)
    manifest_path, teacher_path = Path(spec["manifest"]), Path(spec["teacher_index"])
    manifest, teacher = read(manifest_path), read(teacher_path)
    root = manifest_path.parent
    manifest_sha = sha(manifest_path)
    cameras = manifest["splits"]["train"]
    expected_cameras = [f"cam{i:02d}" for i in range(1, 21) if not (tag == "Cut07" and i == 4)]
    if (manifest["splits"] != {"train": expected_cameras, "dev": [], "test": ["cam00"]}
            or manifest["resolutions"] != {"hr": [1344, 1008], "lr": [336, 252]}
            or manifest.get("scale") != 4
            or manifest.get("physical_lr_derivation", {}).get("formation") != "area4"):
        raise ValueError(f"Registered area4 split/dimensions changed: {tag}")
    rows = manifest["observations"]
    lookup = {(r["camera_id"], int(r["frame_index"])): r for r in rows}
    expected = {(c, f) for c in ["cam00", *cameras] for f in range(300)}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError(f"Incomplete or repeated full300 observations: {tag}")
    for (camera, frame), row in lookup.items():
        if row["split"] != ("test" if camera == "cam00" else "train"):
            raise ValueError("Camera/observation split mismatch")
        if abs(float(row["time"]) - frame / 300.) > 1e-12:
            raise ValueError("Global manifest time must remain frame_index/300")
    train_keys = {(c, f) for c in cameras for f in range(300)}
    entries = {(r["camera"], int(r["frame"])): r for r in teacher["entries"]}
    if (len(entries) != len(teacher["entries"]) or set(entries) != train_keys
            or teacher.get("oracle", False) or teacher["manifest_sha256"] != manifest_sha):
        raise ValueError("Frozen teacher must cover exactly legal training inputs")
    inputs = []
    for camera, frame in sorted(train_keys):
        row, entry = lookup[camera, frame], entries[camera, frame]
        if entry["lr_sha256"] != row["lr_sha256"]:
            raise ValueError("Teacher/source LR identity mismatch")
        lr = checked_path(root, row["lr_path"])
        prior = checked_path(root, entry["relative_path"])
        if verify_training_files and (sha(lr) != row["lr_sha256"] or sha(prior) != entry["sha256"]):
            raise ValueError(f"Training image bytes changed: {camera}/{frame}")
        inputs.append({"camera": camera, "frame": frame, "global_time": row["time"],
                       "lr_path": row["lr_path"], "lr_sha256": row["lr_sha256"],
                       "teacher_path": entry["relative_path"], "teacher_sha256": entry["sha256"],
                       "oracle_hr_path": row["hr_path"], "oracle_hr_sha256": row["hr_sha256"]})
    initialization = Path(spec["init"])
    init_report_path = initialization.parent / "report.json"
    report = read(init_report_path)
    if (report["manifest_sha256"] != manifest_sha or report["npz_sha256"] != sha(initialization)
            or report["source_frames"] != [0] or report["used_hr_images"]
            or report["used_development_or_test_images"]):
        raise ValueError("Initialization is not legal frame-zero training-LR geometry")
    for source in report["source_images"]:
        if (source["camera_id"], source["frame_index"]) not in train_keys or source["frame_index"] != 0:
            raise ValueError("Illegal initialization source")
        if source["sha256"] != lookup[source["camera_id"], 0]["lr_sha256"]:
            raise ValueError("Initialization LR source declaration mismatch")
    roi_path = ROOT / "output/dynamic_sr_soft_motion_20260924/roi_registry" / SCENES[tag] / "roi_protocol.json"
    roi = read(roi_path)
    if roi.get("status") != "fixed_before_prediction_reads":
        raise ValueError("ROI not fixed before prediction reads")
    for reference in roi["reference_images"]:
        if lookup[reference["camera"], reference["frame"]]["hr_sha256"] != reference["sha256"]:
            raise ValueError("ROI reference pixels do not match the current manifest declaration")
    segments = []
    for index in range(6):
        start, stop = 50 * index, 50 * (index + 1)
        selected = [r for r in inputs if start <= r["frame"] < stop]
        segments.append({"segment_id": f"f{start:03d}_{stop - 1:03d}", "index": index,
                         "frame_start_inclusive": start, "frame_stop_exclusive": stop,
                         "training_observations": len(selected), "test_observations": 50,
                         "training_input_declared_hash": digest_json(selected),
                         "time_mapping": {"global": "frame_index / 300",
                                          "local": "(frame_index - frame_start_inclusive) / 50",
                                          "local_first": 0., "local_last": 49 / 50.,
                                          "global_first": start / 300., "global_last": (stop - 1) / 300.},
                         "model_state": "independent_per_segment; no cross-segment weight reuse",
                         "initialization": "shared registered scene-frame0 LR reference; STG-native temporal initialization adaptation pending"})
    refs = {name: baseline(WU / folder / "summary.json", manifest_sha)
            for name, folder in WU_REFERENCES[tag].items()}
    return {"tag": tag, "scene": SCENES[tag], "manifest": identity(manifest_path),
            "multi4d_spec_source": identity(spec_path), "teacher_index": identity(teacher_path),
            "teacher_config": teacher.get("teacher_config"), "train_cameras": cameras,
            "test_cameras": ["cam00"], "dev_cameras": [], "resolutions": manifest["resolutions"],
            "global_frames": list(range(300)), "training_observations": len(inputs),
            "training_input_declared_hash": digest_json(inputs),
            "training_file_content_hashes_verified": verify_training_files,
            "training_paths_exist": True, "oracle_hr_content_hashes_verified": False,
            "initialization": {"npz": identity(initialization), "report": identity(init_report_path),
                               "point_count": report["point_count"], "source_frames": [0],
                               "used_HR_pixels": False, "used_heldout_pixels": False,
                               "official_STG_initialization": False,
                               "pending": "Native STG initialization/time-support compatibility must be audited; no HR substitution."},
            "segments": segments,
            "roi_protocol": {**identity(roi_path), "regions": roi["regions_by_camera_xyxy_exclusive"],
                             "reference_frame": 40, "reference_hash_declarations_match": True,
                             "per_frame_face_text_coverage": "not_verified; fixed frame40 boxes may miss moving faces",
                             "human_blind_acceptance": "pending"},
            "Wu_references": refs,
            "Wu_controlled_teacher_arm": "controlled_areaTm" if tag == "Cut07" else "areaTm"}


def native_identity():
    upstream = ROOT / 'third_party/stg_20261018/source'
    manifest = ROOT / 'deployment/stg_20261018/source_manifest.json'
    adapter = DEFAULT_OUT / 'adapter/identity.json'
    source_record, adapted = read(manifest), read(adapter)
    if source_record['upstream_commit'] != STG_COMMIT or adapted['upstream_commit'] != STG_COMMIT:
        raise ValueError('Native source/adapter commit mismatch')
    for relative, expected in adapted['original_sha256'].items():
        if sha(upstream / relative) != expected:
            raise ValueError('Native source differs from reviewed adapter provenance')
    for relative, expected in adapted['generated_sha256'].items():
        if sha(adapter.parent / relative) != expected:
            raise ValueError('Generated adapter differs from its provenance')
    selected = ['helper_train.py', 'thirdparty/gaussian_splatting/scene/oursfull.py',
                'thirdparty/gaussian_splatting/renderer/__init__.py',
                'configs/n3d_full/cut_roasted_beef.json', 'configs/n3d_full/cook_spinach.json']
    return {'upstream_commit': STG_COMMIT, 'source_manifest': identity(manifest),
            'source_manifest_file_count': source_record['file_count'],
            'adapter_identity': identity(adapter), 'adapter_patch': identity(adapter.parent / 'upstream.patch'),
            'selected_native_sources': [identity(upstream / p) for p in selected],
            'resumable': False, 'checkpoint_kind': 'native PLY and same-stem RGB-decoder .pt; no optimizer/RNG restart state'}


def add_prior_input_verification(scenes, receipt_paths):
    for path in receipt_paths:
        data = read(path)
        scene = next((s for s in scenes if s['scene'] == data.get('scene')), None)
        if (scene is None or data.get('files_verified') is not True
                or data['manifest_sha256'] != scene['manifest']['sha256']):
            raise ValueError('Prior input-transfer receipt is not verified for a registered scene')
        manifest_path = Path(scene['manifest']['path'])
        manifest, teacher = read(manifest_path), read(scene['teacher_index']['path'])
        files = {item['path']: item['sha256'] for item in data['files']}
        if len(files) != len(data['files']):
            raise ValueError('Duplicate paths in prior input-transfer receipt')
        expected = [(row['lr_path'], row['lr_sha256']) for row in manifest['observations'] if row['split'] == 'train']
        expected += [(entry['relative_path'], entry['sha256']) for entry in teacher['entries']]
        for relative, expected_sha in expected:
            key = str((manifest_path.parent / relative).relative_to(ROOT))
            if files.get(key) != expected_sha:
                raise ValueError('Prior transfer receipt does not cover exact registered LR/teacher bytes')
        scene['prior_training_bytes_verification'] = {
            'status': 'verified_via_prior_local_receipt', 'receipt': identity(path),
            'declared_file_count': len(data['files']), 'declared_bytes': data['bytes'],
            'exact_training_LR_teacher_coverage_checked': True,
            'image_bytes_rehashed_by_this_revision': False,
            'remote_verification': 'separate required receipt; local verification does not establish remote completion'}


def build(verify_training_files, transfer_receipts=()):
    scenes = [scene_protocol(tag, verify_training_files) for tag in SCENES]
    add_prior_input_verification(scenes, transfer_receipts)
    sources = ["experiments/dynamic_sr_multi4d_area_20261017/data.py",
               "experiments/dynamic_sr_multi4d_area_20261017/prepare_inputs.py",
               "experiments/dynamic_sr_multi4d_area_20261017/evaluate.py",
               "experiments/dynamic_sr_multi4d_area_20261017/assess.py",
               "experiments/dynamic_sr_20260918/evaluate.py",
               "experiments/dynamic_sr_same_observation_20260930/frequency.py"]
    return {"schema": "stg_area_protocol_v2", "status": "registered_CPU_only",
            "created_utc": datetime.now(timezone.utc).isoformat(), "model": "SpacetimeGaussians (STG)",
            "revision_reason": "Pinned native source uses rendering-only checkpoints, original30k schedule and25k selected pre-update checkpoint; replace provisional full-state-resume gate.",
            "supersedes": identity(DEFAULT_OUT / 'protocol.json'),
            "source_pinning": native_identity(),
            "training_status": "not_started_by_this_script", "implementation_status": "adapter_and_CUDA_validation_pending",
            "scenes": scenes,
            "observation_operator": {"kind": "area4", "target": "4x4 uint8 sRGB mean then numpy.round",
                                     "prediction": "differentiable 4x4 average of HR render; no quantization",
                                     "physical_limit": "synthetic area4 protocol, not a complete optical sensor model"},
            "arms": {"areaL0": {"input": "training LR", "teacher_weight": 0.},
                     "areaTm": {"input": "same LR + registered frozen area-matched SwinIR", "teacher_weight": .2},
                     "areaOracle": {"input": "same LR + training-camera HR", "teacher_weight": .2,
                                    "diagnostic_only": True, "legal_method_ranking": False}},
            "teacher_loss": {"formula": "0.2 * (0.8 * L1(render, teacher) + 0.2 * (1 - SSIM(render, teacher)))",
                             "teacher_updated": False, "native_STG_losses": "preserve; compatibility pending pinned source audit"},
            "sampling": {"native_batch_size": 2, "same_timestamp_per_batch": True,
                         "distinct_training_cameras_per_batch": True,
                         "paired_across_arms": "identical registered frame/camera order and initialization/RNG state",
                         "native_planned_iterations": 30000,
                         "selected_checkpoint_iteration": 25000,
                         "checkpoint_semantics": "Native save precedes controller and optimizer update. Stop at selected25000 checkpoint; 24999 optimizer updates and25000 forward/backward batches per segment.",
                         "full_budget": {"models_per_arm": 6, "frames_per_model": 50,
                                         "forward_backward_batches_per_arm": 150000,
                                         "optimizer_updates_per_arm": 149994,
                                         "training_camera_observations_per_arm": 300000,
                                         "arms_per_scene": 3, "models_per_scene": 18,
                                         "three_arm_forward_backward_batches": 450000}},
            "time_policy": "Keep source manifest unchanged at frame/300. Convert once to (frame-start)/50 only at STG interface; inverse mapping used for evaluation identity.",
            "evaluation": {"test": "cam00 all300 at original global frame identities",
                           "spatial": "same Wu helper: float RGB clamp[0,1], unshaved PSNR, SSIM outer5 exclusion, AlexNet-v0.1 LPIPS",
                           "regions": "fixed full300 HR temporal-variation mask; not semantic person/face segmentation",
                           "frequency": "same registered low/mid/high residual decomposition helper",
                           "training_diagnostic_frames": [0, 150, 299],
                           "preview_frames": [0, 40, 49, 50, 99, 100, 149, 150, 199, 200, 249, 250, 299],
                           "temporal": "all299 consecutive frame-difference residuals, plus 5 inter-segment boundary pairs separately",
                           "boundary_pairs": [[49, 50], [99, 100], [149, 150], [199, 200], [249, 250]],
                           "human_acceptance": "fixed face/text crops plus clean HR, hide labels and randomize; include teacher and final render; pending",
                           "test_input_boundary": "Only calibration/time/zero placeholders enter render; test LR/HR enter evaluator only",
                           "completion_requires": ["all6 segments complete per arm", "all300 test frames", "PSNR/SSIM/LPIPS finite", "all299 temporal pairs", "input/checkpoint hashes verified"]},
            "gates": {"first": {"scene": "Cut07", "segment": "f000_049",
                                 "purpose": "engineering and measured runtime/VRAM feasibility; not full300 quality ranking",
                                 "pass_requires": ["pinned-source review", "legal-input verification", "finite forward/backward", "native PLY plus RGB-decoder save/reload for evaluation", "native segment run and evaluator integration"],
                                 "resume_policy": "No optimizer/RNG restart state exists. Failure records an incomplete stage; do not claim exact resume or silently replay.",
                                 "failure": "record exact failure and cost; stop before multiplying segments"},
                      "Cut07": "complete all6 segments for each of 3 arms, then aggregate cam00 all300",
                      "Cook07": {"require_complete_Cut07": True, "logic": "any absolute difference meets threshold; improvement or degradation both qualify",
                                 "thresholds": {"STG_areaTm_minus_Wu_areaTm_PSNR_dB": .05,
                                                "STG_areaTm_minus_Wu_areaTm_LPIPS": .003,
                                                "STG_vs_Wu_teacher_gain_PSNR_dB": .05},
                                 "missing_or_failed": "incomplete; never zero or tied", "otherwise": "stop; small difference under this development protocol"},
                      "scope": "Cut07 then conditionally Cook07 only; no parameter grid or additional scenes"},
            "comparison_limits": ["STG uses six50-frame independent models while Wu/Multi4D use one300-frame model; improvement cannot be attributed only to representation.",
                                  "Every STG segment reuses registered scene-frame0 LR geometry with local temporal centers initialized at0; this differs from official per-frame SfM point initialization. Poor quality cannot isolate representation from this initialization adaptation.",
                                  "Do not compare first50-frame STG averages with full300 Wu averages.",
                                  "Wu includes training-view posthoc cull; STG native inference must be disclosed separately.",
                                  "Report total points/parameters/storage, per-segment and summed training time, hardware, peak VRAM, teacher preparation cost and optimizer updates.",
                                  "Common evaluation does not remove hardware or native optimization differences; small gains need paired confirmation.",
                                  "Cook and Cut are development scenes; this is not an independent sealed generalization test.",
                                  "Absolute quality, teacher-minus-LR gain and oracle-minus-teacher gap answer different questions.",
                                  "Multi4D full300 results are pending until complete summaries exist; no placeholder predictions or scores."],
            "sources": [identity(ROOT / item) for item in sources], "preparer": identity(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-training-files", action="store_true",
                        help="Hash all registered training LR/teacher image bytes, never test images")
    parser.add_argument('--input-transfer-receipt', type=Path, action='append', default=[],
                        help='Reference already verified local byte-transfer evidence without rehashing images')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("protocol.json", "protocol_audit.json")):
        raise FileExistsError("Registration exists; use a new explicit --out directory to preserve provenance")
    result = build(args.verify_training_files, args.input_transfer_receipt)
    write_new(args.out / "protocol.json", result)
    audit = {"schema": "stg_area_protocol_audit_v1", "status": "passed_CPU_metadata_registration",
             "protocol": identity(args.out / "protocol.json"), "preparer": identity(__file__),
             "scenes": [{"tag": s["tag"], "segments": len(s["segments"]),
                         "training_observations": s["training_observations"],
                         "test_observations": 300, "training_input_declared_hash": s["training_input_declared_hash"],
                         "training_file_content_hashes_verified": s["training_file_content_hashes_verified"],
                         "prior_training_bytes_verification": s.get('prior_training_bytes_verification'),
                         "initialization_point_count": s["initialization"]["point_count"],
                         "Wu_complete_references": sum(v["status"] == "completed" for v in s["Wu_references"].values())} for s in result["scenes"]],
             "CUDA_imports": 0, "image_decodes": 0, "test_image_bytes_read": False,
             "upstream_executed": False, "training_started": False,
             "source_files_modified": False,
             "limitation": "Source metadata/initialization/summary identities checked; STG implementation, runtime, full-state reload and scientific outcomes remain unverified."}
    write_new(args.out / "protocol_audit.json", audit)
    print(json.dumps({"status": result["status"], "protocol": str(args.out / "protocol.json"),
                      "training_file_hashes_verified": args.verify_training_files, "CUDA_imports": 0}))


if __name__ == "__main__":
    main()
