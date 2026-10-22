#!/usr/bin/env python3
"""Freeze CPU-only Yang/4C4D configs and dependency plans for six bicubic scenes.

No staging, CUDA import, remote access, training, queue, or timer is performed.
The old deployment, native source, and old completed endpoints remain immutable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import yaml


SCENES = (
    "cook_spinach", "cut_roasted_beef", "coffee_martini", "flame_steak",
    "meetroom_discussion", "meetroom_vrheadset",
)
METHODS = ("yang", "4c4d")
TRAIN_FRAMES = list(range(0, 300, 10))
RUNTIMES = {
    "yang": "output/dynamic_lr_baselines_20261020/strict4d/runtime/yang_pillowfix",
    "4c4d": "output/dynamic_lr_baselines_20261020/strict4d/runtime/4c4d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != data:
        raise FileExistsError(f"Refuse to overwrite a different frozen plan: {path}")
    path.write_text(data)


def replace_once(text: str, pattern: str, replacement: str) -> str:
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one native configuration field: {pattern}")
    return text


def configure(text: str, dataset: Path, model_output: Path) -> str:
    text = replace_once(text, r'^  source_path:.*$', "  source_path: " + json.dumps(str(dataset)))
    text = replace_once(text, r'^  model_path:.*$', "  model_path: " + json.dumps(str(model_output)))
    text = replace_once(text, r'^  resolution: 2\s*$', "  resolution: 1")
    text = replace_once(text, r'^exhaust_test: True\s*$', "exhaust_test: False")
    if "test_iterations:" in text:
        raise ValueError("Unexpected author test_iterations field; review native template")
    return text.rstrip() + "\n\ntest_iterations: []\n"


def verify_config(original: dict, configured: dict, dataset: Path, model_output: Path) -> None:
    expected = copy.deepcopy(original)
    expected["ModelParams"]["source_path"] = str(dataset)
    expected["ModelParams"]["model_path"] = str(model_output)
    expected["ModelParams"]["resolution"] = 1
    expected["exhaust_test"] = False
    expected["test_iterations"] = []
    if expected != configured:
        raise ValueError("An undeclared scientific parameter changed during path adaptation")
    if configured["OptimizationParams"]["iterations"] != 30000 or configured["batch_size"] != 4:
        raise ValueError("Requires the author full schedule and batch size")
    if configured["time_duration"] != [0.0, 10.0] or configured["gaussian_dim"] != 4:
        raise ValueError("Requires registered native XYZT time convention")


def prepare(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    output = args.output_root.resolve() if args.output_root else Path(__file__).resolve().parent
    summaries = []
    for method in args.methods:
        runtime = root / RUNTIMES[method]
        identity = runtime / "adapter_identity.json"
        if not identity.is_file():
            raise FileNotFoundError(f"Missing pinned native runtime identity: {identity}")
        for scene in args.scenes:
            manifest = root / "data/dynamic_sr/full_time_20261007/prepared" / scene / "manifest_train_ready_seed20261007.json"
            m = json.loads(manifest.read_text())
            if m.get("physical_lr_derivation"):
                raise ValueError("This expansion is explicitly bicubic; area4 is a separate protocol")
            if (m["scene"] != scene or m["scale"] != 4 or m["frame_indices"] != list(range(300))
                    or m["splits"]["test"] != ["cam00"] or m["splits"].get("dev")
                    or "cam00" in m["splits"]["train"]):
                raise ValueError("Unexpected registered scene/split/time protocol")
            train = [r for r in m["observations"] if r["split"] == "train"]
            expected = {(camera, frame) for camera in m["splits"]["train"] for frame in range(300)}
            if len(train) != len(expected) or {(r["camera_id"], r["frame_index"]) for r in train} != expected:
                raise ValueError("Incomplete/duplicate training observations")
            # No author MeetRoom configuration exists. Freeze one disclosed
            # cross-dataset template for both scenes; do not tune on cam00.
            template_scene = "cook_spinach" if scene.startswith("meetroom_") else scene
            template = runtime / "configs/dynerf" / (template_scene + ".yaml")
            original_text = template.read_text()
            original = yaml.safe_load(original_text)
            dataset_rel = f"output/dynamic_lr_expansion_20261022/strict4d/inputs/{method}/{scene}"
            run_rel = f"output/dynamic_lr_expansion_20261022/strict4d/runs/{method}_{scene}"
            dataset, run = root / dataset_rel, root / run_rel
            config = output / "configs/strict4d" / f"{method}_{scene}.yaml"
            text = configure(original_text, dataset, run / "model")
            configured = yaml.safe_load(text)
            verify_config(original, configured, dataset, run / "model")
            config.parent.mkdir(parents=True, exist_ok=True)
            if config.exists() and config.read_text() != text:
                raise FileExistsError(f"Refuse to change an existing frozen configuration: {config}")
            config.write_text(text)
            extra_cli = (["--training_view", ",".join(str(int(c[3:])) for c in m["splits"]["train"]),
                          "--testing_view", "0"] if method == "4c4d" else [])
            plan = {
                "schema": "strict4d_expansion_cpu_preparation_v1",
                "status": "prepared_on_CPU_HOLD_no_GPU_execution",
                "method": method, "scene": scene, "input_protocol": "bicubic",
                "manifest": {"path": str(manifest), "sha256": sha(manifest)},
                "native_runtime": {"project_relative_path": RUNTIMES[method],
                                   "adapter_identity_sha256": sha(identity)},
                "native_template": {"path": str(template), "sha256": sha(template),
                                    "scene": template_scene,
                                    "cross_dataset_transfer": scene.startswith("meetroom_"),
                                    "disclosure": ("MeetRoom uses the unchanged author Cook template, with only registered data/output paths, exact LR resolution, and heldout reporting disabled. Both MeetRoom scenes share it; no cam00-driven tuning."
                                                   if scene.startswith("meetroom_") else "Matching author scene configuration retained.")},
                "configuration": {"path": str(config), "sha256": sha(config),
                                  "scientific_fields_equal_to_template": True,
                                  "adapted_fields": ["ModelParams.source_path", "ModelParams.model_path",
                                                     "ModelParams.resolution", "exhaust_test", "test_iterations"],
                                  "pipeline_params": configured["PipelineParams"],
                                  "author_extra_CLI": extra_cli},
                "expected_dataset_project_relative_path": dataset_rel,
                "expected_run_project_relative_path": run_rel,
                "training": {"iterations": 30000, "batch_size": 4,
                             "training_cameras": m["splits"]["train"], "training_observations": len(train),
                             "native_time": "frame_index/30", "heldout_pixels_read": False,
                             "HR_supervision": False, "execute_allowed_this_turn": False},
                "evaluation": {"train": {"frames": TRAIN_FRAMES, "all_training_cameras": True,
                                          "observations": len(TRAIN_FRAMES) * len(m["splits"]["train"])},
                               "test": {"cameras": ["cam00"], "frames": list(range(300)), "observations": 300},
                               "resolution": m["resolutions"]["hr"], "checkpoint": "model/chkpnt30000.pth",
                               "selection": "fixed author terminal iteration; no heldout selection",
                               "render_validation": "pending actual GPU execution; CPU plan is not a rendered result"},
                "CPU_requirements_pending": ["registered LR SHA/shape staging in isolated dataset",
                                             "native camera projection QA for train LR and all HR evaluation cameras",
                                             "full native CPU loader with CHW from manifest, original 12 workers/batch4",
                                             "CPU deployment contract and environment identity", *(["known-pose CPU COLMAP sparse initialization"] if method == "yang" else ["MASt3R official weights/retrieval/codebook CPU identity gate"])],
                "GPU_requirements_HOLD": (["COLMAP PatchMatch dense initialization"] if method == "yang" else ["native posed MASt3R initialization"])
                                         + ["native training", "train/test HR rendering", "shared metric execution"],
                "GPU_calls": 0, "remote_calls": 0, "parameter_updates": 0,
                "no_scheduler_or_automatic_dispatch": True,
                "preparation_source_sha256": sha(Path(__file__)),
            }
            destination = output / "plans/strict4d" / f"{method}_{scene}.json"
            write_json(destination, plan)
            summaries.append({"method": method, "scene": scene, "plan": str(destination),
                              "plan_sha256": sha(destination), "config": str(config), "config_sha256": sha(config),
                              "native_env_map_res": configured["PipelineParams"]["env_map_res"],
                              "template_scene": template_scene})
    result = {"schema": "strict4d_six_scene_CPU_preparation_index_v1",
              "status": "prepared_on_CPU_HOLD_no_GPU_execution", "input_protocol": "bicubic",
              "configuration_count": len(summaries), "entries": summaries,
              "GPU_calls": 0, "remote_calls": 0, "parameter_updates": 0,
              "native_training_or_rendering_verified": False,
              "source_sha256": sha(Path(__file__))}
    write_json(output / "strict4d_preparation_index.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    result = prepare(parser.parse_args())
    print(json.dumps({"status": result["status"], "configuration_count": result["configuration_count"]}))


if __name__ == "__main__":
    main()
