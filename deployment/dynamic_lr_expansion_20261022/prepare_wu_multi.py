"""Prepare bicubic-LR Wu/Multi4D specifications on CPU; launch nothing.

The default writes auditable specifications and reports missing initialization.
--prepare-initialization additionally runs the existing train-frame-zero CPU
SIFT initializer for the four missing Multi4D scenes, with capped CPU threads.
It never imports torch, queries a GPU, dispatches a process, or contacts a host.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/wu_multi"
SCENES = {"cook_spinach": "Cook07", "cut_roasted_beef": "Cut07",
          "coffee_martini": "Coffee07", "flame_steak": "Flame07",
          "meetroom_discussion": "Discussion07", "meetroom_vrheadset": "VRHeadset07"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def identity(path):
    path = Path(path)
    return dict(path=relative(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == encoded:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def checked_manifest(scene):
    path = ROOT / "data/dynamic_sr/full_time_20261007/prepared" / scene / "manifest_train_ready_seed20261007.json"
    manifest = read(path)
    if manifest.get("physical_lr_derivation"):
        raise ValueError("This preparation preserves original bicubic LR")
    if manifest["scene"] != scene or manifest["splits"]["test"] != ["cam00"] or manifest["splits"].get("dev"):
        raise ValueError("Scene or held-out split changed")
    train = manifest["splits"]["train"]
    expected = {(camera, frame) for camera in [*train, "cam00"] for frame in range(300)}
    keys = [(row["camera_id"], int(row["frame_index"])) for row in manifest["observations"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Every actual camera must have 300 unique registered frames")
    if "cam00" in train or len(set(train)) != len(train):
        raise ValueError("Illegal training split")
    for row in manifest["observations"]:
        if row["split"] != ("test" if row["camera_id"] == "cam00" else "train"):
            raise ValueError("Observation split changed")
        if abs(float(row["time"]) - row["frame_index"] / 300) > 1e-12:
            raise ValueError("Registered frame/300 time required")
    return path, manifest


def required(paths):
    return [dict(path=relative(path), sha256=sha(path)) for path in dict.fromkeys(map(Path, paths))]


def load_cpu_builder():
    """Load the existing CPU helper without importing its training environment."""
    directory = ROOT / "experiments/dynamic_sr_multi4d_area_20261017"
    data_spec = importlib.util.spec_from_file_location("data", directory / "data.py")
    data = importlib.util.module_from_spec(data_spec)
    old_data = sys.modules.get("data")
    sys.modules["data"] = data
    try:
        data_spec.loader.exec_module(data)
        spec = importlib.util.spec_from_file_location("expansion_multi_cpu_builder", directory / "prepare_inputs.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
    finally:
        if old_data is None:
            sys.modules.pop("data", None)
        else:
            sys.modules["data"] = old_data
    return builder


def prepare_multi_inputs(manifest_path, destination, threads):
    if (destination / "complete.json").is_file():
        receipt = read(destination / "complete.json")
        if receipt["manifest_sha256"] != sha(manifest_path):
            raise ValueError("Prepared input belongs to a different manifest")
        for name, key in (("initialization/points_lr.npz", "init_sha256"),
                          ("multi_schedule.json", "schedule_sha256")):
            if sha(destination / name) != receipt[key]:
                raise ValueError("Prepared input changed: " + name)
        return
    builder = load_cpu_builder()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Preserve partial preparation; choose a fresh input directory: " + str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    set_threads = builder.cv2.setNumThreads
    # Execution-only thread cap; the existing algorithm and RNG are unchanged.
    builder.cv2.setNumThreads = lambda requested: set_threads(min(requested, threads))
    try:
        inventory = builder.inventory(manifest_path, None)
        write(destination / "inventory.json", inventory)
        init = builder.initialize(manifest_path, destination / "initialization", 20261007)
        manifest = read(manifest_path)
        records = [row for row in manifest["observations"] if row["split"] == "train"]
        write(destination / "multi_schedule.json", builder.schedule(records, 20261007))
        write(destination / "complete.json", dict(
            status="prepared_original_bicubic_LR_CPU_only", manifest_sha256=sha(manifest_path),
            initializer=init, init_sha256=sha(destination / "initialization/points_lr.npz"),
            schedule_sha256=sha(destination / "multi_schedule.json"),
            source=identity(ROOT / "experiments/dynamic_sr_multi4d_area_20261017/prepare_inputs.py"),
            cpu_threads=threads, CUDA_calls=0, used_HR_pixels=False,
            used_heldout_pixels=False, used_SR=False))
    except BaseException as error:
        write(destination / "failed.json", dict(status="CPU_preparation_failed", error=str(error)))
        raise
    finally:
        builder.cv2.setNumThreads = set_threads


def wu_spec(scene, label, manifest_path, manifest, template):
    directory = ROOT / "output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes" / scene / "seed_20261007"
    plan, complete = read(directory / "config.json"), read(directory / "complete.json")
    state, config = complete["state"], plan["configuration"]
    batch = 4 if scene in {"coffee_martini", "meetroom_discussion", "meetroom_vrheadset"} else 2
    expected = dict(stage="fine", coarse_iteration=3000, fine_iteration=14000,
                    optimizer_resets=2, accepted_RGB=17000 * batch,
                    accepted_backward=17000, accepted_Adam=16999)
    if state != expected or config["OptimizationParams"]["batch_size"] != batch:
        raise ValueError("Wu complete native author budget differs: " + scene)
    if any(complete[key] is not False for key in ("HR_used", "teacher_used", "heldout_pixels_used")):
        raise ValueError("Wu endpoint used forbidden supervision")
    if plan["observation"] != "native_lr" or complete["representation"] != "native_author_GaussianModel_no_children":
        raise ValueError("Endpoint is a modified method, not native Wu")
    if plan["data"]["manifest"] != {key: identity(manifest_path)[key] for key in ("path", "sha256")}:
        raise ValueError("Wu checkpoint plan/manifest mismatch")
    checkpoint = ROOT / complete["checkpoint"]["path"]
    if sha(checkpoint) != complete["checkpoint"]["sha256"]:
        raise ValueError("Wu endpoint checksum differs")
    result = copy.deepcopy(template)
    result.update(schema="wu_bicubic_expansion_20261022", method="wu", scene=scene, label=label,
                  manifest=relative(manifest_path), manifest_identity=identity(manifest_path),
                  parent_plan_directory=relative(directory), checkpoint=identity(checkpoint),
                  execution_state="reuse_completed_endpoint_no_training",
                  planned_training_required=False, train_cameras=manifest["splits"]["train"],
                  test_cameras=["cam00"], train_observations=300 * len(manifest["splits"]["train"]),
                  LR=manifest["resolutions"]["lr"], HR=manifest["resolutions"]["hr"],
                  original_training_hardware=complete["hardware"],
                  original_training_seconds=complete["cost_all_attempts"].get("seconds"),
                  domain_adaptation="pinned Wu generic N3DV batch4 configuration on MeetRoom" if scene.startswith("meetroom_") else "pinned author N3DV scene configuration",
                  gpu_render_acceptance="pending; CPU asset/spec verification only")
    paths = [manifest_path, directory / "config.json", directory / "complete.json", checkpoint,
             manifest_path.parent / "initialization_seed_20261007/points_lr.npz"]
    paths += [ROOT / row["path"] for row in template["required_files"]
              if not row["path"].startswith(("data/", "output/"))]
    for name, expected_sha in plan["runtime_source_sha256"].items():
        runtime_source = ROOT / "third_party/4dgs_843d5ac_runtime" / name
        if sha(runtime_source) != expected_sha:
            raise ValueError("Local Wu author runtime differs: " + name)
        paths.append(runtime_source)
    result["required_files"] = required(paths)
    return result


def multi_spec(scene, label, manifest_path, manifest, template, previous, prepare, threads):
    result = copy.deepcopy(template)
    existing = previous.get("multi4d/" + scene)
    inputs = OLD / "prepared_original" / label if existing else HERE / "prepared/multi4d" / label
    if prepare and not existing:
        prepare_multi_inputs(manifest_path, inputs, threads)
    ready = (inputs / "initialization/report.json").is_file() and (inputs / "multi_schedule.json").is_file()
    result.update(schema="multi4d_bicubic_expansion_20261022", method="multi4d", scene=scene, label=label,
                  manifest=relative(manifest_path), manifest_identity=identity(manifest_path),
                  init=relative(inputs / "initialization/points_lr.npz"),
                  schedule=relative(inputs / "multi_schedule.json"),
                  upstream="output/dynamic_sr_multi4d_area_20261017/runtime_v2/Multi4D-c483e82",
                  adapter=relative(OLD / "multi4d_native/adapter"),
                  execution_state="reuse_completed_endpoint_no_training" if existing else "train_pending_GPU_release",
                  planned_training_required=not bool(existing), initialization_CPU_ready=ready,
                  planned_training_output=relative(ROOT / "output/dynamic_lr_expansion_20261022/multi4d" / label),
                  train_cameras=manifest["splits"]["train"], test_cameras=["cam00"],
                  frames=list(range(300)), time="frame_index/300",
                  LR=manifest["resolutions"]["lr"], HR=manifest["resolutions"]["hr"],
                  domain_adaptation="pinned Multi4D dynerf configuration on calibrated MeetRoom" if scene.startswith("meetroom_") else "pinned author dynerf configuration",
                  gpu_training_and_render_acceptance="pending on future allocated host; no GPU work performed by preparation")
    if existing:
        checkpoint = ROOT / existing["checkpoint"]["path"]
        if sha(checkpoint) != existing["checkpoint"]["sha256"]:
            raise ValueError("Multi4D endpoint checksum differs")
        result["checkpoint"] = identity(checkpoint)
    else:
        result["checkpoint"] = None
    paths = [manifest_path]
    paths += [ROOT / row["path"] for row in template["required_files"]
              if "prepared_original" not in row["path"] and not row["path"].startswith("data/")]
    if ready:
        report = read(inputs / "initialization/report.json")
        if report["manifest_sha256"] != sha(manifest_path) or report["source_frames"] != [0]:
            raise ValueError("Multi4D static initialization must use this scene's train-frame-zero LR")
        if report["used_hr_images"] or report["used_development_or_test_images"]:
            raise ValueError("Multi4D static initialization used forbidden pixels")
        paths += [inputs / "initialization/points_lr.npz", inputs / "initialization/report.json", inputs / "multi_schedule.json"]
        result["static_initial_points"] = report["point_count"]
    repairs = read(ROOT / result["native_repairs_receipt"])
    for name in ("train.py", "arguments/__init__.py", "arguments/dynerf.py", "scene/gaussian_model.py",
                 "scene/deformation.py", "scene/hexplane.py", "gaussian_renderer/__init__.py", "utils/loss_utils.py"):
        paths.append(ROOT / result["upstream"] / name)
    for entry in repairs["patches"]:
        path = ROOT / result["upstream"] / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Local upstream repair identity differs")
        paths.append(path)
    result["required_files"] = required(paths)
    result["remaining_TODO"] = ([] if ready else ["CPU frame-zero initialization and native sampling inventory"]) + [
        "Future host-specific CUDA extensions/environment verification and real GPU smoke under resource admission",
        "Train missing scenes only, then immediately render fixed train/test views and score via shared evaluator"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "specs/wu_multi")
    parser.add_argument("--prepare-initialization", action="store_true")
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    template_wu = read(OLD / "wu_cook_spinach_native_lr.json")
    template_multi = read(OLD / "multi4d_cook_spinach_native_lr.json")
    previous = read(ROOT / "output/dynamic_lr_baselines_20261020/wu_multi_result_index.json")["methods"]
    entries = []
    for scene, label in SCENES.items():
        manifest_path, manifest = checked_manifest(scene)
        for method, spec in (
            ("wu", wu_spec(scene, label, manifest_path, manifest, template_wu)),
            ("multi4d", multi_spec(scene, label, manifest_path, manifest, template_multi,
                                   previous, args.prepare_initialization, args.cpu_threads))):
            path = args.out / f"{method}_{scene}_bicubic.json"
            write(path, spec)
            entries.append(dict(method=method, scene=scene, label=label, spec=identity(path),
                                execution_state=spec["execution_state"],
                                initialization_CPU_ready=spec.get("initialization_CPU_ready", True)))
    if "torch" in sys.modules:
        raise RuntimeError("CPU preparation unexpectedly imported torch")
    report = dict(schema="wu_multi_CPU_preparation_20261022", status="CPU_prepared_no_GPU_no_dispatch",
                  CUDA_calls=0, GPU_queries=0, remote_operations=0, started_jobs=0,
                  train_metric_frames=list(range(0, 300, 10)), test_metric_frames=list(range(300)),
                  training_jobs_pending=sum(row["execution_state"] == "train_pending_GPU_release" for row in entries),
                  entries=entries, preparation_source=identity(__file__),
                  note="Original bicubic LR preserved. GPU work remains deferred to a future explicit resource-admitted execution.")
    write(args.out / "index.json", report)
    print(json.dumps(dict(status=report["status"], specs=len(entries),
                          training_jobs_pending=report["training_jobs_pending"], index=relative(args.out / "index.json"))))


if __name__ == "__main__":
    main()
