"""CPU verification of an SR endpoint or an isolated non-scientific fixture.

Rehash frozen identities, inspect native/RAdam checkpoints on CPU, and verify
all render/metric records. This never renders, fits a correction, or uses CUDA.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path

from build_sr_job import BATCH, EXP, HERE, OLD, ROOT, parent_gate, protocol_module, relative, sha


def identity(path):
    path = Path(path)
    return dict(path=str(relative(path)), sha256=sha(path), bytes=path.stat().st_size)


def optimizer_counters(model, groups):
    assert len(model) == 29, "Wrong native CEM capture format"
    optimizer = model[18]
    assert [item["name"] for item in optimizer["param_groups"]] == groups
    counters = {}
    for group in optimizer["param_groups"]:
        assert len(group["params"]) == 1, "Native CEM group shape changed"
        values = []
        for parameter in group["params"]:
            step = optimizer["state"].get(parameter, {}).get("step", 0)
            if hasattr(step, "item"):
                assert step.device.type == "cpu"
                step = step.item()
            assert float(step).is_integer() and step >= 0
            values.append(int(step))
        counters[group["name"]] = values
    return counters


def checkpoint_contract(path, parent, state, steps, controls, groups, cameras):
    import torch
    assert not torch.cuda.is_initialized(), "Index is CPU-only"
    parent_payload, parent_iteration = torch.load(parent, map_location="cpu", weights_only=False)
    assert parent_iteration == 50000
    initial = optimizer_counters(parent_payload, groups)
    topology = dict(parent=parent_payload[20].clone(), level=parent_payload[21].clone(),
                    static=int(parent_payload[1].shape[0]), motion=int(parent_payload[27].shape[0]),
                    duration=parent_payload[23], interval=parent_payload[24],
                    time_shift=parent_payload[25], keyframe_num=parent_payload[26])
    del parent_payload
    gc.collect()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "CEM_area4_SR_fixed_topology_v1"
    assert payload["iteration"] == 50000 + steps and payload["SR_updates"] == steps
    assert payload["protocol"]["parameter_updates"] == steps and payload["protocol"]["steps_completed"] == steps
    assert payload["protocol"]["controls"] == controls
    assert payload["protocol"]["parent_checkpoint"]["sha256"] == sha(parent)
    final = optimizer_counters(payload["model"], groups)
    model = payload["model"]
    assert torch.equal(model[20], topology["parent"]) and torch.equal(model[21], topology["level"])
    assert int(model[1].shape[0]) == topology["static"] and int(model[27].shape[0]) == topology["motion"]
    assert (model[23], model[24], model[25], model[26]) == (topology["duration"], topology["interval"], topology["time_shift"], topology["keyframe_num"])
    # Every native parameter tensor must remain finite; preserve all native
    # positions/lifetimes/motion rather than inspecting only SH or RGB.
    for index in [1, 2, 3, 4, 5, 6, 7, 8, 22, 27, 28]:
        assert model[index].device.type == "cpu" and bool(torch.isfinite(model[index]).all())
    correction = payload["correction"]
    assert correction["names"] == cameras and "cam00" not in correction["names"] and correction["w4"] == controls["w4"]
    assert list(correction["delta"].shape) == [len(correction["names"]), 3, 4]
    assert list(correction["field"].shape) == [len(correction["names"]), 2, 4, 4]
    assert bool(torch.isfinite(correction["delta"]).all()) and bool(torch.isfinite(correction["field"]).all())
    for key in ["affine_optimizer"] + (["field_optimizer"] if controls["w4"] else []):
        optimizer = correction[key]
        assert len(optimizer["param_groups"]) == 1 and len(optimizer["param_groups"][0]["params"]) == 1
        counter = optimizer["state"][optimizer["param_groups"][0]["params"][0]]["step"]
        assert int(counter.item() if hasattr(counter, "item") else counter) == steps
    if not controls["w4"]:
        assert bool((correction["field"] == 0).all()) and correction["field_optimizer"] is None
    checks = {}
    for name in groups:
        advertised = state["native_RAdam_step_counters"][name]
        updates = [end - start for start, end in zip(initial[name], final[name])]
        assert advertised == dict(initial=initial[name], final=final[name], updates=updates)
        participation = state["native_gradient_participation"][name]["steps_with_grad"]
        assert 0 <= participation <= steps and updates == [participation]
        checks[name] = dict(initial=initial[name], final=final[name], updates=updates, steps_with_grad=participation)
    assert state["initial_point_counts"] == state["final_point_counts"] == dict(rendered_gaussians=topology["static"], virtual_motion_centers=topology["motion"])
    assert state["topology_parent_and_level_unchanged"]
    assert not torch.cuda.is_initialized()
    del payload, model, topology
    gc.collect()
    return checks


def verify_trajectory(path, inputs, steps, groups):
    count = 0
    with path.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["cursor"] == count and row["absolute_iteration"] == 50000 + count
            assert row["keys"] == [list(inputs["record_keys"][i]) for i in inputs["rows"][count - 1]]
            assert row["topology_events"] == 0
            assert set(row["native_LRs"]) == set(groups)
            assert all(math.isfinite(float(value)) for value in row["native_LRs"].values())
            assert all(math.isfinite(float(row[key])) for key in ["LR_term", "teacher_term", "regularizer", "loss"])
    assert count == steps, "Missing or extra parameter-update trajectory records"


def render_contract(run, directory, camera, frames, manifest_hash, checkpoint_hash):
    from PIL import Image
    folder = run / directory
    path = folder / "render_receipt.json"
    value = json.loads(path.read_text())
    assert value["status"] == "completed" and value["kind"] == "sr" and value["iteration"] == 56000
    assert value["camera"] == camera and value["frame_indices"] == frames
    assert value["frames"] == value["finite_before_clip_quantization_passed_frames"] == len(frames)
    assert value["manifest"]["sha256"] == manifest_hash and value["checkpoint"]["sha256"] == checkpoint_hash
    assert value["renderer_source_sha256"] == sha(ROOT / EXP / "render_hr.py")
    assert value["camera_helper_sha256"] == sha(ROOT / EXP / "sr_stage.py")
    assert not value["heldout_pixels_read"] and not value["training_pixels_read"]
    assert value["native_rasterizer_unchanged"] and value["raw_prediction_shape"] == [3, 1008, 1344]
    if camera == "cam00":
        assert value["scope"] == "heldout_300" and value["colour_affine"] == value["warp"] == "identity"
    else:
        assert value["scope"] == "training_view_diagnostic_only_fixed_three_frames"
    assert [entry["frame"] for entry in value["records"]] == frames
    assert value["training_receipt"]["sha256"] == sha(run / "training/training_receipt.json")
    expected_names = {f"{frame:04d}.png" for frame in frames}
    if camera == "cam01":
        expected_names |= {f"{frame:04d}_raw.png" for frame in frames}
    assert {path.name for path in folder.glob("*.png")} == expected_names
    result = []
    for entry in value["records"]:
        assert entry["path"] == f"{entry['frame']:04d}.png"
        png = folder / entry["path"]
        assert sha(png) == entry["sha256"]
        with Image.open(png) as image:
            assert image.mode == "RGB" and image.size == (1344, 1008)
        if camera == "cam00":
            assert entry["raw_path"] is None
        else:
            assert entry["raw_path"] == f"{entry['frame']:04d}_raw.png"
            raw = folder / entry["raw_path"]
            assert sha(raw) == entry["raw_sha256"]
            with Image.open(raw) as image:
                assert image.mode == "RGB" and image.size == (1344, 1008)
        result.append(dict(frame=entry["frame"], sha256=entry["sha256"]))
    return dict(receipt=identity(path), count=len(frames), scope=value["scope"], renders=result)


def metrics_contract(run, manifest, rendered, arm):
    summary_path, csv_path = run / "metrics/summary.json", run / "metrics/per_frame.csv"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed" and summary["frames"] == 300 and summary["camera"] == "cam00"
    assert summary["schema"] == "dynamic_lr_shared_HR_image_metrics_v1"
    assert summary["method"] == "CEM_area4_" + arm
    assert summary["scene"] == json.loads(manifest.read_text())["scene"] and summary["resolution"] == [1344, 1008]
    assert summary["manifest_sha256"] == sha(manifest)
    assert summary["input_representation"] == "8-bit rendered RGB PNG for every method"
    assert summary["device"] == "cuda" and "A100" in summary["gpu"]
    assert summary["evaluator_sha256"] == sha(ROOT / OLD / "evaluate_images.py")
    assert summary["helper_source_sha256"] == sha(ROOT / "experiments/dynamic_sr_20260918/evaluate.py")
    assert Path(summary["render_directory"]).resolve() == (run / "renders_hr").resolve()
    rows = list(csv.DictReader(csv_path.open()))
    assert [int(row["frame"]) for row in rows] == list(range(300))
    data = json.loads(manifest.read_text())
    references = {item["frame_index"]: item for item in data["observations"] if item["split"] == "test"}
    for frame, row in enumerate(rows):
        assert row["render_sha256"] == rendered["renders"][frame]["sha256"]
        assert row["reference_sha256"] == references[frame]["hr_sha256"]
        assert all(math.isfinite(float(row[key])) for key in ["psnr", "ssim", "lpips", "mse"])
    for key in ["psnr", "ssim", "lpips"]:
        assert abs(math.fsum(float(row[key]) for row in rows) / 300 - summary["test"][key]) < 1e-10
    return dict(status="shared_A100_8bit_PNG_complete", summary=identity(summary_path), per_frame=identity(csv_path), values=summary["test"],
                GPU=summary["gpu"], lpips_state_dict_sha256=summary["lpips_state_dict_sha256"])


def build_index(args):
    run, manifest = args.run.resolve(), args.manifest.resolve()
    assert relative(manifest) == Path("data/dynamic_sr/full_time_20261007/prepared") / (json.loads(manifest.read_text())["scene"] + "_area4") / "manifest_train_ready_seed20261007.json"
    assert args.fixture_steps in {0, 3}
    steps, fixture = args.fixture_steps or 6000, bool(args.fixture_steps)
    destination = run / ("fixture_acceptance.json" if fixture else "result_index.json")
    assert not destination.exists(), "Do not replace an accepted index"
    protocol = protocol_module()
    scene = json.loads(manifest.read_text())["scene"]
    parent_run, parent_checkpoint, parent_receipt, parent_state = parent_gate(scene, args.parent_run, relative(manifest))
    job_path = run / "job.json"
    job = json.loads(job_path.read_text())
    assert job["protocol"] == "CEM_area4_backbone_control_v1" and job["arm"] == args.arm and job["scene"] == scene
    assert job["out"] == str(relative(run)) and job["parent_run"] == str(parent_run)
    assert job["SR_actual_updates"] == steps and job["formal_scientific_metrics"] == (not fixture)
    assert job["fixed_topology"] and job["heldout_used_for_training"] is False
    for row in job["required_files"]:
        path = ROOT / row["path"]
        assert path.resolve().is_relative_to(ROOT.resolve()) and sha(path) == row["sha256"], "Frozen job input changed"
    registration, _ = protocol.SCHEDULES[scene]
    schedule = ROOT / "output/dynamic_sr_multiview_footprint_20261007/full_materialized" / registration / scene / "seed_20261007/schedule.json"
    teacher_kind = protocol.ARMS[args.arm]["teacher"]
    teacher = manifest.parent / ("teacher_index_sr_swinir_x4_seed20261007.json" if teacher_kind == "original" else "teacher_index_sr_swinir_x4_areaft_seed20261007.json") if teacher_kind in {"original", "matched"} else None
    inputs = protocol.load_inputs(manifest, schedule, args.arm, teacher)
    training_path = run / "training/training_receipt.json"
    state = json.loads(training_path.read_text())
    assert state["status"] == "completed" and state["arm"] == args.arm
    assert state["result_scope"] == ("wiring_fixture_only" if fixture else "controlled_fixed_topology_SR_continuation")
    assert state["steps_requested"] == state["steps_completed"] == state["parameter_updates"] == steps
    assert state["absolute_start"] == 50001 and state["final_iteration"] == 50000 + steps
    assert state["manifest"]["sha256"] == sha(manifest) and state["schedule"]["sha256"] == sha(schedule)
    assert state["parent_checkpoint"]["sha256"] == sha(ROOT / parent_checkpoint)
    assert state["parent_stage2_receipt"]["sha256"] == sha(ROOT / parent_receipt)
    assert state["controls"] == protocol.ARMS[args.arm]
    assert state["train_HR_used"] == (args.arm == "areaOracle") and not state["heldout_pixels_used"]
    assert state["optimizer_state_inherited"] and not state["native_scheduler_repaired"] and not state["parent_affine"]
    assert not state["gradient_error_statistics_updated"]
    assert all(state[key] == 0 for key in ["topology_events", "CEC_events", "densification_events", "pruning_events"])
    assert [group["name"] for group in state["native_optimizer_groups"]] == protocol.GROUPS
    assert state["device_guard"]["physical_mask"].isdecimal() and "A100" in state["gpu_name"]
    checkpoint = run / "training" / f"checkpoint_{steps:05d}.pt"
    assert sha(checkpoint) == state["checkpoint"]["sha256"]
    counters = checkpoint_contract(checkpoint, ROOT / parent_checkpoint, state, steps, protocol.ARMS[args.arm], protocol.GROUPS, inputs["manifest"]["splits"]["train"])
    verify_trajectory(run / "training/trajectory.jsonl", inputs, steps, protocol.GROUPS)
    index = dict(schema="CEM_area4_SR_verified_result_v1", status="passed_wiring_fixture_no_scientific_metrics" if fixture else "SR_endpoint_and_303_renders_unified_metrics_verified",
        UTC=datetime.now(timezone.utc).isoformat(), scene=scene, arm=args.arm, fixture_only=fixture,
        heldout_used_for_training=False, training_HR=args.arm == "areaOracle", actual_parameter_updates=steps,
        absolute_final_iteration=50000 + steps, fixed_topology=True, native_RAdam_step_counters=counters,
        checkpoint=identity(checkpoint), training_receipt=identity(training_path), trajectory=identity(run / "training/trajectory.jsonl"),
        parent_checkpoint=identity(ROOT / parent_checkpoint), parent_stage2_receipt=identity(ROOT / parent_receipt),
        manifest=identity(manifest), schedule=identity(schedule), teacher_index=identity(teacher) if teacher else None,
        job=identity(job_path), training_GPU=state["gpu_name"], training_seconds=state["elapsed_seconds"],
        torch_peak_reserved_bytes=state["torch_peak_reserved_bytes"], peak_memory_scope=state["peak_memory_scope"],
        indexer_source_sha256=sha(__file__), CPU_checkpoint_verification=True,
        source_identities=[row for row in job["required_files"] if row["path"].endswith((".py", ".json")) and not row["path"].startswith("data/")])
    if fixture:
        assert not any((run / name).exists() for name in ["renders_hr", "train_cam01", "metrics"]), "A wiring fixture cannot enter scientific metrics"
        index["metrics_status"] = "not_applicable_wiring_fixture"
    else:
        heldout = render_contract(run, "renders_hr", "cam00", list(range(300)), sha(manifest), sha(checkpoint))
        diagnostic = render_contract(run, "train_cam01", "cam01", [0, 150, 299], sha(manifest), sha(checkpoint))
        index.update(heldout_rendering=heldout, training_diagnostic=diagnostic,
                     metrics=metrics_contract(run, manifest, heldout, args.arm), metrics_status="shared_A100_8bit_PNG_complete")
    return destination, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=["areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle"], required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--fixture-steps", type=int, default=0)
    args = parser.parse_args()
    destination, index = build_index(args)
    with destination.open("x") as stream:
        stream.write(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=index["status"], path=str(destination), sha256=sha(destination))))


if __name__ == "__main__":
    main()
