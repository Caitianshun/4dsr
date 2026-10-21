"""CTS-only endpoint index; imports unchanged CPU scientific validators.

No original A100 source is patched or monkeypatched. Only the declared profile,
CTS device identity and global common-metric location differ.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from index_sr import checkpoint_contract, verify_trajectory, render_contract
from build_sr_job_cts_v2 import (ROOT, HERE, EXP, OLD, BATCH, PROFILE, CTS_UUID, sha, relative, identity,
    parent_gate, protocol_module, read, endpoint, gate_job, load_profile, check_identity)


def verify_environment(value, job):
    assert value["schema"] == "CEM_area4_CTS_evaluation_environment_v1" and value["status"] == "verified"
    assert value["evaluation_profile"] == PROFILE and value["selected_gpu_uuid"] == CTS_UUID
    assert value["amendment"] == job["amendment"] and value["activation"] == job["activation"]
    assert value["metric_python"] == ".venv/bin/python" and value["CUDA_VISIBLE_DEVICES"] == "0"
    assert value["project_root"] == "/home/cts/Project/4DSR" and value["hostname"] == "cts"
    assert value["job"]["path"] == str(Path(job["out"]) / "job.json")
    assert value["execution_context"]["path"] == str(Path(job["out"]) / "execution_context.json")
    check_identity(value["execution_context"])
    context = read(value["execution_context"]["path"])
    assert context["project_id"] == "4dsr" and context["selected_gpu_uuids"] == [CTS_UUID]
    for key, expected in job["expected_common_environment"].items():
        assert value[key] == expected, "Actual CTS environment differs: " + key
    assert value["execution_context"]["sha256"]
    assert value["job"]["sha256"] == sha(ROOT / job["out"] / "job.json")


def metrics_contract(row, manifest, renders=None, profile=None):
    folder = ROOT / row["metric_dir"]
    summary_path, csv_path = folder / "summary.json", folder / "per_frame.csv"
    summary = read(summary_path)
    assert summary["schema"] == "dynamic_lr_shared_HR_image_metrics_v1" and summary["status"] == "completed"
    assert summary["frames"] == 300 and summary["camera"] == "cam00" and summary["resolution"] == [1344, 1008]
    assert summary["scene"] == row["scene"] and summary["method"] == row["system"] + "_area4_" + row["arm"]
    assert summary["manifest_sha256"] == sha(manifest) and summary["device"] == "cuda" and summary["host"] == "cts"
    assert summary["input_representation"] == "8-bit rendered RGB PNG for every method"
    assert Path(summary["render_directory"]).as_posix() == "/home/cts/Project/4DSR/" + row["render_dir"]
    assert profile is not None, "Explicit validated CTS profile required"
    for key, expected in profile["expected_common_environment"].items():
        assert summary[key] == expected, "CTS metric environment mismatch: " + key
    rows = list(csv.DictReader(csv_path.open()))
    assert [int(item["frame"]) for item in rows] == list(range(300))
    data = read(manifest)
    refs = {o["frame_index"]: o for o in data["observations"] if o["split"] == "test"}
    assert set(refs) == set(range(300)) and {v["camera_id"] for v in refs.values()} == {"cam00"}
    for frame, item in enumerate(rows):
        assert item["reference_sha256"] == refs[frame]["hr_sha256"]
        expected = renders[frame]["sha256"] if renders is not None else sha(ROOT / row["render_dir"] / f"{frame:04d}.png")
        assert item["render_sha256"] == expected
        assert all(math.isfinite(float(item[key])) for key in ["psnr", "ssim", "lpips", "mse"])
    for key in ["psnr", "ssim", "lpips"]:
        assert abs(math.fsum(float(item[key]) for item in rows)/300-summary["test"][key]) < 1e-10
    return dict(status="shared_CTS_8bit_PNG_complete", summary=identity(summary_path), per_frame=identity(csv_path),
        values=summary["test"], GPU=summary["gpu"], lpips_state_dict_sha256=summary["lpips_state_dict_sha256"])

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
    profile, plan = gate_job(job)
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
    assert state["device_guard"]["physical_mask"] == "0" and state["gpu_name"] == profile["expected_common_environment"]["gpu"]
    checkpoint = run / "training" / f"checkpoint_{steps:05d}.pt"
    assert sha(checkpoint) == state["checkpoint"]["sha256"]
    counters = checkpoint_contract(checkpoint, ROOT / parent_checkpoint, state, steps, protocol.ARMS[args.arm], protocol.GROUPS, inputs["manifest"]["splits"]["train"])
    verify_trajectory(run / "training/trajectory.jsonl", inputs, steps, protocol.GROUPS)
    index = dict(schema="CEM_area4_SR_verified_result_CTS_v2", status="passed_wiring_fixture_no_scientific_metrics" if fixture else "SR_endpoint_and_303_renders_unified_metrics_verified",
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
                     metrics=metrics_contract(endpoint(plan, "CEM", scene, args.arm), manifest, heldout["renders"], profile), metrics_status="shared_CTS_8bit_PNG_complete")
    env_path = run / "evaluation_environment.json"
    verify_environment(read(env_path), job)
    index.update(evaluation_profile=PROFILE, amendment=job["amendment"], activation=job["activation"],
        evaluation_environment=profile["expected_common_environment"], evaluation_environment_receipt=identity(env_path))
    return destination, index



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=["naive", "areaT"], required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--fixture-steps", type=int, default=0)
    args = parser.parse_args()
    destination, index = build_index(args)
    with destination.open("x") as stream:
        stream.write(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=index["status"], path=str(destination), sha256=sha(destination))))


if __name__ == "__main__":
    main()
