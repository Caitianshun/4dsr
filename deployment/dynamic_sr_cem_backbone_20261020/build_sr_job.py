"""Freeze one completed-parent CEM SR arm or isolated wiring fixture on CPU.

This only writes a job specification after all identity gates pass. It never
submits work, connects remotely, imports Torch, or accepts a missing parent.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

from build_native_job_v2 import BATCH, EXP, GPUS, HERE, OLD, ROOT, sha


def protocol_module():
    path = ROOT / EXP / "sr_protocol.py"
    spec = importlib.util.spec_from_file_location("cem_SR_protocol_CPU", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve())


def parent_gate(scene, parent_run, manifest):
    parent_run = relative(parent_run)
    assert parent_run == BATCH / "runs" / (scene + "_native_parent_v1"), "Requires this scene's independent native area4 LR parent"
    path = ROOT / parent_run / "stage2.json"
    state = json.loads(path.read_text())
    assert state["status"] == "completed" and state["stage"] == 2, "Parent stage2 is not completed"
    assert state["train_level"] == "lr" and state["endpoint"] == 50000
    assert state["manifest"]["sha256"] == sha(ROOT / manifest)
    assert state["native_schedule_preserved"] and not state["heldout_pixels_used"] and not state["train_HR_used"]
    assert not state["parent_added_affine"] and not state["parent_added_warp"]
    checkpoint = parent_run / "model/chkpnt50000.pth"
    assert sha(ROOT / checkpoint) == state["final_checkpoint"]["sha256"], "Incomplete/different parent checkpoint"
    return parent_run, checkpoint, parent_run / "stage2.json", state


def asset_gate(scene, arm, manifest, path, inputs):
    path = relative(path)
    suffix = "verified_v1" if arm in {"naive", "areaT"} else "existing_verified_v1"
    assert path == BATCH / "assets" / f"a100_{scene}_{suffix}.json", "Wrong arm-specific asset gate"
    verified = json.loads((ROOT / path).read_text())
    assert verified["status"] == "verified" and verified["all_listed_file_bytes_rehashed"]
    assert verified["failure_count"] == 0 and verified["verified_count"] == verified["file_count"]
    assert verified["root"] == "/home/ubuntu/3DGS/4dsr", "Requires A100 byte-verification receipt"
    inventory = Path(verified["inventory_path"])
    assert not inventory.is_absolute() and ".." not in inventory.parts
    assert sha(ROOT / inventory) == verified["inventory_sha256"]
    table = json.loads((ROOT / inventory).read_text())
    assert table["scene"] == scene and table["target"] == "a100"
    assert table["source_manifest"] == dict(path=str(manifest), sha256=sha(ROOT / manifest))
    required = {str(manifest): sha(ROOT / manifest)}
    for observation in inputs["observations"].values():
        required[str(manifest.parent / observation["lr_path"])] = observation["lr_sha256"]
    for target in inputs["targets"].values():
        required[str(relative(target["path"]))] = target["sha256"]
    if inputs["teacher_index"]:
        row = inputs["teacher_index"]
        required[str(relative(row["path"]))] = row["sha256"]
    for observation in inputs["manifest"]["observations"]:
        if observation["split"] == "test":
            assert observation["camera_id"] == "cam00"
            required[str(manifest.parent / observation["hr_path"])] = observation["hr_sha256"]
    for name, expected in required.items():
        assert table["files"][name]["sha256"] == expected, "Asset inventory misses/changes required observation"
        row = verified["files"][name]
        assert row["status"] == "verified" and row["stable_during_read"] and row["sha256"] == expected
    return path, inventory, verified


def build(args):
    assert args.gpu in {0, 1}, "A100 GPU index must be explicit 0 or1"
    assert args.fixture_steps in {0, 3}, "Use 0 for formal6000 or3 for isolated wiring fixture"
    assert not args.output.exists(), "Frozen jobs are never overwritten"
    protocol = protocol_module()
    assert args.arm in protocol.ARMS
    manifest = Path("data/dynamic_sr/full_time_20261007/prepared") / (args.scene + "_area4") / "manifest_train_ready_seed20261007.json"
    parent_run, checkpoint, parent_receipt, parent = parent_gate(args.scene, args.parent_run, manifest)
    registration, expected_schedule_sha = protocol.SCHEDULES[args.scene]
    schedule = Path("output/dynamic_sr_multiview_footprint_20261007/full_materialized") / registration / args.scene / "seed_20261007/schedule.json"
    assert sha(ROOT / schedule) == expected_schedule_sha
    controls = protocol.ARMS[args.arm]
    teacher = None
    if controls["teacher"] in {"original", "matched"}:
        suffix = "sr_swinir_x4" if controls["teacher"] == "original" else "sr_swinir_x4_areaft"
        teacher = manifest.parent / f"teacher_index_{suffix}_seed20261007.json"
    inputs = protocol.load_inputs(ROOT / manifest, ROOT / schedule, args.arm, ROOT / teacher if teacher else None)
    receipt, inventory, verified = asset_gate(args.scene, args.arm, manifest, args.asset_receipt, inputs)
    fixture = args.fixture_steps > 0
    steps = args.fixture_steps if fixture else 6000
    label = f"{args.scene}_{args.arm}_{steps}steps_v1" if fixture else f"{args.scene}_{args.arm}_v1"
    out = BATCH / ("fixtures" if fixture else "runs") / label
    assert not (ROOT / out).exists(), "Select a new version; preserve existing runs/fixtures"
    runtime = BATCH / "runtime/cem-native-area4-v1"
    env = Path("envs/dynamic_lr_baselines_20261020/cem_a100_sm80")
    absolute = lambda value: "{ROOT}/" + str(value)
    python = absolute(env / "bin/python")
    training = out / "training"
    argv = [python, absolute(EXP / "sr_stage.py"), "--runtime", absolute(runtime), "--manifest", absolute(manifest),
            "--schedule", absolute(schedule), "--parent-checkpoint", absolute(checkpoint), "--parent-stage2-receipt", absolute(parent_receipt),
            "--arm", args.arm, "--gpu", str(args.gpu), "--output", absolute(training), "--steps", str(steps)]
    if teacher:
        argv += ["--teacher-index", absolute(teacher)]
    if fixture:
        argv += ["--wiring-fixture"]
    commands = [dict(name="real_GPU_environment", argv=[python, absolute(OLD / "methods/localdygs_cem/check_gpu_environment.py"),
                     "--method", "cem", "--output", absolute(out / "GPU_environment.json")]),
                dict(name="SR_wiring_fixture" if fixture else "SR_6000_updates", argv=argv)]
    if not fixture:
        for camera, directory in [("cam00", "renders_hr"), ("cam01", "train_cam01")]:
            commands.append(dict(name="SR_HR_render_" + camera, argv=[python, absolute(EXP / "render_hr.py"),
                "--runtime", absolute(runtime), "--manifest", absolute(manifest), "--checkpoint", absolute(training / "checkpoint_06000.pt"),
                "--training-receipt", absolute(training / "training_receipt.json"), "--kind", "sr", "--gpu", str(args.gpu),
                "--camera", camera, "--output", absolute(out / directory)]))
        commands.append(dict(name="shared_A100_PNG_metrics", argv=["{ROOT}/.venv/bin/python", absolute(OLD / "evaluate_images.py"),
            "--manifest", absolute(manifest), "--render-dir", absolute(out / "renders_hr"), "--out", absolute(out / "metrics"),
            "--method", "CEM_area4_" + args.arm]))
    commands.append(dict(name="fixture_acceptance" if fixture else "SR_result_index", argv=["{ROOT}/.venv/bin/python", absolute(HERE / "index_sr.py"),
        "--run", absolute(out), "--manifest", absolute(manifest), "--arm", args.arm, "--parent-run", absolute(parent_run),
        "--fixture-steps", str(args.fixture_steps)]))
    paths = [Path(__file__).resolve().relative_to(ROOT), HERE / "build_native_job_v2.py", HERE / "run_job.py", HERE / "index_sr.py",
             EXP / "sr_stage.py", EXP / "sr_protocol.py", EXP / "render_hr.py", OLD / "methods/localdygs_cem/check_gpu_environment.py",
             OLD / "evaluate_images.py", Path("experiments/dynamic_sr_20260918/evaluate.py"),
             receipt, inventory, checkpoint, parent_receipt, manifest, schedule]
    if teacher:
        paths += [teacher]
    runtime_inventory = json.loads((ROOT / runtime / "native_runtime_inventory.json").read_text())
    for item in runtime_inventory["files"]:
        path = runtime / item["path"]
        assert sha(ROOT / path) == item["sha256"], "Native runtime inventory mismatch"
        paths.append(path)
    paths += [runtime / "native_runtime_inventory.json", runtime / "protocol_interface_receipt.json"]
    required = [dict(path=str(path), sha256=sha(ROOT / path)) for path in sorted(set(paths))]
    scene_label = "cook" if args.scene == "cook_spinach" else "cut"
    job_id = f"codex-cem-area4-{scene_label}-{args.arm}-a100g{args.gpu}-20261020-" + ("f3v1" if fixture else "v1")
    job_id = job_id.replace("_", "-").lower()
    assert len(job_id) < 64 and re.fullmatch(r"[a-z0-9][a-z0-9-]*", job_id)
    return dict(protocol="CEM_area4_backbone_control_v1", input_formation="area4", heldout_used_for_training=False,
        job_id=job_id, method="CEM_area4_SR", arm=args.arm, scene=args.scene, gpu_uuid=GPUS["a100-train", args.gpu],
        out=str(out), commands=commands, required_files=required,
        input_receipts=[dict(path=str(receipt), inventory_sha256=verified["inventory_sha256"])],
        env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                 NUMEXPR_NUM_THREADS="1", CEM_PROTOCOL_TRAIN_LEVEL="lr", DYNAMIC_LR_PROTOCOL_MANIFEST=absolute(manifest),
                 LD_LIBRARY_PATH=absolute(env / "cuda-toolkit-overlay/lib64")),
        return_paths=[str(out)], training_HR=args.arm == "areaOracle", parent_run=str(parent_run),
        formal_scientific_metrics=not fixture, SR_actual_updates=steps, SR_absolute_iterations=[50001, 50000 + steps],
        fixed_topology=True, parent_native_endpoint=50000,
        comparison_limit="Same area4 parent/schedule across six arms; CEM fixed-topology continuation, zero-initialized SR affine. Different from historical Wu topology/affine/budget.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arm", choices=["areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle"], required=True)
    parser.add_argument("--fixture-steps", type=int, default=0)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], path=str(args.output), sha256=sha(args.output), required=len(job["required_files"]))))


if __name__ == "__main__":
    main()
