"""Additive CTS contingency freezer and fail-closed admission gate (CPU only).

Original A100 helpers and scientific implementations are unchanged. An explicit
root-owned amendment and separately hashed activation are mandatory; this CLI
never submits, connects, syncs, selects a new experiment, or retries a task.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from build_sr_job import parent_gate, protocol_module
from build_native_job_v2 import BATCH, EXP, HERE, OLD, ROOT, sha

PROFILE = "CTS3090_fallback_v1"
CTS_ROOT = "/home/cts/Project/4DSR"
CTS_UUID = "GPU-fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a910"
SCENES = ("cook_spinach", "cut_roasted_beef")
EVALUATOR = OLD / "evaluate_images.py"
LEGACY = Path("experiments/dynamic_sr_20260918/evaluate.py")


def relative(path):
    path = Path(path)
    if path.is_absolute():
        path = path.resolve().relative_to(ROOT.resolve())
    assert path.parts and not path.is_absolute() and ".." not in path.parts
    assert (ROOT / path).resolve().is_relative_to(ROOT.resolve())
    return path


def read(path):
    return json.loads((ROOT / relative(path)).read_text())


def identity(path):
    path = relative(path)
    return dict(path=str(path), sha256=sha(ROOT / path), bytes=(ROOT / path).stat().st_size)


def check_identity(item):
    actual = identity(item["path"])
    assert actual["sha256"] == item["sha256"], "Identity SHA mismatch: " + item["path"]
    assert "bytes" not in item or actual["bytes"] == item["bytes"]
    return actual


def rooted(path):
    return "{ROOT}/" + str(relative(path))


def manifest_path(scene):
    assert scene in SCENES
    return Path("data/dynamic_sr/full_time_20261007/prepared") / (scene + "_area4") / "manifest_train_ready_seed20261007.json"


def load_profile(path, expected):
    bound = check_identity(dict(path=str(relative(path)), sha256=expected))
    profile = read(path)
    assert profile["schema"] == "CEM_area4_CTS_contingency_amendment_v1"
    assert profile["status"] == "prepared_not_activated" and profile["profile"] == PROFILE
    assert (profile["host"], profile["remote_root"], profile["gpu_index"], profile["gpu_uuid"]) == ("cts", CTS_ROOT, 0, CTS_UUID)
    assert profile["training_python"] == "envs/dynamic_lr_baselines_20261020/cem/bin/python"
    assert profile["metric_python"] == ".venv/bin/python"
    for item in profile["unchanged_numeric_core"]:
        check_identity(item)
    assert {row["path"] for row in profile["unchanged_numeric_core"]} == {str(EXP / name) for name in ["sr_stage.py", "sr_protocol.py", "render_hr.py"]} | {str(EVALUATOR), str(LEGACY)}
    for key in ["source_protocol", "predeclared_matrix", "endpoint_plan"]:
        check_identity(profile[key])
    plan = read(profile["endpoint_plan"]["path"])
    assert plan["schema"] == "CEM_area4_CTS_fixed_endpoint_plan_v1" and plan["expected_count"] == len(plan["endpoints"]) == 28
    expected_keys = {(s, scene, arm) for scene in SCENES for s, arms in [("CEM", ["native_parent", "areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle", "HRref"]), ("Wu", ["areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle"])] for arm in arms}
    assert {(r["system"], r["scene"], r["arm"]) for r in plan["endpoints"]} == expected_keys
    assert len({r["metric_dir"] for r in plan["endpoints"]}) == 28
    for row in plan["endpoints"]:
        assert relative(row["metric_dir"]) == BATCH / "shared_CTS_metrics" / row["system"] / row["scene"] / row["arm"]
        assert relative(row["render_dir"]) == relative(row["run"]) / "renders_hr"
    expected_remaining = {(scene, arm, str(BATCH / "runs" / f"{scene}_{arm}_cts_fallback_v2")) for scene in SCENES for arm in ["naive", "areaT"]}
    assert {(r["scene"], r["arm"], r["out"]) for r in profile["remaining_SR_tasks"]} == expected_remaining
    return profile, plan, bound


def load_activation(profile, amendment, path, expected, phase):
    assert phase in {"engineering_fixture", "production"}
    contract = profile["activation_contract"]
    assert relative(path) == relative(contract[phase + "_path"])
    bound = check_identity(dict(path=str(relative(path)), sha256=expected))
    activation = read(path)
    assert activation["schema"] == contract["schema"]
    assert all(key in activation for key in contract["required_fields"])
    assert activation["phase"] == phase and activation["profile"] == PROFILE
    assert (activation["host"], activation["remote_root"], activation["gpu_index"], activation["gpu_uuid"]) == ("cts", CTS_ROOT, 0, CTS_UUID)
    assert activation["amendment"]["path"] == amendment["path"] and activation["amendment"]["sha256"] == amendment["sha256"]
    assert activation["evidence"] and activation["decision_UTC"]
    if phase == "production":
        fixture_id = check_identity(activation["fixture_acceptance"])
        fixed = profile["engineering_fixture"]
        assert relative(fixture_id["path"]) == relative(fixed["out"]) / "fixture_acceptance.json"
        fixture = read(fixture_id["path"])
        assert fixture["schema"] == "CEM_area4_SR_verified_result_CTS_v2"
        assert fixture["status"] == "passed_wiring_fixture_no_scientific_metrics" and fixture["fixture_only"]
        assert fixture["scene"] == fixed["scene"] and fixture["arm"] == fixed["arm"]
        assert fixture["actual_parameter_updates"] == fixed["steps"] == 3
        assert fixture["metrics_status"] == "not_applicable_wiring_fixture"
        assert fixture["evaluation_profile"] == PROFILE and fixture["amendment"]["sha256"] == amendment["sha256"]
    return activation, bound


def bind(args, phase):
    profile, plan, amendment = load_profile(args.amendment, args.amendment_sha256)
    activation, activation_id = load_activation(profile, amendment, args.activation, args.activation_sha256, phase)
    return profile, plan, amendment, activation_id


def endpoint(plan, system, scene, arm):
    rows = [r for r in plan["endpoints"] if (r["system"], r["scene"], r["arm"]) == (system, scene, arm)]
    assert len(rows) == 1
    return rows[0]


def asset_gate(scene, manifest, path, required_inputs=None):
    path = relative(path)
    assert path == BATCH / "assets" / f"cts_{scene}_SR_fallback_verified_v2.json"
    receipt = read(path)
    assert receipt["status"] == "verified" and receipt["all_listed_file_bytes_rehashed"] is True
    assert receipt["failure_count"] == 0 and receipt["root"] == CTS_ROOT
    inventory_path = BATCH / "assets" / f"cts_{scene}_SR_fallback_inventory_v2.json"
    assert relative(receipt["inventory_path"]) == inventory_path
    assert receipt["inventory_sha256"] == sha(ROOT / inventory_path)
    inventory = read(inventory_path)
    assert inventory["schema"] == "CEM_area4_closed_input_inventory_v1"
    assert (inventory["scene"], inventory["target"]) == (scene, "cts")
    assert inventory["source_manifest"] == dict(path=str(manifest), sha256=sha(ROOT / manifest))
    assert receipt["verified_count"] == receipt["file_count"] == len(inventory["files"])
    assert set(receipt["files"]) == set(inventory["files"])
    for name, row in inventory["files"].items():
        check = receipt["files"][name]
        assert check["status"] == "verified" and check["stable_during_read"]
        assert (check["sha256"], check["bytes"]) == (row["sha256"], row["bytes"])
    needed = {str(manifest): sha(ROOT / manifest)}
    data = read(manifest)
    for row in data["observations"]:
        if row["split"] == "test":
            assert row["camera_id"] == "cam00"
            needed[str(manifest.parent / row["hr_path"])] = row["hr_sha256"]
    if required_inputs is not None:
        for row in required_inputs["observations"].values():
            needed[str(manifest.parent / row["lr_path"])] = row["lr_sha256"]
        for row in required_inputs["targets"].values():
            needed[str(relative(row["path"]))] = row["sha256"]
        row = required_inputs["teacher_index"]
        assert row is not None
        needed[str(relative(row["path"]))] = row["sha256"]
    for name, digest in needed.items():
        assert name in inventory["files"] and inventory["files"][name]["sha256"] == digest, "Missing/changed CTS input coverage: " + name
    return path, inventory_path, receipt, inventory


class Required:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None, size=None):
        row = identity(path)
        assert expected is None or row["sha256"] == expected
        assert size is None or row["bytes"] == size
        assert row["path"] not in self.files or self.files[row["path"]] == row
        self.files[row["path"]] = row
        return row

    def frozen(self, item):
        return self.add(item["path"], item["sha256"], item.get("bytes"))


def shared_required(profile, amendment, activation):
    req = Required()
    for row in [amendment, activation, profile["endpoint_plan"], profile["source_protocol"], profile["predeclared_matrix"], *profile["unchanged_numeric_core"]]:
        req.frozen(row)
    act = read(activation["path"])
    if "fixture_acceptance" in act:
        req.frozen(act["fixture_acceptance"])
    for name in ["build_sr_job_cts_v2.py", "index_sr_cts_v2.py", "build_wu_eval_job_cts_v2.py", "build_sr_job.py", "index_sr.py", "build_native_job_v2.py", "run_job.py"]:
        req.add(HERE / name)
    return req


def target(row):
    return {key: row[key] for key in ["system", "scene", "arm", "metric_dir"]}


def guard_command(out):
    return dict(name="CTS_amendment_activation_gate", argv=[rooted(".venv/bin/python"), rooted(HERE / "build_sr_job_cts_v2.py"), "gate", "--job", rooted(out / "job.json")])


def environment_command(out):
    return dict(name="CTS_common_evaluation_environment", argv=[rooted(".venv/bin/python"), rooted(HERE / "build_wu_eval_job_cts_v2.py"), "environment", "--job", rooted(out / "job.json")])


def evaluation_command(row, manifest):
    return dict(name=("CTS_" + row["system"] + "_" + row["scene"] + "_" + row["arm"]), argv=[rooted(".venv/bin/python"), rooted(EVALUATOR), "--manifest", rooted(manifest), "--render-dir", rooted(row["render_dir"]), "--out", rooted(row["metric_dir"]), "--method", row["system"] + "_area4_" + row["arm"]])


def base_job(profile, amendment, activation, req, out, job_id, scene, arm, method, commands, receipts):
    assert len(job_id) <= 64 and re.fullmatch(r"[a-z0-9][a-z0-9-]*", job_id)
    assert not (ROOT / out).exists(), "Preserve existing output"
    return dict(protocol="CEM_area4_backbone_control_v1", input_formation="area4", heldout_used_for_training=False,
        job_id=job_id, method=method, arm=arm, scene=scene, host="cts", gpu_uuid=CTS_UUID, out=str(out),
        evaluation_profile=PROFILE, amendment=amendment, activation=activation,
        endpoint_plan=profile["endpoint_plan"], expected_common_environment=profile["expected_common_environment"],
        required_files=[req.files[k] for k in sorted(req.files)], input_receipts=receipts,
        commands=[guard_command(out), environment_command(out), *commands],
        env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"))


def gate_job(job, require_remote=False):
    profile, plan, amendment = load_profile(job["amendment"]["path"], job["amendment"]["sha256"])
    phase = "engineering_fixture" if job.get("formal_scientific_metrics") is False else "production"
    load_activation(profile, amendment, job["activation"]["path"], job["activation"]["sha256"], phase)
    assert job["evaluation_profile"] == PROFILE and job["host"] == "cts" and job["gpu_uuid"] == CTS_UUID
    assert job["expected_common_environment"] == profile["expected_common_environment"]
    assert job["endpoint_plan"]["sha256"] == profile["endpoint_plan"]["sha256"]
    assert "CUDA_VISIBLE_DEVICES" not in job["env"]
    if "SR_actual_updates" in job:
        if phase == "engineering_fixture":
            f = profile["engineering_fixture"]
            assert (job["scene"], job["arm"], job["out"], job["SR_actual_updates"], job["job_id"]) == (f["scene"], f["arm"], f["out"], 3, f["job_id"])
            assert job["evaluation_targets"] == []
        else:
            assert any((r["scene"], r["arm"], r["out"]) == (job["scene"], job["arm"], job["out"]) for r in profile["remaining_SR_tasks"])
            assert job["SR_actual_updates"] == 6000
    for item in job["evaluation_targets"]:
        assert item == target(endpoint(plan, item["system"], item["scene"], item["arm"]))
    if require_remote:
        assert str(ROOT) == CTS_ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        context_path = Path(os.environ["CLAUDE_REMOTE_JOB_DIR"]) / "execution_context.json"
        context = json.loads(context_path.read_text())
        assert context["project_id"] == "4dsr" and context["selected_gpu_uuids"] == [CTS_UUID]
    return profile, plan


def add_profile_args(parser):
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--activation-sha256", required=True)


def write_job(output, job):
    assert not output.exists(), "Never overwrite frozen jobs"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(job, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(path=str(output), sha256=sha(output), job_id=job["job_id"], required=len(job["required_files"]), GPU_calls=0)))

def build(args):
    assert args.gpu == 0, "CTS fixed physical GPU0 only"
    profile, plan, amendment, activation = bind(args, "engineering_fixture" if args.fixture_steps else "production")
    assert args.arm in {"naive", "areaT"}, "Only four uncompleted SR tasks"
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
    receipt, inventory, verified, _ = asset_gate(args.scene, manifest, args.asset_receipt, inputs)
    fixture = args.fixture_steps > 0
    steps = args.fixture_steps if fixture else 6000
    row = endpoint(plan, "CEM", args.scene, args.arm)
    fixed = profile["engineering_fixture"]
    if fixture:
        assert (args.scene, args.arm, steps) == (fixed["scene"], fixed["arm"], 3)
    out = Path(fixed["out"] if fixture else row["run"])
    metrics = Path(row["metric_dir"])
    assert fixture or not (ROOT / metrics).exists(), "Do not overwrite common CTS metrics"
    assert not (ROOT / out).exists(), "Select a new version; preserve existing runs/fixtures"
    runtime = BATCH / "runtime/cem-native-area4-v1"
    env = Path("envs/dynamic_lr_baselines_20261020/cem")
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
        commands.append(dict(name="shared_CTS_PNG_metrics", argv=["{ROOT}/.venv/bin/python", absolute(OLD / "evaluate_images.py"),
            "--manifest", absolute(manifest), "--render-dir", absolute(out / "renders_hr"), "--out", absolute(metrics),
            "--method", "CEM_area4_" + args.arm]))
    commands.append(dict(name="fixture_acceptance" if fixture else "SR_result_index", argv=["{ROOT}/.venv/bin/python", absolute(HERE / "index_sr_cts_v2.py"),
        "--run", absolute(out), "--manifest", absolute(manifest), "--arm", args.arm, "--parent-run", absolute(parent_run),
        "--fixture-steps", str(args.fixture_steps)]))
    req = shared_required(profile, amendment, activation)
    paths = [OLD / "methods/localdygs_cem/check_gpu_environment.py", receipt, inventory, checkpoint, parent_receipt, manifest, schedule, teacher]
    runtime_inventory = json.loads((ROOT / runtime / "native_runtime_inventory.json").read_text())
    for item in runtime_inventory["files"]:
        path = runtime / item["path"]
        assert sha(ROOT / path) == item["sha256"], "Native runtime inventory mismatch"
        paths.append(path)
    paths += [runtime / "native_runtime_inventory.json", runtime / "protocol_interface_receipt.json"]
    for path in paths:
        req.add(path)
    short = "cook" if args.scene == "cook_spinach" else "cut"
    job_id = fixed["job_id"] if fixture else f"codex-cem-area4-{short}-{args.arm}-ctsg0-20261020-v2".lower()
    job = base_job(profile, amendment, activation, req, out, job_id, args.scene, args.arm, "CEM_area4_SR_CTS_contingency", commands,
                   [dict(path=str(receipt), inventory_sha256=verified["inventory_sha256"])])
    job["env"].update(CEM_PROTOCOL_TRAIN_LEVEL="lr", DYNAMIC_LR_PROTOCOL_MANIFEST=absolute(manifest), LD_LIBRARY_PATH=absolute(env / "cuda-toolkit-overlay/lib64"))
    job.update(return_paths=[str(out)] + ([] if fixture else [str(metrics)]), training_HR=False, parent_run=str(parent_run),
        formal_scientific_metrics=not fixture, SR_actual_updates=steps, SR_absolute_iterations=[50001, 50000+steps],
        fixed_topology=True, parent_native_endpoint=50000, evaluation_targets=[] if fixture else [target(row)],
        comparison_limit="Only remaining naive/areaT train on CTS; matched arms retain A100 training. Shared CTS evaluation does not remove training hardware confounding.")
    gate_job(job)
    return job

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--job", type=Path, required=True)
    p = sub.add_parser("build")
    add_profile_args(p)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--gpu", type=int, choices=[0], default=0)
    p.add_argument("--arm", choices=["naive", "areaT"], required=True)
    p.add_argument("--fixture-steps", type=int, choices=[0, 3], default=0)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--asset-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gate":
        gate_job(json.loads(args.job.read_text()), require_remote=True)
        print(json.dumps(dict(status="verified", profile=PROFILE, GPU_calls=0)))
    else:
        write_job(args.output, build(args))


if __name__ == "__main__":
    main()
