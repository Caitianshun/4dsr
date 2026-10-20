#!/usr/bin/env python3
"""Create a new immutable job identity with common base-environment evaluation.

Native training commands, configurations and runtime files remain unchanged.
The original frozen job is retained and must never have been submitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
COMMON = "deployment/dynamic_lr_baselines_20261020/methods/strict4d"
SHARED = "deployment/dynamic_lr_baselines_20261020"
p = argparse.ArgumentParser()
p.add_argument("--base-job", type=Path, required=True)
p.add_argument("--new-id", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--include-completed-Cook-metrics", action="store_true")
a = p.parse_args()
a.base_job = a.base_job.resolve()
a.output = a.output.resolve()
assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,126}", a.new_id)
job = json.loads(a.base_job.read_text())
assert not (ROOT / SHARED / "dispatch" / job["job_id"] / "intent.json").exists()
assert not a.output.exists()
old_id = job["job_id"]
job["job_id"] = a.new_id
job["replaces_unsubmitted_job"] = {"job_id": old_id, "path": str(a.base_job.relative_to(ROOT)),
                                    "sha256": hashlib.sha256(a.base_job.read_bytes()).hexdigest()}
scratch = "/tmp/4dsr-s4d-" + hashlib.sha256(a.new_id.encode()).hexdigest()[:12]
old_scratch = job["env"]["TMPDIR"]
for key in ["TMPDIR", "TMP", "TEMP"]:
    job["env"][key] = scratch
for command in job["commands"]:
    command["argv"] = [scratch if v == old_scratch else v for v in command["argv"]]


def path(relative):
    return "{ROOT}/" + relative


def evaluation(method, scene, run, label):
    metrics = f"output/dynamic_lr_baselines_20261020/shared_A100_metrics/{method}/{scene}"
    manifest = f"data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed20261007.json"
    command = {"name": "shared_A100_metrics_" + label, "cwd": ".", "argv": [
        path(".venv/bin/python"), path(SHARED + "/evaluate_images.py"), "--manifest", path(manifest),
        "--render-dir", path(run + "/render/renders_hr"), "--out", path(metrics),
        "--method", method, "--device", "cuda"]}
    attach = {"name": "attach_shared_A100_metrics_" + label, "cwd": ".", "argv": [
        path(".venv/bin/python"), path(COMMON + "/attach_shared_metrics.py"),
        "--run", path(run), "--metrics", path(metrics)]}
    return command, attach, metrics


own_eval, own_attach, own_metrics = evaluation(job["method"], job["scene"], job["out"], "endpoint")
native_render_index = next(i for i, c in enumerate(job["commands"]) if c["name"] == "native_hr_render")
job["commands"].insert(native_render_index + 1, own_eval)
job["commands"].append(own_attach)
job["return_paths"].append(own_metrics)
if a.include_completed_Cook_metrics:
    assert job["method"] == "yang" and job["scene"] == "cut_roasted_beef"
    prior_run = "output/dynamic_lr_baselines_20261020/strict4d/runs/yang_cook_spinach_native_v4"
    cook_eval, cook_attach, cook_metrics = evaluation("yang", "cook_spinach", prior_run, "previous_Cook")
    job["commands"] = [cook_eval, cook_attach, *job["commands"]]
    job["return_paths"] += [cook_metrics, prior_run]
for relative in [SHARED + "/evaluate_images.py", "experiments/dynamic_sr_20260918/evaluate.py",
                 COMMON + "/attach_shared_metrics.py"]:
    assert not any(r["path"] == relative for r in job["required_files"])
    job["required_files"].append({"path": relative, "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()})
job["shared_evaluation"] = {"environment": "A100 project base .venv", "device": "same admitted GPU0, serial after native rendering",
                            "current_scene_output": own_metrics, "training_configuration_changed": False,
                            "endpoint_selected_from_heldout_metrics": False}
a.output.write_text(json.dumps(job, indent=2) + "\n")
print(a.output)
