"""Freeze a fresh native HR run after the isolated short-save fixture failed.

The original CEM save_ply cannot serialize its still-uninitialized motion
array at iteration five. No native stage1/2 has started in that failed job.
This repair removes that artificial early-save fixture, not native training,
and uses a separate output. The native 40k/50k schedule is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed-job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    source = a.failed_job.resolve()
    job = json.loads(source.read_text())
    assert job["scene"] == "cook_spinach" and job["arm"] == "native_HRref"
    old = ROOT / job["out"]
    state = json.loads((old / "status.json").read_text())
    assert state["status"] == "failed" and len(state["stages"]) == 2
    assert state["stages"][-1]["name"] == "native_HR_startup_fixture"
    assert not (old / "model").exists() and not (old / "stage1.json").exists()
    log = (old / "native_HR_startup_fixture.log").read_text()
    assert "[ITER 5] Saving Gaussians" in log and "Dimension out of range" in log
    assert "self._xyz_motion.detach().flatten(start_dim=1)" in log
    dispatch = ROOT / "deployment/dynamic_sr_cem_backbone_20261020/dispatch" / job["job_id"]
    terminal = json.loads((dispatch / "collected.json").read_text())
    assert terminal["status"] == "collected" and terminal["remote_terminal"]["outcome"] == "failure"
    old_out = job["out"]
    new_out = str(Path(old_out).with_name("cook_spinach_native_HRref_v2"))
    job["out"] = new_out
    job["job_id"] = "codex-cem-area4-cook-native-hrref-ctsg0-20261020-v4"
    job["commands"] = [x for x in job["commands"] if x["name"] != "native_HR_startup_fixture"]
    for command in job["commands"]:
        command["argv"] = [x.replace(old_out, new_out) for x in command["argv"]]
    job["return_paths"] = [new_out]
    job["required_files"].append(dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(__file__)))
    fixture = json.loads((old / "startup_fixture/native_fixture_execution.json").read_text())
    job["short_fixture_failure_disposition"] = dict(
        failed_job_sha256=sha(source), failure_log_sha256=sha(old / "native_HR_startup_fixture.log"),
        failure="native early save of empty one-dimensional motion array at fixture iteration5",
        native_full_stage_updates_before_failure=0,
        observed_native_HR_startup_iterations_before_save=5,
        fixture_peak_Torch_reserved_bytes=fixture["torch_peak_reserved_bytes"],
        evidence_scope="startup HR forward/backward only; no mature memory or convergence guarantee",
        repair="remove artificial early endpoint fixture; fresh independent native original 0->40k->50k run",
        native_source_changed=False, native_training_schedule_changed=False, output_reused=False)
    assert len(job["job_id"]) <= 64 and re.fullmatch(r"[a-z0-9][a-z0-9-]*", job["job_id"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], output=str(a.output), sha256=sha(a.output))))


if __name__ == "__main__":
    main()
