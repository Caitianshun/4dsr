"""On a verified initialization collector exit, submit one frozen Local job.

The parent service watches the existing collector PID exit, not model polls.
Any failure is terminal and recorded. There is no automatic replay or CEM job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def properties(unit):
    raw = subprocess.check_output(["systemctl", "--user", "show", unit, "--property=MainPID,ActiveState,ExecMainStatus"], text=True)
    return dict(line.split("=", 1) for line in raw.splitlines())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initialization-unit", required=True)
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists() or a.job.exists():
        raise SystemExit("Refusing an existing follow-on identity; inspect instead of replaying")
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status="waiting_for_existing_initialization_exit", scene=a.scene,
                 initialization_collector_unit=a.initialization_unit,
                 only_method="localdygs", automatic_failure_replay=False, began_unix=time.time())

    def save():
        a.receipt.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        before = properties(a.initialization_unit)
        state["collector_before_wait"] = before
        save()
        pid = int(before["MainPID"])
        if pid:
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                descriptor = None
            if descriptor is not None:
                try:
                    select.select([descriptor], [], [])
                finally:
                    os.close(descriptor)
        after = properties(a.initialization_unit)
        # systemd may not have committed ExecMainStatus at the exact pidfd edge.
        # Synchronize its terminal job state once, without polling the model.
        if after["ActiveState"] in ("active", "deactivating"):
            subprocess.run(["systemctl", "--user", "is-active", "--quiet", a.initialization_unit], check=False)
            after = properties(a.initialization_unit)
        state["collector_terminal"] = after
        if after["ActiveState"] not in ("inactive", "failed") or after["ExecMainStatus"] != "0":
            raise RuntimeError("Initialization collector did not exit successfully; no training submitted")
        init = ROOT / f"output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/{a.scene}/native_init"
        assert json.loads((init / "complete.json").read_text())["status"] == "completed"
        assert json.loads((init / "camera_initialization_acceptance.json").read_text())["status"] == "passed"
        build = [sys.executable, str(HERE / "build_job.py"), "--scene", a.scene, "--method", "localdygs",
                 "--skip-initialization", "--input-receipt", "deployment/dynamic_lr_baselines_20261020/input_verified.json",
                 "--output", str(a.job)]
        subprocess.run(build, cwd=ROOT, check=True)
        state.update(status="frozen_local_job", job=str(a.job), job_sha256=hashlib.sha256(a.job.read_bytes()).hexdigest())
        save()
        # Only our interface sources and the newly frozen job are synchronized.
        subprocess.run(["rsync", "-a", str(HERE) + "/", "cts:/home/cts/Project/4DSR/" + str(HERE.relative_to(ROOT)) + "/"], check=True)
        subprocess.run(["rsync", "-a", str(a.job), "cts:/home/cts/Project/4DSR/deployment/dynamic_lr_baselines_20261020/jobs/"], check=True)
        state["status"] = "broker_execution_and_exit_artifact_return"
        save()
        subprocess.run([sys.executable, str(HERE.parents[1] / "submit_remote.py"), "--host", "cts", "--gpu-index", "0",
                        "--job", str(a.job), "--memory-mib", "21000", "--timeout-seconds", "172800"], cwd=ROOT, check=True)
        index = ROOT / f"output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/{a.scene}/localdygs/result_index.json"
        assert json.loads(index.read_text())["status"] == "completed"
        state.update(status="completed_and_artifacts_returned", result_index=str(index),
                     result_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    except BaseException:
        state.update(status="failed_no_replay", error=traceback.format_exc())
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["began_unix"]
        save()


if __name__ == "__main__":
    main()
