"""Wait for the existing successful Local collector, then submit CEM once."""
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


def properties(unit):
    result = subprocess.check_output(["systemctl", "--user", "show", unit,
        "--property=MainPID,ActiveState,ExecMainStatus"], text=True)
    return dict(line.split("=", 1) for line in result.splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-unit", required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--job-sha256", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists():
        raise SystemExit("Refusing an existing follow-on identity; no replay")
    state = dict(status="waiting_for_existing_Local_collector_exit", local_unit=a.local_unit,
        job=str(a.job), job_sha256=a.job_sha256, automatic_failure_replay=False, began_unix=time.time())
    state.update(source=str(Path(__file__).resolve()), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        a.receipt.write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        assert hashlib.sha256(a.job.read_bytes()).hexdigest() == a.job_sha256
        before = properties(a.local_unit)
        state["local_service_before_wait"] = before
        save()
        pid = int(before["MainPID"])
        if pid:
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                fd = None
            if fd is not None:
                try:
                    select.select([fd], [], [])
                finally:
                    os.close(fd)
        local = ROOT / "output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/cook_spinach/localdygs/result_index.json"
        assert json.loads(local.read_text())["status"] == "completed"
        remote_check = """import fcntl,json,os,subprocess
from pathlib import Path
root=Path('/home/cts/Project/4DSR')
p=root/'.claude-training/jobs/codex-dynlr-localdygs-cts0-cook-spinach-train-20261020/status.json'
terminal=json.loads(p.read_text())
assert terminal.get('state')=='ended' and terminal.get('outcome')=='success'
fd=os.open(root/'deployment/cts_20260921/gpu0.lock',os.O_RDONLY|os.O_NOFOLLOW)
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
finally:os.close(fd)
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not compute,compute
print(json.dumps({'status':'passed','existing_Local_broker_outcome':'success','GPU0_lock_released':True,'CUDA_compute_processes':compute}))
"""
        gate = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "cts", "/usr/bin/python3 -"],
            input=remote_check, text=True, capture_output=True, timeout=60, check=True)
        state.update(status="Local_completed_and_GPU_released", pre_CEM_gate=json.loads(gate.stdout))
        save()
        # The root broker repeats resource admission. No canonical helper or
        # running Local file is changed by this watcher.
        state["status"] = "CEM_broker_execution_and_exit_artifact_return"
        save()
        subprocess.run([sys.executable, str(ROOT / "deployment/dynamic_lr_baselines_20261020/submit_remote.py"),
            "--host", "cts", "--gpu-index", "0", "--job", str(a.job), "--memory-mib", "21000",
            "--timeout-seconds", "172800"], cwd=ROOT, check=True)
        index = ROOT / "output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/cook_spinach/cem/result_index.json"
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
