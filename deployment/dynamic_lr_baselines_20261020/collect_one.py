"""Persistent one-job submission/collection with a truthful terminal receipt.

This wrapper never retries. A reattachment is an explicitly selected operation
against the exact existing broker request, without submitting another GPU job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", choices=["a100-train", "cts"], required=True)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--memory-mib", type=int, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--reattach", action="store_true")
    a = p.parse_args()
    assert not a.receipt.exists() and not a.log.exists(), "Preserve an existing attempt"
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.log.parent.mkdir(parents=True, exist_ok=True)
    job = json.loads(a.job.read_text())
    state = dict(status="running", job_id=job["job_id"], host=a.host, gpu_index=a.gpu_index,
                 job=str(a.job.resolve()), job_sha256=hashlib.sha256(a.job.read_bytes()).hexdigest(),
                 started_utc=datetime.now(timezone.utc).isoformat(), reattach_only=a.reattach,
                 automatic_replay=False)
    save(a.receipt, state)
    try:
        argv = [sys.executable, "-u", str(Path(__file__).with_name("submit_remote.py")),
                "--host", a.host, "--gpu-index", str(a.gpu_index), "--job", str(a.job.resolve()),
                "--memory-mib", str(a.memory_mib), "--timeout-seconds", "172800"]
        if a.reattach:
            argv.append("--reattach")
        with a.log.open("x") as log:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["collector_returncode"] = result.returncode
        collected = Path(__file__).parent / "dispatch" / job["job_id"] / "collected.json"
        if collected.is_file():
            state["collection"] = json.loads(collected.read_text())
        terminal = state.get("collection", {}).get("remote_terminal", {})
        assert result.returncode == 0 and terminal.get("outcome") == "success", "Task or collection did not succeed; inspect the preserved log/terminal state"
        state["status"] = "completed_and_artifacts_returned"
    except BaseException:
        state.update(status="failed_no_automatic_retry", error=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(a.receipt, state)


if __name__ == "__main__":
    main()
