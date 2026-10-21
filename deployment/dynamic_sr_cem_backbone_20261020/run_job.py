"""Run frozen CEM area4 controls under the existing remote GPU broker.

Training, rendering, evaluation and indexing are consecutive commands. A
nonzero exit records failure and stops; no automatic replay or GPU polling.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            result.update(data)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    source = args.job.resolve()
    job = json.loads(source.read_text())
    assert job["protocol"] == "CEM_area4_backbone_control_v1"
    assert job["input_formation"] == "area4" and job["heldout_used_for_training"] is False
    assert job["arm"] in {"native_parent", "native_HRref", "areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle", "wu_reference"}
    context = json.loads((Path(os.environ["CLAUDE_REMOTE_JOB_DIR"]) / "execution_context.json").read_text())
    assert context["project_id"] == "4dsr" and context["selected_gpu_uuids"] == [job["gpu_uuid"]]
    out = ROOT / job["out"]
    assert out.resolve().is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, out / "job.json")
    record = dict(schema="CEM_area4_remote_execution_v1", job_id=job["job_id"], method=job["method"],
                  scene=job["scene"], arm=job["arm"], input_formation="area4", heldout_used_for_training=False,
                  host=os.uname().nodename, job_sha256=sha(source), gpu_uuid=job["gpu_uuid"],
                  broker_context=context, started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    began, locks = time.monotonic(), []
    try:
        for entry in job["required_files"]:
            path = ROOT / entry["path"]
            assert path.resolve().is_relative_to(ROOT)
            assert sha(path) == entry["sha256"], "Frozen identity mismatch: " + entry["path"]
        for item in job.get("input_receipts", []):
            receipt = json.loads((ROOT / item["path"]).read_text())
            assert receipt["status"] == "verified" and receipt["inventory_sha256"] == item["inventory_sha256"]
        if (ROOT / "activate_cts.sh").exists():
            shared = ROOT / "deployment/cts_20260921/project_access.lock"
            for path, mode in [(shared, fcntl.LOCK_SH), (shared.with_name("gpu0.lock"), fcntl.LOCK_EX)]:
                descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
                locks.append(descriptor)
            assert not (ROOT / "deployment/cts_20260921/sync_pending.json").exists(), "Unverified CTS shared sync"
        environment = os.environ.copy()
        environment.update({k: str(v).replace("{ROOT}", str(ROOT)) for k, v in job.get("env", {}).items()})
        environment.setdefault("CUDA_HOME", str(ROOT / "tools/cuda-12.8-minimal"))
        environment.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
        environment.setdefault("MPLBACKEND", "Agg")
        environment.setdefault("PYTHONFAULTHANDLER", "1")
        environment["PATH"] = environment["CUDA_HOME"] + "/bin:" + environment.get("PATH", "/usr/bin:/bin")
        environment["LD_LIBRARY_PATH"] = ":".join([str(ROOT / "tools/runtime-libs"),
            environment["CUDA_HOME"] + "/lib64", environment.get("LD_LIBRARY_PATH", "")])
        for command in job["commands"]:
            argv = [str(value).replace("{ROOT}", str(ROOT)) for value in command["argv"]]
            cwd = ROOT / command.get("cwd", ".")
            assert cwd.resolve().is_relative_to(ROOT)
            stage = dict(name=command["name"], argv=argv, cwd=str(cwd), started_utc=datetime.now(timezone.utc).isoformat())
            record["stages"].append(stage)
            write(out / "status.json", dict(record, status="running", elapsed_seconds=time.monotonic()-began))
            before = time.monotonic()
            with (out / (command["name"] + ".log")).open("x") as stream:
                process = subprocess.run(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
            stage.update(returncode=process.returncode, seconds=time.monotonic()-before)
            write(out / "status.json", dict(record, status="stage_completed" if process.returncode == 0 else "failed",
                                           elapsed_seconds=time.monotonic()-began))
            assert process.returncode == 0, "Stage failed: " + command["name"] + " rc=" + str(process.returncode)
        result = dict(record, status="completed", elapsed_seconds=time.monotonic()-began,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(out / "status.json", result)
        write(out / "complete.json", result)
    except BaseException:
        write(out / "status.json", dict(record, status="failed", error=traceback.format_exc(),
                                       elapsed_seconds=time.monotonic()-began))
        raise
    finally:
        for descriptor in locks:
            os.close(descriptor)


if __name__ == "__main__":
    main()
