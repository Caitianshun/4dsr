"""Return only essential assets of one already submitted A100 native LR parent.

This never submits, retries, probes GPUs, modifies remote files or cancels work.
The original full collector continues independently. Local rsync uses temporary
files and --ignore-existing; existing bytes are subsequently verified, never
replaced. A terminal receipt is written once and is not a full-return receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path("output/dynamic_sr_cem_backbone_20261020")
REMOTE = "/home/ubuntu/3DGS/4dsr"
ALLOWED = {"cook_native_parent_v1.json": "cook_spinach", "cut_native_parent_v2.json": "cut_roasted_beef"}
FILES = ["job.json", "status.json", "complete.json", "stage1.json", "stage2.json", "result_index.json",
         "GPU_environment.json", "metrics/summary.json", "model/chkpnt50000.pth"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def contained(path):
    result = (ROOT / path).resolve()
    assert result.is_relative_to(ROOT) and result != ROOT, "Path must remain within ROOT"
    return result


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # O_EXCL-equivalent final publication, no overwrite.
    finally:
        temporary.unlink()


WATCH = r'''import ctypes,errno,json,os,select,time
from pathlib import Path
p=Path(__PATH__)
deadline=time.monotonic()+86400
libc=ctypes.CDLL(None,use_errno=True)
libc.inotify_init1.argtypes=[ctypes.c_int]
libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
if fd<0:raise OSError(ctypes.get_errno())
watched={}
def watch():
 parent=p.parent
 while not parent.is_dir():parent=parent.parent
 stat=parent.stat();identity=(str(parent),stat.st_dev,stat.st_ino)
 if identity in watched.values():return False
 wd=libc.inotify_add_watch(fd,os.fsencode(parent),0x100|0x80|0x8|0x2|0x4|0x400|0x800)
 if wd<0:
  error=ctypes.get_errno()
  if error in (errno.ENOENT,errno.ENOTDIR):return True
  raise OSError(error)
 watched[wd]=identity;return True
try:
 while True:
  watch()
  try:value=json.loads(p.read_text())
  except (FileNotFoundError,json.JSONDecodeError):value={}
  if value.get('state') in ('ended','failed','completed','rejected','outcome_unknown'):
   print(json.dumps(value),flush=True);break
  if watch():continue
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Native parent remote exit event not received; no replay')
  if select.select([fd],[],[],remaining)[0]:
   try:os.read(fd,65536)
   except BlockingIOError:pass
finally:os.close(fd)
'''


def verify(job, job_path, run, terminal):
    assert terminal["state"] in {"ended", "completed"} and terminal["outcome"] == "success" and terminal["training_exit_code"] == 0
    assert terminal["job_id"] == job["job_id"]
    assert sha(run / "job.json") == sha(job_path), "Returned job does not match the submitted frozen job"
    status = json.loads((run / "status.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    for value in (status, complete):
        assert value["status"] == "completed" and value["job_id"] == job["job_id"]
        assert value["job_sha256"] == sha(job_path) and value["arm"] == "native_parent" and value["scene"] == job["scene"]
        assert len(value["stages"]) == len(job["commands"]) and all(stage["returncode"] == 0 for stage in value["stages"])
    manifest_relative = Path("data/dynamic_sr/full_time_20261007/prepared") / (job["scene"] + "_area4") / "manifest_train_ready_seed20261007.json"
    manifest_sha = sha(ROOT / manifest_relative)
    index = json.loads((run / "result_index.json").read_text())
    assert index["status"] == "native_endpoints_and_300_renders_verified" and index["train_level"] == "lr"
    assert index["manifest_sha256"] == manifest_sha and index["heldout_used_for_training"] is False
    assert index["metrics_status"] == "shared_A100_complete"
    source_hash = next(row["sha256"] for row in job["required_files"] if row["path"] == "experiments/dynamic_sr_cem_area_20261020/native_stage.py")
    runtime_hash = next(row["sha256"] for row in job["required_files"] if row["path"].endswith("runtime/cem-native-area4-v1/protocol_interface_receipt.json"))
    initialization_command = next(command for command in job["commands"] if command["name"] == "native_stage1")["argv"]
    source = initialization_command[initialization_command.index("--source") + 1].replace("{ROOT}/", "")
    initialization_sha = sha(ROOT / source / "colmap_0/sparse/0/points3D.ply")
    for number, endpoint in ((1, 40000), (2, 50000)):
        stage_path = run / f"stage{number}.json"
        state = json.loads(stage_path.read_text())
        assert state["status"] == "completed" and state["stage"] == number and state["endpoint"] == endpoint and state["train_level"] == "lr"
        assert state["manifest"]["sha256"] == manifest_sha and state["native_schedule_preserved"]
        assert state["heldout_pixels_used"] is False and state["train_HR_used"] is False
        assert not state["parent_added_affine"] and not state["parent_added_warp"]
        assert state["entry_sha256"] == source_hash and state["runtime_receipt"]["sha256"] == runtime_hash
        assert state["source"]["LR_initialization_pointcloud_sha256"] == initialization_sha
        row = index["stages"][number - 1]
        assert row["sha256"] == sha(stage_path) and row["checkpoint_sha256"] == state["final_checkpoint"]["sha256"]
        assert state["final_checkpoint"]["path"] == REMOTE + "/" + job["out"] + f"/model/chkpnt{endpoint}.pth"
        if number == 2:
            assert state["author_optimizer_restored"] and state["start_checkpoint"]["sha256"] == index["stages"][0]["checkpoint_sha256"]
    checkpoint = run / "model/chkpnt50000.pth"
    checkpoint_sha = sha(checkpoint)
    stage2 = json.loads((run / "stage2.json").read_text())
    assert checkpoint_sha == stage2["final_checkpoint"]["sha256"] == index["stages"][1]["checkpoint_sha256"]
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary == index["metrics"] and summary["status"] == "completed" and summary["frames"] == 300
    assert summary["manifest_sha256"] == manifest_sha and summary["input_representation"] == "8-bit rendered RGB PNG for every method"
    return dict(manifest=dict(path=str(manifest_relative), sha256=manifest_sha), checkpoint=dict(path=str(checkpoint.relative_to(ROOT)), sha256=checkpoint_sha, bytes=checkpoint.stat().st_size),
                stage2=dict(path=str((run / "stage2.json").relative_to(ROOT)), sha256=sha(run / "stage2.json")),
                native_entry_sha256=source_hash, runtime_receipt_sha256=runtime_hash, LR_initializer_sha256=initialization_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    job_path, receipt = contained(args.job), contained(args.receipt)
    assert not receipt.exists(), "A terminal essential receipt must never be overwritten or replayed"
    began = time.monotonic()
    record = dict(schema="CEM_area4_parent_essential_collection_v1", status="running", started_utc=utc(),
                  source_path=str(Path(__file__).resolve()), source_sha256=sha(__file__), pid=os.getpid(),
                  full_artifact_return_complete=False, collection_scope="essential metadata and final 50000 checkpoint only; original full collector remains responsible for all assets",
                  no_GPU_dispatch=True, no_GPU_queries=True, retry_policy="none")
    try:
        assert job_path.parent == ROOT / BATCH / "jobs" and job_path.name in ALLOWED
        job = json.loads(job_path.read_text())
        assert job["protocol"] == "CEM_area4_backbone_control_v1" and job["arm"] == "native_parent" and job["scene"] == ALLOWED[job_path.name]
        assert job["input_formation"] == "area4" and job["heldout_used_for_training"] is False
        assert receipt == ROOT / BATCH / "controllers" / (job["scene"] + "_parent_essential_v1.json")
        record.update(job_id=job["job_id"], scene=job["scene"], job_path=str(job_path.relative_to(ROOT)), job_sha256=sha(job_path), host="a100-train", remote_root=REMOTE)
        intent = json.loads((Path(__file__).parent / "dispatch" / job["job_id"] / "intent.json").read_text())
        assert intent["job_sha256"] == sha(job_path) and intent["host"] == "a100-train", "Only attach to the exact existing submission"
        required = []
        for row in job["required_files"]:
            path = contained(row["path"])
            assert sha(path) == row["sha256"], "Frozen parent input/source identity changed: " + row["path"]
            required.append(row)
        record["required_files"] = required
        watch = WATCH.replace("__PATH__", repr(REMOTE + "/.claude-training/jobs/" + job["job_id"] + "/status.json"))
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "a100-train", "/usr/bin/python3 -"]
        result = subprocess.run(ssh, input=watch, text=True, capture_output=True, timeout=86500)
        watch_log = receipt.with_name(receipt.stem + ".watch.log")
        with watch_log.open("x") as stream:
            stream.write(result.stdout + result.stderr)
        assert result.returncode == 0, "Remote inotify wait failed; no automatic reconnect or replay"
        terminal = json.loads(result.stdout.strip().splitlines()[-1])
        record["remote_terminal"] = terminal
        assert terminal["job_id"] == job["job_id"]
        assert terminal.get("state") in {"ended", "completed"} and terminal.get("outcome") == "success" and terminal.get("training_exit_code") == 0, "Parent broker terminal is not successful"
        run = contained(job["out"])
        record["returned_files"] = []
        for name in FILES:
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--ignore-existing", "--", "a100-train:" + REMOTE + "/" + job["out"] + "/" + name,
                            str(target)], check=True, timeout=7200)
            record["returned_files"].append(dict(path=str(target.relative_to(ROOT)), sha256=sha(target), bytes=target.stat().st_size))
        record.update(verify(job, job_path, run, terminal), status="parent_essential_returned_verified")
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
    finally:
        record.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - began)
        write_once(receipt, record)
    print(json.dumps({key: record.get(key) for key in ["status", "job_id", "scene", "checkpoint", "error"]}), flush=True)
    return 0 if record["status"] == "parent_essential_returned_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
