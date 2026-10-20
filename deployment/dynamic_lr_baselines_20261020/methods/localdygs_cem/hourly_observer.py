"""Observe one already authorized finite queue without dispatch or retries.

The existing collector and follow-on services own all completion actions. This
CPU observer waits on the follow-on PID exit and records a bounded read-only
snapshot once per hour as an abnormal-state fallback. It never starts CUDA,
replays a job, kills a process, changes a lock, or sends a chat notification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parents[4]
REMOTE_ROOTS = {"cts": "/home/cts/Project/4DSR", "a100-train": "/home/ubuntu/3DGS/4dsr"}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {"unavailable": type(error).__name__, "path": str(path)}


def service_state(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=Id,MainPID,ActiveState,SubState,ExecMainStatus,ExecMainCode,Result", "--no-pager"],
        text=True, capture_output=True, timeout=20)
    return dict(properties=dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line),
                returncode=result.returncode, stderr_tail=result.stderr[-4096:])


def remote_snapshot(host, jobs):
    # Only frozen identity metadata is embedded in Python source. No shell
    # interpolation, remote mutation, GPU inventory, or model import is used.
    script = """import json
from pathlib import Path
root=Path(REMOTE_ROOT)
jobs=JOBS
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError) as error:return {'unavailable':type(error).__name__,'path':str(path)}
def tail(path):
 try:
  with path.open('rb') as stream:
   stream.seek(max(0,path.stat().st_size-8192))
   return stream.read().decode('utf-8',errors='replace')[-4096:]
 except FileNotFoundError:return None
records=[]
for job in jobs:
 state=read(root/job['out']/'status.json')
 broker=read(root/'.claude-training/jobs'/job['job_id']/'status.json')
 record={'job_id':job['job_id'],'author_queue_status':state,'remote_broker_terminal_or_running':broker}
 failed=state.get('status')=='failed' or broker.get('state')=='failed' or broker.get('outcome') in ['failed','error']
 if failed:
  stages=state.get('stages',[])
  if stages:record['last_stage_error_tail']=tail(root/job['out']/(stages[-1]['name']+'.log'))
 records.append(record)
print(json.dumps(records))
""".replace("REMOTE_ROOT", repr(REMOTE_ROOTS[host])).replace("JOBS", repr(jobs))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "/usr/bin/python3 -"],
                            input=script, text=True, capture_output=True, timeout=60)
    if result.returncode:
        return dict(status="read_failed_no_retry", returncode=result.returncode, stderr_tail=result.stderr[-4096:])
    try:
        return dict(status="read_only_snapshot", jobs=json.loads(result.stdout))
    except json.JSONDecodeError:
        return dict(status="invalid_read_no_retry", stdout_tail=result.stdout[-4096:], stderr_tail=result.stderr[-4096:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=REMOTE_ROOTS, required=True)
    p.add_argument("--units", nargs="+", required=True)
    p.add_argument("--terminal-unit", required=True)
    p.add_argument("--jobs", nargs="+", type=Path, required=True)
    p.add_argument("--follow-on-receipt", type=Path, required=True)
    p.add_argument("--first-check-utc", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--snapshots", type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists() or a.snapshots.exists():
        raise SystemExit("Refusing an existing observer identity; inspect instead of replacing it")
    if a.terminal_unit not in a.units:
        raise SystemExit("Terminal service must be one of the explicitly observed services")
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.snapshots.parent.mkdir(parents=True, exist_ok=True)
    first = datetime.fromisoformat(a.first_check_utc).timestamp()
    state = dict(status="watching_existing_finite_queue", started_utc=datetime.now(timezone.utc).isoformat(),
        source=str(Path(__file__).resolve()), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host=a.host, units=a.units, terminal_unit=a.terminal_unit, interval_seconds=3600,
        first_check_utc=a.first_check_utc, snapshots=str(a.snapshots),
        read_only_CPU_observer=True, automatic_retries=False, GPU_dispatch=False, chat_notification=False,
        snapshot_count=0)
    write(a.receipt, state)

    def snapshot(reason):
        jobs, local_dispatch = [], []
        for path in a.jobs:
            job = read_json(path)
            if "job_id" not in job:
                local_dispatch.append(dict(job_source=str(path), not_frozen_yet=job))
                continue
            assert not Path(job["out"]).is_absolute() and ".." not in Path(job["out"]).parts
            jobs.append({key: job[key] for key in ["job_id", "out"]})
            dispatch = ROOT / "deployment/dynamic_lr_baselines_20261020/dispatch" / job["job_id"]
            local_dispatch.append(dict(job_id=job["job_id"], source=str(path),
                broker_transport=read_json(dispatch / "transport_terminal.json"),
                remote_terminal=read_json(dispatch / "remote_terminal.json"),
                collection=read_json(dispatch / "collected.json")))
        try:
            remote = remote_snapshot(a.host, jobs)
        except Exception:
            remote = dict(status="read_failed_no_retry", error=traceback.format_exc())
        value = dict(utc=datetime.now(timezone.utc).isoformat(), reason=reason,
            services={unit: service_state(unit) for unit in a.units},
            follow_on_receipt=read_json(a.follow_on_receipt), local_dispatch=local_dispatch, remote=remote)
        with a.snapshots.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        state.update(snapshot_count=state["snapshot_count"] + 1, last_snapshot_utc=value["utc"], last_snapshot_reason=reason)
        write(a.receipt, state)

    descriptor = None
    try:
        terminal = service_state(a.terminal_unit)["properties"]
        pid = int(terminal.get("MainPID", "0"))
        if pid:
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                pass
        next_check = first
        if descriptor is not None:
            while True:
                timeout = max(0, next_check - time.time())
                if select.select([descriptor], [], [], timeout)[0]:
                    break
                snapshot("hourly_abnormal_state_fallback")
                next_check += 3600
                # A late launch never causes multiple immediate snapshots.
                next_check = max(next_check, time.time() + 3600)
        snapshot("follow_on_process_exit_event")
        result = read_json(a.follow_on_receipt)
        state.update(status="completed_and_observer_stopped" if result.get("status") == "completed_and_artifacts_returned"
                     else "failed_or_incomplete_and_observer_stopped", follow_on_terminal=result)
    except BaseException:
        state.update(status="observer_failed_no_retry", error=traceback.format_exc())
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(a.receipt, state)


if __name__ == "__main__":
    main()
