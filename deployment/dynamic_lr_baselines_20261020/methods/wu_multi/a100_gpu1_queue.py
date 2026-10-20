"""Persistent finite GPU1 queue: Wu evaluation, then native Multi4D training.

Start only after root's final GPU admission and verified common-input receipt.
Submissions happen once. Any job, source, environment or collection failure
stops this queue; it never retries a training job or clears another job's lock.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "deployment/dynamic_lr_baselines_20261020"
OUT = ROOT / "output/dynamic_lr_baselines_20261020/a100_gpu1_queue"


def write(value):
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUT / "status.json")


def wait_multi_cpu_build():
    # The remote atomic JSON completion is an inotify event, not GPU polling.
    code = '''import ctypes,json,os,select,time
from pathlib import Path
p=Path('/home/ubuntu/3DGS/4dsr/output/dynamic_lr_baselines_20261020/runtime/multi_build_status.json')
libc=ctypes.CDLL('libc.so.6',use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno())
wd=libc.inotify_add_watch(fd,os.fsencode(p.parent),0x100|0x80|0x8)
if wd<0:raise OSError(ctypes.get_errno())
deadline=time.monotonic()+7200
try:
 while True:
  if p.exists():
   v=json.loads(p.read_text())
   if v.get('status')=='completed_CPU_build_no_CUDA_initialized':
    print(json.dumps(v));break
   if v.get('status')=='failed':raise RuntimeError(json.dumps(v))
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Private Multi4D CPU build did not finish')
  if select.select([fd],[],[],remaining)[0]:os.read(fd,65536)
finally:os.close(fd)
'''
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "a100-train", "/usr/bin/python3 -"],
                       input=code, text=True, capture_output=True, timeout=7300)
    (OUT / "multi_build_wait.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError("Private Multi4D CPU build failed or its event watcher disconnected")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    record = dict(status="starting", started_unix=time.time(), queue=["wu", "multi4d"], completed=[],
                  policy="finite, once-only broker submissions; exit-driven evaluation/return; failure stops")
    try:
        for method, memory in (("wu", 8000), ("multi4d", 34000)):
            if method == "multi4d":
                record["status"] = "waiting_private_Multi4D_CPU_build"
                write(record)
                record["multi_build"] = wait_multi_cpu_build()
                contract = json.loads((Path(__file__).parent / "multi4d_native/cpu_contract.json").read_text())
                assert contract["status"] == "CPU_native_loss_contract_passed_CUDA_not_tested"
            job_path = BASE / "jobs" / f"a100_gpu1_{method}_original_lr.json"
            job = json.loads(job_path.read_text())
            assert job.get("input_receipt") and job.get("input_inventory_sha256")
            record.update(status="executing_" + method, active_job_id=job["job_id"])
            write(record)
            command = [sys.executable, str(BASE / "submit_remote.py"), "--host", "a100-train", "--gpu-index", "1",
                       "--job", str(job_path), "--memory-mib", str(memory), "--timeout-seconds", "86400"]
            with (OUT / (method + "_dispatch.log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            result = json.loads((ROOT / job["out"] / "complete.json").read_text())
            assert result["status"] == "completed"
            subprocess.run([sys.executable, str(Path(__file__).with_name("update_result_index.py"))], cwd=ROOT, check=True)
            record["completed"].append(dict(method=method, job_id=job["job_id"], collected=True))
        record.update(status="completed", finished_unix=time.time())
        write(record)
    except BaseException:
        record.update(status="failed_no_automatic_replay", error=traceback.format_exc(), finished_unix=time.time())
        write(record)
        raise


if __name__ == "__main__":
    main()
