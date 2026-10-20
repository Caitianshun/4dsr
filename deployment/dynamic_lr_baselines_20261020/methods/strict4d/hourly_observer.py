#!/usr/bin/env python3
"""Read-only hourly fallback for the finite strict4d queue.

The observer never submits, retries, signals, evaluates, imports CUDA or edits
remote files. Normal completion/collection still uses the existing exit events.
Local inotify stops this observer when that finite queue reaches a terminal state.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import time


REMOTE_READER = r'''
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
request=json.load(sys.stdin)
root=Path(request["remote_project_root"]).resolve()
result={"utc":datetime.now(timezone.utc).isoformat(),"host":os.uname().nodename,
        "operation":"read_only_hourly_anomaly_snapshot", "CUDA_imported":False,"jobs":[]}
def read_json(path):
    if not path.is_file(): return {"file_absent":True,"path":str(path)}
    try: return json.loads(path.read_text())
    except Exception as error: return {"read_error":repr(error),"path":str(path)}
def tail(path):
    if not path.is_file(): return None
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-32768))
        value=stream.read().decode("utf-8",errors="replace")
    lines=value.replace("\r","\n").splitlines()
    return {"path":str(path),"bytes":path.stat().st_size,"last_lines":lines[-12:],
            "error_lines_in_tail":[line for line in lines if any(word in line.lower()
                 for word in ["traceback", "error", "exception", "out of memory", "killed"])][-20:]}
for item in request["jobs"]:
    run=root/item["run_relative_path"]
    assert run.is_relative_to(root)
    status=read_json(run/"status.json")
    stages=status.get("stages",[])
    row={"job_id":item["job_id"],"method":item["method"],"scene":item["scene"],
         "run":str(run),"status":status,"resources":read_json(run/"train_resources.json")}
    if stages:
        stage=stages[-1]
        row["current_or_last_stage_log_tail"]=tail(run/(stage["name"]+".log"))
        if stage["name"]=="native_train": row["native_training_log_tail"]=tail(run/"native_train.log")
    result["jobs"].append(row)
for key,argv in [("GPU0_inventory",["nvidia-smi","-i",request["gpu_uuid"],
        "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu","--format=csv,noheader"]),
        ("compute_process_inventory",["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"])]:
    try:
        query=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        result[key]={"returncode":query.returncode,"stdout":query.stdout,"stderr":query.stderr}
    except Exception as error: result[key]={"read_error":repr(error)}
print(json.dumps(result,allow_nan=False))
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/"observer.lock").open("a")
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=json.loads(a.ledger.read_text())
    jobs=ledger["jobs"]
    assert jobs and len({item["remote_project_root"] for item in jobs})==1
    assert len({item["gpu_uuid"] for item in jobs})==1
    request={"jobs":jobs,"remote_project_root":jobs[0]["remote_project_root"],"gpu_uuid":jobs[0]["gpu_uuid"]}
    first=datetime.fromisoformat(a.first_utc.replace("Z","+00:00"))
    assert first.tzinfo is not None and a.interval_seconds>=3600
    next_snapshot=first.timestamp()
    record={"status":"waiting_first_hourly_snapshot","started_utc":utc(),"pid":os.getpid(),
            "host":a.host,"ledger":str(a.ledger.resolve()),"queue_status":str(a.queue_status.resolve()),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "first_snapshot_utc":first.isoformat(),"interval_seconds":a.interval_seconds,
            "snapshot_count":0,"GPU_submissions":0,"remote_mutations":0,"automatic_replay":False,
            "chat_notification":False,"completion_path":"existing immediate exit-event collection remains authoritative"}
    write(a.out/"status.json",record)
    library=ctypes.CDLL("libc.so.6",use_errno=True)
    fd=library.inotify_init1(os.O_CLOEXEC)
    if fd<0: raise OSError(ctypes.get_errno())
    watch=library.inotify_add_watch(fd,os.fsencode(a.queue_status.parent),0x100|0x80|0x8|0x4)
    if watch<0: raise OSError(ctypes.get_errno())
    try:
        while True:
            queue=json.loads(a.queue_status.read_text())
            if queue["status"] in ("completed","failed_stopped"):
                record.update(status="stopped_queue_terminal",queue_terminal=queue,stopped_utc=utc())
                write(a.out/"status.json",record)
                return
            if time.time()>=next_snapshot:
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                command="python3 -c "+shlex.quote(REMOTE_READER)
                try:
                    process=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20",a.host,command],
                         input=json.dumps(request),capture_output=True,text=True,timeout=90)
                    snapshot={"utc":utc(),"queue":queue,"ssh_returncode":process.returncode,
                              "stderr":process.stderr,"automatic_retry":False}
                    if process.returncode==0: snapshot["remote"]=json.loads(process.stdout)
                    else: snapshot["stdout"]=process.stdout
                except Exception as error:
                    snapshot={"utc":utc(),"queue":queue,"read_error":repr(error),"automatic_retry":False}
                write(a.out/(stamp+".json"),snapshot)
                record.update(status="waiting_next_hourly_snapshot",last_snapshot_utc=utc(),last_snapshot=stamp+".json",
                              snapshot_count=record["snapshot_count"]+1)
                write(a.out/"status.json",record)
                # A delayed reader does not replay missed intervals.
                next_snapshot=max(next_snapshot+a.interval_seconds,time.time()+a.interval_seconds)
            remaining=max(0,min(60,next_snapshot-time.time()))
            if select.select([fd],[],[],remaining)[0]: os.read(fd,65536)
    finally:
        os.close(fd)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--queue-status",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--host",default="a100-train")
    parser.add_argument("--first-utc",required=True)
    parser.add_argument("--interval-seconds",type=int,default=3600)
    main(parser.parse_args())
