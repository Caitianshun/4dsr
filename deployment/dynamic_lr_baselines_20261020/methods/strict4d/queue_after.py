#!/usr/bin/env python3
"""Wait for successful collection, then submit next frozen jobs serially.

Waits use filesystem exit events, never model polling or admission retries.
Every next GPU operation enters the shared broker again. Any unknown/failure
outcome stops the chain and leaves a persistent status record.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def wait_file(path):
    library=ctypes.CDLL("libc.so.6",use_errno=True)
    fd=library.inotify_init1(os.O_CLOEXEC)
    if fd<0:
        raise OSError(ctypes.get_errno())
    watched=None
    try:
        while True:
            parent=path.parent
            while not parent.exists():
                parent=parent.parent
            if parent!=watched:
                if watched is not None:
                    library.inotify_rm_watch(fd,watch)
                watch=library.inotify_add_watch(fd,os.fsencode(parent),0x100|0x80|0x8)
                if watch<0:
                    raise OSError(ctypes.get_errno())
                watched=parent
            if path.exists():
                return json.loads(path.read_text())
            if select.select([fd],[],[])[0]:
                os.read(fd,65536)
    finally:
        os.close(fd)


def main(a):
    root=Path(__file__).resolve().parents[4]
    shared=root/"deployment/dynamic_lr_baselines_20261020"
    state={"status":"waiting_previous_success","after_collected":str(a.after_collected),
           "jobs":[str(p) for p in a.jobs],"started_utc":datetime.now(timezone.utc).isoformat(),
           "automatic_replay":False,"gpu_index":a.gpu_index,"completed":[]}

    def save():
        a.status.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(a.status)

    save()
    try:
        previous=wait_file(a.after_collected)
        if previous.get("status")!="collected" or previous["remote_terminal"].get("outcome")!="success":
            raise RuntimeError("Previous job failed/unknown; next GPU job will not be submitted")
        for path in a.jobs:
            job=json.loads(path.read_text())
            state.update(status="submitting_and_waiting_on_exit_event",current_job=job["job_id"])
            save()
            argv=[sys.executable,str(shared/"submit_remote.py"),"--host","a100-train","--gpu-index",str(a.gpu_index),
                  "--job",str(path.resolve()),"--memory-mib",str(a.memory_mib)]
            with a.status.with_name(job["job_id"]+"_submit_watch.log").open("x") as log:
                subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            collected=json.loads((shared/"dispatch"/job["job_id"]/"collected.json").read_text())
            if collected["remote_terminal"].get("outcome")!="success":
                raise RuntimeError("Native job failed/unknown; chain stops")
            state["completed"].append(job["job_id"])
            save()
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed_stopped",error=repr(error))
        raise
    finally:
        state["updated_utc"]=datetime.now(timezone.utc).isoformat()
        save()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--after-collected",type=Path,required=True)
    p.add_argument("--jobs",type=Path,nargs="+",required=True)
    p.add_argument("--memory-mib",type=int,default=37000)
    p.add_argument("--gpu-index",type=int,choices=[0,1],default=0)
    p.add_argument("--status",type=Path,required=True)
    main(p.parse_args())
