#!/usr/bin/env python3
"""Sequence GPU jobs by verified native exit; return large artifacts in parallel.

Every new job still enters the shared broker exactly once. The existing
submitter/collector continues independently until all registered files return.
This controller never treats remote completion as local artifact collection.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys


VERIFY_REMOTE = r'''
import hashlib,json,subprocess,sys
from pathlib import Path
r=json.load(sys.stdin);root=Path(r["root"]);job=r["job"];run=root/job["out"]
def read(path): return json.loads(path.read_text())
assert hashlib.sha256((root/r["job_relative_path"]).read_bytes()).hexdigest()==r["job_sha256"]
state=read(run/"status.json");index=read(run/"result_index.json");render=read(run/"render/summary.json")
assert state["job_id"]==job["job_id"] and state["status"]=="completed"
assert state["job_sha256"]==r["job_sha256"]
assert index["status"]=="complete_native_training_and_300_HR_PNGs" and index["checkpoint"]["iteration"]==30000
assert index["resources"]["status"]=="complete"
assert index["manifest"]["sha256"]==hashlib.sha256(Path(index["manifest"]["path"]).read_bytes()).hexdigest()
assert render["finite_and_shape_guard_before_clamp_and_quantization"]=={"status":"passed","frames":300,"shape":[3,1008,1344]}
assert {p.name for p in (run/"render/renders_hr").glob("*.png")}=={f"{f:04d}.png" for f in range(300)}
if job.get("shared_evaluation"):
 assert index["metrics_status"]=="complete_shared_A100_base_evaluator_on_300_RGB_PNGs"
 assert index["shared_metrics"]["frames"]==300
query=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],capture_output=True,text=True,check=True,timeout=20)
owned_gpu=[line.strip() for line in query.stdout.splitlines() if line.split(",")[0].strip()==job["gpu_uuid"]]
unexpected=[line for line in owned_gpu if not (line.split(",")[1].strip()=="971823" and line.split(",")[2].strip()=="/usr/libexec/gnome-remote-desktop-daemon")]
assert not unexpected,unexpected
print(json.dumps({"status":"verified_remote_endpoint_success_not_local_collection","job_id":job["job_id"],"job_sha256":r["job_sha256"],"native_status":state,"result_index":index,"render_summary":render,"GPU_compute_inventory":owned_gpu,"no_training_GPU_residual":True,"used_HR_pixels":False}))
'''


def utc(): return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def wait_terminal(path,process=None):
    library=ctypes.CDLL("libc.so.6",use_errno=True)
    fd=library.inotify_init1(os.O_CLOEXEC)
    if fd<0: raise OSError(ctypes.get_errno())
    watched=None;pidfd=os.pidfd_open(process.pid) if process else None
    try:
        while True:
            parent=path.parent
            while not parent.exists():parent=parent.parent
            if parent!=watched:
                if watched is not None:library.inotify_rm_watch(fd,watch)
                watch=library.inotify_add_watch(fd,os.fsencode(parent),0x100|0x80|0x8)
                if watch<0:raise OSError(ctypes.get_errno())
                watched=parent
            if path.is_file():
                terminal=json.loads(path.read_text())
                if terminal.get("outcome")!="success":raise RuntimeError("Existing native job terminal failed/unknown; never replay")
                return terminal
            if process and process.poll() is not None:
                raise RuntimeError("Submit/watch process exited before a confirmed remote terminal; inspect existing identity, never replay")
            ready=select.select([fd,*([pidfd] if pidfd is not None else [])],[],[])[0]
            if fd in ready:os.read(fd,65536)
    finally:
        os.close(fd)
        if pidfd is not None:os.close(pidfd)


def main(a):
    root=Path(__file__).resolve().parents[4];shared=root/"deployment/dynamic_lr_baselines_20261020"
    state={"status":"verifying_previous_remote_success","started_utc":utc(),"initial_job":str(a.initial_job),
           "jobs":[str(p) for p in a.jobs],"gpu_index":a.gpu_index,"automatic_replay":False,
           "remote_endpoints_complete":[],"artifact_collectors":[],"reassigned":[],
           "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    collectors=[]
    def save():state["updated_utc"]=utc();write(a.status,state)
    def verify(path,process=None):
        job=json.loads(path.read_text());directory=shared/"dispatch"/job["job_id"]
        terminal=wait_terminal(directory/"remote_terminal.json",process)
        assert terminal["job_id"]==job["job_id"]
        request={"root":"/home/ubuntu/3DGS/4dsr","job":job,"job_relative_path":str(path.relative_to(root)),"job_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20","a100-train","python3 -c "+shlex.quote(VERIFY_REMOTE)],input=json.dumps(request),capture_output=True,text=True,timeout=60)
        if result.returncode:raise RuntimeError("Remote endpoint CPU verification failed: "+result.stderr)
        receipt=json.loads(result.stdout);receipt["utc"]=utc();receipt["remote_terminal"]=terminal
        dest=a.receipts/(job["job_id"]+"_remote_endpoint_ready.json");write(dest,receipt)
        return dest
    save()
    try:
        state["initial_remote_success_receipt"]=str(verify(a.initial_job));save()
        for path in a.jobs:
            job=json.loads(path.read_text())
            reassignment=json.loads(a.reassign_file.read_text()) if a.reassign_file.exists() else {"jobs":[]}
            row=next((r for r in reassignment["jobs"] if r["job_id"]==job["job_id"]),None)
            if row:
                assert not (shared/"dispatch"/job["job_id"]/"intent.json").exists()
                state["reassigned"].append(row);save();continue
            assert not (shared/"dispatch"/job["job_id"]).exists(),"An existing identity must never be automatically submitted again"
            state.update(status="submitting_and_waiting_remote_exit_event",current_job=job["job_id"]);save()
            log=(a.receipts/(job["job_id"]+"_submit_collect.log")).open("x")
            argv=[sys.executable,str(shared/"submit_remote.py"),"--host","a100-train","--gpu-index",str(a.gpu_index),"--job",str(path),"--memory-mib",str(a.memory_mib)]
            process=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            collectors.append((job["job_id"],process,log))
            state["artifact_collectors"].append({"job_id":job["job_id"],"pid":process.pid,"operation":"original submitter and registered artifact collection; no resubmission"});save()
            receipt=verify(path,process)
            state["remote_endpoints_complete"].append({"job_id":job["job_id"],"receipt":str(receipt)});save()
        state.update(status="remote_endpoints_complete_collecting",current_job=None);save()
        for job_id,process,log in collectors:
            code=process.wait();log.close()
            if code:raise RuntimeError("Artifact collector failed for "+job_id+"; native success remains recorded, never replay")
        state["status"]="completed";save()
    except BaseException as error:
        state.update(status="failed_stopped",error=repr(error));save()
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--initial-job",type=Path,required=True)
    p.add_argument("--jobs",type=Path,nargs="+",required=True)
    p.add_argument("--status",type=Path,required=True)
    p.add_argument("--receipts",type=Path,required=True)
    p.add_argument("--reassign-file",type=Path,required=True)
    p.add_argument("--gpu-index",type=int,choices=[0,1],default=0)
    p.add_argument("--memory-mib",type=int,default=37000)
    main(p.parse_args())
