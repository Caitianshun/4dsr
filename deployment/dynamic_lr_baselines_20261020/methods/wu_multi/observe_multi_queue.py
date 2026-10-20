"""Finite read-only hourly fallback for the existing GPU1 queue.

This observer cannot submit, retry, train, clear a lock, or send a message.
The actual queue already chains rendering, evaluation, and collection on exit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
QUEUE = ROOT / "output/dynamic_lr_baselines_20261020/a100_gpu1_queue/status.json"
OUT = ROOT / "output/dynamic_lr_baselines_20261020/a100_gpu1_queue_observer"

REMOTE = '''import json,subprocess,time
from pathlib import Path
root=Path('/home/ubuntu/3DGS/4dsr')
base=root/'output/dynamic_lr_baselines_20261020'
value=dict(checked_unix=time.time(),scenes={})
for name in ['Cook','Cut']:
 folder=base/'a100_multi4d_original_lr'/name
 info={}
 p=folder/'status.json'
 if p.is_file():
  state=json.loads(p.read_text());info.update(status=state.get('status'),seconds=state.get('seconds'),stages=state.get('stages',[]),status_age_seconds=time.time()-p.stat().st_mtime)
 for phase in ['coarse','train']:
  p=folder/phase/'training.jsonl'
  if p.is_file() and p.stat().st_size:
   with p.open('rb') as stream:
    stream.seek(max(0,p.stat().st_size-65536));last=stream.read().splitlines()[-1]
   info[phase]=dict(last_row=json.loads(last),log_age_seconds=time.time()-p.stat().st_mtime)
 value['scenes'][name]=info
for name,args in [('GPU1_readonly',['nvidia-smi','--id=GPU-b79cd3fe-81f0-f449-30be-432e2857e517','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits']),('compute_processes_readonly',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])]:
 result=subprocess.run(args,text=True,capture_output=True,timeout=15)
 value[name]=dict(returncode=result.returncode,output=result.stdout.strip(),error=result.stderr.strip())
print(json.dumps(value))
'''


def append(value):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue-pid", type=int, required=True)
    p.add_argument("--first-check-utc", required=True)
    a = p.parse_args()
    OUT.mkdir(exist_ok=False)
    fd = os.pidfd_open(a.queue_pid)
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    next_check = datetime.fromisoformat(a.first_check_utc).timestamp()
    deadline = time.time() + 86400
    (OUT / "process.json").write_text(json.dumps(dict(pid=os.getpid(),queue_pid=a.queue_pid,
        started_utc=datetime.now(timezone.utc).isoformat(),first_check_utc=a.first_check_utc,
        scope="read-only hourly health; no retry/GPU submission/chat notification"), indent=2)+"\n")
    try:
        while True:
            now = time.time()
            if now >= deadline:
                append(dict(event="finite_observer_deadline",checked_unix=now,queue=json.loads(QUEUE.read_text())))
                break
            wait_ms = max(0,int((min(next_check,deadline)-now)*1000))
            if poll.poll(wait_ms):
                append(dict(event="queue_exit",checked_unix=time.time(),queue=json.loads(QUEUE.read_text())))
                break
            if time.time() < next_check:
                continue
            try:
                result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","a100-train",
                    "python3","-c",shlex.quote(REMOTE)],text=True,capture_output=True,timeout=55)
                event=dict(event="hourly_readonly_health",checked_unix=time.time(),queue=json.loads(QUEUE.read_text()),
                           returncode=result.returncode)
                if result.returncode == 0:
                    event["remote"]=json.loads(result.stdout)
                else:
                    event.update(stderr=result.stderr,stdout=result.stdout)
                append(event)
            except BaseException:
                append(dict(event="observer_health_error",checked_unix=time.time(),error=traceback.format_exc()))
            next_check += 3600
    finally:
        os.close(fd)
        (OUT / "complete.json").write_text(json.dumps(dict(status="observer_stopped",stopped_unix=time.time(),
                      no_retry_or_GPU_submission_performed=True),indent=2)+"\n")


if __name__ == "__main__":
    main()
