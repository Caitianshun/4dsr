"""Local event-driven half-rung/seed07 validation supervisor.

This file does not start work on import, --make-spec, or --validate-only. The
root operator starts --spec through a persistent local user service. No gate
failure, interrupted output, or transfer conflict is retried automatically.
Original input/receipt bytes and their recorded paths are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import subprocess
import statistics
import sys
import time
import traceback

from common import DATA, HERE, OUT, ROOT, disk_guard, entry, local, mkdir, read, sha, write

SEED = 20261007
BINDINGS = {
    "long360": dict(condition="l3H_long360", host="a100-train",
                    remote_root="/home/ubuntu/3DGS/4dsr/workspaces/highres_20261011",
                    remote_python="/home/ubuntu/3DGS/4dsr/.venv/bin/python", indices=[0, 1]),
    "deskgames_cube": dict(condition="dgH_cube", host="cts",
                           remote_root="/home/cts/Project/4DSR/workspaces/highres_20261011",
                           remote_python="/home/cts/Project/4DSR/.venv/bin/python", indices=[0]),
}
DEPENDENCIES = ["dynamic_sr_20260918", "dynamic_sr_multiview_footprint_20261007",
                "dynamic_sr_cull_retrain_20261009", "dynamic_sr_modules_20261010",
                "dynamic_sr_native_20261010"]
METRIC_SOURCES = ["experiments/dynamic_sr_detail_supervision_20260924/evaluate.py",
                  "experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py",
                  "experiments/dynamic_sr_same_observation_20260930/frequency.py",
                  "experiments/dynamic_sr_prior_diagnosis_20260929/evaluation_adapter.py"]
SCHEMA = "highres_local_event_orchestrator_v1"
CHECKPOINT_SHA = "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215"
NETWORK_SHA = "1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_identity(pid):
    p = Path("/proc") / str(int(pid))
    # Field22 follows the parenthesized comm, which may contain spaces.
    stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
    import hashlib
    cmdline = (p / "cmdline").read_bytes()
    return dict(pid=int(pid), start_ticks=int(stat[19]),
                cmdline_sha256=hashlib.sha256(cmdline).hexdigest(),
                cmdline=cmdline.replace(b"\0", b" ").decode(errors="replace"))


def wait_local_exit(identity):
    """Bind Linux PID/start identity, then wait for the exit event once."""
    if identity is None:
        return dict(status="receipt_already_available", event_wait=False)
    try:
        fd = os.pidfd_open(identity["pid"])
    except ProcessLookupError:
        return dict(status="process_already_exited_receipt_required", event_wait=False)
    try:
        current = process_identity(identity["pid"])
        if any(current[k] != identity[k] for k in ("pid", "start_ticks", "cmdline_sha256")):
            raise ValueError("Supervisor PID identity changed; never wait on reused PID")
        select.select([fd], [], [])
        return dict(status="supervisor_exit_event_received", event_wait=True)
    finally:
        os.close(fd)


def namespace_inventory():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob("*.py"))}


def safe_local_member(rel):
    rel = Path(rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Only safe project-relative members are accepted")
    p = ROOT / rel
    for q in [p, *p.parents]:
        if q == ROOT:
            break
        if q.is_symlink():
            raise ValueError("Symlink member/ancestor rejected")
    if not p.resolve().is_relative_to(ROOT):
        raise ValueError("Member escapes the project")
    return p


def check_freeze(freeze):
    if namespace_inventory() != freeze["namespace_python"]:
        raise ValueError("Frozen fresh namespace changed, including added/deleted Python files")
    for rel, digest in freeze["inputs"].items():
        if sha(local(rel)) != digest:
            raise ValueError("Frozen input changed: " + rel)
    z1 = read(local(freeze["z1_path"]))
    if z1.get("passed") is not True or z1.get("rung") != "half":
        raise ValueError("Passed half-rung Z1 is required")
    if z1["reprojection"]["median_px"] > .5 or z1["reprojection"]["p90_px"] > 1.5:
        raise ValueError("Z1 geometric threshold differs")
    if z1["NCC"]["median"] < .7 or z1["NCC"]["median"] <= z1["NCC"]["wrong_mapping_median"] + .05:
        raise ValueError("Z1 mapping threshold differs")


def validate_spec(spec):
    if spec.get("schema") != SCHEMA or spec.get("dataset") not in BINDINGS:
        raise ValueError("Unsupported orchestrator schema/dataset")
    b = BINDINGS[spec["dataset"]]
    for k in ("condition", "host", "remote_root", "remote_python"):
        if spec.get(k) != b[k]:
            raise ValueError("Unapproved dataset/host/workspace binding: " + k)
    if spec.get("seed") != SEED or spec.get("rung") != "half":
        raise ValueError("Only half rung and seed07 are authorized")
    gpus = spec.get("gpus", [])
    if [g["index"] for g in gpus] != b["indices"]:
        raise ValueError("Use A100 GPU0/1 for Long360 or CTS GPU0 for DeskGames")
    if len({g["uuid"] for g in gpus}) != len(gpus) or not all(re.fullmatch(r"GPU-[0-9a-fA-F-]+", g["uuid"]) for g in gpus):
        raise ValueError("Each allocated GPU needs a distinct explicit UUID")
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", spec["run_id"]):
        raise ValueError("Unsafe run_id")
    if spec.get("reserve_bytes") != 60 * 10**9 or spec.get("remote_future_gb") != 65:
        raise ValueError("Fixed 60GB reserve and65GB teacher/checkpoint budget required")
    if spec.get("workers") not in (1, 2, 4) or spec.get("cpu_threads") not in (1, 2):
        raise ValueError("Bound CPU preparation concurrency required")
    if not Path(spec["local_python"]).is_absolute() or not Path(spec["local_python"]).is_file():
        raise ValueError("Existing absolute local Python is required; do not install environments")
    out = local(spec["out"])
    if out.parent != OUT / "orchestration" or out.name != spec["run_id"]:
        raise ValueError("Fresh orchestration output must match run_id")
    ident = spec.get("supervisor")
    if ident is not None and "z5_after_samples.py" not in ident.get("cmdline", ""):
        raise ValueError("Only the owned Z5 supervisor may be waited on")
    check_freeze(spec["freeze"])
    return b


def make_spec(args):
    b = BINDINGS[args.dataset]
    zero = OUT / "zero_checks" / args.dataset
    z1, window = zero / "z1_half.json", zero / "window_selection.json"
    cfg = HERE / "datasets" / (args.dataset + ".json")
    run_id = args.run_id or (args.dataset + "_seed07_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower())
    freeze = dict(namespace_python=namespace_inventory(), inputs={str(p.relative_to(ROOT)): sha(p) for p in (z1, window, cfg)}, z1_path=str(z1.relative_to(ROOT)))
    if len(args.gpu_uuid) != len(b["indices"]):
        raise ValueError("Provide one UUID for every fixed GPU index")
    supervisor = process_identity(args.supervisor_pid) if args.supervisor_pid else None
    if supervisor is None and not (zero / "gpu_screen_half" / "complete.json").is_file():
        raise ValueError("Missing completed Z5 receipt; provide its live owned supervisor PID")
    spec = dict(schema=SCHEMA, dataset=args.dataset, **{k: b[k] for k in ("condition", "host", "remote_root", "remote_python")},
                seed=SEED, rung="half", run_id=run_id, out=str((OUT / "orchestration" / run_id).relative_to(ROOT)),
                local_python=args.local_python, workers=args.workers, cpu_threads=args.cpu_threads,
                reserve_bytes=60 * 10**9, remote_future_gb=65,
                gpus=[dict(index=i, uuid=u) for i, u in zip(b["indices"], args.gpu_uuid)],
                supervisor=supervisor, freeze=freeze, created_utc=now(),
                expansion_policy="no automatic retry, full rung, second seed, or module branches")
    validate_spec(spec)
    write(local(args.spec_out), spec)
    return spec


def freeze_screening(spec):
    """Return immutable Z5 binding, or a semantic rejection without training."""
    check_freeze(spec["freeze"])
    zero = OUT / "zero_checks" / spec["dataset"]
    complete_path = zero / "gpu_screen_half" / "complete.json"
    if not complete_path.is_file():
        return None, dict(status="rejected_Z5_missing_or_failed_receipt_no_training", failures=[entry(p) for p in sorted((zero / "gpu_screen_half").glob("failure_*.json"))])
    complete = read(complete_path)
    if complete.get("status") not in ("completed_passed_Z5", "completed_rejected_Z5"):
        raise ValueError("Unexpected Z5 completion status")
    zpath = zero / "gpu_screen_half" / "z5_half.json"
    samples_path = zero / "samples_half" / "samples.json"
    z, s = read(zpath), read(samples_path)
    if complete["result"] != entry(zpath) or z["samples_identity"]["sha256"] != sha(samples_path):
        raise ValueError("Z5 output/sample bytes differ from their completion binding")
    window = read(zero / "window_selection.json")
    cfg = HERE / "datasets" / (spec["dataset"] + ".json")
    if s["dataset_config"] != entry(cfg) or s["dataset"] != spec["dataset"] or z["dataset"] != spec["dataset"]:
        raise ValueError("Z5 dataset configuration identity differs")
    if s["rung"] != "half" or z["rung"] != "half" or s["start"] != window["selected_start"] or s["test_views_used"] or s["teacher_cache"]:
        raise ValueError("Z5 grid/window/diagnostic boundary differs")
    if any(ref != entry(HERE / "zero_checks.py") for ref in (s["source"], z["source"])):
        raise ValueError("Z5 inference/sample source changed")
    if z["frozen"]["network_SHA256"] != NETWORK_SHA or z["frozen"]["checkpoint_SHA256"] != CHECKPOINT_SHA or z["frozen"]["tile"] != 0 or z["frozen"]["precision"] != "float32":
        raise ValueError("Frozen SwinIR recipe differs")
    expected_network = ROOT / "third_party/swinir_frozen_20261010/network_swinir.py"
    expected_weights = expected_network.parent / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    if sha(expected_network) != NETWORK_SHA or sha(expected_weights) != CHECKPOINT_SHA:
        raise ValueError("Pinned local SwinIR source or checkpoint differs")
    keys = [(e["camera"], e["frame"]) for e in s["entries"]]
    metric_keys = [(e["camera"], e["frame"]) for e in z["rows"]]
    if len(keys) != len(set(keys)) or set(keys) != set(metric_keys) or len(metric_keys) != len(keys):
        raise ValueError("Incomplete or duplicated Z5 population")
    camera_count = 32 if spec["dataset"] == "long360" else 20
    expected = {(f"cam{c:02d}", f) for c in range(1, camera_count + 1) for f in (15, 150, 285)}
    if set(keys) != expected:
        raise ValueError("Z5 does not cover every preregistered train camera/frame")
    for e in s["entries"]:
        if e["source_frame"] != s["start"] + e["frame"] or e["frame"] not in (15, 150, 285) or e["camera"] == "cam00":
            raise ValueError("Wrong Z5 train-camera/frame population")
        for k, rel in e["paths"].items():
            if sha(samples_path.parent / rel) != e["sha256"][k]:
                raise ValueError("Z5 sample pixels changed")
    a = z["aggregate"]
    if not all(math.isfinite(v) for v in [a["bicubic_PSNR"], a["SwinIR_PSNR"], *a["band_PSNR_gain"].values()]):
        raise ValueError("Z5 aggregate contains nonfinite values")
    passed = a["bicubic_PSNR"] <= 34 and all(a["band_PSNR_gain"][k] >= 1 for k in ("mid", "high"))
    if passed != z["Z5_pass"] or passed != (complete["status"] == "completed_passed_Z5"):
        raise ValueError("Z5 numeric gate and receipt status disagree")
    inputs = dict(spec["freeze"]["inputs"])
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in (complete_path, zpath, samples_path)})
    frozen = dict(spec["freeze"], inputs=inputs)
    binding = dict(status="passed_Z5" if passed else "rejected_Z5_no_training", z5=entry(zpath), samples=entry(samples_path), completion=entry(complete_path), aggregate=a,
                   rule="bicubic_PSNR<=34 and radial mid/high gains EACH>=1dB; conservative image screening only", freeze=frozen)
    return frozen if passed else None, binding


# A small remote helper is sent as Python code. File content is hashed in1MiB
# blocks; the JSON inventory carries only paths, sizes and digests.
REMOTE_HELPER = r'''
import csv,hashlib,json,os,pathlib,select,shutil,sys,time
root=pathlib.Path(sys.argv[1]).resolve();mode=sys.argv[2];args=json.load(sys.stdin)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def safe(rel):
 p=pathlib.PurePosixPath(rel)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe relative member')
 q=root/rel
 for parent in [q,*q.parents]:
  if parent==root:break
  if parent.is_symlink():raise ValueError('Symlink destination/source rejected')
 if not q.resolve().is_relative_to(root):raise ValueError('Member escapes workspace')
 return q
def guard(needed):
 free=shutil.disk_usage(root).free
 if free-needed<60*10**9:raise RuntimeError('Remote60GB disk reserve would be violated')
 return dict(free_bytes=free,needed_bytes=needed,reserve_bytes=60*10**9)
if mode in ('preflight','verify'):
 missing=[];reused=[]
 for rel,v in args['files'].items():
  q=safe(rel)
  if q.exists():
   if not q.is_file() or q.stat().st_size!=v['bytes'] or sha(q)!=v['sha256']:raise ValueError('Preserve differing remote member '+rel)
   reused.append(rel)
  else:missing.append(rel)
 if mode=='verify' and missing:raise ValueError('Missing synchronized members')
 disk=guard(sum(args['files'][r]['bytes'] for r in missing)+args['future_bytes'])
 print(json.dumps(dict(status=mode+'_accepted',missing=missing,reused_count=len(reused),disk=disk)))
elif mode=='inventory':
 files={}
 for rel in args['paths']:
  p=safe(rel)
  if not p.exists():continue
  for q in ([p] if p.is_file() else sorted(p.rglob('*'))):
   if q.is_symlink():raise ValueError('Symlink result rejected')
   if not q.is_file() or '__pycache__' in q.parts:continue
   if not q.resolve().is_relative_to(root):raise ValueError('Result outside workspace')
   files[str(q.relative_to(root))]=dict(bytes=q.stat().st_size,sha256=sha(q))
 print(json.dumps(dict(files=files,host=os.uname().nodename,captured_unix=time.time())))
elif mode=='resources':
 import subprocess
 def capture():
  def rows(kind,fields):
   text=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
   return [dict(zip(fields,[x.strip() for x in row])) for row in csv.reader(text.splitlines())]
  return dict(gpus=rows('gpu',['index','uuid','name','memory.used','memory.total','utilization.gpu']),processes=rows('compute-apps',['gpu_uuid','pid','process_name','used_gpu_memory']),time_unix=time.time(),load=os.getloadavg())
 states=[capture()];time.sleep(20);states.append(capture());expected=args['gpus']
 for state in states:
  for exp in expected:
   matches=[g for g in state['gpus'] if int(g['index'])==exp['index'] and g['uuid']==exp['uuid']]
   if len(matches)!=1:raise ValueError('GPU index/UUID no longer matches')
   g=matches[0]
   if '5090' in g['name'] or ('A100' not in g['name'] if args['host']=='a100-train' else '3090' not in g['name']):raise ValueError('Wrong GPU model')
   if float(g['memory.used'])>1024 or float(g['utilization.gpu'])>10:raise RuntimeError('GPU no longer idle')
   for p in state['processes']:
    if p['gpu_uuid']!=exp['uuid']:continue
    allowed=args['host']=='a100-train' and exp['index']==0 and pathlib.Path(p['process_name']).name=='gnome-remote-desktop-daemon' and float(p['used_gpu_memory'])<=600
    if not allowed:raise RuntimeError('Existing GPU compute job protected')
 print(json.dumps(dict(status='two_read_idle_acceptance',states=states,disk=guard(args['future_bytes']))))
else:raise ValueError('Unknown helper operation')
'''


class Supervisor:
    def __init__(self, spec):
        self.spec = spec
        self.out = local(spec["out"])
        self.freeze = spec["freeze"]
        self.stage = "created"
        self.remote_started = False
        self.smoke_started = False
        self.formal_returned = False
        self.serial = 0
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ControlMaster=no", "-o", "ControlPath=none", spec["host"]]

    def record(self, name, **value):
        self.serial += 1
        return write(self.out / "events" / f"{self.serial:03d}_{name}.json", dict(stage=self.stage, recorded_utc=now(), **value))

    def frozen(self):
        check_freeze(self.freeze)

    def remote(self, mode, value):
        argv = [self.spec["remote_python"], "-B", "-c", REMOTE_HELPER, self.spec["remote_root"], mode]
        result = subprocess.run([*self.ssh, shlex.join(argv)], input=json.dumps(value), text=True, capture_output=True)
        self.record("remote_" + mode, exit_code=result.returncode, stderr=result.stderr[-8000:])
        if result.returncode:
            raise RuntimeError("Remote " + mode + " failed: " + result.stderr[-2000:])
        return json.loads(result.stdout)

    def cpu(self, name, script, argv, estimated_gb=0):
        self.stage = name
        self.frozen()
        guard = disk_guard(ROOT, estimated_gb)
        cmd = [self.spec["local_python"], "-u", "-B", str(HERE / script), *map(str, argv)]
        log = self.out / "logs" / (name + ".log")
        with log.open("xb") as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
            self.record("started", argv=cmd, pid=proc.pid, disk=guard, completion_trigger="subprocess.wait exit event")
            started = time.monotonic()
            code = proc.wait()
        self.record("returned", exit_code=code, seconds=time.monotonic() - started, log=entry(log))
        self.frozen()
        if code:
            raise RuntimeError(name + " exited " + str(code) + "; preserve outputs, no retry")

    def add_input_freeze(self, paths):
        self.frozen()
        self.freeze = dict(self.freeze, inputs=dict(self.freeze["inputs"]))
        for path in paths:
            self.freeze["inputs"][str(Path(path).relative_to(ROOT))] = sha(path)
        self.record("input_freeze", frozen_inputs=self.freeze["inputs"])

    def collect_inputs(self, condition, task_table):
        paths = [DATA / "prepared" / condition, DATA / "hr_full" / condition,
                 OUT / "registry" / "full_LR_prefixes" / condition / f"seed_{SEED}",
                 OUT / "registry" / "teachers" / condition / "plan.json", task_table,
                 OUT / "zero_checks" / self.spec["dataset"] / "samples_half",
                 OUT / "zero_checks" / self.spec["dataset"] / "gpu_screen_half",
                 OUT / "zero_checks" / self.spec["dataset"] / "z1_half.json",
                 OUT / "zero_checks" / self.spec["dataset"] / "window_selection.json"]
        data_files = {}
        def add(p):
            if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
                raise ValueError("Sync symlinks/outside paths are forbidden")
            safe_local_member(p.relative_to(ROOT))
            if p.is_file() and "__pycache__" not in p.parts:
                data_files[str(p.relative_to(ROOT))] = dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in paths:
            if not p.exists():
                raise FileNotFoundError("Required synchronization source " + str(p))
            for q in ([p] if p.is_file() else sorted(p.rglob("*"))):
                add(q)
        # Source only for old generic dependencies; no old audit JSON/results.
        for directory in [HERE, *[ROOT / "experiments" / n for n in DEPENDENCIES], ROOT / "third_party/4dgs_843d5ac_runtime"]:
            for q in sorted(directory.rglob("*.py")):
                if not any(x in q.parts for x in ("__pycache__", "build", ".git")):
                    add(q)
        for q in (HERE / "datasets").glob("*.json"):
            add(q)
        for q in (ROOT / "third_party/swinir_frozen_20261010").glob("*"):
            if q.is_file() and q.suffix in (".py", ".pth"):
                add(q)
        plan_path = OUT / "registry" / "teachers" / condition / "plan.json"
        plan = read(plan_path)
        # The original frozen generator is a source snapshot, not an old result
        # tree. Retain exactly its bound path; never rewrite its provenance.
        for key, pin in (("generator", "generator_sha256"), ("network", "network_sha256"), ("checkpoint", "checkpoint_sha256")):
            ref = plan[key]
            p = safe_local_member(ref["path"])
            digest = sha(p)
            if digest != ref["sha256"] or digest != plan["frozen_recipe"][pin]:
                raise ValueError("Frozen teacher plan source/weight identity differs: " + key)
            if key == "generator" and p.suffix != ".py":
                raise ValueError("Only the pinned generator source snapshot is allowed")
            add(p)
        extra_sources = [safe_local_member(r) for r in METRIC_SOURCES]
        for p in extra_sources:
            add(p)
        self.add_input_freeze([*extra_sources, safe_local_member(plan["generator"]["path"]), safe_local_member(plan["network"]["path"]), safe_local_member(plan["checkpoint"]["path"])])
        return dict(schema="highres_additive_stream_sync_v1", files=data_files, future_bytes=self.spec["remote_future_gb"] * 10**9,
                    host=self.spec["host"], destination=self.spec["remote_root"], freeze=self.freeze)

    def sync(self, inventory):
        self.stage = "verified_additive_sync"
        self.frozen()
        self.record("sync_inventory", file_count=len(inventory["files"]), total_bytes=sum(v["bytes"] for v in inventory["files"].values()))
        write(self.out / "sync_inputs.json", inventory)
        pre = self.remote("preflight", inventory)
        self.record("sync_preflight_accepted", **pre)
        names = self.out / "sync_missing.txt"
        names.write_text("".join(r + "\n" for r in pre["missing"]))
        if pre["missing"]:
            # --ignore-existing preserves any new conflict created after preflight.
            # Post-transfer SHA validation then detects that race and rejects.
            cmd = ["rsync", "-a", "--no-owner", "--no-group", "--relative", "--ignore-existing", "--files-from=" + str(names),
                   "-e", shlex.join(self.ssh[:-1]), str(ROOT) + "/", self.spec["host"] + ":" + self.spec["remote_root"] + "/"]
            with (self.out / "logs" / "sync.log").open("xb") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("Additive synchronization failed, no retry")
        verified = self.remote("verify", inventory)
        self.record("sync_verified", **verified)
        self.frozen()

    def runner(self, task_table):
        self.stage = "remote_validate_and_launch"
        self.frozen()
        remote = self.spec["remote_root"]
        condition = self.spec["condition"]
        pipeline_rel = f"output/dynamic_sr_highres_20261011/pipelines/{condition}_seed{SEED}_{self.spec['run_id']}"
        self.pipeline_rel = pipeline_rel
        task_rel = str(task_table.relative_to(ROOT))
        base = [self.spec["remote_python"], "-u", "-B", remote + "/experiments/dynamic_sr_highres_20261011/persistent_runner.py", "--tasks", task_rel, "--out", pipeline_rel, "--gpus", *[g["uuid"] for g in self.spec["gpus"]]]
        if self.spec["host"] == "a100-train":
            base += ["--allowed-graphics-name", "gnome-remote-desktop-daemon"]
        subprocess.run([*self.ssh, shlex.join([*base, "--validate-only"])], check=True)
        resources = self.remote("resources", dict(gpus=self.spec["gpus"], host=self.spec["host"], future_bytes=self.spec["remote_future_gb"] * 10**9))
        self.record("resources_verified", **resources)
        self.frozen()
        # A unit name and output directory unique to this frozen spec.
        unit = "highres-headroom-" + self.spec["run_id"]
        logfile = remote + "/output/dynamic_sr_highres_20261011/pipelines/" + self.spec["run_id"] + ".service.log"
        cmd = ["systemd-run", "--user", "--wait", "--unit=" + unit, "--property=WorkingDirectory=" + remote,
               "--property=StandardOutput=append:" + logfile, "--property=StandardError=append:" + logfile,
               "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=OMP_NUM_THREADS=4", *base]
        self.remote_started = True
        self.unit = unit + ".service"
        self.record("remote_dispatch_attempted", argv=cmd, unit=self.unit, completion_trigger="systemd-run --wait native D-Bus unit exit event; remote job persists across SSH loss")
        self.stage = "remote_exit_event_wait"
        result = subprocess.run([*self.ssh, shlex.join(cmd)], capture_output=True, text=True)
        self.record("remote_exit_or_SSH_returned", exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            self.pull()
            raise RuntimeError("Persistent remote service/SSH returned " + str(result.returncode) + "; returned artifacts decide completion; no resubmission")
        return pipeline_rel, logfile

    def startup(self):
        self.stage = "GPU_diagnostic50steps_startup"
        self.frozen()
        spec = self.spec
        remote = spec["remote_root"]
        selected = spec["gpus"][-1]  # A100 GPU1 or CTS GPU0.
        self.smoke_rel = f"output/dynamic_sr_highres_20261011/smoke/{spec['condition']}_seed{SEED}_{spec['run_id']}"
        zero = f"output/dynamic_sr_highres_20261011/zero_checks/{spec['dataset']}"
        argv = [spec["remote_python"], "-u", "-B", remote + "/experiments/dynamic_sr_highres_20261011/gpu_startup.py",
                "--condition", spec["condition"], "--seed", str(SEED), "--gpu", str(selected["index"]), "--uuid", selected["uuid"],
                "--out", self.smoke_rel, "--z5-receipt", zero + "/gpu_screen_half/z5_half.json", "--z5-samples", zero + "/samples_half/samples.json"]
        subprocess.run([*self.ssh, shlex.join([*argv, "--validate-only"])], check=True)
        resources = self.remote("resources", dict(gpus=[selected], host=spec["host"], future_bytes=spec["remote_future_gb"] * 10**9))
        self.record("startup_resources_verified", **resources)
        unit = "highres-startup-" + spec["run_id"]
        logfile = remote + "/output/dynamic_sr_highres_20261011/pipelines/" + spec["run_id"] + ".startup.service.log"
        cmd = ["systemd-run", "--user", "--wait", "--unit=" + unit, "--property=WorkingDirectory=" + remote,
               "--property=StandardOutput=append:" + logfile, "--property=StandardError=append:" + logfile,
               "--setenv=PYTHONDONTWRITEBYTECODE=1", *argv]
        self.frozen()
        self.smoke_started = True
        self.record("startup_dispatch_attempted", argv=cmd, unit=unit + ".service", completion_trigger="systemd-run --wait native D-Bus unit exit event")
        result = subprocess.run([*self.ssh, shlex.join(cmd)], capture_output=True, text=True)
        self.record("startup_exit_or_SSH_returned", exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        self.pull_paths([self.smoke_rel, f"output/dynamic_sr_highres_20261011/pipelines/{spec['run_id']}.startup.service.log"], "startup")
        if result.returncode:
            raise RuntimeError("Startup persistent service/SSH returned " + str(result.returncode) + "; diagnostic failures returned, no formal task started")
        smoke = local(self.smoke_rel)
        complete = read(smoke / "complete.json")
        if complete.get("status") != "completed_diagnostic50steps_and_eval3" or complete.get("eligible_for_formal_parent") is not False or complete.get("diagnostic50steps") is not True:
            raise ValueError("Startup acceptance incomplete; no formal teacher or training allowed")
        cp = local(complete["checkpoint"]["path"])
        if sha(cp) != complete["checkpoint"]["sha256"] or read(local(complete["smoke_override"]["path"])).get("not_eligible_for_formal_parent") is not True:
            raise ValueError("Startup checkpoint hash/diagnostic marker differs")
        write(self.out / "startup_acceptance.json", dict(status="passed_runtime_gradient_and_eval3_only", complete=entry(smoke / "complete.json"), checkpoint=entry(cp), diagnostic_only=True, formal_parent_eligible=False))
        self.frozen()

    def result_paths(self):
        condition = self.spec["condition"]
        return [f"output/dynamic_sr_highres_20261011/runs/{condition}_seed{SEED}",
                f"output/dynamic_sr_highres_20261011/registry/full_SR_refinement/{condition}/seed_{SEED}",
                f"output/dynamic_sr_highres_20261011/registry/teachers/{condition}",
                f"output/dynamic_sr_highres_20261011/registry/schedules/{condition}/seed_{SEED}",
                self.pipeline_rel,
                f"output/dynamic_sr_highres_20261011/pipelines/{self.spec['run_id']}.service.log"]

    def pull(self):
        result = self.pull_paths(self.result_paths(), "formal")
        self.formal_returned = True
        return result

    def pull_paths(self, paths, label):
        self.stage = "return_artifacts"
        self.frozen()
        inventory = self.remote("inventory", dict(paths=paths))
        if not inventory["files"]:
            raise RuntimeError("No remote completion/failure artifacts to return")
        inventory_path = self.out / ("remote_result_inventory.json" if label == "formal" else label + "_result_inventory.json")
        write(inventory_path, inventory)
        needed = 0
        for rel, ref in inventory["files"].items():
            p = safe_local_member(rel)
            if not p.is_relative_to(OUT) or p.is_symlink():
                raise ValueError("Result path outside fresh output namespace")
            if p.exists():
                if not p.is_file() or p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
                    raise ValueError("Preserve differing local result " + rel)
            else:
                needed += ref["bytes"]
        disk = disk_guard(ROOT, needed / 10**9)
        names = self.out / (label + "_pull_files.txt")
        names.write_text("".join(r + "\n" for r in inventory["files"]))
        cmd = ["rsync", "-a", "--no-owner", "--no-group", "--relative", "--ignore-existing", "--files-from=" + str(names),
               "-e", shlex.join(self.ssh[:-1]), self.spec["host"] + ":" + self.spec["remote_root"] + "/", str(ROOT) + "/"]
        with (self.out / "logs" / (label + "_pull.log")).open("xb") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Result return failed, remote artifacts preserved")
        for rel, ref in inventory["files"].items():
            p = safe_local_member(rel)
            if p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
                raise ValueError("Returned result hash differs " + rel)
        checkpoints = []
        for rel, ref in inventory["files"].items():
            if not rel.endswith(".pt"):
                continue
            cp = safe_local_member(rel)
            complete = cp.parent / "complete.json"
            completed = complete.is_file() and read(complete).get("status") == "completed"
            checkpoints.append(dict(local_path=str(cp), remote_path=self.spec["remote_root"] + "/" + rel,
                                    host=self.spec["host"], condition=self.spec["condition"], seed=SEED,
                                    task=cp.parent.name, status="task_completed_checkpoint_returned" if completed else "partial_or_failed_task_checkpoint_preserved",
                                    training_completion=entry(complete) if complete.is_file() else None, **ref))
        index_path = self.out / ("checkpoint_index.json" if label == "formal" else label + "_checkpoint_index.json")
        write(index_path, dict(status="returned_bytes_SHA_verified", checkpoints=checkpoints, inventory=entry(inventory_path), diagnostic_only=label=="startup"))
        self.record("return_verified", files=len(inventory["files"]), checkpoints=len(checkpoints), disk=disk)
        return inventory

    def summarize(self):
        self.stage = "headroom_analysis"
        self.frozen()
        pipeline = local(self.pipeline_rel)
        if not (pipeline / "complete.json").is_file():
            raise RuntimeError("Remote DAG did not complete; returned failures remain authoritative")
        if read(pipeline / "complete.json")["status"] != "awaiting_review":
            raise ValueError("Unexpected DAG completion status")
        condition = self.spec["condition"]
        key = f"{condition}_seed{SEED}"
        runs = OUT / "runs" / key
        arms = {a: str(runs / ("eval_LRparent17k" if a == "LRparent" else "eval_" + a)) for a in ("LRparent", "lam0", "base", "oracleHR", "SRrecon", "HRref")}
        for name, path in arms.items():
            d = Path(path)
            summary = read(d / "summary.json")
            if summary.get("status") != "completed" or summary.get("frames") != 300:
                raise ValueError("Full300 evaluation missing for " + name)
            run = read(d / "run.json")
            checkpoint = local(run["checkpoint"]["path"])
            training = read(checkpoint.parent / "run.json")
            if training.get("seed") != SEED or training.get("scene") != condition or sha(checkpoint) != run["checkpoint"]["sha256"]:
                raise ValueError("Arm seed/condition/checkpoint binding differs")
        fitting = self.summarize_prior_fit(runs)
        hspec = dict(condition=condition, seed=SEED, arms=arms, protocol=dict(scope="one seed half-rung development screening, six endpoints",
                     common_window=entry(OUT / "zero_checks" / self.spec["dataset"] / "window_selection.json"),
                     Z5=entry(OUT / "zero_checks" / self.spec["dataset"] / "gpu_screen_half" / "z5_half.json"),
                     budgets=dict(LRparent=17000, SRrecon=17000, HRref=17000, lam0=23000, base=23000, oracleHR=23000),
                     extras_scope="cam00 stride10 ROI30frames; main full300; training/evaluation hardware recorded separately",
                     hardware=self.spec["gpus"], host=self.spec["host"], no_geometric_or_mathematical_bound_claim=True))
        hspec["protocol"]["train_view_prior_fit"] = entry(fitting)
        target = self.out / "headroom_spec.json"
        write(target, hspec)
        result = self.out / "headroom.json"
        self.cpu("headroom", "headroom.py", ["--spec", target, "--out", result])
        write(self.out / "complete.json", dict(status="completed_screening_awaiting_review", headroom=entry(result), train_view_prior_fit=entry(fitting), startup_acceptance=entry(self.out / "startup_acceptance.json"), checkpoint_index=entry(self.out / "checkpoint_index.json"), pipeline=entry(pipeline / "complete.json"), ended_utc=now(),
              final_dataset_selection=False, automatic_second_seed=False, automatic_module_training=False))

    def summarize_prior_fit(self, runs):
        arms = {}
        identities = {}
        population = None
        for arm in ("LRparent17k", "SRrecon", "HRref", "base"):
            directory = runs / ("prior_fit_" + arm)
            completion = read(directory / "prior_fit_complete.json")
            if completion.get("status") != "completed_train_view_prior_fit" or completion.get("seed") != SEED or completion.get("scene") != self.spec["condition"] or completion.get("heldout_pixels_used") is not False or completion.get("parameter_updates") != 0:
                raise ValueError("Prior-fit receipt role/seed/condition differs")
            for name in ("metrics", "csv", "observation_index"):
                p = local(completion[name]["path"])
                if sha(p) != completion[name]["sha256"]:
                    raise ValueError("Prior-fit returned artifact identity differs")
            with local(completion["csv"]["path"]).open() as stream:
                rows = list(csv.DictReader(stream))
            keys = [(r["camera"], int(r["frame"])) for r in rows]
            expected = {tuple(k) for k in completion["population_keys"]}
            camera_count = 32 if self.spec["dataset"] == "long360" else 20
            required_keys = {(f"cam{c:02d}", f) for c in range(1, camera_count + 1) for f in (15, 150, 285)}
            if expected != required_keys:
                raise ValueError("Prior-fit does not contain all preregistered training-camera samples")
            if len(keys) != len(set(keys)) or set(keys) != expected or any(r["scope"] != "train_prior_fit_diagnostic" for r in rows):
                raise ValueError("Prior-fit observation population differs")
            if population is None:
                population = set(keys)
            if set(keys) != population:
                raise ValueError("All prior-fit arms require identical training-view positions")
            values = {key: {k: float(v) for k, v in row.items() if k not in ("scope", "camera", "frame") and v != ""} for key, row in zip(keys, rows)}
            if not all(math.isfinite(v) for row in values.values() for v in row.values()):
                raise ValueError("Nonfinite prior-fit metrics")
            arms[arm] = values
            identities[arm] = dict(completion=entry(directory / "prior_fit_complete.json"), metrics=completion["metrics"], csv=completion["csv"])
        means = {arm: {metric: statistics.mean(row[metric] for row in rows.values()) for metric in sorted(next(iter(rows.values())))} for arm, rows in arms.items()}
        def difference(left_arm, left_prefix, right_arm, right_prefix):
            # Positive means the right target has better image quality. This
            # convention is an observation, not a causal optimization budget.
            result = {}
            for metric in ("psnr", "ssim", "lpips", "mse"):
                lk, rk = left_prefix + "_" + metric, right_prefix + "_" + metric
                sign = -1 if metric in ("lpips", "mse") else 1
                result[metric] = dict(left_mean=means[left_arm][lk], right_mean=means[right_arm][rk],
                                      improvement_mean=statistics.mean(sign * (arms[right_arm][key][rk] - arms[left_arm][key][lk]) for key in sorted(population)))
            return dict(left=dict(arm=left_arm, metric=left_prefix), right=dict(arm=right_arm, metric=right_prefix), metrics=result)
        comparison = {
            "image_teacher_vs_bicubic": difference("SRrecon", "bicubic_vs_HR", "SRrecon", "teacher_vs_HR"),
            "image_teacher_minus_SRrecon_render": difference("SRrecon", "render_raw_vs_HR", "SRrecon", "teacher_vs_HR"),
            "HRref_minus_SRrecon_same_recipe": difference("SRrecon", "render_raw_vs_HR", "HRref", "render_raw_vs_HR"),
            "SRrecon_minus_LRparent_same_budget": difference("LRparent17k", "render_raw_vs_HR", "SRrecon", "render_raw_vs_HR"),
            "HRref_minus_LRparent_same_budget": difference("LRparent17k", "render_raw_vs_HR", "HRref", "render_raw_vs_HR"),
        }
        path = self.out / "train_view_fit_summary.json"
        write(path, dict(status="awaiting_review_train_view_diagnostic", condition=self.spec["condition"], seed=SEED,
                         observations=len(population), sampled_frames=[15, 150, 285], source_identity=identities,
                         aggregate_equal_observation_mean=means, comparisons=comparison,
                         limitations=["Train-view fitting is separate from full300 heldout quality.",
                                      "Raw and learned affine-corrected metrics remain separately named; no affine is refit during evaluation.",
                                      "A teacher/render difference is empirical and cannot identify a causal module gain or a guaranteed bound.",
                                      "One seed and sampled frames do not estimate seed uncertainty."]))
        return path

    def run(self):
        validate_spec(self.spec)
        if self.out.exists():
            raise FileExistsError("Preserve existing orchestration output; no automatic restart")
        mkdir(self.out); mkdir(self.out / "events"); mkdir(self.out / "logs")
        write(self.out / "spec.json", self.spec)
        lock = mkdir(OUT / "orchestration" / "claims") / (self.spec["condition"] + "_seed07.lock")
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.stage = "Z5_supervisor_exit_wait"
                self.record("wait_binding", supervisor=self.spec["supervisor"])
                exited = wait_local_exit(self.spec["supervisor"])
                self.record("supervisor_exited", **exited)
                frozen, gate = freeze_screening(self.spec)
                write(self.out / "Z5_gate.json", gate)
                if frozen is None:
                    write(self.out / "reject.json", dict(**gate, ended_utc=now(), training_launched=False, automatic_retry=False))
                    return
                self.freeze = frozen
                write(self.out / "screening_freeze.json", frozen)
                condition = self.spec["condition"]
                zero = OUT / "zero_checks" / self.spec["dataset"]
                dataset = HERE / "datasets" / (self.spec["dataset"] + ".json")
                producer = DATA / "prepared" / condition / "manifest.json"
                self.cpu("prepare_all_video_half", "prepare_colmap_dataset.py", ["--dataset", dataset, "--rungs", "half", "--workers", self.spec["workers"]], 65)
                prepared = read(producer)
                complete = read(producer.parent / "complete.json")
                if prepared.get("full_time_decode_verified") is not True or complete.get("GPU_calls") != 0:
                    raise ValueError("Preparation actual full decode CPU acceptance missing")
                cfg = read(dataset)
                expected_cameras = 36 if self.spec["dataset"] == "long360" else 21
                if len(prepared["decode_audit"]) != expected_cameras or any(r["actual_decoded_count"] != cfg["frames"] or r["packet_count"] != cfg["frames"] or r["selected_frame_count"] != 300 for r in prepared["decode_audit"]):
                    raise ValueError("Not every source video passed full decode and PTS count")
                self.add_input_freeze([producer, producer.parent / "complete.json"])
                self.cpu("initialize_seed07", "initialize_highres.py", ["--manifest", producer, "--seeds", SEED, "--cpu-threads", self.spec["cpu_threads"]], 3)
                ready = producer.parent / f"manifest_train_ready_seed{SEED}.json"
                self.add_input_freeze([ready, producer.parent / read(ready)["initialization"]["npz_path"]])
                self.cpu("expand_privileged_HR", "expand_hr_highres.py", ["--manifest", producer, "--workers", self.spec["workers"]], 45)
                hri = DATA / "hr_full" / condition / "hr_full_storage.json"
                if read(hri).get("LR_roundtrip_all_frames_SHA_equal") is not True:
                    raise ValueError("Privileged HR does not reproduce prepared LR bytes")
                self.add_input_freeze([hri])
                self.cpu("register_LR", "register_highres.py", ["lr", "--manifest", ready, "--seed", SEED])
                self.cpu("plan_teacher", "teacher_highres.py", ["plan", "--manifest", producer, "--seeds", SEED])
                parent = OUT / "registry" / "full_LR_prefixes" / condition / f"seed_{SEED}" / "config.json"
                plan = OUT / "registry" / "teachers" / condition / "plan.json"
                self.add_input_freeze([parent, plan])
                table = OUT / "pipelines" / f"tasks_{condition}_seed{SEED}_{self.spec['run_id']}.json"
                self.cpu("make_screening_tasks", "make_screening_tasks.py", ["--condition", condition, "--seed", SEED, "--z5-receipt", str((zero / "gpu_screen_half" / "z5_half.json").relative_to(ROOT)), "--z5-samples", str((zero / "samples_half" / "samples.json").relative_to(ROOT)), "--out", table])
                self.add_input_freeze([table])
                self.sync(self.collect_inputs(condition, table))
                self.startup()
                self.runner(table)
                self.pull()
                self.summarize()
            except BaseException:
                failure = dict(status="failed_preserved_no_retry", stage=self.stage, error=traceback.format_exc(), ended_utc=now(), remote_started=self.remote_started,
                               outputs_preserved=True, automatic_retry=False, automatic_second_seed=False)
                write(self.out / f"failure_{time.time_ns()}.json", failure)
                # When waiting/validation fails after launch, attempt a bounded
                # return of whatever exists; never stop or relaunch healthy work.
                if self.smoke_started and not (self.out / "startup_acceptance.json").is_file() and self.stage != "return_artifacts":
                    try:
                        self.pull_paths([self.smoke_rel, f"output/dynamic_sr_highres_20261011/pipelines/{self.spec['run_id']}.startup.service.log"], "startup_failure_return")
                    except BaseException:
                        write(self.out / f"startup_return_failure_{time.time_ns()}.json", dict(status="failed_startup_return_remote_preserved", error=traceback.format_exc()))
                if self.remote_started and not self.formal_returned and self.stage != "return_artifacts":
                    try:
                        self.pull()
                    except BaseException:
                        write(self.out / f"return_failure_{time.time_ns()}.json", dict(status="failed_return_remote_preserved", error=traceback.format_exc()))
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--make-spec", action="store_true")
    p.add_argument("--dataset", choices=list(BINDINGS))
    p.add_argument("--supervisor-pid", type=int)
    p.add_argument("--gpu-uuid", action="append", default=[])
    p.add_argument("--run-id")
    p.add_argument("--spec-out", type=Path)
    p.add_argument("--local-python", default="/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    a = p.parse_args()
    if a.make_spec:
        if not a.dataset or not a.spec_out or a.spec:
            p.error("--make-spec requires --dataset/--spec-out and excludes --spec")
        result = make_spec(a)
        print(json.dumps(dict(status="spec_frozen_not_launched", spec=str(local(a.spec_out)), run_id=result["run_id"])))
        return
    if not a.spec:
        p.error("--spec required unless --make-spec")
    spec = read(local(a.spec))
    if a.validate_only:
        validate_spec(spec)
        print(json.dumps(dict(status="orchestrator_spec_validated_not_launched", dataset=spec["dataset"], condition=spec["condition"], seed=SEED, host=spec["host"], gpus=spec["gpus"])))
        return
    try:
        Supervisor(spec).run()
    except BaseException:
        # Startup validation failures also get an immediate record, provided
        # the requested fresh receipt directory itself has a safe identity.
        target = local(spec.get("out", ""))
        if target.parent == OUT / "orchestration" and target.name == spec.get("run_id") and not target.exists():
            mkdir(target)
            write(target / "startup_failure.json", dict(status="failed_startup_no_training", error=traceback.format_exc(), ended_utc=now(), automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
