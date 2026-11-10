"""CTS-only sequential continuation of the frozen RD4 lanes.

This entry point is run by the fixed remote training service. It changes no
scientific recipe, starts one lane at a time, and stops on the first failure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def verify_inputs(config, common):
    receipts = []
    for scene in common.SCENES:
        manifest, data = common.metadata(scene, verify_init=True)
        rows = [x for x in data["observations"] if x["split"] == "train"]
        for row in rows:
            p = manifest.parent / row["lr_path"]
            require(sha(p) == row["lr_sha256"], "Training LR SHA mismatch: " + str(p))
        receipts.append(dict(scene=scene, manifest=common.identity(manifest),
                             training_LR_files=len(rows), HR_pixels_opened=False,
                             heldout_pixels_opened=False))
    frozen = ROOT / config["task_root"] / "experiments/static_sr_wu_20261022"
    for filename, digest in config["expected_frozen_sources"].items():
        require(sha(frozen / filename) == digest, "Frozen method source changed: " + filename)
    for filename, digest in (
        ("threedsr/stablesr_turbo.ckpt", config["stable_sha256"]),
        ("stablesr/vqgan_cfw_00011.ckpt", config["vqgan_sha256"]),
    ):
        p = ROOT / config["task_root"] / "weight/static_sr_wu_20261022" / filename
        require(sha(p) == digest, "Author teacher weight changed: " + filename)
    return dict(status="passed", scenes=receipts,
                frozen_sources=config["expected_frozen_sources"],
                stable_sha256=config["stable_sha256"], vqgan_sha256=config["vqgan_sha256"])


def main():
    config = json.loads((HERE / "cts_queue.json").read_text())
    require(str(ROOT) == config["root"], "Wrong project root")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CTS physical GPU0 required")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / config["baselines"]))
    import rd_common as common
    from disk_guard import check
    check(ROOT, 20)
    out = ROOT / config["output"]
    require(not (out / "status.json").exists(), "Existing queue attempt must be reconciled first")
    record = dict(schema="rd_takeover_cts_queue_run_v1", status="verifying_inputs",
                  began_unix=time.time(), config_sha256=sha(HERE / "cts_queue.json"),
                  script_sha256=sha(__file__), host="cts", CUDA_VISIBLE_DEVICES="0",
                  ordered_jobs=config["ordered_jobs"], completed_jobs=[])
    emit(out / "status.json", record)
    try:
        checked = verify_inputs(config, common)
        emit(out / "input_acceptance.json", checked)
        print(json.dumps(dict(event="input_acceptance", scenes=len(checked["scenes"]),
                              training_LR_files=sum(x["training_LR_files"] for x in checked["scenes"]))), flush=True)
        jobs = {x["job_id"]: x for x in common.read(config["baselines"] + "/matrix.json")["jobs"]}
        pins = {x["path"]: x["sha256"] for x in common.read(config["baselines"] + "/artifact_lock.json")["files"]}
        for job_id in config["ordered_jobs"]:
            check(ROOT, 20)
            job = jobs[job_id]
            require(job["host"] == "cts", "Foreign host job in CTS queue")
            lane_path = ROOT / job["lane"]
            require(sha(lane_path) == pins[job["lane"]], "Lane SHA changed: " + job_id)
            complete = ROOT / common.output(job["method"], job["scene"]) / "jobs" / job_id / "complete.json"
            require(not complete.exists(), "Preexisting completion needs explicit reconciliation: " + job_id)
            record.update(status="running", current_job=job_id, current_lane_sha256=sha(lane_path),
                          current_job_began_unix=time.time())
            emit(out / "status.json", record)
            print(json.dumps(dict(event="lane_started", job_id=job_id, time_unix=time.time())), flush=True)
            result = subprocess.run([config["python"], "-u", str(ROOT / config["baselines"] / "run_lane.py"),
                                     "--lane", job["lane"], "--execute"], cwd=ROOT)
            require(result.returncode == 0, "Lane failed: " + job_id)
            done = json.loads(complete.read_text())
            require(done["status"] == "completed" and done["job_id"] == job_id
                    and done["lane"]["sha256"] == sha(lane_path), "Invalid completion: " + job_id)
            record["completed_jobs"].append(dict(job_id=job_id, complete_sha256=sha(complete),
                                                  ended_unix=time.time()))
            emit(out / "status.json", record)
            print(json.dumps(dict(event="lane_completed", job_id=job_id, time_unix=time.time())), flush=True)
            # Full cam00 renders are checked by run_lane. Create the immutable
            # minimum return manifest immediately after a cell finishes.
            if job_id.endswith("-render") or job["method"] in ("cem", "splatsure"):
                result = subprocess.run(["/usr/bin/python3", str(ROOT / config["baselines"] / "returns_rd.py"),
                                         "manifest", "--method", job["method"], "--scene", job["scene"]], cwd=ROOT)
                require(result.returncode == 0, "Return validation failed: " + job_id)
        record.update(status="completed", ended_unix=time.time())
        emit(out / "status.json", record)
    except BaseException:
        record.update(status="failed", ended_unix=time.time(), error=traceback.format_exc())
        emit(out / "status.json", record)
        print(record["error"], flush=True)
        raise


if __name__ == "__main__":
    main()
