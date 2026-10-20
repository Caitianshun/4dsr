"""Evaluate an audited, already completed Wu original-LR parent at HR.

GPU selection, live admission, persistence and result return are the caller's
responsibility. This worker neither selects nor claims a GPU. It preserves the
audited author's full endpoint. No model parameters are updated by this worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("External dispatcher must explicitly bind CUDA_VISIBLE_DEVICES")
    for entry in spec["required_files"]:
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen input/source changed: " + entry["path"])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(schema="wu_native_lr_20261020", spec=str(spec_path), spec_sha256=sha(spec_path),
                  started_unix=started, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                  training_HR_or_SR_pixels_used=False, target_camera_pixels_used_for_training=False,
                  author_final_Adam_skip_preserved=True,
                  inherited_complete_checkpoint=spec["checkpoint"], training_reused=True)
    write(out / "status.json", dict(record, status="starting"))
    entrypoint = Path(__file__).with_name("evaluate_native_lr.py")
    evaluate = [sys.executable, str(entrypoint), "--method", "wu", "--spec", str(spec_path),
                "--checkpoint", str(ROOT / spec["checkpoint"]["path"]), "--out", str(out / "evaluation")]
    try:
        for stage, command in (("evaluation", evaluate),):
            write(out / "status.json", dict(record, status=stage, command=command))
            with (out / (stage + ".log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        complete = json.loads((ROOT / spec["parent_plan_directory"] / "complete.json").read_text())
        summary = json.loads((out / "evaluation/summary.json").read_text())
        assert complete["HR_used"] is False and complete["teacher_used"] is False
        assert complete["heldout_pixels_used"] is False and summary["posthoc_cull"] is False
        assert complete["state"]["fine_iteration"] == 14000 and summary["frames"] == 300
        result = dict(record, status="completed", elapsed_seconds=time.time() - started,
                      metrics=summary["test"], checkpoint=complete["checkpoint"],
                      summary=dict(path=str(out / "evaluation/summary.json"), sha256=sha(out / "evaluation/summary.json")))
        write(out / "status.json", result)
        write(out / "complete.json", result)
    except BaseException:
        write(out / "status.json", dict(record, status="failed", error=traceback.format_exc(), elapsed_seconds=time.time() - started))
        raise


if __name__ == "__main__":
    main()
