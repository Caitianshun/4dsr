"""Allocated Multi4D native original-LR smoke, training and exit-driven HR eval.

This worker uses one caller-selected GPU and performs no resource admission,
environment installation, remote operations or changes to another job.
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
HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Dispatcher must bind CUDA_VISIBLE_DEVICES")
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    for item in spec["required_files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Prepared input/source changed: " + item["path"])
    manifest = json.loads((ROOT / spec["manifest"]).read_text())
    if manifest.get("physical_lr_derivation"):
        raise ValueError("Original registered LR required; area4 is forbidden")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(schema="multi4d_original_native_lr_20261020", spec=str(spec_path),
                  spec_sha256=sha(spec_path), started_unix=started,
                  gpu_binding=os.environ["CUDA_VISIBLE_DEVICES"], stages=[],
                  training_pixels="Original LR only; no SR teacher and no custom enhancement modules")
    train = [sys.executable, str(HERE / "multi4d_native/train.py"), "--spec", str(spec_path)]
    commands = [
        ("smoke_coarse", train + ["--method", "coarse", "--stop", "20", "--out", str(out / "smoke_coarse")]),
        ("coarse", train + ["--method", "coarse", "--out", str(out / "coarse")]),
        ("smoke_fine", train + ["--method", "nativeLR", "--resume", str(out / "coarse/fine0.pt"), "--stop", "20", "--out", str(out / "smoke_fine")]),
        ("fine", train + ["--method", "nativeLR", "--resume", str(out / "coarse/fine0.pt"), "--out", str(out / "train")]),
        ("evaluation", [sys.executable, str(HERE / "evaluate_native_lr.py"), "--spec", str(spec_path),
                        "--checkpoint", str(out / "train/fine_20000.pt"), "--method", "multi4d", "--out", str(out / "evaluation")]),
    ]
    try:
        for stage, command in commands:
            write(out / "status.json", dict(record, status=stage, command=command))
            with (out / (stage + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            record["stages"].append(dict(stage=stage, completed_unix=time.time()))
        summary = json.loads((out / "evaluation/summary.json").read_text())
        assert summary["status"] == "completed" and summary["frames"] == 300
        assert summary["checkpoint_metadata"]["teacher_weight"] == 0
        checkpoint = out / "train/fine_20000.pt"
        result = dict(record, status="completed", seconds=time.time() - started, metrics=summary["test"],
                      checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
                      summary=dict(path=str(out / "evaluation/summary.json"), sha256=sha(out / "evaluation/summary.json")))
        write(out / "status.json", result)
        write(out / "complete.json", result)
    except BaseException:
        write(out / "status.json", dict(record, status="failed", seconds=time.time() - started, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
