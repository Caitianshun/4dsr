#!/usr/bin/env python3
"""Attach completed common PNG metrics to a completed native endpoint index.

This edits result metadata only after all native training/rendering has ended.
It never loads a model, changes a checkpoint, or selects an iteration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--metrics", type=Path, required=True)
a = p.parse_args()
index_path = a.run / "result_index.json"
index = json.loads(index_path.read_text())
summary_path = a.metrics / "summary.json"
summary = json.loads(summary_path.read_text())
assert index["status"] == "complete_native_training_and_300_HR_PNGs"
assert summary["status"] == "completed" and summary["frames"] == 300
assert summary["method"] == index["method"] and summary["scene"] == index["scene"]
assert summary["manifest_sha256"] == index["manifest"]["sha256"]
assert Path(summary["render_directory"]).resolve() == Path(index["renders"]["directory"]).resolve()
assert all(math.isfinite(v) for v in summary["test"].values())
root = Path(__file__).resolve().parents[4]
assert summary["evaluator_sha256"] == sha(root / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py")
assert summary["helper_source_sha256"] == sha(root / "experiments/dynamic_sr_20260918/evaluate.py")
index["metrics_status"] = "complete_shared_A100_base_evaluator_on_300_RGB_PNGs"
index["shared_metrics"] = {
    "directory": str(a.metrics.resolve()),
    "summary": str(summary_path.resolve()),
    "summary_sha256": sha(summary_path),
    "test": summary["test"],
    "frames": 300,
    "input_representation": summary["input_representation"],
    "evaluator_sha256": summary["evaluator_sha256"],
    "helper_source_sha256": summary["helper_source_sha256"],
    "lpips_state_dict_sha256": summary["lpips_state_dict_sha256"],
    "environment": {k: summary[k] for k in ["host", "torch", "numpy", "opencv", "device", "gpu"]},
}
temporary = index_path.with_suffix(".shared-metrics.tmp")
temporary.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
temporary.replace(index_path)
print(json.dumps({"status": index["metrics_status"], "method": index["method"],
                  "scene": index["scene"], "test": summary["test"]}), flush=True)
