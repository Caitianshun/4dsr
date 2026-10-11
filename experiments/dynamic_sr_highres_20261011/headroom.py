"""Audit empirical reconstruction headroom on exactly paired observations.

No training or model imports. A spec maps arm names to existing evaluation
directories (metrics_per_observation.csv + run.json). Missing arms remain NA.
PSNR/SSIM gains are target minus baseline; LPIPS gains are baseline minus target.
These measurements are conditional references, never mathematical bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


PAIRS = {
    "SR_source_vs_LR_same_budget": ("LRparent", "SRrecon"),
    "HR_source_vs_LR_same_budget": ("LRparent", "HRref"),
    "prior_realized": ("lam0", "base"),
    "teacher_reference_gap": ("base", "oracleHR"),
    "HR_input_reference_gap": ("base", "HRref"),
    "SR_then_reconstruct_HR_gap": ("SRrecon", "HRref"),
    "mixed_vs_SR_then_reconstruct": ("SRrecon", "base"),
}
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_arm(path):
    path = Path(path)
    table = path / "metrics_per_observation.csv"
    rows = {}
    with table.open() as f:
        for row in csv.DictReader(f):
            if row["scope"] != "test":
                continue
            key = (row["camera"], int(row["frame"]))
            if key in rows:
                raise ValueError(f"Duplicate observation {key} in {table}")
            rows[key] = {k: float(v) for k, v in row.items()
                         if k not in ("scope", "camera", "frame") and v != ""}
    if not rows:
        raise ValueError(f"No test observations in {table}")
    if set(rows) != {('cam00', f) for f in range(300)}:
        raise ValueError('Formal G-H requires cam00 all 300 frames; smoke results are refused')
    if any(not math.isfinite(value) for row in rows.values() for value in row.values()):
        raise ValueError('Nonfinite evaluation metric')
    run = json.loads((path / "run.json").read_text())
    checkpoint = Path(run['checkpoint']['path'])
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    training_path = checkpoint.parent / 'run.json'
    training = json.loads(training_path.read_text())
    complete_path = checkpoint.parent / 'complete.json'
    complete = json.loads(complete_path.read_text())
    if complete.get('status') != 'completed' or complete.get('checkpoint') != run['checkpoint']:
        raise ValueError('Evaluation is not bound to a completed training endpoint')
    if complete.get('kind') == 'sr':
        if complete.get('cursor') != 6000:
            raise ValueError('SR continuation budget is incomplete')
    elif complete.get('kind') == 'lr':
        state = complete.get('state', {})
        if state.get('coarse_iteration') != 3000 or state.get('fine_iteration') != 14000:
            raise ValueError('Source reconstruction budget is incomplete')
    else:
        raise ValueError('Unknown endpoint training recipe')
    return dict(rows=rows, run=run, training=training, identity=dict(
        csv=dict(path=str(table), sha256=sha(table)),
        run=dict(path=str(path / "run.json"), sha256=sha(path / "run.json")),
        training=dict(path=str(training_path), sha256=sha(training_path)),
        completed=dict(path=str(complete_path), sha256=sha(complete_path))))


def paired(left, right):
    if set(left["rows"]) != set(right["rows"]):
        raise ValueError("Pair requires identical camera/frame observations")
    lman = left["run"].get("manifest", {}).get("sha256")
    rman = right["run"].get("manifest", {}).get("sha256")
    if lman != rman or not lman:
        raise ValueError("Pair requires one identical bound manifest")
    if left['run'].get('posthoc_cull', False) != right['run'].get('posthoc_cull', False):
        raise ValueError('Paired checkpoints must use identical evaluation culling')
    if 'training' in left and 'training' in right:
        for field in ('seed', 'scene', 'gpu'):
            if left['training'].get(field) != right['training'].get(field) or left['training'].get(field) is None:
                raise ValueError('Pair requires identical training ' + field)
    keys = sorted(left["rows"])
    metrics = set.intersection(*(set(left["rows"][k]) & set(right["rows"][k]) for k in keys))
    gains = {}
    for metric in sorted(metrics):
        if metric not in ("psnr", "ssim", "lpips", "mse", "low_mse", "mid_mse", "high_mse"):
            continue
        sign = -1 if metric == "lpips" or metric.endswith("mse") else 1
        deltas = [sign * (right["rows"][k][metric] - left["rows"][k][metric]) for k in keys]
        lm = statistics.mean(left["rows"][k][metric] for k in keys)
        rm = statistics.mean(right["rows"][k][metric] for k in keys)
        gains[metric] = dict(left_mean=lm, right_mean=rm, improvement_mean=statistics.mean(deltas),
                             frame_std=statistics.stdev(deltas) if len(deltas) > 1 else None,
                             frame_std_is_not_seed_uncertainty=True,
                             improvement_positive_fraction=sum(d > 0 for d in deltas) / len(deltas))
        if metric.endswith("mse"):
            gains[metric]["relative_MSE_reduction"] = (lm - rm) / lm if lm else None
    return dict(observations=len(keys), paired_keys=[list(k) for k in keys], metrics=gains,
                same_evaluation_gpu_model=left["run"].get("gpu") == right["run"].get("gpu"))


def analyze(spec):
    arms = {name: read_arm(path) for name, path in spec["arms"].items() if Path(path).exists()}
    for name, arm in arms.items():
        if arm['training'].get('seed') != spec['seed'] or arm['training'].get('scene') != spec['condition']:
            raise ValueError('Spec and measured training identity differ for ' + name)
    continuation_parents = {arm['training']['parent_checkpoint']['sha256']
                            for name, arm in arms.items() if name in ('lam0', 'base', 'oracleHR')}
    if len(continuation_parents) > 1:
        raise ValueError('Continuation arms do not share the same LR parent checkpoint')
    comparisons = {}
    for name, (left, right) in PAIRS.items():
        if left in arms and right in arms:
            comparisons[name] = dict(left_arm=left, right_arm=right, **paired(arms[left], arms[right]))
        else:
            comparisons[name] = dict(status="not_measured", missing=[a for a in (left, right) if a not in arms])
    oracle = comparisons["teacher_reference_gap"].get("metrics", {})
    teacher_gate = None if not oracle else (
        oracle["psnr"]["improvement_mean"] >= .5 or oracle["lpips"]["improvement_mean"] >= .02)
    return dict(condition=spec["condition"], seed=spec["seed"], status="awaiting_review",
                arms={k: v["identity"] for k, v in arms.items()}, comparisons=comparisons,
                provisional_teacher_module_gate=teacher_gate,
                protocol=spec.get("protocol", {}),
                limitations=["All gaps are empirical and conditional on this backbone, recipe, data and budget.",
                             "HRref/SRrecon use 3000 coarse + 14000 fine; base/lam0/oracle use a shared LR parent plus 6000 updates.",
                             "HRref minus base is budget-confounded; HRref versus SRrecon matches the 17000-update source-reconstruction recipe.",
                             "Direct SR on heldout LR is an image-quality diagnostic and cannot be an upper bound for novel-view reconstruction.",
                             "A single seed supports screening only; seed variation and fixed confirmation data remain necessary.",
                             "No additive causal error decomposition is implied by metric subtraction."])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    result = analyze(json.loads(a.spec.read_text()))
    result["spec"] = dict(path=str(a.spec), sha256=sha(a.spec))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
