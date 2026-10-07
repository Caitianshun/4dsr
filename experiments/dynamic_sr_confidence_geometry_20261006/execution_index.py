"""Finite budget/provenance ledger; never rank or select by quality scores.

The default checks receipt identity, JSONL accounting, asset presence and hashes
of small records. Checkpoint bytes are optionally verified explicitly because a
full pass over every multi-GB checkpoint is expensive. Unverified bytes remain
labelled unverified. A failed attempted update is not silently counted as zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from depth_prior import sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "output" / HERE.name
CORE = ["C1", "Jperm", "B2perm", "R", "G", "RG"]
MECHANISM = ["RG_time", "RG_ST", "RG_mass", "G_FullSR_ST", "G_Q_uniform"]
PROBES = ["P_joint", "P_xyz", "P_SH"]


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def asset(path):
    path = Path(path)
    return dict(path=str(path), exists=path.exists(), bytes=path.stat().st_size if path.exists() else None,
                sha256=sha256(path) if path.exists() else None)


def resolve_receipt_path(run_root, recorded):
    path = Path(recorded)
    if path.is_absolute():
        return path
    if str(path).startswith("output/"):
        return run_root.parents[1] / path
    return run_root / path


def training_rows(path):
    rows = []
    failures = []
    if Path(path).exists():
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    failures.append(dict(line=line_number, bytes=len(line), final_partial_line=True))
    return rows, failures


def inspect_run(directory, run_root, verify_checkpoints=False):
    config = read(directory / "config.json")
    if config is None:
        return None
    rows, jsonl_errors = training_rows(directory / "training.jsonl")
    complete = read(directory / "complete.json")
    failed = read(directory / "failed.json")
    hook_failure = read(directory / "probe_hook_failed.json")
    reload_audit = read(directory / "reload_audit.json")
    start = reload_audit.get("fork_cursor", 0) if reload_audit else 0
    steps = [int(row["suffix_step"]) for row in rows]
    expected = list(range(start + 1, start + len(rows) + 1))
    contiguous = steps == expected
    last_attempt = read(directory / "last_attempt.json")
    ambiguous_attempt = bool(last_attempt and int(last_attempt["suffix_step"]) > (steps[-1] if steps else start))
    errors = []
    if not contiguous:
        errors.append("training JSONL suffix steps are not contiguous")
    if jsonl_errors:
        errors.append("training JSONL has an incomplete/invalid line")
    if complete is not None and complete.get("updates") != len(rows):
        errors.append("complete updates differ from logged completed optimizer rounds")
    checkpoints = []
    for receipt_path in sorted(directory.glob("checkpoint_*.json")):
        receipt = read(receipt_path)
        recorded = receipt.get("path", receipt.get("checkpoint", {}).get("path"))
        digest = receipt.get("sha256", receipt.get("checkpoint", {}).get("sha256"))
        path = resolve_receipt_path(run_root, recorded)
        exists = path.exists()
        verification = "exists_receipt_hash_not_recomputed" if exists else "missing"
        actual = None
        if verify_checkpoints and exists:
            actual = sha256(path)
            verification = "verified" if actual == digest else "hash_mismatch"
        if not exists or verification == "hash_mismatch":
            errors.append(f"checkpoint integrity failure: {path.name}")
        checkpoints.append(dict(path=str(path), receipt=asset(receipt_path), recorded_sha256=digest,
            actual_sha256=actual, bytes=path.stat().st_size if exists else None, integrity=verification))
    method, repeat = config["method"], str(config["repeat"])
    status = "completed" if complete is not None and not failed and not hook_failure else "failed" if failed or hook_failure else "running_or_interrupted"
    if errors:
        status = "integrity_attention"
    logs_rgb = sum(int(row.get("rgb_forwards", 0)) for row in rows)
    logs_moments = sum(int(row.get("moment_forwards", 0)) for row in rows)
    logs_adam = sum(int(row.get("adam_calls", 0)) for row in rows)
    loss_log = rows[-1] if rows else {}
    source_snapshot = config.get("source_identity", {})
    snapshot_missing = [name for name in source_snapshot if not (directory / "sources" / name).exists()]
    hook = read(directory / "probe_hook_complete.json")
    if method in PROBES and complete is not None and hook is None:
        errors.append("Completed cause probe lacks observational hook receipt")
        status = "integrity_attention"
    evaluation = run_root / "evaluation" / f"r{repeat}_{method}"
    # Keep only metric file identities/presence, never read score values to make
    # any execution or method-selection decision.
    metrics = {split: asset(evaluation / split / "metrics.json") for split in ["test", "dev", "train_fixed"]}
    return dict(task=directory.name, method=method, repeat=repeat, status=status,
        category="core" if method in CORE else "cause_probe" if method in PROBES else "conditional_mechanism" if method in MECHANISM else "other",
        completed_logged_updates=len(rows), suffix_start=start, suffix_last=steps[-1] if steps else start,
        ambiguous_partial_Adam_attempt=ambiguous_attempt,
        partial_attempt_rule="Not assumed zero; exact completed partial Adam/forward cost requires failure trace/last_attempt review" if ambiguous_attempt else None,
        actual_logged_RGB=logs_rgb, actual_logged_moments=logs_moments, actual_logged_Adam_calls=logs_adam,
        train_seconds=complete.get("train_s") if complete else loss_log.get("train_s"),
        wall_seconds=complete.get("wall_s") if complete else loss_log.get("wall_s"),
        peak_gb=complete.get("peak_gb") if complete else loss_log.get("peak_gb"),
        gpu=config.get("gpu"), physical_gpu=config.get("physical_gpu"),
        complete_receipt=asset(directory / "complete.json"), reload_audit=reload_audit,
        schedule_sha256=config.get("schedule_sha256"), protocol_sha256=config.get("protocol_sha256"),
        parent=config.get("parent"), sources_missing_from_snapshot=snapshot_missing,
        source_snapshot_count=len(source_snapshot), checkpoints=checkpoints,
        optimizer_probe_hook=hook, evaluation_metric_files=metrics, errors=errors,
        quality_values_used_for_selection=False)


def cache_receipts(run_root):
    results = []
    for path in sorted((run_root / "cache").glob("**/*index.json")) + sorted((run_root / "cache").glob("**/cache_manifest.json")):
        value = read(path)
        entries = value.get("entries", value.get("rows", []))
        results.append(dict(receipt=asset(path), status=value.get("status"), entries=len(entries),
            manifest_sha256=value.get("manifest_sha256"),
            frozen=value.get("frozen"), HR_derived_training=value.get("hr_derived_training_items", value.get("privileged_train_hr")),
            source_sha256=value.get("source_sha256", value.get("identity", {}).get("source_sha256")),
            payload_hashes_recomputed=False))
    return results


def build(run_root, output, verify_checkpoints=False):
    run_root, output = Path(run_root), Path(output)
    started = time.monotonic()
    protocol = read(run_root / "protocol.json")
    if protocol is None:
        raise FileNotFoundError("A registered protocol is required")
    runs = []
    for directory in sorted((run_root / "runs").glob("**/config.json")):
        row = inspect_run(directory.parent, run_root, verify_checkpoints)
        if row is not None:
            runs.append(row)
    planned = [dict(method=method, repeat=str(repeat), updates=6000, authorized_category="core")
               for repeat in [1, 2] for method in CORE]
    planned += [dict(method=method, repeat="1", updates=500, authorized_category="cause_probe") for method in PROBES]
    sums = {category: dict(completed_logged_updates=sum(r["completed_logged_updates"] for r in runs if r["category"] == category),
        logged_RGB=sum(r["actual_logged_RGB"] for r in runs if r["category"] == category),
        logged_moments=sum(r["actual_logged_moments"] for r in runs if r["category"] == category),
        ambiguity_count=sum(r["ambiguous_partial_Adam_attempt"] for r in runs if r["category"] == category))
        for category in ["core", "cause_probe", "conditional_mechanism", "other"]}
    allowed = protocol["budget"]
    warnings = []
    if sums["core"]["completed_logged_updates"] > allowed["core_updates"]:
        warnings.append("Core update allowance exceeded, including retries")
    if sums["cause_probe"]["completed_logged_updates"] > allowed["cause_probe_updates"]:
        warnings.append("1500 cause-probe allowance exceeded, including retries")
    if sums["conditional_mechanism"]["completed_logged_updates"] and not (run_root / "phase13_decision.json").exists():
        warnings.append("Mechanism updates exist without an explicit saved core-quality decision receipt")
    if any(r["ambiguous_partial_Adam_attempt"] for r in runs):
        warnings.append("At least one unfinished before-Adam attempt has ambiguous partial update cost; not treated as zero")
    report = dict(status="indexed_with_explicit_verification_limits", generated_unix=time.time(),
        protocol=asset(run_root / "protocol.json"), budget=allowed, planned_authorized_core_and_causes=planned,
        actual=sums, runs=runs, caches=cache_receipts(run_root), warnings=warnings,
        calibration=asset(run_root / "calibration.json"), phase13_decision=asset(run_root / "phase13_decision.json"),
        phase13_flow_coverage=asset(run_root / "phase13/formal_flow_coverage.json"),
        provenance_incidents=[asset(path) for path in sorted((run_root / "source_incidents").glob("**/incident.json"))],
        verification=dict(checkpoint_byte_hashes_recomputed=verify_checkpoints,
            small_receipt_hashes_recomputed=True, source_snapshot_bytes_verified=False,
            cache_payload_hashes_recomputed=False, quality_values_used_for_selection=False,
            running_jobs_not_polled=True),
        source_sha256=sha256(__file__), seconds=time.monotonic() - started)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_index.json", report)
    # Separate checkpoint location index is useful for later cross-host transfer.
    checkpoint_rows = [dict(task=run["task"], method=run["method"], repeat=run["repeat"],
        status=run["status"], gpu=run["gpu"], **checkpoint) for run in runs for checkpoint in run["checkpoints"]]
    write_json(output / "checkpoint_index.json", dict(status="indexed", checkpoints=checkpoint_rows,
        hash_verification_requested=verify_checkpoints, source_execution_index_sha256=sha256(output / "execution_index.json")))
    print(json.dumps(dict(status=report["status"], runs=len(runs), actual=sums, warnings=warnings)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    build(args.run_root, args.out, args.verify_checkpoints)


if __name__ == "__main__":
    main()
