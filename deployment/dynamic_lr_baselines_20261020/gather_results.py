"""Validate returned endpoints and assemble the original-LR comparison table.

The registry explicitly selects fixed author endpoints. This program does not
search checkpoints or select a run using its held-out scores.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("wu", "yang", "4c4d", "multi4d", "localdygs", "cem")
SCENES = ("cook_spinach", "cut_roasted_beef")
REMOTE_ROOTS = ("/home/ubuntu/3DGS/4dsr", "/home/cts/Project/4DSR")


def local_path(value):
    path = Path(value)
    for prefix in REMOTE_ROOTS:
        try:
            return ROOT / path.relative_to(prefix)
        except ValueError:
            pass
    return path if path.is_absolute() else ROOT / path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked_identity(item):
    path = local_path(item["path"])
    assert path.is_file(), f"Returned file missing: {path}"
    digest = sha(path)
    assert digest == item["sha256"], f"Returned hash mismatch: {path}"
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def validate(row):
    from PIL import Image

    source = local_path(row["index"])
    index = envelope = json.loads(source.read_text())
    if row.get("key"):
        index = index["methods"][row["key"]]
    assert index["method"] == row["method"] and index["scene"] == row["scene"]
    assert index.get("original_LR", envelope.get("original_LR")) is True
    assert index.get("no_proposed_modules", envelope.get("no_proposed_modules")) is True
    assert index["status"] in ("completed_returned_unified_PNG_metrics", "complete_native_training_and_300_HR_PNGs", "completed")
    manifest_identity = checked_identity(index["manifest"])
    manifest_path = Path(manifest_identity["path"])
    manifest = json.loads(manifest_path.read_text())
    observations = sorted((x for x in manifest["observations"] if x["split"] == "test"), key=lambda x: x["frame_index"])
    assert [x["frame_index"] for x in observations] == list(range(300))
    assert {x["camera_id"] for x in observations} == {"cam00"}
    render = index.get("renders") or index.get("rendered_pngs")
    directory = local_path(render["directory"] if render else index["PNG_directory"])
    pattern = render["pattern"] if render else index["PNG_pattern"]
    expected = {pattern.format(frame=i) for i in range(300)}
    assert {p.name for p in directory.glob("*.png")} == expected, f"PNG count/names: {directory}"
    summary_path = local_path(row.get("metrics") or index["unified_metrics"]["path"])
    if "unified_metrics" in index:
        checked_identity(index["unified_metrics"])
    metrics = json.loads(summary_path.read_text())
    assert metrics["status"] == "completed" and metrics["scene"] == row["scene"]
    assert metrics["camera"] == "cam00" and metrics["frames"] == 300
    assert metrics["manifest_sha256"] == manifest_identity["sha256"]
    assert metrics["resolution"] == [1344, 1008]
    assert local_path(metrics["render_directory"]) == directory
    with (summary_path.parent / "per_frame.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    assert [int(x["frame"]) for x in frames] == list(range(300))
    for reference, frame in zip(observations, frames):
        pred = directory / pattern.format(frame=int(frame["frame"]))
        assert sha(pred) == frame["render_sha256"], f"PNG differs from evaluated bytes: {pred}"
        gt = manifest_path.parent / reference["hr_path"]
        assert sha(gt) == reference["hr_sha256"] == frame["reference_sha256"]
        with Image.open(pred) as image:
            assert image.mode == "RGB" and image.size == (1344, 1008)
        assert all(math.isfinite(float(frame[k])) for k in ("psnr", "ssim", "lpips", "mse"))
    for name in ("psnr", "ssim", "lpips"):
        mean = sum(float(frame[name]) for frame in frames) / 300
        assert abs(mean - metrics["test"][name]) < 1e-9
    endpoints = index.get("formal_checkpoint") or [index["checkpoint"]]
    checkpoints = [checked_identity(x) for x in endpoints]
    evidence = []
    for name in ("spec", "execution_job", "config", "source_config", "native_initialization", "source_and_patch",
                 "native_calibration_acceptance", "GPU_environment_acceptance"):
        if name in index:
            evidence.append(dict(kind=name, **checked_identity(index[name])))
    if "source_metadata" in index:
        metadata_identity = checked_identity(index["source_metadata"])
        evidence.append(dict(kind="source_metadata", **metadata_identity))
        metadata = json.loads(Path(metadata_identity["path"]).read_text())
        assert metadata["status"] == "preserved"
        for item in metadata["files"]:
            destination = source.parent / item["path_relative_to_result"]
            evidence.append(dict(kind="preserved_source_metadata",
                                 **checked_identity(dict(path=str(destination), sha256=item["sha256"]))))
    for name in ("config_source_patch", "training_receipts"):
        for item in index.get(name, []):
            evidence.append(dict(kind=name, **checked_identity(item)))
    return dict(method=row["method"], scene=row["scene"], status="completed_verified_returned",
                metrics=metrics["test"], metric_identity=dict(path=str(summary_path), sha256=sha(summary_path)),
                manifest=manifest_identity, checkpoints=checkpoints,
                PNG_directory=str(directory), PNG_pattern=pattern, PNG_count=300,
                source_index=dict(path=str(source), sha256=sha(source)),
                verified_provenance_files=evidence,
                source_index_key=row.get("key"), original_LR=True, no_proposed_modules=True,
                evaluator_contract={k: metrics[k] for k in ("input_representation", "psnr_aggregation", "ssim_definition",
                    "lpips_definition", "helper_source_sha256", "evaluator_sha256", "lpips_state_dict_sha256",
                    "host", "torch", "numpy", "opencv", "device", "gpu")},
                native_endpoint_metadata=index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    registry = json.loads(a.registry.read_text())
    assert len(registry) == 12 and {(r["method"], r["scene"]) for r in registry} == {(m, s) for m in METHODS for s in SCENES}
    records = []
    for row in registry:
        if row.get("index") is None:
            records.append(dict(method=row["method"], scene=row["scene"], status="pending"))
        else:
            records.append(validate(row))
    completed = [row for row in records if row["status"] == "completed_verified_returned"]
    if a.require_complete:
        assert len(completed) == 12, f"Only {len(completed)}/12 validated endpoints"
    contracts = [x["evaluator_contract"] for x in completed]
    assert all(value == contracts[0] for value in contracts), "Main table evaluator environment/metric contracts differ"
    if contracts:
        assert contracts[0]["device"] == "cuda" and contracts[0]["gpu"] == "NVIDIA A100-SXM4-40GB"
        assert contracts[0]["evaluator_sha256"] == sha(Path(__file__).with_name("evaluate_images.py"))
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema="original_LR_six_dynamic_methods_comparison_v1", status="complete" if len(completed) == 12 else "in_progress",
                   completed=len(completed), expected=12, registry_sha256=sha(a.registry), records=records)
    (a.out / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (a.out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "scene", "status", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in ("method", "scene", "status")} | row.get("metrics", {}))
    visual_rows = [{k: row[k] for k in ("method", "scene", "PNG_directory", "PNG_pattern")} for row in completed]
    (a.out / "visual_runs.json").write_text(json.dumps(visual_rows, indent=2) + "\n")
    print(json.dumps(dict(status=receipt["status"], validated_endpoints=len(completed), expected=12)))


if __name__ == "__main__":
    main()
