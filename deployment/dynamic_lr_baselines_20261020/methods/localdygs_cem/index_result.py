"""Index final native model and all 300 HR PNGs; never select by test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["localdygs", "cem"], required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--training-receipts", nargs="+", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    metrics = p.add_mutually_exclusive_group()
    metrics.add_argument("--provisional-metrics", type=Path, help="CTS base evaluator output; excluded from main table")
    metrics.add_argument("--unified-metrics", type=Path, help="A100 base evaluator output on these same PNGs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    endpoint = 30000 if a.method == "localdygs" else 50000
    final = a.model / f"point_cloud/iteration_{endpoint}"
    if not final.is_dir():
        raise RuntimeError("Missing the author final-iteration model")
    required_assets = ["point_cloud.ply"]
    if a.method == "localdygs":
        required_assets += ["mlp_opacity.pth", "mlp_cov.pth", "mlp_color.pth", "mlp_offset.pth", "time_embedding.pth"]
        if not any((final / name).is_file() for name in ["FDHash.pth", "hexplane.pth"]):
            raise RuntimeError("Missing native final dynamic-field weights")
    if any(not (final / name).is_file() or (final / name).stat().st_size == 0 for name in required_assets):
        raise RuntimeError("Missing or empty native final model asset")
    if {x.name for x in a.renders.glob("*.png")} != {f"{f:04d}.png" for f in range(300)}:
        raise RuntimeError("Missing or unexpected HR frame names")
    for path in a.renders.glob("*.png"):
        with Image.open(path) as image:
            if image.size != (1344, 1008) or image.mode != "RGB":
                raise RuntimeError(f"HR RGB PNG mismatch: {path}")
    rendered = json.loads((a.renders / "render_receipt.json").read_text())
    assert rendered["status"] == "completed" and rendered["frames"] == 300
    assert rendered["method"] == a.method and rendered["scene"] == a.scene and rendered["iteration"] == endpoint
    assert rendered["finite_before_clip_quantization_passed_frames"] == 300
    assert rendered["raw_prediction_shape"] == [3, 1008, 1344]
    assert [x["frame"] for x in rendered["records"]] == list(range(300))
    for row in rendered["records"]:
        assert row["path"] == f"{row['frame']:04d}.png"
        assert identity(a.renders / row["path"])["sha256"] == row["sha256"]
    receipts = [json.loads(path.read_text()) for path in a.training_receipts]
    if any(x["status"] != "completed" for x in receipts):
        raise RuntimeError("Training did not complete")
    metadata = a.output.parent / "source_metadata"
    metadata.mkdir(exist_ok=False)
    copied = []

    def preserve(source, relative):
        source = Path(source)
        destination = metadata / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record = identity(destination)
        record.update(original_source_path=str(source), path_relative_to_result=str(destination.relative_to(a.output.parent)))
        copied.append(record)
        return destination

    interface = preserve(a.runtime / "protocol_interface_receipt.json", "protocol_interface_receipt.json")
    preserve(a.runtime / "protocol_interface.patch", "protocol_interface.patch")
    preserve(a.runtime / "protocol_interface.py", "protocol_interface.py")
    upstream = json.loads(interface.read_text())["upstream_commit"]
    (metadata / "upstream_commit.txt").write_text(upstream + "\n")
    copied.append(dict(identity(metadata / "upstream_commit.txt"), original_source_path=str(a.runtime / "protocol_interface_receipt.json") + "#upstream_commit",
                       path_relative_to_result="source_metadata/upstream_commit.txt"))
    configs = ([a.runtime / f"arguments/dynerf/{a.scene}.py", a.runtime / "arguments/dynerf/default.py"] if a.method == "localdygs" else
               [a.runtime / f"configs/N3V_s{stage}/{a.scene}_prepared_lr.json" for stage in [1, 2]])
    for config in configs:
        preserve(config, "configs/" + str(config.relative_to(a.runtime)))
    for name in ["run_author.py", "render_hr.py", "index_result.py"]:
        preserve(Path(__file__).parent / name, "execution_helpers/" + name)
    initialization = preserve(a.native_source / "native_initialization_receipt.json", "native_initialization_receipt.json")
    calibration = preserve(a.native_source / "camera_initialization_acceptance.json", "camera_initialization_acceptance.json")
    transfer = a.native_source.parent / "transferred_initialization_acceptance.json"
    if transfer.is_file():
        preserve(transfer, "transferred_initialization_acceptance.json")
    cfg_args = preserve(a.model / "cfg_args", "cfg_args")
    # Copy the exact environment text identities captured by the training
    # process. The full third-party installation tree is deliberately omitted.
    environment_files = {}
    for receipt in receipts:
        for record in receipt["isolated_environment_identity"]:
            source = Path(record["path"])
            if identity(source)["sha256"] != record["sha256"]:
                raise RuntimeError("Environment metadata changed since author training: " + str(source))
            if source.name in environment_files:
                assert environment_files[source.name]["sha256"] == record["sha256"]
                continue
            environment_files[source.name] = record
            preserve(source, "environment/" + source.name)
    assert {"installed_packages.txt", "upstream_commit.txt", "compiler.txt"} <= set(environment_files)
    assert Path(environment_files["upstream_commit.txt"]["path"]).read_text().strip() == upstream
    environment = Path(receipts[0]["python_executable"]).parent.parent
    for name in ["build_identity.json", "cuda_profiler_header_receipt.json", "cuda-toolkit-overlay/overlay_receipt.json"]:
        source = environment / name
        if source.is_file():
            preserve(source, "environment/" + name)
    for name in ["cfloat_compatibility.patch", "header_compatibility_receipt.json"]:
        source = environment / "build_sources/simple-knn-cfloat" / name
        if source.is_file():
            preserve(source, "environment/" + name)
    metric_summary = a.provisional_metrics or a.unified_metrics
    metric_record = None
    if metric_summary:
        score = json.loads(metric_summary.read_text())
        assert score["status"] == "completed" and score["frames"] == 300
        assert score["manifest_sha256"] == identity(a.manifest)["sha256"]
        assert Path(score["render_directory"]).resolve() == a.renders.resolve()
        scope = "provisional_CTS" if a.provisional_metrics else "unified_A100"
        assert ("3090" if a.provisional_metrics else "A100") in score["gpu"]
        project = next(parent for parent in Path(__file__).resolve().parents
                       if (parent / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py").is_file()
                       and (parent / "experiments/dynamic_sr_20260918/evaluate.py").is_file())
        evaluator = project / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py"
        legacy = project / "experiments/dynamic_sr_20260918/evaluate.py"
        assert score["evaluator_sha256"] == identity(evaluator)["sha256"]
        assert score["helper_source_sha256"] == identity(legacy)["sha256"]
        preserve(evaluator, "evaluation/evaluate_images.py")
        preserve(legacy, "evaluation/legacy_evaluate.py")
        directory = a.output.parent / ("metrics_" + scope)
        directory.mkdir()
        for name in ["summary.json", "per_frame.csv"]:
            shutil.copy2(metric_summary.parent / name, directory / name)
        metric_record = dict(scope=scope, main_table_eligible=bool(a.unified_metrics),
                             original_source_path=str(metric_summary), summary=identity(directory / "summary.json"),
                             per_frame=identity(directory / "per_frame.csv"))
    (metadata / "metadata_index.json").write_text(json.dumps(dict(status="preserved", upstream_commit=upstream,
          files=copied, full_third_party_installation_copied=False), indent=2) + "\n")
    checkpoints = [identity(p) for p in sorted(final.iterdir()) if p.is_file()]
    resume = a.model / "chkpnt50000.pth"
    if a.method == "cem":
        if not resume.is_file():
            raise RuntimeError("Missing native final CEM optimizer/model checkpoint")
        checkpoints.append(identity(resume))
    value = dict(status="completed", method=a.method, scene=a.scene, original_LR=True, no_proposed_modules=True,
                 manifest=identity(a.manifest), final_iteration=endpoint, checkpoint_selection="fixed author final endpoint; no test-score selection",
                 formal_checkpoint=checkpoints, rendered_pngs=dict(directory=str(a.renders), pattern="{frame:04d}.png", count=300,
                 receipt=identity(a.renders / "render_receipt.json")),
                 source_and_patch=identity(interface), source_metadata=identity(metadata / "metadata_index.json"),
                 source_metadata_relative_directory="source_metadata", upstream_commit=upstream,
                 native_initialization=identity(initialization),
                 native_calibration_acceptance=identity(calibration),
                 source_config=identity(cfg_args), gpu_uuid=a.gpu_uuid,
                 GPU_environment_acceptance=identity(a.model.parent / "GPU_environment_acceptance.json"),
                 training_seconds=sum(x["elapsed_seconds"] for x in receipts),
                 peak_memory_bytes=max(x.get("torch_peak_reserved_bytes", 0) for x in receipts),
                 peak_memory_scope="Torch reserved allocator; excludes external Faiss/COLMAP allocations",
                 training_receipts=[identity(x) for x in a.training_receipts], image_metrics=metric_record,
                 metrics_status="completed unified A100 evaluator" if a.unified_metrics else "pending unified A100 evaluator; CTS values are provisional" if a.provisional_metrics else "pending unified A100 evaluator")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
