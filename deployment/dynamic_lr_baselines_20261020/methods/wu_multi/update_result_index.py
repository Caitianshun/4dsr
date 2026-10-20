"""Index returned Wu/Multi4D endpoints; missing data remains explicitly pending."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/dynamic_lr_baselines_20261020"


def identity(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return dict(path=str(path), sha256=value.hexdigest(), bytes=path.stat().st_size)


def main():
    result = dict(schema="original_LR_baseline_returned_result_index_v1",
                  original_LR=True, no_proposed_modules=True, methods={},
                  main_metric_protocol="shared A100 base environment; legacy evaluator on 8-bit rendered RGB PNG; historical raw-float means excluded")
    for method in ("wu", "multi4d"):
        for scene, label in (("cook_spinach", "Cook"), ("cut_roasted_beef", "Cut")):
            spec_path = HERE / f"{method}_{scene}_native_lr.json"
            spec = json.loads(spec_path.read_text())
            path = OUT / f"a100_{method}_original_lr" / label
            manifest = (ROOT / spec["parent_plan_directory"] / "config.json") if method == "wu" else ROOT / spec["manifest"]
            if method == "wu":
                plan = json.loads(manifest.read_text())
                manifest = ROOT / plan["data"]["manifest"]["path"]
            source_paths = ([ROOT / spec["parent_plan_directory"] / "config.json"] if method == "wu" else
                            [HERE / "multi4d_native/adapter/train_native_lr.patch", HERE / "multi4d_native/adapter/inherited_runtime_fixes.patch",
                             HERE / "multi4d_native/adapter/identity.json"])
            record = dict(status="pending_execution_or_return", method=method, scene=scene,
                          manifest=identity(manifest), spec=identity(spec_path),
                          execution_job=identity(ROOT / "deployment/dynamic_lr_baselines_20261020/jobs" / f"a100_gpu1_{method}_original_lr.json"),
                          PNG_directory=str(path / "evaluation/renders_hr"), PNG_pattern="{frame:04d}.png", expected_PNG_count=300,
                          config_source_patch=[identity(source) for source in source_paths],
                          required_source_and_asset_identities=spec["required_files"])
            execution_complete = OUT / f"a100_{method}_original_lr_job/complete.json"
            if execution_complete.exists():
                execution = json.loads(execution_complete.read_text())
                record.update(remote_execution=identity(execution_complete), execution_host=execution["host"],
                              execution_GPU_UUID=execution["gpu_uuid"])
            if method == "wu":
                record.update(checkpoint=identity(ROOT / spec["checkpoint"]["path"]),
                              training_GPU=spec["original_training_hardware"], training_seconds=spec["original_training_seconds"],
                              training_peak_GB=None, training_peak_note="Not recorded in the audited historical parent completion; remains unknown",
                              historical_training_reused=True, current_A100_training_performed=False)
            shared = path / "shared_PNG_metrics/summary.json"
            if shared.exists():
                metrics = json.loads(shared.read_text())
                pngs = sorted((path / "evaluation/renders_hr").glob("*.png"))
                assert metrics["status"] == "completed" and metrics["frames"] == 300 and len(pngs) == 300
                record.update(status="completed_returned_unified_PNG_metrics", main_metrics=metrics["test"],
                              unified_metrics=identity(shared), evaluation_GPU=metrics["gpu"],
                              evaluation_seconds=metrics["seconds"], PNG_count=300)
                if method == "multi4d":
                    complete = json.loads((path / "train/complete.json").read_text())
                    coarse = json.loads((path / "coarse/complete.json").read_text())
                    smoke = [json.loads((path / name / "complete.json").read_text()) for name in ("smoke_coarse", "smoke_fine")]
                    worker_complete = json.loads((path / "complete.json").read_text())
                    assert all(v["teacher_samples"] == 0 and v["metadata"]["teacher_weight"] == 0 for v in [coarse, complete, *smoke])
                    assert complete["metadata"]["method"] == "nativeLR"
                    record.update(checkpoint=identity(path / "train/fine_20000.pt"), training_GPU=complete["metadata"]["gpu"],
                                  training_seconds=coarse["elapsed_s"] + complete["elapsed_s"],
                                  training_timing_scope="Native loop context after scene initialization; includes checkpointing, excludes pre-context scene loading/initialization",
                                  worker_wall_seconds_including_loading_smokes_and_raw_evaluation=worker_complete["seconds"],
                                  coarse_seconds=coarse["elapsed_s"], fine_seconds=complete["elapsed_s"],
                                  smoke_seconds=sum(v["elapsed_s"] for v in smoke),
                                  training_peak_GB=max(coarse["peak_gb"], complete["peak_gb"]),
                                  actual_method="nativeLR", teacher_samples=0, teacher_weight=0,
                                  native_loss_AST_contract=identity(HERE / "multi4d_native/cpu_contract.json"),
                                  inherited_template_name_note="Frozen docstrings retain area4 and unused teacher diagnostics; actual CLI/nativeLR and zero teacher samples determine this run",
                                  native_configuration=identity(path / "train/config.json"))
            else:
                worker = path / "status.json"
                if worker.exists():
                    record["worker_status"] = json.loads(worker.read_text()).get("status")
            result["methods"][method + "/" + scene] = record
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "wu_multi_result_index.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(target)
    print(target)


if __name__ == "__main__":
    main()
