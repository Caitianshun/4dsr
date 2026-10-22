#!/usr/bin/env python3
"""Freeze LocalDyGS/CEM expansion metadata and author configs on CPU only.

This program never imports Torch, creates a CUDA context, runs COLMAP, starts
training, submits work, contacts a host, or creates a scheduler/queue. Existing
October 20 outputs are read only. --write-configs copies small text configs to
this new preparation directory; it does not alter an existing runtime.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/localdygs_cem"
SCENES = ("cook_spinach", "cut_roasted_beef", "coffee_martini", "flame_steak",
          "meetroom_discussion", "meetroom_vrheadset")
TRAIN_FRAMES = list(range(0, 300, 10))
UPSTREAM = {"localdygs": ROOT / "third_party/localdygs_20261020",
            "cem": ROOT / "third_party/cem4dgs_20261020"}
COMMITS = {"localdygs": "39dacdcd8ef6d2b93824df79041713b4a29fb828",
           "cem": "7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    try:
        name = str(path.resolve().relative_to(ROOT))
    except ValueError:
        name = str(path.resolve())
    return dict(path=name, exists=path.is_file(),
                sha256=sha(path) if path.is_file() else None,
                bytes=path.stat().st_size if path.is_file() else None)


def python_dicts(path):
    """Read the pinned author's literal dict(...) configs without executing them."""
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dict" and not node.value.args):
            values[target.id] = {x.arg: ast.literal_eval(x.value) for x in node.value.keywords}
        else:
            values[target.id] = ast.literal_eval(node.value)
    return values


def config_recipe(method, scene):
    if method == "localdygs":
        relative = (f"arguments/dynerf/{scene}.py" if not scene.startswith("meetroom_")
                    else "arguments/meetroom/discussion.py")
        source = UPSTREAM[method] / relative
        parsed = python_dicts(source)
        return dict(method=method, scene=scene, source_files=[str(source), str(source.parent / "default.py")],
                    relative=relative, endpoint=parsed["OptimizationParams"]["iterations"],
                    stage_endpoints=[parsed["OptimizationParams"]["iterations"]],
                    configuration_status=("official_scene_config" if scene != "meetroom_vrheadset"
                                          else "disclosed_MeetRoom_Discussion_config_transfer_without_tuning"))
    source_scene = "cook_spinach" if scene.startswith("meetroom_") else scene
    files = [UPSTREAM[method] / f"configs/N3V_s{stage}/{source_scene}.json" for stage in (1, 2)]
    endpoints = [json.loads(path.read_text())["iterations"] for path in files]
    return dict(method=method, scene=scene, source_scene=source_scene, source_files=[str(p) for p in files],
                endpoint=endpoints[-1], stage_endpoints=endpoints,
                stage2_additional_iterations=endpoints[1] - endpoints[0],
                configuration_status=("official_scene_config" if not scene.startswith("meetroom_") else
                                      "disclosed_fixed_N3V_Cook_recipe_transfer_no_official_MeetRoom_config"))


def existing_endpoint(method, scene):
    if scene not in ("cook_spinach", "cut_roasted_beef"):
        return None
    base = "a100_native_cem" if method == "cem" and scene == "cut_roasted_beef" else "cts_native_localdygs_cem"
    result = ROOT / f"output/dynamic_lr_baselines_20261020/{base}/{scene}/{method}"
    path = result / "result_index.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    assets = []
    for item in record["formal_checkpoint"]:
        local = result / "model" / item["path"].split("/model/", 1)[1]
        actual = identity(local)
        actual.update(expected_sha256=item["sha256"],
                      matches_preserved_result=actual["sha256"] == item["sha256"])
        assets.append(actual)
    return dict(result_index=identity(path), model_path=str(result / "model"),
                checkpoint_assets=assets, all_assets_SHA256_verified=all(x["matches_preserved_result"] for x in assets),
                final_iteration=record["final_iteration"], training_seconds=record["training_seconds"],
                input="original_bicubic_LR", reuse_without_retraining=True,
                train_view_metrics="missing; native HR render and independent HR scoring required",
                test_metrics=identity(ROOT / f"output/dynamic_lr_baselines_20261020/shared_A100_metrics/{method}/{scene}/summary.json"))


def copy_config(source, destination, content=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = Path(source).read_bytes() if content is None else content.encode()
    if destination.exists():
        if destination.read_bytes() != expected:
            raise RuntimeError(f"Refusing changed frozen config: {destination}")
    else:
        destination.write_bytes(expected)
    return identity(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "notes/local_cem_audit.json")
    parser.add_argument("--write-configs", action="store_true")
    args = parser.parse_args()
    rows = []
    for scene in SCENES:
        manifest_path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed20261007.json"
        value = json.loads(manifest_path.read_text())
        assert value["frame_indices"] == list(range(300)) and value["splits"]["test"] == ["cam00"]
        assert "cam00" not in value["splits"]["train"] and "physical_lr_derivation" not in value
        for method in ("localdygs", "cem"):
            recipe = config_recipe(method, scene)
            recipe["source_config_identities"] = [identity(p) for p in recipe["source_files"]]
            copies = []
            if args.write_configs:
                if method == "localdygs":
                    source, default = [Path(p) for p in recipe["source_files"]]
                    dest = HERE / "configs/localdygs" / scene / "scene.py"
                    copies = [copy_config(source, dest), copy_config(default, dest.parent / "default.py")]
                else:
                    for stage, source in enumerate(recipe["source_files"], 1):
                        config = json.loads(Path(source).read_text())
                        config["resolution"] = 1
                        copies.append(copy_config(source, HERE / f"configs/cem/{scene}_s{stage}.json",
                                                  json.dumps(config, indent=2) + "\n"))
            recipe["frozen_config_copies"] = copies
            recipe["changes_from_author"] = ([] if method == "localdygs" else
                                             ["resolution=1: input pixels and calibration are already LR; no second resize"])
            existing = existing_endpoint(method, scene)
            initialization_root = ROOT / f"output/dynamic_lr_expansion_20261022/native/{scene}"
            if existing:
                init_source = ROOT / "output/dynamic_lr_baselines_20261020" / (
                    "cts_native_localdygs_cem" if scene == "cook_spinach" else "a100_native_localdygs_cem") / scene / "native_init"
            else:
                init_source = initialization_root / "native_init"
            native_source = init_source / ("localdygs_data" if method == "localdygs" else "cem_data")
            if method == "localdygs" and scene.startswith("meetroom_"):
                native_source = initialization_root / "meetroom_localdygs_data"
            runtime = ROOT / ("third_party/localdygs_calibrated_runtime_v3_20261020" if method == "localdygs" else
                              "third_party/cem4dgs_calibrated_runtime_v3_20261020")
            if scene.startswith("meetroom_"):
                runtime = HERE / f"runtime/{method}_meetroom"
            model = Path(existing["model_path"]) if existing else ROOT / f"output/dynamic_lr_expansion_20261022/runs/{method}/{scene}/model"
            frozen = ([HERE / "configs/localdygs" / scene / "scene.py"] if method == "localdygs" else
                      [HERE / f"configs/cem/{scene}_s{stage}.json" for stage in (1, 2)])
            author_commands = []
            if not existing:
                prefix = ["<ADMITTED_METHOD_PYTHON>", str(OLD / "run_author.py"), "--runtime", str(runtime)]
                if method == "localdygs":
                    author_commands.append(prefix + ["--receipt", str(model.parent / "training_receipt.json"), "--",
                        "-s", str(native_source), "-m", str(model), "--frames_start_end", "0", "300",
                        "--configs", str(frozen[0]), "--test_iterations", str(recipe["endpoint"]),
                        "--save_iterations", str(recipe["endpoint"])])
                else:
                    for stage, config in enumerate(frozen, 1):
                        command = prefix + ["--receipt", str(model.parent / f"training_stage{stage}_receipt.json"), "--",
                            "--configpath", str(config), "--source_path", str(native_source), "--model_path", str(model),
                            "--gpu", "<BROKER_PHYSICAL_GPU>"]
                        if stage == 2:
                            command += ["--start_checkpoint", str(model / f"chkpnt{recipe['stage_endpoints'][0]}.pth")]
                        author_commands.append(command)
            initialization = dict(
                CPU_prepare_argv=["python3", str(OLD / "prepare_native_colmap.py"), "--manifest", str(manifest_path),
                                  "--output", str(initialization_root / "native_inputs"), "--include-frame0"],
                CPU_prepare_executed=False,
                native_COLMAP_status="WAIT_GPU_RELEASE; not executed or queued",
                planned_native_COLMAP_output=str(init_source),
                planned_native_source=str(native_source),
                MeetRoom_layout_plan_argv=(["python3", str(HERE / "prepare_meetroom_layout.py"),
                    "--manifest", str(manifest_path), "--native-init", str(init_source),
                    "--native-inputs", str(initialization_root / "native_inputs"), "--output", str(native_source), "--plan-only"]
                    if method == "localdygs" and scene.startswith("meetroom_") else None),
                MeetRoom_source_layout_status=("Required CPU author sparse-layout/provenance stage before training; old N3DV localdygs_data is not used."
                                               if method == "localdygs" and scene.startswith("meetroom_") else None),
                MeetRoom_CEM_provenance_argv=(["<CPU_METHOD_PYTHON>", str(HERE / "freeze_meetroom_initialization.py"),
                    "--method", "cem", "--manifest", str(manifest_path), "--source", str(native_source),
                    "--asset", "colmap_0/sparse/0/points3D.ply", "--input-receipt", str(initialization_root / "native_inputs/input_receipt.json"),
                    "--native-receipt", str(init_source / "complete.json"), "--calibration-acceptance", str(init_source / "camera_initialization_acceptance.json")]
                    if method == "cem" and scene.startswith("meetroom_") else None),
                LocalDyGS_N3DV="training-LR multi-time COLMAP dense union at 10,20,...,290,299; author voxel rule; no HR or heldout pixels",
                CEM="training-LR frame0 COLMAP sparse initialization; no LocalDyGS dense substitution",
                MeetRoom_caveat=("Official LocalDyGS MeetRoom path uses its COLMAP/pcds loader; existing calibrated runtime only adapts N3DV. "
                                  "Reviewable meetroom_adapter.py and source patch now preserve that entry and pcds/sparse-fallback contract. "
                                  "Materialize the new runtime and verify actual LR initialization before GPU dispatch. "
                                  "CEM has no official MeetRoom loader/config; disclose calibrated-input and fixed-config transfer."
                                  if scene.startswith("meetroom_") else None))
            row = dict(schema="local_cem_future_task_spec_v1", method=method, scene=scene, dataset="MeetRoom" if scene.startswith("meetroom_") else "N3DV",
                             mode="evaluation_only_existing_endpoint" if existing else "native_author_training_then_evaluation_HOLD",
                             manifest=identity(manifest_path), area4_manifest_for_identity_separation_only=identity(
                                 manifest_path.parent.parent / (scene + "_area4") / manifest_path.name),
                             resolutions=value["resolutions"], train_cameras=value["splits"]["train"],
                             train_metric_frames=TRAIN_FRAMES, train_metric_observations=len(value["splits"]["train"]) * len(TRAIN_FRAMES),
                             test_metric_frames=list(range(300)), source_commit=COMMITS[method], author_config=recipe,
                             existing_endpoint=existing, initialization=initialization,
                             later_execution_plan=dict(runtime=str(runtime), model=str(model), source=str(native_source),
                                 config=[str(x) for x in frozen], manifest_environment={"DYNAMIC_LR_PROTOCOL_MANIFEST": str(manifest_path)},
                                 inert_author_argv=author_commands, submitted=False,
                                 preconditions=["fresh GPU resource admission after Claude release", "SHA-verified file sync", "native LR initialization accepted",
                                                "real native environment and camera contract accepted"],
                                 completion="author exit success -> native HR render train/test -> independent metrics -> integrity and local return; no polling queue created"),
                             status="reuse_completed_endpoint_and_prepare_train_metrics" if existing else
                                    "CPU_recipe_frozen_GPU_runtime_and_initialization_acceptance_pending",
                             missing_acceptance=["new renderer real native CUDA model load and HR render",
                                                 "training-view camera/frame/projection numeric validation",
                                                 "post-render finite output and independent metrics"] +
                                                ([] if existing else ["author-specific LR initialization", "native runtime data/projection acceptance",
                                                                     "real GPU environment admission", "author schedule training endpoint"]))
            spec = HERE / f"plans/local_cem/{method}_{scene}.json"
            render_base = ["<ADMITTED_METHOD_PYTHON>", str(HERE / "render_local_cem.py"), "--method", method,
                           "--manifest", str(manifest_path), "--runtime", str(runtime), "--model", str(model),
                           "--source", str(native_source), "--config", str(frozen[-1])]
            row["render_cli_plan_only"] = {scope: render_base + ["--scope", scope, "--output",
                str(ROOT / f"output/dynamic_lr_expansion_20261022/renders/{method}/{scene}/{scope}"), "--plan-only"] for scope in ("train", "test")}
            row["render_cli_later_GPU"] = {scope: ["--execute" if token == "--plan-only" else token for token in command]
                                           for scope, command in row["render_cli_plan_only"].items()}
            if scene.startswith("meetroom_"):
                for scope in ("train", "test"):
                    row["render_cli_later_GPU"][scope] += ["--runtime-acceptance", str(native_source / f"{method}_runtime_camera_acceptance.json")]
            row["spec_path"] = str(spec)
            spec.parent.mkdir(parents=True, exist_ok=True)
            spec.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            rows.append(row)
    output = dict(schema="local_cem_expansion_CPU_audit_20261022_v2", created_utc=datetime.now(timezone.utc).isoformat(),
                  status="CPU_PREPARATION_ONLY_NO_GPU_NO_TRAIN_NO_REMOTE_NO_QUEUE", selected_input="original_bicubic_LR",
                  training_frames=list(range(300)), metric_protocol=dict(train_frames=TRAIN_FRAMES, test_frames=list(range(300)),
                       train="all train cameras at 30 fixed timestamps; evaluation-only HR references", test="cam00 all 300 timestamps"),
                  no_training_additions=["SR teacher", "HR training pixels", "heldout camera pixels", "P/PC modules", "warp", "color fitting", "culling"],
                  schedule_caveat="Local Coffee 30k, Flame/MeetRoom 40k. CEM Coffee 40k to total50k; Flame 20k to total30k. Do not hardcode Cook/Cut schedules.",
                  preparation_sources=[identity(HERE / name) for name in
                      ("prepare_local_cem.py", "render_local_cem.py", "meetroom_adapter.py", "prepare_meetroom_adapter.py",
                       "prepare_meetroom_layout.py", "freeze_meetroom_initialization.py", "audit_meetroom_adapter_cpu.py", "audit_local_cem_plans_cpu.py")],
                  old_helpers_readonly={name: identity(OLD / name) for name in
                      ("prepare_native_colmap.py", "run_native_colmap.py", "run_author.py", "render_hr.py", "build_job.py", "index_result.py")},
                  rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=output["status"], rows=len(rows), reused_endpoints=sum(bool(r["existing_endpoint"]) for r in rows),
                          missing_training_endpoints=sum(not bool(r["existing_endpoint"]) for r in rows), output=str(args.output))))


if __name__ == "__main__":
    main()
