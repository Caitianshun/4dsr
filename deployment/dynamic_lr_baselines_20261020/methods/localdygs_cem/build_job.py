"""Freeze the CTS GPU0 native COLMAP -> LocalDyGS -> CEM serial queue.

Run only after both isolated environments pass import/build acceptance. The
root broker still checks GPU admission; this script does not submit anything.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def full(path):
    return "{ROOT}/" + str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-receipt", required=True)
    p.add_argument("--inventory", type=Path, default=ROOT / "deployment/dynamic_lr_baselines_20261020/input_inventory.json")
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef", "both"], default="both")
    p.add_argument("--method", choices=["localdygs", "cem", "both"], default="both")
    p.add_argument("--initialization-only", action="store_true")
    p.add_argument("--skip-initialization", action="store_true")
    p.add_argument("--inputs-already-prepared", action="store_true")
    p.add_argument("--native-initialization-root", type=Path, help="Existing single-scene native_init copied from another host; source receipts remain unchanged")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("Refusing existing frozen job; do not automatically replay")
    inventory = a.inventory
    if a.initialization_only and a.skip_initialization:
        raise SystemExit("Initialization-only cannot skip initialization")
    if a.native_initialization_root and (not a.skip_initialization or a.scene == "both"):
        raise SystemExit("An explicit initialization root requires one scene and --skip-initialization")
    suffix = ("-" + a.scene if a.scene != "both" else "") + ("-init" if a.initialization_only else "-train" if a.skip_initialization else "")
    job = dict(job_id="codex-dynlr-localdygs-cem-ctsg0-20261020" + suffix.replace("_", "-"), method="LocalDyGS_and_CEM_native",
        scene=a.scene + "_full300", out="output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem_job" + suffix,
        original_LR=True, no_proposed_modules=True, gpu_uuid="GPU-fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a910",
        input_receipt=a.input_receipt, input_inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp",
                 OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                 LD_LIBRARY_PATH="{ROOT}/envs/dynamic_lr_baselines_20261020/cem/cuda-toolkit-overlay/lib64"),
        required_files=[], commands=[], return_paths=[])
    if a.method != "both" and not a.initialization_only:
        phase = "train" if a.skip_initialization else "full"
        job["job_id"] = f"codex-dynlr-{a.method}-cts0-{a.scene.replace('_', '-')}-{phase}-20261020"
        job["method"] = a.method + "_native"
        job["out"] += "-" + a.method
    selected_methods = ["localdygs", "cem"] if a.method == "both" else [a.method]
    required = set(x for x in HERE.iterdir() if x.is_file())
    required.update([ROOT / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py", ROOT / "experiments/dynamic_sr_20260918/evaluate.py"])
    for method in ["localdygs", "cem4dgs"]:
        runtime = ROOT / f"third_party/{method}_calibrated_runtime_v3_20261020"
        required.update(x for x in runtime.rglob("*.py") if "__pycache__" not in x.parts)
        required.update([runtime / "protocol_interface_receipt.json", runtime / "protocol_interface.patch"])
        required.update(runtime.rglob("*_prepared_lr.json"))
    command = lambda name, argv: job["commands"].append(dict(name=name, argv=[str(x) for x in argv]))
    helper = full(HERE.relative_to(ROOT))
    env = full("envs/dynamic_lr_baselines_20261020")
    target_base = Path("output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem")
    for scene in (["cook_spinach", "cut_roasted_beef"] if a.scene == "both" else [a.scene]):
        manifest = Path(f"data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed20261007.json")
        poses = Path(f"output/dynamic_lr_baselines_20261020/localdygs_cem/native_metadata/{scene}_poses_bounds.npy")
        required.update([ROOT / manifest, ROOT / poses])
        result = target_base / scene
        inputs, init = result / "native_inputs", result / "native_init"
        if a.native_initialization_root:
            init = a.native_initialization_root
        if a.skip_initialization:
            for directory in ["localdygs_data", "cem_data"]:
                prepared = ROOT / init / directory
                if not (prepared / "native_initialization_receipt.json").is_file():
                    raise SystemExit(f"Completed initialization must be returned before freezing training: {prepared}")
                required.update(x for x in prepared.rglob("*") if x.is_file())
        prefix = ["/usr/bin/env", f"DYNAMIC_LR_PROTOCOL_MANIFEST={full(manifest)}"]
        if not a.skip_initialization:
            if not a.inputs_already_prepared:
                command(f"{scene}_prepare_native_inputs", ["{ROOT}/.venv/bin/python", helper + "/prepare_native_colmap.py", "--manifest", full(manifest), "--output", full(inputs), "--include-frame0"])
            command(f"{scene}_native_COLMAP", ["{ROOT}/.venv/bin/python", helper + "/run_native_colmap.py", "--inputs", full(inputs), "--output", full(init), "--poses-bounds", full(poses)])
        if a.initialization_only:
            job["return_paths"] += [str(init / "localdygs_data"), str(init / "cem_data"), str(init)]
            continue
        if a.skip_initialization:
            command(f"{scene}_transferred_initialization_acceptance", ["{ROOT}/.venv/bin/python", helper + "/audit_transferred_initialization.py", "--manifest", full(manifest), "--initialization", full(init)])
        else:
            command(f"{scene}_native_COLMAP_calibration_acceptance", ["{ROOT}/.venv/bin/python", helper + "/audit_native_initialization.py", "--manifest", full(manifest), "--inputs", full(inputs), "--initialization", full(init)])
        for method in selected_methods:
            command(f"{scene}_{method}_real_GPU_environment", [env + f"/{method}/bin/python", helper + "/check_gpu_environment.py", "--method", method, "--output", full(result / method / "GPU_environment_acceptance.json")])
        for method, code_name, endpoint in [("localdygs", "localdygs", 30000), ("cem", "cem4dgs", 50000)]:
            if method not in selected_methods:
                continue
            runtime = f"third_party/{code_name}_calibrated_runtime_v3_20261020"
            model = result / method / "model"
            renders = result / method / "renders_hr"
            native_source = init / ("localdygs_data" if method == "localdygs" else "cem_data")
            interpreter = env + f"/{method}/bin/python"
            if method == "localdygs":
                receipts = [result / method / "training_receipt.json"]
                command(f"{scene}_{method}_author_train", [*prefix, interpreter, helper + "/run_author.py", "--runtime", full(runtime), "--receipt", full(receipts[0]), "--", "-s", full(native_source), "-m", full(model), "--frames_start_end", "0", "300", "--configs", f"arguments/dynerf/{scene}.py", "--test_iterations", str(endpoint), "--save_iterations", str(endpoint)])
            else:
                receipts = [result / method / f"training_stage{n}_receipt.json" for n in [1, 2]]
                for stage in [1, 2]:
                    argv = [*prefix, interpreter, helper + "/run_author.py", "--runtime", full(runtime), "--receipt", full(receipts[stage-1]), "--", "--configpath", f"configs/N3V_s{stage}/{scene}_prepared_lr.json", "--source_path", full(native_source), "--model_path", full(model), "--gpu", "0"]
                    if stage == 2:
                        argv += ["--start_checkpoint", full(model / "chkpnt40000.pth")]
                    command(f"{scene}_{method}_author_stage{stage}", argv)
            command(f"{scene}_{method}_render_HR", [*prefix, interpreter, helper + "/render_hr.py", "--method", method, "--runtime", full(runtime), "--model", full(model), "--source", full(native_source), "--scene", scene, "--output", full(renders)])
            provisional = Path("output/dynamic_lr_baselines_20261020/provisional_CTS_metrics") / method / scene
            command(f"{scene}_{method}_provisional_CTS_HR_metrics", ["{ROOT}/.venv/bin/python", "{ROOT}/deployment/dynamic_lr_baselines_20261020/evaluate_images.py", "--manifest", full(manifest), "--render-dir", full(renders), "--out", full(provisional), "--method", method + "_original_LR_provisional_CTS", "--device", "cuda"])
            command(f"{scene}_{method}_index", ["{ROOT}/.venv/bin/python", helper + "/index_result.py", "--method", method, "--scene", scene, "--manifest", full(manifest), "--runtime", full(runtime), "--model", full(model), "--native-source", full(native_source), "--renders", full(renders), "--training-receipts", *[full(x) for x in receipts], "--gpu-uuid", job["gpu_uuid"], "--provisional-metrics", full(provisional / "summary.json"), "--output", full(result / method / "result_index.json")])
            job["return_paths"].append(str(result / method))
            job["return_paths"].append(str(provisional))
        job["return_paths"] += [str(init / "localdygs_data"), str(init / "cem_data")]
    job["required_files"] = [identity(x) for x in sorted(required)]
    job["required_files"].append(dict(path=".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        sha256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"))
    job["return_paths"].insert(0, job["out"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], commands=len(job["commands"]), required_files=len(required), submitted=False), indent=2))


if __name__ == "__main__":
    main()
