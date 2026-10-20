"""Freeze only the two author CEM stages for one A100 GPU; never submit.

Native initialization must already have completed, passed calibration audit,
and returned locally. Data and author runtime bytes are frozen by SHA256.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUNTIME_SOURCE = ROOT / "third_party/cem4dgs_calibrated_runtime_v3_20261020"
RUNTIME_REMOTE = Path("output/dynamic_lr_baselines_20261020/runtime/CEM-4DGS-calibrated-v3")
LEAF = Path("envs/dynamic_lr_baselines_20261020/cem_a100_sm80")
GPU_UUIDS = {0: "GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b",
             1: "GPU-b79cd3fe-81f0-f449-30be-432e2857e517"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    path = Path(path)
    return path.resolve().relative_to(ROOT) if path.is_absolute() else path


def full(path):
    return "{ROOT}/" + str(relative(path))


def initialization_gate(manifest, native_source):
    value = json.loads((native_source / "native_initialization_receipt.json").read_text())
    camera = json.loads((native_source / "camera_initialization_acceptance.json").read_text())
    assert value["status"] == "completed"
    assert value["used_HR"] is False and value["used_test_pixels"] is False
    assert value["native_frames"] == [0, *range(10, 300, 10), 299]
    # Receipt paths retain their original host, while the returned copy is
    # checked at its explicitly supplied, relocatable project-relative path.
    pointcloud = native_source / "colmap_0/sparse/0/points3D.ply"
    assert sha(pointcloud) == value["cem_initialization"]["sha256"]
    assert camera["status"] == "passed" and camera["manifest_sha256"] == sha(manifest)
    assert camera["initialization_receipt_sha256"] == sha(native_source / "native_initialization_receipt.json")
    assert camera["COLMAP_poses_intrinsics_unchanged"] is True
    assert camera["camera_optimization_used_to_fit_data"] is False
    assert camera["max_K_parameter_error"] < 1e-9 and camera["max_pose_matrix_error"] < 1e-9
    expected_cameras = set(json.loads(manifest.read_text())["splits"]["train"])
    assert {(row["frame"], row["camera"]) for row in camera["camera_records"]} == {
        (frame, camera_id) for frame in value["native_frames"] for camera_id in expected_cameras}
    return dict(receipt_sha256=sha(native_source / "native_initialization_receipt.json"),
                camera_acceptance_sha256=sha(native_source / "camera_initialization_acceptance.json"),
                initialization_pointcloud_sha256=sha(pointcloud))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, choices=["cook_spinach", "cut_roasted_beef"])
    p.add_argument("--gpu-index", type=int, choices=[0, 1], default=1,
                   help="Physical A100 index, frozen with its UUID; dispatch remains separately admitted")
    p.add_argument("--native-source", type=Path, required=True, help="Returned completed cem_data directory")
    p.add_argument("--input-receipt", default="deployment/dynamic_lr_baselines_20261020/input_verified.json")
    p.add_argument("--inventory", type=Path, default=ROOT / "deployment/dynamic_lr_baselines_20261020/input_inventory.json")
    p.add_argument("--helper-directory", type=Path, default=HERE / "frozen_execution_helpers_v2")
    p.add_argument("--build-identity", type=Path, default=ROOT / "output/dynamic_lr_baselines_20261020/a100_cem_cpu_build/leaf_metadata/build_identity.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    gpu_physical, gpu_uuid = str(a.gpu_index), GPU_UUIDS[a.gpu_index]
    if a.output.exists():
        raise SystemExit("Refusing existing frozen job; no automatic replay")
    native = ROOT / relative(a.native_source)
    helpers = ROOT / relative(a.helper_directory)
    manifest = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{a.scene}/manifest_train_ready_seed20261007.json"
    init_gate = initialization_gate(manifest, native)
    build_identity = json.loads(a.build_identity.read_text())
    assert build_identity["status"] == "CEM_sm80_CPU_build_verified_no_GPU_execution"
    assert build_identity["CUDA_initialized"] is False
    protocol = json.loads((RUNTIME_SOURCE / "protocol_interface_receipt.json").read_text())
    assert protocol["native_readonly_test_evaluation_disabled"]
    assert protocol["core_losses_sampling_optimization_schedule_unchanged"]
    guard_source = (helpers / "run_author.py").read_text()
    assert "verify_cem_broker_device" in guard_source
    assert "source_metadata" in (helpers / "index_result.py").read_text()
    for stage, final in [(1, 40000), (2, 50000)]:
        config = json.loads((RUNTIME_SOURCE / f"configs/N3V_s{stage}/{a.scene}_prepared_lr.json").read_text())
        assert config["iterations"] == final and config["model"] == "virtual"
    result = Path("output/dynamic_lr_baselines_20261020/a100_native_cem") / a.scene / "cem"
    shared_metrics = Path("output/dynamic_lr_baselines_20261020/shared_A100_metrics/cem") / a.scene
    model, renders = result / "model", result / "renders_hr"
    receipts = [result / f"training_stage{i}_receipt.json" for i in [1, 2]]
    job = dict(job_id=f"codex-dynlr-cem-a100g{a.gpu_index}-{a.scene.replace('_', '-')}-20261020",
               method="CEM_native", scene=a.scene + "_full300",
               out=str(result / "execution"), original_LR=True, no_proposed_modules=True,
               gpu_uuid=gpu_uuid, input_receipt=a.input_receipt,
               input_inventory_sha256=sha(a.inventory), physical_gpu_argument=gpu_physical,
               initialization_gate=init_gate,
               env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                        LD_LIBRARY_PATH=full(LEAF / "cuda-toolkit-overlay/lib64")),
               required_files=[], commands=[], return_paths=[str(result), str(shared_metrics)])
    required = {}
    def require(local, remote=None):
        local = Path(local)
        remote = relative(local) if remote is None else Path(remote)
        if not local.is_file():
            raise FileNotFoundError(local)
        required[str(remote)] = dict(path=str(remote), sha256=sha(local))
    require(Path(__file__))
    for name in ["run_author.py", "render_hr.py", "index_result.py", "check_gpu_environment.py"]:
        require(helpers / name)
    require(helpers / "frozen_helper_identity.json")
    require(a.build_identity, LEAF / "build_identity.json")
    for entry in build_identity["frozen_environment_files"]:
        required[entry["path"]] = dict(path=entry["path"], sha256=entry["sha256"])
    require(manifest)
    transfer = native.parent / "transferred_initialization_acceptance.json"
    if transfer.is_file():
        # index_result preserves this receiver-side receipt. Freeze it alongside
        # the original initialization provenance rather than copying unchecked
        # metadata only after author training has finished.
        require(transfer)
    for path in sorted(native.rglob("*")):
        if path.is_file():
            require(path)
    for path in sorted(RUNTIME_SOURCE.rglob("*")):
        if path.is_file() and not any(part in {".git", "__pycache__", "build", ".venv"} for part in path.relative_to(RUNTIME_SOURCE).parts) and path.suffix not in {".so", ".pyc"}:
            require(path, RUNTIME_REMOTE / path.relative_to(RUNTIME_SOURCE))
    evaluator = ROOT / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py"
    require(ROOT / "deployment/dynamic_lr_baselines_20261020/run_job.py")
    require(evaluator)
    require(ROOT / "experiments/dynamic_sr_20260918/evaluate.py")
    # The input inventory omits model/cache files by design. Gate the only
    # external evaluation backbone cache explicitly; no training prior is used.
    required[".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"] = dict(
        path=".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        sha256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02")
    def command(name, argv):
        job["commands"].append(dict(name=name, argv=[str(x) for x in argv]))
    prefix = ["/usr/bin/env", "DYNAMIC_LR_PROTOCOL_MANIFEST=" + full(manifest)]
    python = full(LEAF / "bin/python")
    command("CEM_real_GPU_environment", [python, full(helpers / "check_gpu_environment.py"), "--method", "cem", "--output", full(result / "GPU_environment_acceptance.json")])
    for stage in [1, 2]:
        argv = [*prefix, python, full(helpers / "run_author.py"), "--runtime", full(RUNTIME_REMOTE),
                "--receipt", full(receipts[stage-1]), "--", "--configpath", f"configs/N3V_s{stage}/{a.scene}_prepared_lr.json",
                "--source_path", full(native), "--model_path", full(model), "--gpu", gpu_physical]
        if stage == 2:
            argv += ["--start_checkpoint", full(model / "chkpnt40000.pth")]
        command(f"CEM_author_stage{stage}", argv)
    command("CEM_render_HR", [*prefix, python, full(helpers / "render_hr.py"), "--method", "cem", "--runtime", full(RUNTIME_REMOTE),
            "--model", full(model), "--source", full(native), "--scene", a.scene, "--output", full(renders)])
    command("CEM_shared_PNG_metrics", ["{ROOT}/.venv/bin/python", full(evaluator), "--manifest", full(manifest),
            "--render-dir", full(renders), "--out", full(shared_metrics), "--method", "CEM_native_original_LR"])
    command("CEM_index", ["{ROOT}/.venv/bin/python", full(helpers / "index_result.py"), "--method", "cem", "--scene", a.scene,
            "--manifest", full(manifest), "--runtime", full(RUNTIME_REMOTE), "--model", full(model), "--native-source", full(native),
            "--renders", full(renders), "--training-receipts", *[full(x) for x in receipts], "--gpu-uuid", gpu_uuid,
            "--unified-metrics", full(shared_metrics / "summary.json"), "--output", full(result / "result_index.json")])
    job["required_files"] = [required[key] for key in sorted(required)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], required_files=len(required), commands=len(job["commands"]),
                         submitted=False, fixed_author_endpoints=[40000, 50000], physical_gpu_argument=gpu_physical), indent=2))


if __name__ == "__main__":
    main()
