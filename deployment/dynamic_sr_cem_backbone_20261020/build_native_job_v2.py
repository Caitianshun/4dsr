"""Freeze one native area4 CEM parent or independent HR reference job.

No submission or CUDA import occurs. All generated paths are separate from
the bicubic-LR benchmark, including initialization and native model files.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path("output/dynamic_sr_cem_backbone_20261020")
HERE = Path("deployment/dynamic_sr_cem_backbone_20261020")
EXP = Path("experiments/dynamic_sr_cem_area_20261020")
OLD = Path("deployment/dynamic_lr_baselines_20261020")
GPUS = {("a100-train", 0): "GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b",
        ("a100-train", 1): "GPU-b79cd3fe-81f0-f449-30be-432e2857e517",
        ("cts", 0): "GPU-fd2728bf-7bb9-4ce0-fc1b-6ab34ae0a910"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    p.add_argument("--level", choices=["lr", "hr"], required=True)
    p.add_argument("--host", choices=["a100-train", "cts"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--initialization", type=Path, required=True, help="Returned, verified initialization directory")
    p.add_argument("--asset-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    assert a.host == "a100-train" if a.level == "lr" else a.host == "cts"
    arm = "native_parent" if a.level == "lr" else "native_HRref"
    runtime = BATCH / "runtime/cem-native-area4-v1"
    model_out = BATCH / "runs" / (a.scene + "_" + arm + "_v1")
    manifest = Path("data/dynamic_sr/full_time_20261007/prepared") / (a.scene + "_area4") / "manifest_train_ready_seed20261007.json"
    initialization = a.initialization.resolve().relative_to(ROOT)
    initial = json.loads((ROOT / initialization / "complete.json").read_text())
    assert initial["status"] == "completed" and initial["manifest_sha256"] == sha(ROOT / manifest)
    assert initial["native_frames"] == [0] and not initial["used_HR"] and not initial["used_test_pixels"]
    assert initial["COLMAP_poses_intrinsics_unchanged"]
    source = initialization / "cem_data"
    assert sha(ROOT / source / "colmap_0/sparse/0/points3D.ply") == initial["cem_initialization"]["sha256"]
    receipt = a.asset_receipt.resolve().relative_to(ROOT)
    verified = json.loads((ROOT / receipt).read_text())
    assert verified["status"] == "verified" and verified["all_listed_file_bytes_rehashed"]
    inventory = Path(verified["inventory_path"])
    assert sha(ROOT / inventory) == verified["inventory_sha256"]
    prefix = "a100" if a.host == "a100-train" else "cts"
    assert inventory.name.startswith(prefix + "_" + a.scene)
    env = Path("envs/dynamic_lr_baselines_20261020") / ("cem_a100_sm80" if prefix == "a100" else "cem")
    python = "{ROOT}/" + str(env / "bin/python")
    absroot = lambda value: "{ROOT}/" + str(value)
    commands = [dict(name="real_GPU_environment", argv=[python, absroot(OLD / "methods/localdygs_cem/check_gpu_environment.py"),
                     "--method", "cem", "--output", absroot(model_out / "GPU_environment.json")])]
    if a.level == "hr":
        commands.append(dict(name="native_HR_startup_fixture", argv=[python, absroot(EXP / "native_capacity_fixture.py"),
            "--runtime", absroot(runtime), "--manifest", absroot(manifest), "--source", absroot(source),
            "--gpu", str(a.gpu), "--iterations", "5", "--output", absroot(model_out / "startup_fixture")]))
    for stage in [1, 2]:
        commands.append(dict(name=f"native_stage{stage}", argv=[python, absroot(EXP / "native_stage.py"),
            "--runtime", absroot(runtime), "--manifest", absroot(manifest), "--source", absroot(source),
            "--model", absroot(model_out / "model"), "--stage", str(stage), "--level", a.level,
            "--gpu", str(a.gpu), "--receipt", absroot(model_out / f"stage{stage}.json")]))
    for camera, directory in [("cam00", "renders_hr"), ("cam01", "train_cam01")]:
        commands.append(dict(name="native_HR_render_" + camera, argv=[python, absroot(EXP / "render_hr.py"),
            "--runtime", absroot(runtime), "--manifest", absroot(manifest),
            "--checkpoint", absroot(model_out / "model/chkpnt50000.pth"),
            "--training-receipt", absroot(model_out / "stage2.json"), "--kind", "native", "--gpu", str(a.gpu),
            "--camera", camera, "--output", absroot(model_out / directory)]))
    if a.host == "a100-train":
        commands.append(dict(name="shared_A100_PNG_metrics", argv=["{ROOT}/.venv/bin/python", absroot(OLD / "evaluate_images.py"),
            "--manifest", absroot(manifest), "--render-dir", absroot(model_out / "renders_hr"),
            "--out", absroot(model_out / "metrics"), "--method", "CEM_area4_" + arm]))
    commands.append(dict(name="native_result_index", argv=["{ROOT}/.venv/bin/python", absroot(HERE / "index_native.py"),
        "--run", absroot(model_out), "--manifest", absroot(manifest), "--level", a.level]))
    paths = [Path(__file__).resolve().relative_to(ROOT), HERE / "run_job.py", HERE / "index_native.py",
             EXP / "native_stage.py", OLD / "methods/localdygs_cem/check_gpu_environment.py",
             EXP / "render_hr.py", EXP / "sr_stage.py", EXP / "sr_protocol.py", OLD / "evaluate_images.py",
             Path("experiments/dynamic_sr_20260918/evaluate.py"), receipt, inventory,
             initialization / "complete.json"]
    if a.level == "hr":
        paths += [EXP / "native_capacity_fixture.py", OLD / "methods/localdygs_cem/run_author.py"]
    paths += [f.relative_to(ROOT) for f in sorted((ROOT / source).rglob("*")) if f.is_file()]
    runtime_inventory = json.loads((ROOT / runtime / "native_runtime_inventory.json").read_text())
    paths += [runtime / item["path"] for item in runtime_inventory["files"]]
    paths += [runtime / "native_runtime_inventory.json", runtime / "protocol_interface_receipt.json"]
    required = [dict(path=str(f), sha256=sha(ROOT / f)) for f in sorted(set(paths))]
    job = dict(protocol="CEM_area4_backbone_control_v1", input_formation="area4", heldout_used_for_training=False,
        job_id=f"codex-cem-area4-{a.scene.split('_')[0]}-{arm}-{prefix}g{a.gpu}-20261020-v2".replace("_", "-"),
        method="CEM_native_area4", arm=arm, scene=a.scene, gpu_uuid=GPUS[a.host, a.gpu],
        out=str(model_out), commands=commands, required_files=required,
        input_receipts=[dict(path=str(receipt), inventory_sha256=verified["inventory_sha256"])],
        env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CEM_PROTOCOL_TRAIN_LEVEL=a.level,
            DYNAMIC_LR_PROTOCOL_MANIFEST=absroot(manifest),
            LD_LIBRARY_PATH=absroot(env / "cuda-toolkit-overlay/lib64")),
        return_paths=[str(model_out)], training_HR=a.level == "hr", native_schedule="0->40000->50000",
        comparison_limit="Native CEM initialization/optimizer/topology retained; HRref privileged independent diagnostic")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], path=str(a.output), sha256=sha(a.output), required=len(required))))


if __name__ == "__main__":
    main()
