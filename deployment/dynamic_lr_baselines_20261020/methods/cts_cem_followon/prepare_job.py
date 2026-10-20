"""Freeze the one newly authorized CTS CEM Cook job; never submit it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "localdygs_cem"


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("Refusing an existing job identity")
    runtime = ROOT / "third_party/cem4dgs_calibrated_runtime_v3_20261020"
    train = ast.parse((runtime / "train.py").read_text())
    training = next(n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == "training")
    loads = [n for n in ast.walk(training) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load"]
    assert len(loads) == 1 and any(k.arg == "weights_only" and isinstance(k.value, ast.Constant)
                                 and k.value.value is False for k in loads[0].keywords)
    configs = [json.loads((runtime / f"configs/N3V_s{stage}/cook_spinach_prepared_lr.json").read_text()) for stage in [1, 2]]
    assert [c["iterations"] for c in configs] == [40000, 50000]
    assert [c["start_duration"] for c in configs] == [10, 300]
    assert all(c["resolution"] == 1 for c in configs)
    subprocess.run([sys.executable, str(SHARED / "build_job.py"), "--method", "cem", "--scene", "cook_spinach",
        "--skip-initialization", "--input-receipt", "deployment/dynamic_lr_baselines_20261020/input_verified.json",
        "--output", str(a.output)], cwd=ROOT, check=True)
    job = json.loads(a.output.read_text())
    stages = job["commands"]
    first = next(i for i, c in enumerate(stages) if c["name"].endswith("_author_stage1"))
    second = next(i for i, c in enumerate(stages) if c["name"].endswith("_author_stage2"))
    assert second == first + 1
    assert any(c["name"].endswith("_real_GPU_environment") for c in stages[:first])
    stage2 = stages[second]["argv"]
    checkpoint = "{ROOT}/output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/cook_spinach/cem/model/chkpnt40000.pth"
    assert stage2[stage2.index("--start_checkpoint") + 1] == checkpoint
    for c in [stages[first], stages[second]]:
        assert c["argv"][c["argv"].index("--gpu") + 1] == "0"
    base = "{ROOT}/output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/cook_spinach/cem"
    stages.insert(second, dict(name="cook_spinach_cem_native_40k_checkpoint_boundary", argv=[
        "{ROOT}/envs/dynamic_lr_baselines_20261020/cem/bin/python", "{ROOT}/" + str(HERE.relative_to(ROOT)) + "/checkpoint_boundary.py",
        "--checkpoint", checkpoint, "--stage1-receipt", base + "/training_stage1_receipt.json",
        "--runtime", "{ROOT}/third_party/cem4dgs_calibrated_runtime_v3_20261020", "--output", base + "/stage_boundary_acceptance.json"]))
    # The returned CPU build identity refers to the real CTS leaf. Freeze all
    # runtime binaries and build metadata inside that project, not historical
    # provenance locations on another host.
    source = ROOT / "output/dynamic_lr_baselines_20261020/localdygs_cem/cem_environment/build_identity.json"
    build = json.loads(source.read_text())
    assert build["status"] == "CPU_build_and_import_passed"
    remote_root = Path("/home/cts/Project/4DSR")
    extra = [{"path": "envs/dynamic_lr_baselines_20261020/cem/build_identity.json", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]
    extra.append(dict(path="envs/dynamic_lr_baselines_20261020/cem/upstream_commit.txt",
        sha256=hashlib.sha256(source.with_name("upstream_commit.txt").read_bytes()).hexdigest()))
    for key in ["compiler", "installed_packages", "official_profiler_header", "profiler_header_provenance", "overlay", "build_recipe", "build_log", "dependency_source_inventory", "dependency_source_verification"]:
        row = build[key]
        extra.append(dict(path=str(Path(row["path"]).relative_to(remote_root)), sha256=row["sha256"]))
    extra += [dict(path=str(Path(row["path"]).relative_to(remote_root)), sha256=row["sha256"]) for row in build["extension_binaries"]]
    extra.append(identity(HERE / "checkpoint_boundary.py"))
    extra.append(identity(__file__))
    combined = {row["path"]: row for row in job["required_files"]}
    for row in extra:
        if row["path"] in combined:
            assert combined[row["path"]] == row
        combined[row["path"]] = row
    job["required_files"] = sorted(combined.values(), key=lambda row: row["path"])
    job["checkpoint_stage_connection"] = dict(stage1_endpoint=40000, stage2_total_endpoint=50000,
        stage2_start_checkpoint=checkpoint, native_weights_only_false_verified=True,
        readonly_CPU_40k_checkpoint_schema_gate=True, native_optimization_schedule_unchanged=True)
    job["isolated_CTS_CEM_build_identity_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], commands=len(stages), required_files=len(combined),
        job_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), submitted=False), indent=2))


if __name__ == "__main__":
    main()
