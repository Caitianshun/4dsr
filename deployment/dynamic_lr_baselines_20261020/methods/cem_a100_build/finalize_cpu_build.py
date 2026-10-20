"""Record the completed CEM private build without importing CUDA libraries."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/dynamic_lr_baselines_20261020/a100_cem_cpu_build"
LEAF = ROOT / "envs/dynamic_lr_baselines_20261020/cem_a100_sm80"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU provenance step requires hidden devices"
    status = OUT / "recovery_01/status.json"
    state = json.loads(status.read_text())
    assert state["status"] == "completed_CPU_build_no_GPU_execution"
    assert state["acceptance"]["CUDA_initialized"] is False
    assert not (LEAF / "build_identity.json").exists(), "Refusing changed build identity"
    inventory = json.loads((OUT / "source_inventory.json").read_text())
    for entry in inventory["files"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    for path, expected in inventory["git_commits"].items():
        actual = subprocess.check_output(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True).strip()
        assert actual == expected, (path, actual)
    shutil.copy2(OUT / "cuda_profiler_header_receipt.json", LEAF / "cuda_profiler_header_receipt.json")
    files = [LEAF / name for name in ["installed_packages.txt", "upstream_commit.txt", "compiler.txt",
             "cuda_profiler_header_receipt.json", "cuda-toolkit-overlay/overlay_receipt.json", "cuda-toolkit-overlay/include/cuda_profiler_api.h"]]
    for entry in state["acceptance"]["extensions"]:
        path = Path(entry["path"])
        assert sha(path) == entry["sha256"]
        assert path.resolve().is_relative_to(LEAF.resolve()), path
        files.append(path)
    value = dict(status="CEM_sm80_CPU_build_verified_no_GPU_execution", CUDA_initialized=False,
                 native_GPU_operators_execution="pending admitted GPU environment fixture",
                 source_inventory=identity(OUT / "source_inventory.json"), verified_source_files=len(inventory["files"]),
                 source_bytes_unchanged_after_build=True, git_commits=inventory["git_commits"],
                 author_upstream_commit="7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd",
                 official_dependencies=dict(pytorch3d=dict(version="0.7.9", commit="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"),
                      faiss=dict(version="1.12.0", commit="e8234e563f1ecef5f036e83c3cfee366d3f1fbca", native_GPU_classes_present=True)),
                 original_failure=identity(OUT / "status.json"), completed_recovery=identity(status),
                 CPU_import_acceptance=state["acceptance"], build_device_environment="CUDA_VISIBLE_DEVICES=''",
                 TORCH_CUDA_ARCH_LIST="8.0", FAISS_CUDA_ARCHITECTURES="80", MAX_JOBS=4,
                 compiler_scope="private leaf CUDA overlay; base Torch/toolkit read-only",
                 source_patches=[], external_header_dependency=json.loads((LEAF / "cuda_profiler_header_receipt.json").read_text()),
                 build_helpers=[identity(HERE / name) for name in ["build_isolated_env.sh", "prepare_cuda_overlay.py", "run_cpu_build.py", "continue_cpu_build.py", "finalize_cpu_build.py"]],
                 frozen_environment_files=[identity(p) for p in sorted(set(files))])
    payload = json.dumps(value, indent=2) + "\n"
    (LEAF / "build_identity.json").write_text(payload)
    metadata = OUT / "leaf_metadata"
    metadata.mkdir(exist_ok=False)
    for name in ["build_identity.json", "installed_packages.txt", "upstream_commit.txt", "compiler.txt", "cuda_profiler_header_receipt.json"]:
        shutil.copy2(LEAF / name, metadata / name)
    shutil.copy2(LEAF / "cuda-toolkit-overlay/overlay_receipt.json", metadata / "overlay_receipt.json")
    (OUT / "finalized.json").write_text(json.dumps(dict(status=value["status"], identity=identity(LEAF / "build_identity.json"),
                                finalized_without_GPU_import=True), indent=2) + "\n")
    print(json.dumps(dict(status=value["status"], frozen_environment_files=len(files), source_files_verified=len(inventory["files"]),
                         build_identity=identity(LEAF / "build_identity.json")), indent=2))


if __name__ == "__main__":
    main()
