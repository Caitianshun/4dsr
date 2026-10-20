"""Build a private CEM sm80 runtime with CUDA devices hidden; never train."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/dynamic_lr_baselines_20261020/a100_cem_cpu_build"
LEAF = ROOT / "envs/dynamic_lr_baselines_20261020/cem_a100_sm80"
UPSTREAM = ROOT / "output/dynamic_lr_baselines_20261020/runtime/CEM-4DGS-7b5dfe3"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(value):
    p = OUT / "status.tmp"
    p.write_text(json.dumps(value, indent=2) + "\n")
    p.replace(OUT / "status.json")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only build requires hidden CUDA devices"
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(status="verifying_sources", pid=os.getpid(), started_unix=time.time(),
                  CPU_only=True, CUDA_VISIBLE_DEVICES="", no_GPU_training_authorized=True)
    try:
        write(record)
        inventory_path = OUT / "source_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        for entry in inventory["files"]:
            assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]
        for path, expected in inventory["git_commits"].items():
            actual = subprocess.check_output(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True).strip()
            assert actual == expected, (path, actual)
        record.update(source_inventory_sha256=sha(inventory_path), verified_source_files=len(inventory["files"]),
                      builder_sha256=sha(HERE / "build_isolated_env.sh"))
        base = ROOT / ".venv/bin/python"
        python = LEAF / "bin/python"
        base_site = subprocess.check_output([str(base), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
        leaf_site = Path(subprocess.check_output([str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip())
        (leaf_site / "remote_base_torch_readonly.pth").write_text(base_site + "\n")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", FORCE_CUDA="1", FOURDSR_ENV_RESUME="1",
                           FOURDSR_DEPS_PREVERIFIED="1", FOURDSR_CUDA_OVERLAY="1", TORCH_CUDA_ARCH_LIST="8.0",
                           FAISS_CUDA_ARCHITECTURES="80", MAX_JOBS="4", OMP_NUM_THREADS="4",
                           TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", PIP_REQUIRE_VIRTUALENV="true")
        stages = [
            ("pinned_build_dependencies", [str(python), "-m", "pip", "install", "setuptools==80.9.0", "wheel",
              "ninja==1.11.1.4", "packaging", "cmake==3.31.10", "swig==4.5.1",
              "fvcore==0.1.5.post20221221", "iopath==0.1.10"]),
            ("native_sm80_build", ["bash", str(HERE / "build_isolated_env.sh"), "cem", str(base), str(LEAF),
              str(UPSTREAM), str(ROOT / "tools/cuda-12.8-minimal")]),
        ]
        for stage, command in stages:
            record.update(status=stage)
            write(record)
            with (OUT / (stage + ".log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        probe = '''import hashlib,json,torch,faiss,pytorch3d,pytorch3d._C,diff_gaussian_rasterization_df,simple_knn._C
from pathlib import Path
assert not torch.cuda.is_initialized()
assert hasattr(faiss,'StandardGpuResources') and hasattr(faiss,'GpuIndexFlatL2')
modules=[pytorch3d._C,diff_gaussian_rasterization_df._C,simple_knn._C]
v=dict(status='CEM_sm80_CPU_import_passed_GPU_execution_not_tested',torch=str(torch.__version__),pytorch3d=pytorch3d.__version__,faiss=faiss.__version__,faiss_compile_options=faiss.get_compile_options(),CUDA_initialized=False,extensions=[dict(path=m.__file__,sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for m in modules])
print(json.dumps(v,indent=2))
'''
        result = subprocess.run([str(python), "-c", probe], cwd=ROOT, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (OUT / "CPU_import.json").write_text(result.stdout)
        (OUT / "CPU_import.stderr.log").write_text(result.stderr)
        record.update(status="completed_CPU_build_no_GPU_execution", seconds=time.monotonic() - started,
                      import_acceptance=json.loads(result.stdout), leaf=str(LEAF),
                      calibrated_runtime=str(ROOT / "output/dynamic_lr_baselines_20261020/runtime/CEM-4DGS-calibrated-v3"))
        write(record)
    except BaseException:
        record.update(status="failed_CPU_build", seconds=time.monotonic() - started, error=traceback.format_exc())
        write(record)
        raise


if __name__ == "__main__":
    main()
