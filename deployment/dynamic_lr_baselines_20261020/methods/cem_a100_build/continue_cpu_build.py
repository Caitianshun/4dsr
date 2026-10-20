"""Continue the diagnosed header-only CEM CPU build failure, no CUDA device.

The original failure is retained. The added official CUDA 12.8 profiler header
belongs only to the leaf overlay; native author/PyTorch3D/Faiss source is intact.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output/dynamic_lr_baselines_20261020/a100_cem_cpu_build"
RECOVERY = OUT / "recovery_01"
LEAF = ROOT / "envs/dynamic_lr_baselines_20261020/cem_a100_sm80"
UPSTREAM = ROOT / "output/dynamic_lr_baselines_20261020/runtime/CEM-4DGS-7b5dfe3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(value):
    path = RECOVERY / "status.tmp"
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.replace(RECOVERY / "status.json")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    RECOVERY.mkdir(exist_ok=False)
    old = json.loads((OUT / "status.json").read_text())
    assert old["status"] == "failed_CPU_build"
    assert "fatal error: cuda_profiler_api.h: No such file" in (OUT / "native_sm80_build.log").read_text()
    header = LEAF / "cuda-toolkit-overlay/include/cuda_profiler_api.h"
    assert sha(header) == "bb8e03c217df1309ccfe3249c4fac1bcd577bdcff5c65944217548195204118e"
    inventory = json.loads((OUT / "source_inventory.json").read_text())
    for entry in inventory["files"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    started = time.monotonic()
    record = dict(status="continuing_CPU_build", CPU_only=True, CUDA_VISIBLE_DEVICES="", pid=os.getpid(),
                  started_unix=time.time(), original_failure_sha256=sha(OUT / "status.json"),
                  source_inventory_sha256=sha(OUT / "source_inventory.json"), verified_source_files=len(inventory["files"]),
                  private_header_dependency=json.loads((OUT / "cuda_profiler_header_receipt.json").read_text()),
                  no_author_or_dependency_source_patch=True, no_GPU_execution_authorized=True)
    write(record)
    python = LEAF / "bin/python"
    toolkit = LEAF / "cuda-toolkit-overlay"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(toolkit), CUDAToolkit_ROOT=str(toolkit),
               PATH=f"{LEAF}/bin:{toolkit}/bin:" + env.get("PATH", "/usr/bin:/bin"),
               CPATH=str(toolkit / "include"), LIBRARY_PATH=str(toolkit / "lib64"),
               LD_LIBRARY_PATH=f"{toolkit}/lib64:{ROOT}/tools/runtime-libs", TORCH_CUDA_ARCH_LIST="8.0",
               MAX_JOBS="4", OMP_NUM_THREADS="4", FORCE_CUDA="1", TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp",
               PIP_REQUIRE_VIRTUALENV="true")
    faiss = LEAF / "build_sources/faiss"
    commands = [("faiss_sm80_continue", [str(LEAF / "bin/cmake"), "--build", str(faiss / "build"), "--target", "swigfaiss", "--parallel", "4"]),
                ("faiss_install", [str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(faiss / "build/faiss/python")]),
                ("native_raster_KNN_install", [str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                    str(UPSTREAM / "submodules/diff_gaussian_rasterization_df"), str(UPSTREAM / "submodules/simple-knn")])]
    try:
        for name, argv in commands:
            record["status"] = name
            write(record)
            with (RECOVERY / f"{name}.log").open("x") as stream:
                subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        probe = '''import hashlib,json,torch,faiss,pytorch3d,pytorch3d._C,diff_gaussian_rasterization_df,simple_knn._C
from pathlib import Path
assert not torch.cuda.is_initialized()
assert hasattr(faiss,'StandardGpuResources') and hasattr(faiss,'GpuIndexFlatL2')
modules=[pytorch3d._C,diff_gaussian_rasterization_df._C,simple_knn._C]
files=[Path(m.__file__) for m in modules]+list(Path(faiss.__file__).parent.glob('*.so'))
v=dict(status='CEM_sm80_CPU_import_passed_GPU_execution_not_tested',torch=str(torch.__version__),pytorch3d=pytorch3d.__version__,faiss=faiss.__version__,faiss_compile_options=faiss.get_compile_options(),CUDA_initialized=False,extensions=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
print(json.dumps(v,indent=2))
'''
        result = subprocess.run([str(python), "-c", probe], cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (RECOVERY / "CPU_import.json").write_text(result.stdout)
        (RECOVERY / "CPU_import.stderr.log").write_text(result.stderr)
        (LEAF / "installed_packages.txt").write_text(subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True))
        (LEAF / "upstream_commit.txt").write_text(subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True))
        (LEAF / "compiler.txt").write_text(subprocess.check_output([str(toolkit / "bin/nvcc"), "--version"], env=env, text=True))
        record.update(status="completed_CPU_build_no_GPU_execution", seconds=time.monotonic()-started,
                      acceptance=json.loads(result.stdout), leaf=str(LEAF))
    except BaseException:
        record.update(status="failed_CPU_build_recovery", seconds=time.monotonic()-started, error=traceback.format_exc())
        raise
    finally:
        write(record)


if __name__ == "__main__":
    main()
