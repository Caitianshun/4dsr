"""Build the pinned IE-SRGS RADE extension in a fresh isolated output directory.

Uses CPU host compilation and nvcc, hides all GPUs, never installs into an
existing Python environment. The only patch is an explicit C++ standard header.
The resulting receipt distinguishes CPU import acceptance from GPU acceptance.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_COMMIT = "74d60d5bb0f49974ac0efa3497dbbd9e246480f4"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def finalize_receipt(source, output, patch_path, cuda_home, architectures, jobs):
    """Record a successful local build; does not run any GPU call."""
    import torch
    source, output, patch_path = Path(source).resolve(), Path(output).resolve(), Path(patch_path).resolve()
    binaries = list((output / "lib/diff_gaussian_rasterization_rade").glob("_C*.so"))
    if len(binaries) != 1:
        raise ValueError("expected exactly one isolated RADE binary")
    receipt = dict(source_commit=SOURCE_COMMIT, build_status="success",
                   source_root=str(source), runtime_root=str(output),
                   torch_version=torch.__version__, python_abi=sys.implementation.cache_tag,
                   cuda_home=str(cuda_home), architectures=architectures, max_jobs=jobs,
                   gpu_executed=False, gpu_verified=False,
                   patch=dict(path=str(patch_path), sha256=sha(patch_path),
                              reason="explicit <cstdint> needed for std::uintptr_t and uint32_t with CUDA12.8"),
                   wrapper=dict(sha256=sha(output / "lib/diff_gaussian_rasterization_rade/__init__.py")),
                   binary=dict(name=binaries[0].name, sha256=sha(binaries[0])),
                   cuda_sources=[dict(path=str(p.relative_to(source)), sha256=sha(p))
                                 for p in sorted(source.rglob("*"))
                                 if p.is_file() and p.suffix in (".cu", ".h", ".cpp", ".py")
                                 and "third_party" not in p.parts],
                   cpu_import_verified=False)
    path = output / "build_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-home", default="/usr/local/cuda-12.8")
    p.add_argument("--architectures", default="8.0;8.6;12.0")
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    repo, out = Path(a.upstream).resolve(), Path(a.out).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        raise ValueError("IE-SRGS checkout is not the pinned source")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("source checkout must be clean")
    out.mkdir(parents=True, exist_ok=False)
    original = repo / "submodules/diff-gaussian-rasterization-rade"
    source = out / "source_cuda128"
    shutil.copytree(original, source, ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info", "*.so"))
    header = source / "cuda_rasterizer/rasterizer_impl.h"
    before = header.read_text()
    after = before.replace("#pragma once\n", "#pragma once\n#include <cstdint>\n", 1)
    if before == after or "#include <cstdint>" in before:
        raise ValueError("unexpected header contents; inspect source before patching")
    header.write_text(after)
    patch = out / "cuda128_cstdint.patch"
    patch.write_text("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                     fromfile="a/cuda_rasterizer/rasterizer_impl.h", tofile="b/cuda_rasterizer/rasterizer_impl.h")))
    package = out / "lib/diff_gaussian_rasterization_rade"
    package.mkdir(parents=True)
    shutil.copy2(original / "diff_gaussian_rasterization_rade/__init__.py", package / "__init__.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               CUDA_HOME=a.cuda_home, TORCH_CUDA_ARCH_LIST=a.architectures, MAX_JOBS=str(a.jobs))
    command = [sys.executable, "setup.py", "build_ext", "--build-lib", str(out / "lib"),
               "--build-temp", str(out / "build_cuda128")]
    with (out / "build_cuda128.log").open("x") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = finalize_receipt(source, out, patch, a.cuda_home, a.architectures, a.jobs)
    print(receipt)


if __name__ == "__main__":
    main()
