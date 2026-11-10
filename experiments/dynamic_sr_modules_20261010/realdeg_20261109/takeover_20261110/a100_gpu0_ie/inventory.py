"""Targeted read-only IE-SRGS asset inventory and optional CUDA acceptance.

Run only within the fixed service's actual deployment/preflight job.  This file
does not build, install, copy, or change an asset.  Its JSON is evidence of disk
identity, not proof that a CUDA extension imports or runs on the selected GPU.
--gpu-check additionally imports Torch and checks one scalar CUDA autograd
operation; it does not import an external method or update a model parameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

DA_SHA = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
WRAPPER_SHA = "c1d57f2c1f3c64a31d8d9f289c7e6e965cb4639dfec370474a7fb18d1996c8ec"
SOURCE_COMMIT = "74d60d5bb0f49974ac0efa3497dbbd9e246480f4"
GPU0_UUID = "GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b"


def fingerprint(path, expected=None):
    path = Path(path)
    result = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
    if not path.is_file():
        return result
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        result.update(bytes=stat.st_size, sha256=digest.hexdigest())
        if expected is not None:
            result.update(expected_sha256=expected, sha_matches=result["sha256"] == expected)
    except (OSError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def runtime_inventory(runtime):
    receipt_path = runtime / "build_receipt.json"
    result = {"root": str(runtime), "receipt": fingerprint(receipt_path)}
    if not receipt_path.is_file():
        return result
    try:
        if receipt_path.stat().st_size > (2 << 20):
            raise ValueError("unexpected receipt exceeds 2 MiB")
        receipt = json.loads(receipt_path.read_text())
        keys = ("source_commit", "build_status", "source_root", "runtime_root", "torch_version",
                "python_abi", "cuda_home", "architectures", "max_jobs", "gpu_executed",
                "gpu_verified", "cpu_import_verified", "patch", "wrapper", "binary")
        result["recorded"] = {key: receipt[key] for key in keys if key in receipt}
        package = runtime / "lib/diff_gaussian_rasterization_rade"
        result["wrapper"] = fingerprint(package / "__init__.py", WRAPPER_SHA)
        binary = receipt.get("binary", {})
        name = binary.get("name")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".so"):
            raise ValueError("unsafe or missing receipt binary basename")
        result["binary"] = fingerprint(package / name, binary.get("sha256"))
        result["identity_checks"] = {
            "pinned_source_commit": receipt.get("source_commit") == SOURCE_COMMIT,
            "recorded_build_success": receipt.get("build_status") == "success",
            "python_abi_matches_running_python": receipt.get("python_abi") == sys.implementation.cache_tag,
            "wrapper_sha_matches": result["wrapper"].get("sha_matches", False),
            "binary_sha_matches_receipt": result["binary"].get("sha_matches", False),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def collect(project):
    task = project / "tasks/static_sr_shared_wu_20261024_a100"
    bases = [task, project]
    result = {
        "schema": "rd4_ie_targeted_readonly_inventory_v1",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project), "task_root": str(task),
        "python": {"executable": sys.executable, "abi": sys.implementation.cache_tag,
                   "version": sys.version, "prefix": sys.prefix},
        "scope": "targeted filesystem SHA plus nvcc --version; optional explicitly requested scalar Torch CUDA check; no model import, build, install, model training, or asset mutation",
        "weights": [], "runtimes": [], "sources": [], "compilers": [],
        "acceptance_boundary": "Presence, source hashes and a build receipt do not prove Torch/CUDA ABI compatibility or GPU execution; actual owner preflight must check those.",
    }
    for base in bases:
        result["weights"].append(fingerprint(base / "weight/static_sr_wu_20261022/depth_anything_v2_large/depth_anything_v2_vitl.pth", DA_SHA))
        result["runtimes"].append(runtime_inventory(base / "output/static_sr_wu_20261022/runtime/iesrgs"))
        upstream = base / "tmp/static_sr_wu_20261022/upstream/iesrgs"
        for relative in (
            "submodules/Depth-Anything-V2/depth_anything_v2/dpt.py",
            "submodules/Depth-Anything-V2/depth_anything_v2/dinov2.py",
            "submodules/diff-gaussian-rasterization-rade/setup.py",
            "submodules/diff-gaussian-rasterization-rade/cuda_rasterizer/rasterizer_impl.h",
            "submodules/diff-gaussian-rasterization-rade/diff_gaussian_rasterization_rade/__init__.py",
        ):
            result["sources"].append(fingerprint(upstream / relative))
        result["sources"].append(fingerprint(base / "output/static_sr_wu_20261024/next_teacher_gates_cpu_v1/iesrgs_clean_74d60d5/submodules/Depth-Anything-V2/depth_anything_v2/dpt.py"))
    candidates = [project / "tools/cuda-12.8-minimal/bin/nvcc",
                  task / "tools/cuda-12.8-minimal/bin/nvcc",
                  Path("/usr/local/cuda-12.8/bin/nvcc"), Path("/usr/local/cuda/bin/nvcc")]
    for compiler in candidates:
        item = fingerprint(compiler)
        if compiler.is_file():
            try:
                version = subprocess.run([str(compiler), "--version"], text=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         timeout=10, check=False)
                item.update(version_returncode=version.returncode, version_output=version.stdout[:4096])
            except (OSError, subprocess.TimeoutExpired) as error:
                item["version_error"] = f"{type(error).__name__}: {error}"
        result["compilers"].append(item)
    return result


def gpu_acceptance(expected_uuid):
    """Check the fixed-service visible CUDA device using a scalar, not a model."""
    result = {"requested": True, "expected_uuid": expected_uuid, "status": "failed"}
    try:
        import torch
        result.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      python_abi=sys.implementation.cache_tag,
                      cuda_available=torch.cuda.is_available(),
                      visible_device_count=torch.cuda.device_count())
        if not result["cuda_available"] or result["visible_device_count"] != 1:
            raise RuntimeError("expected exactly one service-visible CUDA device")
        torch.cuda.set_device(0)
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        raw_uuid = getattr(properties, "uuid", None)
        actual_uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
        result.update(device_index=device, device_name=properties.name,
                      device_uuid=actual_uuid, total_memory_bytes=properties.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(device)))
        if actual_uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("Torch-visible device UUID does not match the assigned GPU")
        value = torch.ones(1, device="cuda:0", dtype=torch.float32, requires_grad=True)
        loss = value.square().sum()
        loss.backward()
        torch.cuda.synchronize(0)
        result.update(scalar_forward=float(loss.item()), scalar_gradient=float(value.grad.item()))
        if result["scalar_forward"] != 1.0 or result["scalar_gradient"] != 2.0:
            raise RuntimeError("scalar CUDA forward/backward result mismatch")
        result.update(status="passed", synchronized=True,
                      model_parameters_updated=False, external_cuda_extension_imported=False)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/home/ubuntu/3DGS/4dsr")
    parser.add_argument("--gpu-check", action="store_true")
    parser.add_argument("--expected-gpu-uuid", default=GPU0_UUID)
    args = parser.parse_args()
    result = collect(Path(args.project_root))
    result["gpu_check"] = gpu_acceptance(args.expected_gpu_uuid) if args.gpu_check else {"requested": False}
    actual_torch = result["gpu_check"].get("torch_version")
    if actual_torch:
        for runtime in result["runtimes"]:
            if "identity_checks" in runtime:
                runtime["identity_checks"]["torch_version_matches_running_python"] = runtime["recorded"].get("torch_version") == actual_torch
    print(json.dumps(result, indent=2))
    return 1 if args.gpu_check and result["gpu_check"]["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
