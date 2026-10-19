#!/usr/bin/env python3
"""Build the pinned STG extensions in an isolated target without using a GPU.

Run on the A100 host after transferring the audited source snapshot, its
deployment manifest, and the pinned dependency wheels. Existing shared Python
packages and the upstream source snapshot are never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume-failed-build", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "third_party/stg_20261018/source"
    runtime = source.parent / "runtime"
    target = source.parent / "pydeps"
    out = root / "deployment/stg_20261018"
    wheels = out / "wheels"
    python = root / ".venv/bin/python"
    cuda = root / "tools/cuda-12.8-minimal"
    if root != Path("/home/ubuntu/3DGS/4dsr"):
        raise ValueError("This audited deployment targets the fixed A100 project root")
    if not args.resume_failed_build and (runtime.exists() or target.exists()):
        raise FileExistsError("Isolated runtime/pydeps already exists; inspect instead of overwriting")
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        file = source / item["path"]
        if file.is_symlink() or not file.is_file() or file.stat().st_size != item["bytes"] or digest(file) != item["sha256"]:
            raise ValueError("Source identity mismatch: " + item["path"])
    receipt = {"started_utc": now(), "state": "preparing", "host": "a100-train",
               "upstream_commit": manifest["upstream_commit"], "source_manifest_sha256": digest(manifest_path),
               "source": str(source), "runtime": str(runtime), "pydeps": str(target),
               "python": str(python), "CUDA_HOME": str(cuda), "CUDA_VISIBLE_DEVICES": "",
               "TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "4", "commands": [],
               "gpu_execution_performed": False, "shared_environment_modified": False,
               "upstream_setup_shell_executed": False}
    status = out / "build_receipt.json"
    previous = None
    if args.resume_failed_build:
        previous = json.loads(status.read_text())
        if previous.get("state") != "failed" or previous.get("source_manifest_sha256") != digest(manifest_path):
            raise ValueError("Only this failed build with an unchanged source manifest may resume")
        archive = out / "build_receipt_attempt1.json"
        with archive.open("x") as stream:
            json.dump(previous, stream, indent=2)
        receipt["prior_attempt"] = str(archive)
    save(status, receipt)
    started = time.monotonic()
    try:
        if not args.resume_failed_build:
            shutil.copytree(source, runtime, symlinks=False)
            target.mkdir()
        # CUDA 12.8 no longer exposes FLT_MAX through an incidental include.
        # Add its standard declaration only in the disposable runtime copy.
        relative = Path("thirdparty/gaussian_splatting/submodules/simple-knn/simple_knn.cu")
        original = (source / relative).read_text()
        if original.count("#include <vector>\n") != 1:
            raise ValueError("Pinned simple-knn header anchor changed")
        patched = original.replace("#include <vector>\n", "#include <vector>\n#include <cfloat>\n")
        current = (runtime / relative).read_text()
        if current not in (original, patched):
            raise ValueError("Unexpected runtime simple-knn modification")
        (runtime / relative).write_text(patched)
        patch = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                            fromfile="source/" + str(relative), tofile="runtime/" + str(relative)))
        (out / "simple_knn_cfloat.patch").write_text(patch)
        receipt["compatibility_patch"] = {"path": str(relative), "source_sha256": digest(source / relative),
                                          "runtime_sha256": digest(runtime / relative),
                                          "patch_sha256": digest(out / "simple_knn_cfloat.patch"),
                                          "reason": "Explicit cfloat declaration of FLT_MAX for CUDA 12.8; algorithm unchanged"}
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="8.0",
                           MAX_JOBS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                           PYTHONPATH=str(target), PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                           PATH=str(cuda / "bin") + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
                           LD_LIBRARY_PATH=str(cuda / "lib64"),
                           TORCH_EXTENSIONS_DIR=str(out / "torch_extensions"))
        def run(label: str, argv: list[str], cwd: Path) -> None:
            if previous and any(c["label"] == label and c.get("returncode") == 0 for c in previous["commands"]):
                receipt["commands"].append({"label": label, "returncode": 0, "reused_success_from": receipt["prior_attempt"]})
                return
            record = {"label": label, "argv": argv, "cwd": str(cwd), "started_utc": now()}
            receipt["commands"].append(record)
            receipt["state"] = label
            save(status, receipt)
            log_path = out / (label + ("_retry1" if args.resume_failed_build else "") + ".log")
            record["log_path"] = str(log_path)
            with log_path.open("xb") as log:
                result = subprocess.run(argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, finished_utc=now())
            save(status, receipt)
            if result.returncode:
                raise RuntimeError(f"{label} failed with exit code {result.returncode}")
        dependencies = sorted(wheels.glob("natsort-8.4.0-*.whl")) + sorted(wheels.glob("kornia-0.6.12-*.whl"))
        if len(dependencies) != 2:
            raise FileNotFoundError("Exactly the two pinned dependency wheels are required")
        run("install_dependencies", [str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                                      "--no-compile", "--target", str(target), *map(str, dependencies)], root)
        for label, folder in [("ch9", "gaussian_rasterization_ch9"), ("simple_knn", "simple-knn")]:
            extension = runtime / "thirdparty/gaussian_splatting/submodules" / folder
            run("build_" + label, [str(python), "setup.py", "bdist_wheel", "--dist-dir", str(wheels)], extension)
        extension_wheels = sorted(wheels.glob("diff_gaussian_rasterization_ch9-*.whl")) + sorted(wheels.glob("simple_knn-*.whl"))
        if len(extension_wheels) != 2:
            raise FileNotFoundError("Expected exactly one wheel for each compiled extension")
        run("install_extensions", [str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                                    "--no-compile", "--target", str(target), *map(str, extension_wheels)], root)
        # No tensor is allocated on CUDA, and no CUDA availability/device API is
        # called here. is_initialized only inspects PyTorch's Python flag.
        probe = """import json,sys,torch,natsort,kornia
import diff_gaussian_rasterization_ch9 as raster
import diff_gaussian_rasterization_ch9._C as raster_c
import simple_knn._C as knn
assert not torch.cuda.is_initialized()
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'natsort':natsort.__version__,'kornia':kornia.__version__,'raster_path':raster.__file__,'raster_extension':raster_c.__file__,'knn_extension':knn.__file__,'cuda_initialized':torch.cuda.is_initialized(),'scope':'CPU import only; no rendering/training acceptance'}))
"""
        run("cpu_import_check", [str(python), "-c", probe], root)
        import_log = out / ("cpu_import_check" + ("_retry1" if args.resume_failed_build else "") + ".log")
        receipt["cpu_import_check"] = json.loads(import_log.read_text().splitlines()[-1])
        for key in ("raster_extension", "knn_extension"):
            if target not in Path(receipt["cpu_import_check"][key]).parents:
                raise ValueError("CPU import resolved outside the isolated dependency target")
        receipt["wheels"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(wheels.glob("*.whl"))]
        receipt["extensions"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(target.rglob("*.so"))]
        receipt["state"] = "cpu_build_and_import_verified"
    except BaseException as error:
        receipt.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(finished_utc=now(), elapsed_seconds=round(time.monotonic()-started, 3))
        save(status, receipt)
    print(json.dumps({"state": receipt["state"], "receipt": str(status)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
