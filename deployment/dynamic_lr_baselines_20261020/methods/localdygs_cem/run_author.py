"""Run an unchanged author entrypoint and save elapsed time/Torch memory peak."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import resource
import sys
import time
import traceback


def verify_cem_broker_device(arguments, selected):
    gpu_args = [arguments[i+1] for i, token in enumerate(arguments[:-1]) if token == "--gpu"]
    gpu_args += [token.split("=", 1)[1] for token in arguments if token.startswith("--gpu=")]
    if len(gpu_args) != 1 or not selected.isdecimal() or gpu_args[0] != selected:
        raise ValueError("CEM physical --gpu must exactly equal the single broker CUDA_VISIBLE_DEVICES mask; no CUDA started")
    return dict(status="passed_before_author_import", broker_physical_mask=selected,
                original_author_gpu_argument=gpu_args[0], native_operator_logical_device=0,
                native_CEM_source_environment_assignment_unchanged=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--entry", default="train.py")
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("author_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.receipt.exists():
        raise SystemExit("Refusing to overwrite author execution receipt")
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    runtime = a.runtime.resolve()
    protocol = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    assert protocol["native_readonly_test_evaluation_disabled"]
    assert protocol["core_losses_sampling_optimization_schedule_unchanged"]
    assert os.environ.get("DYNAMIC_LR_PROTOCOL_MANIFEST")
    receipt = a.receipt.resolve()
    arguments = a.author_args[1:] if a.author_args[:1] == ["--"] else a.author_args
    device_guard = None
    if (runtime / "scene/v_gaussian_model.py").is_file():
        # CEM's unchanged __main__ assigns CUDA_VISIBLE_DEVICES=args.gpu.
        # Its CLI is therefore a host physical mask, while native CUDA/Faiss
        # operators address logical device 0 after that mask is installed.
        selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_guard = verify_cem_broker_device(arguments, selected)
    state = dict(status="running", argv=[str(runtime / a.entry), *arguments], cwd=str(runtime),
                 native_readonly_test_evaluation_disabled=True, heldout_pixels_used=False,
                 runtime_interface_receipt=protocol,
                 broker_device_guard=device_guard,
                 CPU_thread_environment={key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
                 started_utc=datetime.now(timezone.utc).isoformat(), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    start = time.monotonic()
    old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    new_soft = old_soft if old_soft == resource.RLIM_INFINITY else max(old_soft, requested)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    state["RLIMIT_NOFILE"] = dict(previous_soft=old_soft, new_soft=resource.getrlimit(resource.RLIMIT_NOFILE)[0], hard=hard,
                                   scope="this author subprocess only; native worker count and sampling unchanged")
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    sys.argv = [str(runtime / a.entry), *arguments]
    try:
        runpy.run_path(str(runtime / a.entry), run_name="__main__")
        state["status"] = "completed"
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        import torch
        state.update(elapsed_seconds=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat(),
                     python_executable=sys.executable, torch_version=torch.__version__, torch_CUDA=torch.version.cuda)
        environment = Path(sys.prefix)
        state["isolated_environment_identity"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [environment / "installed_packages.txt", environment / "upstream_commit.txt", environment / "compiler.txt"] if path.is_file()]
        if torch.cuda.is_initialized():
            properties = torch.cuda.get_device_properties(0)
            state.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         gpu_name=properties.name, gpu_total_memory=properties.total_memory,
                         peak_memory_scope="Torch allocator only; external Faiss/COLMAP allocations are not measured")
        receipt.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
