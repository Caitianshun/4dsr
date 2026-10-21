"""Run one original CEM stage using the independent area4 observation runtime.

This entry does not dispatch or select a free GPU. A separately admitted broker
must supply exactly the physical CUDA mask passed by --gpu. Native CEM keeps
that mask and uses logical device zero internally, including Faiss.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--level", choices=["lr", "hr"], required=True)
    parser.add_argument("--gpu", required=True, help="Broker-assigned physical mask, never logical index")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert args.gpu.isdecimal() and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
    assert not args.receipt.exists(), "Refusing to overwrite an execution receipt"
    runtime, manifest, source, model, receipt = [p.resolve() for p in [args.runtime, args.manifest, args.source, args.model, args.receipt]]
    protocol = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    assert protocol["version"] == "native_area4_v1"
    assert protocol["core_losses_sampling_optimization_schedule_unchanged"]
    assert protocol["native_readonly_test_evaluation_disabled"]
    assert sha(runtime / "protocol_interface.py") == protocol["interface_sha256"]
    for item in protocol["core_source_files"] + protocol["configurations"]:
        assert sha(runtime / item["path"]) == item["sha256"]
    data = json.loads(manifest.read_text())
    assert data["physical_lr_derivation"]["formation"] == "area4"
    assert data["frame_indices"] == list(range(300)) and data["splits"]["test"] == ["cam00"]
    assert "cam00" not in data["splits"]["train"]
    assert (source / "colmap_0/sparse/0/points3D.ply").is_file(), "Native LR frame0 initializer is required"
    config_path = Path(f"configs/N3V_s{args.stage}/{data['scene']}_prepared_lr.json")
    endpoint = 40000 if args.stage == 1 else 50000
    assert json.loads((runtime / config_path).read_text())["iterations"] == endpoint
    checkpoint = model / "chkpnt40000.pth" if args.stage == 2 else None
    if checkpoint is not None:
        assert checkpoint.is_file(), "Stage2 restores this branch's native 40k checkpoint and RAdam"
    else:
        assert not (model / "cfg_args").exists(), "Never start over an existing native run"
    argv = [str(runtime / "train.py"), "--configpath", str(config_path), "--source_path", str(source),
            "--model_path", str(model), "--gpu", args.gpu]
    if checkpoint is not None:
        argv.extend(["--start_checkpoint", str(checkpoint)])
    os.environ.update(DYNAMIC_LR_PROTOCOL_MANIFEST=str(manifest), CEM_PROTOCOL_TRAIN_LEVEL=args.level,
                      TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
    # Do not let an inherited renderer switch alter held-out constructor sizes.
    os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    new_soft = soft if soft == resource.RLIM_INFINITY else max(soft, requested)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    state = dict(status="running", protocol_type="CEM_area4_backbone_control_v1", stage=args.stage,
        train_level=args.level, arm="native_area4_parent" if args.level == "lr" else "native_HR_reference",
        argv=argv, cwd=str(runtime), manifest=dict(path=str(manifest), sha256=sha(manifest)),
        source=dict(path=str(source), LR_initialization_pointcloud_sha256=sha(source / "colmap_0/sparse/0/points3D.ply")),
        runtime_receipt=dict(path=str(runtime / "protocol_interface_receipt.json"), sha256=sha(runtime / "protocol_interface_receipt.json")),
        start_checkpoint=None if checkpoint is None else dict(path=str(checkpoint), sha256=sha(checkpoint)),
        endpoint=endpoint, native_schedule_preserved=True, parent_added_affine=False, parent_added_warp=False,
        heldout_pixels_used=False, train_HR_used=args.level == "hr", author_optimizer_restored=args.stage == 2,
        device_guard=dict(status="passed_before_CUDA_import", physical_mask=args.gpu, native_logical_device=0),
        RLIMIT_NOFILE=dict(previous_soft=soft, new_soft=new_soft, hard=hard),
        CPU_thread_environment={key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        started_utc=datetime.now(timezone.utc).isoformat(), entry_sha256=sha(__file__))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    sys.argv = argv
    try:
        runpy.run_path(str(runtime / "train.py"), run_name="__main__")
        assert (model / f"chkpnt{endpoint}.pth").is_file()
        state["status"] = "completed"
        state["final_checkpoint"] = dict(path=str(model / f"chkpnt{endpoint}.pth"), sha256=sha(model / f"chkpnt{endpoint}.pth"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        import torch
        state.update(elapsed_seconds=time.monotonic() - began, finished_utc=datetime.now(timezone.utc).isoformat(),
                     python_executable=sys.executable, torch_version=torch.__version__, torch_CUDA=torch.version.cuda)
        if torch.cuda.is_initialized():
            properties = torch.cuda.get_device_properties(0)
            state.update(gpu_name=properties.name, gpu_total_memory=properties.total_memory,
                         torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         peak_memory_scope="Torch allocator only; external Faiss allocations excluded")
        receipt.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
