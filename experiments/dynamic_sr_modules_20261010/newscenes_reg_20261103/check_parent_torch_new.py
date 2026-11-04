"""Owner-run CPU tensor acceptance of a real gateaff parent; never uses CUDA.

This file is syntax-checked by P2 but must be executed later using Claude's
absolute mml_bw interpreter. The system CPU interpreter has no torch.
It does not convert the checkpoint into the old native-parent schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import protocol_new as protocol
from materialize_bsync_new import physical_parent


def check(scene, parent, out):
    out = output_path(out)
    require(not out.exists(), "Existing tensor acceptance preserved")
    mp = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    _, data, _ = protocol.validate_manifest(mp, formation="area4")
    parent_value, lr_plan = physical_parent(scene, parent, data)
    import torch
    require(not torch.cuda.is_initialized(), "CPU-only acceptance requires no initialized CUDA context")
    payload = torch.load(local(parent), map_location="cpu", weights_only=False)
    require(payload["schema"] == "cullretrain_lr_v1" and payload["plan"] == lr_plan
            and payload["metadata"]["state"] == parent_value["state"]
            and payload["metadata"]["arm"] == "gateaff", "Tensor payload/sidecar/plan differs")
    model = payload["model"]
    require(len(model) == 14, "Expected author fourteen-item Gaussian capture")
    n = model[1].shape[0]
    require(tuple(model[1].shape) == (n, 3) and n > 0, "Invalid actual parent point shape")
    groups = model[12]["param_groups"]
    require([g["name"] for g in groups] ==
            ["xyz", "deformation", "grid", "f_dc", "f_rest", "opacity", "scaling", "rotation"],
            "Actual author eight-group optimizer differs")
    sampler = payload["sampler"]
    require(sampler["length"] == data["training_observations"]
            and sampler["batch_size"] == lr_plan["configuration"]["OptimizationParams"]["batch_size"],
            "Checkpoint sampler differs from the full scene/author batch")
    affine = payload["affine"]
    require(affine["names"] == data["train_cameras"] and tuple(affine["delta"].shape) ==
            (len(data["train_cameras"]), 3, 4), "Actual gateaff camera inventory differs")

    tensors = []

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(payload)
    require(all(t.device.type == "cpu" for t in tensors)
            and all(bool(torch.isfinite(t).all()) for t in tensors if t.is_floating_point()),
            "Nonfinite or non-CPU tensor payload")
    require(entry(parent) == parent_value["checkpoint"] and not torch.cuda.is_initialized(),
            "Parent changed during CPU acceptance or CUDA initialized")
    result = dict(status="completed_CPU_actual_gateaff_parent_tensor_acceptance",
                  source=entry(__file__), parent=parent_value, points=n, tensors=len(tensors),
                  interpreter=sys.executable, torch=torch.__version__,
                  checkpoint_schema="cullretrain_lr_v1", converted_to_native_schema=False,
                  CUDA_initialized=False, GPU_calls=0, training_submitted=False)
    write_new(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=tuple(NEW), required=True)
    p.add_argument("--parent")
    p.add_argument("--out")
    a = p.parse_args()
    parent = a.parent or OUT / f"lr/runs/{NEW[a.scene]}07_area4_gateaff/checkpoint_fine_14000.pt"
    out = a.out or OUT / f"lr/validation/{a.scene}_parent_tensors.json"
    result = check(a.scene, parent, out)
    print(__import__("json").dumps(dict(status=result["status"], points=result["points"],
                                       CUDA_initialized=False)))


if __name__ == "__main__":
    main()
