"""Read the trusted native stage-1 endpoint before native CEM stage 2."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stage1-receipt", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("Refusing an existing checkpoint boundary receipt")
    before = time.monotonic()
    completed = json.loads(a.stage1_receipt.read_text())
    assert completed["status"] == "completed"
    assert a.checkpoint.name == "chkpnt40000.pth"
    checkpoint_sha = digest(a.checkpoint)
    import torch
    # Only this task's just-produced, successful native checkpoint is trusted.
    # This process never imports a model, calls CUDA, or changes its tensors.
    model_args, iteration = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    assert iteration == 40000 and isinstance(model_args, tuple) and len(model_args) == 29
    assert isinstance(model_args[1], torch.Tensor) and model_args[1].ndim == 2 and model_args[1].shape[1] == 3
    optimizer = model_args[18]
    assert isinstance(optimizer, dict) and {"state", "param_groups"} <= set(optimizer)
    assert not torch.cuda.is_initialized()
    assert digest(a.checkpoint) == checkpoint_sha
    source = a.checkpoint.parent / "cfg_args"
    cfg_copy = a.output.parent / "stage1_cfg_args.txt"
    cfg_copy.write_bytes(source.read_bytes())
    value = dict(status="passed", trusted_scope="only this task's own successful native stage-1 checkpoint",
        checkpoint=str(a.checkpoint), checkpoint_sha256=checkpoint_sha, iteration=iteration,
        native_capture_tuple_fields=len(model_args), static_points=int(model_args[1].shape[0]),
        native_optimizer_groups=len(optimizer["param_groups"]), optimizer_state_entries=len(optimizer["state"]),
        stage1_receipt=str(a.stage1_receipt), stage1_receipt_sha256=digest(a.stage1_receipt),
        stage1_cfg_args_copy=str(cfg_copy), stage1_cfg_args_sha256=digest(cfg_copy),
        helper_sha256=digest(__file__), native_train_sha256=digest(a.runtime / "train.py"),
        native_capture_restore_source_sha256=digest(a.runtime / "scene/v_gaussian_model.py"),
        load_options=dict(map_location="cpu", weights_only=False), CUDA_initialized=False,
        model_or_optimizer_updated=False, checkpoint_unchanged=True, elapsed_seconds=time.monotonic() - before)
    a.output.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
