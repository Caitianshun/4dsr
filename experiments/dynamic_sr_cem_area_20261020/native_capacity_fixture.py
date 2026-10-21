"""Isolated 3..10-iteration native CEM HR wiring/cost fixture, never a result.

Only the author stage1 iteration endpoint is shortened in a copied config.
The native final iteration is saved before optimizer.step: N iterations mean
N-1 optimizer updates. Initial points are smaller than the mature 50k model,
so this cannot establish the final model's memory or convergence.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from sr_protocol import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUTHOR_RUNNER = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/localdygs_cem/run_author.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--iterations", type=int, choices=range(3, 11), default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.gpu.isdecimal() and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
    assert not args.output.exists()
    runtime, source, output = args.runtime.resolve(), args.source.resolve(), args.output.resolve()
    value = json.loads(args.manifest.read_text())
    assert value["physical_lr_derivation"]["formation"] == "area4"
    assert (source / "colmap_0/sparse/0/points3D.ply").is_file()
    protocol = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    assert protocol["version"] == "native_area4_v1"
    for item in protocol["core_source_files"] + protocol["configurations"]:
        assert sha(runtime / item["path"]) == item["sha256"]
    assert sha(runtime / "protocol_interface.py") == protocol["interface_sha256"]
    original = runtime / f"configs/N3V_s1/{value['scene']}_prepared_lr.json"
    config = json.loads(original.read_text())
    assert config["iterations"] == 40000
    config["iterations"] = args.iterations
    output.mkdir(parents=True)
    derived = output / "fixture_config.json"
    derived.write_text(json.dumps(config, indent=2) + "\n")
    (output / "fixture_scope.json").write_text(json.dumps(dict(status="wiring_fixture_requested", scientific_result=False,
        train_level="hr", train_HR_used=True, heldout_pixels_used=False, author_core_modified=False,
        only_config_change=dict(iterations=dict(original=40000, fixture=args.iterations)),
        native_optimizer_updates_if_success=args.iterations - 1,
        scope="native startup/HR loss/backward wiring; not mature50k capacity, convergence or quality",
        source_config=dict(path=str(original), sha256=sha(original)), fixture_config_sha256=sha(derived),
        source_helper=dict(path=str(AUTHOR_RUNNER), sha256=sha(AUTHOR_RUNNER)),
        manifest=dict(path=str(args.manifest.resolve()), sha256=sha(args.manifest)),
        initialization=dict(path=str(source), sha256=sha(source / "colmap_0/sparse/0/points3D.ply")),
        fixture_entry_sha256=sha(__file__)), indent=2) + "\n")
    os.environ.update(DYNAMIC_LR_PROTOCOL_MANIFEST=str(args.manifest.resolve()), CEM_PROTOCOL_TRAIN_LEVEL="hr",
                      TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
    os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
    spec = importlib.util.spec_from_file_location("accepted_CEM_native_runner", AUTHOR_RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sys.argv = [str(AUTHOR_RUNNER), "--runtime", str(runtime), "--receipt", str(output / "native_fixture_execution.json"),
        "--", "--configpath", str(derived), "--source_path", str(source), "--model_path", str(output / "model"), "--gpu", args.gpu]
    try:
        runner.main()
    finally:
        receipt = output / "native_fixture_execution.json"
        execution = json.loads(receipt.read_text()) if receipt.is_file() else {}
        (output / "fixture_result.json").write_text(json.dumps(dict(status=execution.get("status", "failed_before_author_receipt"),
            scientific_result=False, native_iterations=args.iterations,
            native_optimizer_updates=args.iterations - 1 if execution.get("status") == "completed" else None,
            execution_receipt=dict(path=str(receipt), sha256=sha(receipt)) if receipt.is_file() else None,
            elapsed_seconds=execution.get("elapsed_seconds"),
            torch_peak_reserved_bytes=execution.get("torch_peak_reserved_bytes"),
            scope="startup HR wiring/cost only; no mature50k memory or convergence conclusion"), indent=2) + "\n")


if __name__ == "__main__":
    main()
