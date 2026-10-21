"""Freeze an independent area4 CEM runtime from the accepted calibrated v3.

CPU file preparation only. No training, CUDA imports, SSH, or environment edits.
The immutable v3 remains untouched. Parent and independent HRref share native
40k -> total50k configurations; only observation paths/pixel-grid metadata vary.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM_COMMIT = "7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd"
SOURCE = ROOT / "third_party/cem4dgs_calibrated_runtime_v3_20261020"
DEFAULT_OUTPUT = ROOT / "output/dynamic_sr_cem_backbone_20261020/runtime/cem-native-area4-v1"
EXPECTED = {
    "protocol_interface.py": "25c94202de1cff22300e581980ed10ffa4ef69087e72393e3e128c9573e10d5f",
    "train.py": "e7c6f39900e4f5cda237defb5aee378041825edcedb751ad4668d5ad9a9b0a6d",
    "scene/v_gaussian_model.py": "0e87c364f6daceae947e9d4a6ae5d530fd973e714b6387d22d8a443c55f1461a",
}
CORE = ["train.py", "scene/v_gaussian_model.py", "gaussian_renderer/__init__.py",
        "scene/__init__.py", "scene/cameras.py", "scene/dataset_readers.py",
        "utils/stereo_helper.py", "utils/stereo_utils.py", "arguments/__init__.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_receipt = json.loads((SOURCE / "protocol_interface_receipt.json").read_text())
    assert source_receipt["upstream_commit"] == UPSTREAM_COMMIT
    assert all(sha(SOURCE / name) == expected for name, expected in EXPECTED.items())
    assert not args.output.exists(), "Never overwrite a frozen runtime"
    shutil.copytree(SOURCE, args.output, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.so", "build", ".venv"))
    shutil.copy2(SOURCE / "protocol_interface_receipt.json", args.output / "calibrated_v3_parent_receipt.json")
    before = (SOURCE / "protocol_interface.py").read_text()
    shutil.copy2(HERE / "protocol_interface.py", args.output / "protocol_interface.py")
    after = (args.output / "protocol_interface.py").read_text()
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile="calibrated-v3/protocol_interface.py", tofile="area4-v1/protocol_interface.py"))
    (args.output / "area4_interface_v1.patch").write_text(patch)
    core = []
    for name in CORE:
        assert (SOURCE / name).read_bytes() == (args.output / name).read_bytes()
        core.append(dict(path=name, sha256=sha(SOURCE / name), unchanged_from_calibrated_v3=True))
    configs = []
    for scene in ["cook_spinach", "cut_roasted_beef"]:
        for stage, endpoint in [(1, 40000), (2, 50000)]:
            name = f"configs/N3V_s{stage}/{scene}_prepared_lr.json"
            config = json.loads((args.output / name).read_text())
            assert config["iterations"] == endpoint and config["resolution"] == 1
            assert (SOURCE / name).read_bytes() == (args.output / name).read_bytes()
            configs.append(dict(path=name, sha256=sha(args.output / name), unchanged_from_calibrated_v3=True))
    receipt = dict(method="cem", version="native_area4_v1", upstream_commit=UPSTREAM_COMMIT,
        source_runtime=str(SOURCE), source_receipt_sha256=sha(SOURCE / "protocol_interface_receipt.json"),
        interface_sha256=sha(args.output / "protocol_interface.py"), patch_sha256=sha(args.output / "area4_interface_v1.patch"),
        core_source_files=core, configurations=configs,
        native_readonly_test_evaluation_disabled=True, heldout_pixels_read_by_training=False,
        core_losses_sampling_optimization_schedule_unchanged=True,
        training_executed=False, CUDA_imported=False,
        native_parent_training="area4 LR only, original RAdam/state, 0->40000->total50000, no added affine or warp",
        HR_reference_training="independent from same LR-only frame0 initialization; CEM_PROTOCOL_TRAIN_LEVEL=hr; train HR only",
        schedule_policy="author absolute clock; no scheduler repair, topology reset, densification extension, or CEC extension",
        test_data_policy="metadata and protocol-zero placeholders only; independent evaluation opens held-out HR",
        changes=[dict(path="protocol_interface.py", reason="accept exact area4 manifests; explicit independent train-grid level; preserve K/pose/time/split")])
    write(args.output / "protocol_interface_receipt.json", receipt)
    files = [dict(path=str(p.relative_to(args.output)), sha256=sha(p)) for p in sorted(args.output.rglob("*")) if p.is_file()]
    write(args.output / "native_runtime_inventory.json", dict(schema="cem_area4_runtime_v1", files=files))
    print(json.dumps(dict(runtime=str(args.output.resolve()), receipt_sha256=sha(args.output / "protocol_interface_receipt.json"),
                         inventory_sha256=sha(args.output / "native_runtime_inventory.json"), files=len(files))))


if __name__ == "__main__":
    main()
