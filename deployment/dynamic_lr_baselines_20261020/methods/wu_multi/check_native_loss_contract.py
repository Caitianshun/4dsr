"""Rebuild the native Multi4D loss AST contract without importing Torch/CUDA.

Reads the fixed author Git source and a locally generated adapter. Output is a
new CPU receipt; existing deployment/acceptance evidence is never overwritten.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
COMMIT = "c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0"


def reconstruction(source):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "scene_reconstruction")


def loss_terms(function):
    terms = []
    for node in ast.walk(function):
        assign = isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "loss" for target in node.targets)
        add = isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "loss"
        if not (assign or add):
            continue
        value = node.value
        if ast.unparse(value) == "native_loss + teacher_term":
            continue
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "native_term":
            assert isinstance(value.func.value, ast.Name) and value.func.value.id == "adapter_context"
            assert len(value.args) == 4
            value = value.args[3]
        terms.append(["assign" if assign else "add", ast.unparse(value)])
    return terms


def assignments(function, name):
    return [ast.dump(node.value, include_attributes=False) for node in ast.walk(function)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


def rgb_calls(function):
    return Counter((node.func.id, *[ast.unparse(arg) for arg in node.args]) for node in ast.walk(function)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"l1_loss", "ssim", "ssim_raw", "psnr"} and len(node.args) >= 2
                   and "gt_image_tensor" in ast.unparse(node.args[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, default=HERE / "multi4d_native/adapter/adapted_train.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing existing CPU contract; preserve frozen evidence")
    head = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    assert head == COMMIT, (head, COMMIT)
    original_bytes = subprocess.check_output(["git", "-C", str(args.upstream), "show", COMMIT + ":train.py"])
    adapted_bytes = args.adapted.read_bytes()
    original, adapted = original_bytes.decode(), adapted_bytes.decode()
    author_fn, adapted_fn = reconstruction(original), reconstruction(adapted)
    author_terms, adapted_terms = loss_terms(author_fn), loss_terms(adapted_fn)
    assert len(author_terms) == 19 and author_terms == adapted_terms
    for name in ["_mask", "_l1", "loss_smooth_sta"]:
        assert assignments(author_fn, name) and assignments(author_fn, name) == assignments(adapted_fn, name), name
    original_rgb, adapted_rgb = rgb_calls(author_fn), rgb_calls(adapted_fn)
    assert sum(original_rgb.values()) == 17 and original_rgb == adapted_rgb
    endpoint = "iteration < optimization_params.iterations"
    conditions = lambda fn: sum(ast.unparse(node.test) == endpoint for node in ast.walk(fn) if isinstance(node, ast.If))
    assert conditions(author_fn) == conditions(adapted_fn) == 3
    assert "lr_project(" not in adapted and "avg_pool2d(" not in adapted and "iteration <= final_iter" not in adapted
    result = dict(status="CPU_native_loss_contract_passed_CUDA_not_tested", native_loss_assignments_equal=19,
                  region_mask_and_l1_and_depth_assignments_exact=True, no_training_RGB_resampling=True,
                  author_optimizer_endpoint_conditions=3, upstream_loss_terms=author_terms,
                  input_policy="original registered LR only", no_HR_or_SR_training=True,
                  source_sha256={"original_train.py": hashlib.sha256(original_bytes).hexdigest(),
                                 "adapted_train.py": hashlib.sha256(adapted_bytes).hexdigest()},
                  target_RGB_call_expressions_equal=17, CPU_only=True, imported_Torch_or_CUDA=False,
                  scope="Loss/region/depth/RGB AST and endpoint conditions only; input legality and zero teacher samples require separate runtime receipts",
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "native_loss_assignments_equal", "target_RGB_call_expressions_equal", "CPU_only", "source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
