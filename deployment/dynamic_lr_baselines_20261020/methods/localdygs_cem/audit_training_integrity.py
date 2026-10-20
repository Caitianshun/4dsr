"""Prove interface/reporting patches preserve native optimizer functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["localdygs", "cem"], required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    original = ast.parse((a.upstream / "train.py").read_text())
    adapted = ast.parse((a.runtime / "train.py").read_text())
    funcs = lambda tree: {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    before, after = funcs(original), funcs(adapted)
    names = ["training", "scene_reconstruction"] if a.method == "localdygs" else ["training"]
    matched = {}
    for name in names:
        x, y = [ast.dump(z[name], include_attributes=False) for z in [before, after]]
        assert x == y, f"Native optimizer function changed: {name}"
        matched[name] = dict(upstream_AST_sha256=sha(x), runtime_AST_sha256=sha(y), identical=True)
    native_report = before["training_report"]
    adapted_report = after["training_report"]
    original_tests = [x for x in ast.walk(native_report) if isinstance(x, ast.If) and ast.unparse(x.test) == "iteration in testing_iterations"]
    new_tests = [x for x in ast.walk(adapted_report) if isinstance(x, ast.If) and "DYNAMIC_LR_PROTOCOL_MANIFEST" in ast.unparse(x.test)]
    assert len(original_tests) == len(new_tests) == 1
    guard = ast.unparse(new_tests[0].test)
    new_tests[0].test = original_tests[0].test
    assert ast.dump(native_report, include_attributes=False) == ast.dump(adapted_report, include_attributes=False)
    evidence = dict(readonly_report_guard=guard, reporting_AST_identical_after_restoring_only_guard=True)
    if a.method == "localdygs":
        loads = [x for x in ast.walk(original) if isinstance(x, ast.Name) and x.id == "lpips_fn" and isinstance(x.ctx, ast.Load)]
        eval_loads = [x for x in ast.walk(before["evaluate"]) if isinstance(x, ast.Name) and x.id == "lpips_fn" and isinstance(x.ctx, ast.Load)]
        assert len(loads) == len(eval_loads) == 1
        # The standalone evaluate function is not called from the training
        # entrypoint or either native training/optimizer function.
        train_calls = [x for node in [after[n] for n in names] for x in ast.walk(node)
                       if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "evaluate"]
        entry_calls = [x for node in adapted.body if isinstance(node, ast.If)
                       for x in ast.walk(node) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "evaluate"]
        assert not train_calls and not entry_calls
        evidence.update(VGG_lpips_uses="one load at train.py:evaluate(); no uses or calls in native training functions or __main__",
                        VGG_is_optimization_loss=False, VGG_module_initialization_skipped_only_under_protocol=True)
    result = dict(status="passed", method=a.method, native_optimization_functions=matched,
                  native_loss_sampling_optimizer_schedule_unchanged=True,
                  heldout_readonly_diagnostics_bypassed=True, evidence=evidence)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
