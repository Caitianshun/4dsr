#!/usr/bin/env python3
"""Check all 12 specs, 24 renderer CLIs, and in-place plan continuation on CPU.

Torch/NumPy imports are blocked. No --execute branch is called. The exact
output-directory preflight used by --execute is checked directly instead.
"""
import contextlib
import copy
import importlib.abc
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile

from prepare_local_cem import HERE, ROOT, TRAIN_FRAMES, identity
from render_local_cem import validate_output_directory


class NoGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "cv2", "lpips", "tinycudann"}:
            raise RuntimeError("Forbidden plan-only import: " + fullname)
        return None


def main():
    sys.meta_path.insert(0, NoGpuImports())
    reports = []
    plan_path = HERE / "notes/local_cem_audit.json"
    specs = json.loads(plan_path.read_text())["rows"]
    assert len(specs) == 12
    with tempfile.TemporaryDirectory(prefix="local-cem-cli-check-", dir=HERE / "notes") as scratch:
        for spec in specs:
            assert Path(spec["spec_path"]).is_file()
            for scope, argv in spec["render_cli_plan_only"].items():
                output = ROOT / "output/dynamic_lr_expansion_20261022/cpu_render_plans" / spec["method"] / spec["scene"] / scope
                argv = list(argv)
                argv[argv.index("--output") + 1] = str(output)
                sys.argv = argv[1:]
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(HERE / "render_local_cem.py"), run_name="__main__")
                plan = json.loads((output / "expected_views.json").read_text())
                validate_output_directory(output, plan)
                expected_view_identity = identity(output / "expected_views.json")
                # Negative cases use disposable metadata only, keeping the
                # long-lived plan referenced by root assembly intact.
                fixture = Path(scratch) / spec["method"] / spec["scene"] / scope
                fixture.mkdir(parents=True)
                (fixture / "expected_views.json").write_text((output / "expected_views.json").read_text())
                pending_changed = copy.deepcopy(plan)
                pending_changed["missing_assets"] = []
                pending_changed["checkpoint_assets"] = []
                validate_output_directory(output, pending_changed)
                expected_count = spec["train_metric_observations"] if scope == "train" else 300
                assert len(plan["records"]) == expected_count
                assert sorted({r["frame"] for r in plan["records"]}) == (TRAIN_FRAMES if scope == "train" else list(range(300)))
                rejected = 0
                for filename in ("foreign.txt", "0000.png", "render_index.json", "render_receipt.json"):
                    added = fixture / filename
                    added.write_text("CPU negative fixture")
                    try:
                        validate_output_directory(fixture, plan)
                    except ValueError:
                        rejected += 1
                    else:
                        raise AssertionError("Foreign/result file was accepted")
                    added.unlink()
                for field, value in (("manifest_sha256", "0" * 64), ("method", "wrong_method"), ("scope", "wrong_scope"),
                                     ("iteration", -1), ("resolution", [1, 1]), ("records", [])):
                    changed = copy.deepcopy(plan)
                    changed[field] = value
                    try:
                        validate_output_directory(fixture, changed)
                    except ValueError:
                        rejected += 1
                    else:
                        raise AssertionError("Changed core plan identity accepted")
                assert not (output / "render_index.json").exists()
                reports.append(dict(method=spec["method"], scene=spec["scene"], scope=scope,
                                    views=expected_count, rejected_invalid_continuations=rejected,
                                    expected_views=expected_view_identity, missing_model_assets=len(plan["missing_assets"]),
                                    mode=spec["mode"], spec=identity(Path(spec["spec_path"]))))
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    result = dict(schema="local_cem_CPU_preparation_acceptance_v2", status="passed_CPU_only", spec_count=len(specs),
                  CLI_plans_checked=len(reports), planned_observations=sum(r["views"] for r in reports),
                  Torch_imported=False, NumPy_imported=False, GPU_actions=0, training_actions=0, remote_actions=0,
                  plan_to_execution_directory_preflight_passed=True,
                  source=[identity(HERE / name) for name in ("prepare_local_cem.py", "render_local_cem.py", "audit_local_cem_plans_cpu.py")], plans=reports)
    (HERE / "notes/local_cem_cpu_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("plans", "source")}))


if __name__ == "__main__":
    main()
