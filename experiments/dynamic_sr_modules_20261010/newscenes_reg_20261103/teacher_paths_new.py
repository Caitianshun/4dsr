"""Run the original SR entry with a bounded, external teacher-cache path map.

Only exact image paths from the validated areaft index may escape the dataset
directory, and they must land in this task's teacher_cache. Other safe_child
calls retain the original check. Driver children are wrapped explicitly because
in-memory changes do not propagate through subprocesses. No loss, seed, sampler,
model, optimizer, rounding or training argv is changed.

--inspect is CPU-only. Execution is for Claude's torch environment, after fixed
executor acceptance; this module never chooses a GPU or submits a job.
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
from materialize_bsync_new import teacher_files

GEO = ROOT / "experiments/dynamic_sr_modules_20261010/geoprobe_20261017"
THIRD = ROOT / "experiments/dynamic_sr_modules_20261010/third_20261022"
DRIVERS = {GEO / "run_modkit7_remote.py", GEO / "run_modkit4_remote.py"}
DIRECT = {GEO / "modkit4.py", GEO / "modkit6.py", THIRD / "pvcd_modkit.py"}


def resolver(original, dataset_root, mapping):
    dataset_root = Path(dataset_root).resolve()

    def safe_child(root, relative):
        # No generic allow-outside-root switch: only one registered dataset and
        # exact inventory tokens get the output-only teacher-cache exception.
        if Path(root).resolve() == dataset_root and str(relative) in mapping:
            target = Path(mapping[str(relative)]).resolve()
            require(target.is_relative_to((OUT / "teacher_cache").resolve()),
                    "Teacher mapping escaped the new cache")
            return target
        return original(root, relative)
    return safe_child


def child_command(command, kit, index):
    """Pure argv transform, testable without importing a tensor/runtime."""
    require(len(command) >= 3 and command[1] == "-u"
            and local(command[2]) == local(kit), "Unexpected historical driver child interface")
    args = command[3:]
    require("--out" in args, "SR child lacks an output directory")
    receipt = output_path(local(args[args.index("--out") + 1]) / "teacher_paths_adapter.json")
    return [command[0], "-u", str(CODE / "teacher_paths_new.py"),
            "--index", rel(index), "--script", rel(kit), "--receipt", rel(receipt),
            "--", *args]


@contextlib.contextmanager
def script_argv(script, args):
    previous = sys.argv
    sys.argv = [str(script), *args]
    try:
        yield
    finally:
        sys.argv = previous


def invoke(function):
    try:
        return function()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def execute(index, script, args, receipt):
    script, index, receipt = local(script), local(index), output_path(receipt)
    require(script in DRIVERS | DIRECT, "Entry outside the four historical SR stage families")
    require(not receipt.exists(), "Existing path-adapter receipt preserved")
    inventory = read(index)
    scene = inventory["scene"]
    require(scene in NEW, "Only new scenes supported")
    manifest_path = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    files = teacher_files(index, read(manifest_path), manifest_path)
    mapping = {row["relative_path"]: local(f["path"])
               for row, f in zip(inventory["entries"], files)}
    record = dict(source=entry(__file__), original_script=entry(script),
                  original_argv=list(args), teacher_index=entry(index),
                  teacher_files=len(mapping), science_arguments_unchanged=True)
    if script in DRIVERS:
        module = load("_newscene_original_driver", script)
        original_run = subprocess.run
        children = []

        def run(command, **kwargs):
            if len(command) > 2 and local(command[2]) == Path(module.KIT):
                adapted = child_command(command, module.KIT, index)
                children.append(dict(original=command, adapted=adapted))
                command = adapted
            return original_run(command, **kwargs)

        # Limit this replacement to the loaded driver, not the stdlib module.
        module.subprocess = SimpleNamespace(run=run)
        with script_argv(script, args):
            invoke(module.main)
        record["driver_child_adapters"] = children
    else:
        cull = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
        sys.path.insert(0, str(cull))
        import rt
        import cullretrain as cr
        require(Path(cr.__file__).resolve() == cull / "cullretrain.py",
                "Resolved another cullretrain runtime")
        safe = resolver(cr.native.safe_child, manifest_path.parent, mapping)
        # rt.check_runtime resets torch.hub's directory; retain its original
        # runtime checks, then redirect only its cache directory.
        original_check = rt.check_runtime

        def check_runtime():
            value = original_check()
            import torch
            torch.hub.set_dir(str(OUT / "model_cache/torch_home"))
            return value

        with attributes(cr.native, safe_child=safe), attributes(rt, check_runtime=check_runtime):
            sys.path.insert(0, str(script.parent))
            with script_argv(script, args):
                invoke(lambda: runpy.run_path(str(script), run_name="__main__"))
        record["only_safe_child_exact_inventory_exceptions"] = True
    record["status"] = "completed_original_stage_with_external_teacher_paths"
    write_new(receipt, record)
    return record


def main():
    argv = sys.argv[1:]
    if argv == ["--inspect"]:
        print(__import__("json").dumps(dict(
            source=entry(__file__), torch_imported=False,
            original_safe_child=entry(FP / "full_refine_registered_v2.py"),
            driver_children_explicitly_wrapped=True, arbitrary_path_escape_allowed=False)))
        return
    require("--" in argv, "Use -- before unchanged original stage argv")
    cut = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "script", "receipt"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args(argv[:cut])
    execute(a.index, a.script, argv[cut + 1:], a.receipt)


if __name__ == "__main__":
    main()
