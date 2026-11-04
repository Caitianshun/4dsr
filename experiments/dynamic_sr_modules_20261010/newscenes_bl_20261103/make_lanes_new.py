"""Historical lane-format files, shell equivalents and token-by-token audit."""
from __future__ import annotations

import copy
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *
import cem_frame0_new as frame0
import cem_new
from teacher_index_new import index_path, producer_path, cache_path

TASK_ROOT = "tasks/static_sr_shared_wu_20261024_a100"
TORCH_HOME = rel(OUT / "model_cache/torch_home")


def parse(line):
    args = shlex.split(line)
    return args[1:] if args[:1] == ["CRITICAL"] else args


def compare(old, new, mapping, evidence, **metadata):
    """Only exact, explicitly enumerated argv tokens can change."""
    require(len(old) == len(new), "Unexpected argv length change")
    expected = [mapping.get(t, t) for t in old]
    require(expected == new, "Unexplained argv difference")
    return dict(evidence=evidence, **metadata, old_argv=old, new_argv=new,
                token_comparison=[dict(position=i, old=a, new=b, equal=a == b,
                                       reason="identical" if a == b else "explicit path/scene/author-endpoint mapping")
                                  for i, (a, b) in enumerate(zip(old, new))],
                status="passed")


def map_for(label, scene, method):
    oldscene = "flame_steak" if label == "Flame07" else "coffee_martini"
    mapping = {
        f"data/dynamic_sr/full_time_20261007/prepared/{oldscene}_area4/manifest_train_ready_seed{SEED}.json":
        rel(manifest_path(scene)),
        "output/dynamic_sr_modules_20261010/bl4_20261024/torch_home": TORCH_HOME}
    if method in ("splatsure", "iesrgs"):
        c = f"experiments/dynamic_sr_modules_20261010/static_sr_v2_20261026/configs/{label}"
        mapping[c + f"/{method}.json"] = rel(CODE / f"configs/{scene}/{method}.json")
        mapping[c + "/areaft_index.json"] = rel(index_path(scene))
        o = f"output/dynamic_sr_modules_20261010/static_sr_v2_20261026"
        for old, new in ((f"runs/{label}/{method}_lr", "lr"), (f"runs/{label}/{method}_internal", "internal"),
                         (f"runs/{label}/{method}_sr", "sr"), (f"runs/{label}/{method}_weights", "weights"),
                         (f"priors/{label}/ie_depth", "depth")):
            for suffix in ("", "/checkpoint.pt", "/receipt.json", "/index.json"):
                mapping[o + "/" + old + suffix] = rel(OUT / method / scene / (new + suffix))
        for scope in ("test", "train"):
            mapping[f"{o}/eval/{label}/{method}_render_{scope}"] = rel(OUT / method / scene / f"renders_{scope}")
            mapping[f"{o}/eval/{label}/{method}_score_{scope}"] = rel(OUT / method / scene / f"scores_{scope}")
    else:
        o = "output/dynamic_sr_modules_20261010/sr4d_area4_20261028"
        mapping[f"{o}/runs/{label}/sr4d_b"] = rel(OUT / method / scene / "model")
        mapping[f"{o}/checks/{label}_a100_b.json"] = rel(OUT / method / scene / "check.json")
        mapping[oldscene] = scene
        for scope in ("test", "train"):
            mapping[f"{o}/eval/{label}/sr4d_b_render_{scope}"] = rel(OUT / method / scene / f"renders_{scope}")
            mapping[f"{o}/eval/{label}/sr4d_b_score_{scope}"] = rel(OUT / method / scene / f"scores_{scope}")
    return mapping


def historical_lane(label, method):
    if method == "sr4d":
        source = SR4D / f"lanes/a1b_{label}.txt"
    else:
        source = STATIC / f"lanes/c_a100_{label}_{method}.txt"
    lines = []
    omitted = []
    for n, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("TOUCH ") or "/threedsr.json " in line:
            omitted.append(dict(line=n, text=line, reason="historical completion marker" if line.startswith("TOUCH ")
                                else "unrelated 3DSR LR job; outside requested methods"))
            continue
        lines.append((n, line, parse(line)))
    return source, lines, omitted


def interpreter(host, argv):
    if host == "cts" and Path(argv[0]).name in ("cem_new.py", "cem_frame0_new.py"):
        return REMOTE[host] + "/envs/dynamic_lr_baselines_20261020/cem/bin/python"
    return REMOTE[host] + "/.venv/bin/python"


def resolved_task(host, method, scene, line_number, args):
    return dict(name=f"{host}_{scene}_{method}_{line_number:02d}", host=host, cwd=REMOTE[host],
                interpreter=interpreter(host, args), argv=args,
                executable_argv=[interpreter(host, args), "-u", *args],
                environment=dict(PYTHONDONTWRITEBYTECODE="1", TORCH_HOME=TORCH_HOME,
                                 HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                resource="CPU" if Path(args[0]).name in ("teacher_index_new.py", "cem_frame0_new.py",
                                                        "collect_results_new.py", "sv2_cleanup.py") else "GPU",
                dispatch_status="prepared_not_submitted",
                device_binding="inherit single GPU from fixed executor; no GPU chosen by this file")


def emit_lane(name, host, method, scene, lines):
    lane = CODE / "lanes" / (name + ".txt")
    text_new(lane, "\n".join(lines) + "\n")
    tasks = [resolved_task(host, method, scene, i, parse(line))
             for i, line in enumerate(lines, 1)]
    # .txt retains historical <script> argv format; .sh supplies each exact
    # absolute interpreter and makes failures stop the sequential chain.
    shell = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(REMOTE[host]),
             "export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
             "export TORCH_HOME=" + shlex.quote(TORCH_HOME)]
    shell.extend(shlex.join(task["executable_argv"]) for task in tasks)
    text_new(CODE / "lanes_exec" / (name + ".sh"), "\n".join(shell) + "\n")
    return dict(name=name, host=host, method=method, scene=scene, lane=rel(lane),
                shell=rel(CODE / "lanes_exec" / (name + ".sh")), tasks=tasks)


def build():
    lanes, comparisons, omissions = [], [], []
    for scene in SCENES:
        for method in ("sr4d", "splatsure", "iesrgs"):
            source, lines, skipped = historical_lane("Flame07", method)
            mapping = map_for("Flame07", scene, method)
            newlines = []
            scientific = []
            for n, text, args in lines:
                new = [mapping.get(t, t) for t in args]
                scientific.append(new)
                prefix = "CRITICAL " if text.startswith("CRITICAL ") else ""
                newlines.append(prefix + shlex.join(new))
                comparisons.append(compare(args, new, mapping, identity(source), scene=scene,
                                           method=method, historical_label="Flame07", historical_line=n))
            # Coffee07 is the independent batch4 reference for salmon. Compare
            # every command for both scenes so differences cannot be hidden.
            coffee_source, coffee_lines, _ = historical_lane("Coffee07", method)
            coffee_mapping = map_for("Coffee07", scene, method)
            require(len(coffee_lines) == len(scientific), "Historical lane structure changed")
            for (n, _, args), new in zip(coffee_lines, scientific):
                comparisons.append(compare(args, new, coffee_mapping, identity(coffee_source), scene=scene,
                                           method=method, historical_label="Coffee07", historical_line=n))
            omissions.extend(dict(scene=scene, method=method, source=rel(source), **row) for row in skipped)
            newlines.append(shlex.join([rel(CODE / "collect_results_new.py"), "--method", method, "--scene", scene]))
            lanes.append(emit_lane(f"a100_{scene}_{method}", "a100", method, scene, newlines))

        cemlines = []
        for mode in ("prepare", "run", "check"):
            args = [rel(CODE / "cem_frame0_new.py"), mode, "--scene", scene]
            if mode != "check":
                args += ["--execute"]
            cemlines.append(shlex.join(args))
        for stage in (1, 2):
            cemlines.append(shlex.join([rel(CODE / "cem_new.py"), "train", "--scene", scene,
                                       "--stage", str(stage), "--execute"]))
        for scope in ("test", "train"):
            cemlines.append(shlex.join([rel(CODE / "cem_new.py"), "render", "--scene", scene,
                                       "--scope", scope, "--execute"]))
        for scope in ("test", "train"):
            base = OUT / "cem" / scene
            cemlines.append(shlex.join([
                rel(OLD / "score_views_area4.py"), "--manifest", rel(manifest_path(scene)),
                "--scope", scope, "--render-dir", rel(base / f"renders_{scope}"),
                "--out", rel(base / f"scores_{scope}"), "--method", "cem_area4", "--device", "cuda",
                "--execute", "--torch-home", TORCH_HOME]))
            cemlines.append(shlex.join([rel(STATIC / "sv2_cleanup.py"), rel(base / f"renders_{scope}"),
                                       rel(base / f"scores_{scope}")]))
        cemlines.append(shlex.join([rel(CODE / "collect_results_new.py"), "--method", "cem", "--scene", scene]))
        lanes.append(emit_lane(f"cts_{scene}_cem", "cts", "cem", scene, cemlines))

        hist_base = ROOT / "output/dynamic_sr_modules_20261010/area4_baselines_20261024/localdygs_cem/flame_steak"
        steps = [__import__("json").loads(l) for l in (hist_base / "cem/steps.jsonl").read_text().splitlines()]
        # Compare expanded CEM entry argv, not just the new wrapper's CLI.
        for row in steps:
            old = row["argv"]
            if row["step"].startswith("train_stage"):
                stage = int(row["step"][-1])
                args = cem_new.training_argv(scene, stage, "0")
                new = [REMOTE["cts"] + "/envs/dynamic_lr_baselines_20261020/cem/bin/python",
                       str(OLD / "localdygs_cem/run_area4.py"), "stage-b", *args]
            elif row["step"].startswith("render_"):
                scope = row["step"].split("_", 1)[1]
                new = [REMOTE["cts"] + "/envs/dynamic_lr_baselines_20261020/cem/bin/python",
                       str(OLD / "localdygs_cem/run_area4.py"), "stage-b",
                       str(OLD / "localdygs_cem/render_local_cem_area4.py"), "--",
                       *cem_new.render_argv(scene, scope, True)]
            else:
                # Shared scorer is unchanged. The explicit cache location is
                # compared separately, because history used its default.
                scope = row["step"].split("_", 1)[1]
                new = [REMOTE["cts"] + "/.venv/bin/python", str(OLD / "score_views_area4.py"),
                       "--manifest", str(manifest_path(scene)), "--scope", scope,
                       "--render-dir", str(OUT / "cem" / scene / f"renders_{scope}"),
                       "--out", str(OUT / "cem" / scene / f"scores_{scope}"),
                       "--method", "cem_area4", "--device", "cuda", "--execute"]
            # Expand all target project-relative or local-root paths to CTS.
            new = [t.replace(str(ROOT) + "/", REMOTE["cts"] + "/") for t in new]
            mapping = {}
            # Explicit token roles from the exact historical command position.
            path_flags = {"--configpath", "--source_path", "--model_path", "--start_checkpoint",
                          "--receipt", "--manifest", "--model", "--source", "--output", "--render-dir", "--out"}
            for i, (a, b) in enumerate(zip(old, new)):
                if a != b:
                    require(i > 0 and old[i - 1] in path_flags, "Unexplained CEM expanded argv change")
                    mapping[a] = b
            comparisons.append(compare(old, new, mapping, identity(hist_base / "cem/steps.jsonl"),
                                       scene=scene, method="cem", historical_label="Flame07",
                                       historical_step=row["step"], expanded_wrapper_argv=True,
                                       extra_actual_args=["--torch-home", TORCH_HOME]
                                       if row["step"].startswith("score_") else [],
                                       wrapper_change="new init acceptance and output paths; original stage-b/author/renderer retained"))
        native = read(hist_base / "native_init/complete.json")
        original_commands = {r["name"]: r for r in native["commands"]}
        for name, argv in frame0.commands(*frame0.paths(scene)):
            new = [rel(t) if str(t).startswith(str(ROOT) + "/") else t for t in argv]
            old = original_commands[name]["argv"]
            mapping = {t: t.replace(
                "output/dynamic_sr_modules_20261010/area4_baselines_20261024/localdygs_cem/flame_steak/",
                rel(OUT / "cem" / scene) + "/") for t in old if
                t.startswith("output/dynamic_sr_modules_20261010/area4_baselines_20261024/localdygs_cem/flame_steak/")}
            comparisons.append(compare(old, new, mapping, identity(hist_base / "native_init/complete.json"),
                                       method="cem_init", scene=scene, historical_label="Flame07",
                                       historical_step=name))

    prep = []
    for scene in SCENES:
        prep.append(emit_lane(f"a100_{scene}_teacher_a", "a100", "teacher_a", scene, [
            shlex.join([rel(CODE / "teacher_index_new.py"), "--scene", scene])]))
        cache = cache_path(scene, True)
        prep.append(emit_lane(f"a100_{scene}_teacher_b", "a100", "teacher_b", scene, [
            shlex.join([rel(CODE / "teacher_bl_new.py"), "--manifest", rel(manifest_path(scene)),
                        "--batch", "6", "--weights", "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1",
                        "--out-root", rel(cache)]),
            shlex.join([rel(CODE / "teacher_index_new.py"), "--scene", scene,
                        "--producer-index", rel(cache / f"teacher_index_sr_swinir_x4_areaft_seed{SEED}.json"),
                        "--cache-root", rel(cache), "--reference-index", rel(producer_path(scene))])]))
    record = dict(schema="newscene_baseline_commands_v1", dispatch_status="not_submitted",
                  lanes=lanes, teacher_alternatives=prep,
                  interpreter_evidence=dict(
                      a100="historical job logs and SR4D config.environment.python: project .venv",
                      cts_cem="Flame07 training_stage{1,2}_receipt.python_executable",
                      cts_score="Flame07 steps.jsonl score argv"),
                  conditions=["fixed executor admission and exact remote inventory/SHA acceptance",
                              "one teacher alternative per scene; all required inputs verified",
                              "single admitted GPU inherited for every GPU command"])
    write_new(OUT / "commands.json", record)
    write_new(OUT / "comparisons/argv_comparison.json", comparisons)
    write_new(OUT / "comparisons/omissions_and_adaptations.json", dict(
        omitted=omissions,
        CEM_initialization="only frame0 sparse producer; no image_undistorter/dense/poses_bounds copy; new receipt schema",
        CEM_gpu="plan comparisons use historical numeric mask0 only as a comparison fixture; execute uses broker mask",
        scoring_cache="all --torch-home paths point at new namespace, same frozen LPIPS weights required",
        execution="shell equivalents run sequentially with set -e; no scheduling, GPU selection or submission",
        new_commands=["frame0 check", "teacher index verification", "result collection/acceptance"]))
    print(__import__("json").dumps(dict(lanes=len(lanes), teacher_lanes=len(prep),
                                      comparisons=len(comparisons),
                                      tokens=sum(len(r["token_comparison"]) for r in comparisons))))


if __name__ == "__main__":
    build()
