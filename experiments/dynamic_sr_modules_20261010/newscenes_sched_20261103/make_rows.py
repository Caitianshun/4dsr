"""Write reviewable pipeline rows and exact commands; never submit a job."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *

PIPELINE = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"


def row(name, script, argv, deps, done):
    return dict(name=name, script=os.path.relpath(CODE / script, PIPELINE),
                argv=[str(v) for v in argv], deps=deps, done=rel(done))


def command(task):
    return [TRAINPY, "-u", rel((PIPELINE / task["script"]).resolve()), *task["argv"]]


def contexts():
    result = []
    for scene in ("sear_steak", "flame_salmon"):
        result.append(dict(key=scene, scene=scene, family="cullretrain_lr_v1",
                           parent=REGOUT / f"wu/{SCENES[scene]}_area4_wu/checkpoint_fine_14000.pt"))
    result.extend([
        dict(key="flame_steak_area4", scene="flame_steak", family="cullretrain_lr_v1",
             parent=ROOT / "output/dynamic_lr_area4_20261024/wu/Flame07_area4_wu/checkpoint_fine_14000.pt"),
        dict(key="flame_steak_native", scene="flame_steak", family="full_author_LR_prefix_checkpoint_v1",
             parent=FPOUT / "full_LR_prefixes/flame_steak/seed_20261007/checkpoint_fine_14000.pt"),
    ])
    return result


def build():
    tasks = []
    groups = {k: [] for k in ("plan", "gpu", "schedules", "finalize", "controls")}
    owners = []
    external = []
    replacements = {}
    for ctx in contexts():
        key, scene, parent = ctx["key"], ctx["scene"], ctx["parent"]
        name = "sched_" + key
        plan = OUT / f"plans/{key}/registered_plan.json"
        run = OUT / f"overlap/{key}/run_v1"
        admission = OUT / f"owner_inputs/{key}/admission.json"
        service_exit = OUT / f"owner_inputs/{key}/service_exit.json"
        coarse_acceptance = OUT / f"acceptance/{key}/coarse.json"
        schedule_dir = OUT / f"schedules/{key}/v1"
        assignment = OUT / f"acceptance/{key}/assignment.json"
        plan_row = row(name + "_plan", "overlap_new.py",
            ["plan", "--scene", scene, "--family", ctx["family"], "--parent", rel(parent),
             "--registered-plan", rel(plan)], [], plan)
        gpu_row = row(name + "_cuda", "overlap_new.py",
            ["prepare", "--registered-plan", rel(plan), "--authorization", rel(admission),
             "--out", rel(run), "--cpu-threads", "2"], [plan_row["name"]],
             run / "all_output_validation.json")
        accept_row = row(name + "_coarse_accept", "acceptance_new.py",
            ["overlap", "--complete", rel(run / "complete.json"), "--service-exit", rel(service_exit),
             "--out", rel(coarse_acceptance)], [gpu_row["name"]], coarse_acceptance)
        schedule_row = row(name + "_schedule", "schedule_new.py",
            ["generate", "--index", rel(run / "index.json"), "--coarse-acceptance", rel(coarse_acceptance),
             "--out", rel(schedule_dir)], [accept_row["name"]], schedule_dir / "complete.json")
        groups["plan"].append(plan_row)
        groups["gpu"].append(gpu_row)
        groups["schedules"].extend([accept_row, schedule_row])
        tasks.extend([plan_row, gpu_row, accept_row, schedule_row])
        external.extend([
            dict(context=key, required_before="plan", path=rel(parent), purpose="completed exact parent and sidecar/registry"),
            dict(context=key, required_before="cuda", path=rel(admission), purpose="actual fixed-service admission"),
            dict(context=key, required_before="coarse_accept", path=rel(service_exit), purpose="normal Exit0 bound to unedited service result"),
        ])
        if scene in ("sear_steak", "flame_salmon"):
            train = REGOUT / f"lr/runs/{SCENES[scene]}_area4_gateaff/checkpoint_fine_14000.pt"
            teacher = REGOUT / f"teacher_cache/{scene}_area4/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json"
            clean = OUT / f"clean_bsync/{scene}/seed{SEED}/Bsync/scientific_plan.json"
            selection = read(HIST_BSYNC)["dependencies"]["selection"]["path"]
            common = ["--scene", scene, "--parent", rel(train), "--overlap-parent", rel(parent),
                      "--schedule", rel(schedule_dir / "schedule.json"),
                      "--assignment-acceptance", rel(assignment)]
            required = row(name + "_require_schedule", "bsync_new.py",
                ["require-schedule", *common, "--out", rel(schedule_dir / "validated.json")],
                [schedule_row["name"]], schedule_dir / "validated.json")
            materialized = row(name + "_materialize", "bsync_new.py",
                ["materialize", *common, "--teacher", rel(teacher), "--selection", selection,
                 "--out", rel(clean)], [required["name"]], clean)
            target = OUT / f"sr/registry/{scene}_area4/seed_{SEED}/areaft_teacher/Bsync/config.json"
            registered = row(name + "_sr_register", "bsync_new.py",
                ["sr-register", "--scene", scene, "--parent", rel(train), "--teacher", rel(teacher),
                 "--clean", rel(clean)], [materialized["name"]], target)
            groups["finalize"].extend([required, materialized, registered])
            tasks.extend([required, materialized, registered])
            owners.append(dict(context=key, after=schedule_row["name"],
                review=[rel(schedule_dir / "assignment_statistics.json"), rel(coarse_acceptance)],
                argv=[TRAINPY, "-u", rel(CODE / "acceptance_new.py"), "assignment",
                      "--candidate", rel(schedule_dir / "assignment_acceptance_candidate.json"),
                      "--source-registration-accepted", "--storage-and-cost-accepted",
                      "--out", rel(assignment)],
                note="Only Claude executes these flags after actual storage/cost and source review"))
            external.extend([
                dict(context=key, required_before="require_schedule", path=rel(train), purpose="shared gateaff SR parent"),
                dict(context=key, required_before="require_schedule", path=rel(assignment), purpose="explicit owner assignment acceptance"),
                dict(context=key, required_before="materialize", path=rel(teacher), purpose="complete areaft teacher inventory"),
                dict(context=key, required_before="materialize", path=selection, purpose="original frozen Bsync selection and bound evidence"),
            ])
            replacements[rel(REGOUT / f"sr/registry/{scene}_area4/seed_{SEED}/areaft_teacher/Bsync")] = rel(target.parent)
        else:
            control = OUT / f"controls/{key}/v1"
            args = ["--baseline", rel(HIST_SCHEDULE), "--baseline-index", rel(HIST_INDEX),
                    "--candidate", rel(schedule_dir / "schedule.json"),
                    "--candidate-index", rel(run / "index.json"), "--out", rel(control)]
            if key.endswith("_native"):
                args += ["--compare-raw", "--expect-exact"]
            compared = row(name + "_compare", "compare_new.py", args, [schedule_row["name"]], control / "complete.json")
            groups["controls"].append(compared)
            tasks.append(compared)
    return tasks, groups, owners, external, replacements


def validate_rows(tasks):
    names = {t["name"] for t in tasks}
    require(len(names) == len(tasks), "Duplicate task names")
    visited = set()
    for task in tasks:
        require(set(task) == {"name", "script", "argv", "deps", "done"}, "Pipeline row fields differ")
        require((PIPELINE / task["script"]).resolve().is_file(), "Pipeline script missing")
        require(not Path(task["script"]).is_absolute() and not Path(task["done"]).is_absolute()
                and not any(v.startswith("/") for v in task["argv"]), "Absolute row path")
        require(set(task["deps"]) <= visited, "Rows not topologically ordered/unknown dependency")
        require(local(task["done"]).is_relative_to(OUT), "Done outside new namespace")
        visited.add(task["name"])
    return dict(tasks=len(tasks), names_unique=True, DAG=True, exact_five_fields=True,
                scripts_resolved=True, argv_and_done_project_relative=True,
                interpreter=TRAINPY, training_submitted=False, GPU_assignment=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=rel(OUT / "rows"))
    a = p.parse_args()
    tasks, groups, owners, external, replacements = build()
    validation = validate_rows(tasks)
    directory = output_path(a.out)
    write_new(directory / "all.json", tasks)
    for group, rows in groups.items():
        selected = {t["name"] for t in rows}
        subset = [dict(t, deps=[d for d in t["deps"] if d in selected]) for t in rows]
        validate_rows(subset)
        write_new(directory / f"{group}.json", subset)
    for ctx in contexts():
        subset = [t for t in tasks if t["name"].startswith("sched_" + ctx["key"] + "_")]
        validate_rows(subset)
        write_new(directory / f"{ctx['key']}.json", subset)
    write_new(OUT / "ROW_VALIDATION.json", validation)
    write_new(OUT / "CLAUDE_COMMANDS.json", dict(
        cwd=str(ROOT), interpreter=TRAINPY, dispatched=False,
        phases=["plan", "actual service admission", "gpu (workers=1)", "actual service Exit0",
                "schedules", "review actual assignment cost/source and run explicit owner command",
                "finalize", "controls"],
        commands=[dict(name=t["name"], argv=command(t), shell=shlex.join(command(t)), deps=t["deps"]) for t in tasks],
        owner_assignment_commands=owners, external_inputs=external,
        downstream_registry_path_replacements=replacements,
        note="Phase subsets have only internal deps; external gates remain mandatory. Do not run all.json unattended. "
             "Existing downstream task files are preserved; Claude applies the listed registry mapping in a new task file."))
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
