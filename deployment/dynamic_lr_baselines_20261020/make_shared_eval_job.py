"""Freeze a render-only A100 evaluation job for explicitly returned endpoints.

This prepares a job and a return plan. It neither submits GPU work nor edits
the result registry. Already scored entries are not evaluated a second time.
"""
import argparse
import json
from pathlib import Path
import re

from gather_results import METHODS, ROOT, SCENES, local_path, sha


GPU_UUIDS = {0: "GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b", 1: "GPU-b79cd3fe-81f0-f449-30be-432e2857e517"}
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--method", action="append", choices=METHODS, required=True)
    p.add_argument("--scene", action="append", choices=SCENES,
                   help="Only these explicitly returned scenes; default is both")
    p.add_argument("--job-id", required=True)
    p.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,63}", a.job_id)
    registry = json.loads(a.registry.read_text())
    selected = [r for r in registry if r["method"] in a.method and r.get("index")
                and r["scene"] in (a.scene or SCENES)]
    assert selected, "No explicit returned endpoints to evaluate"
    required = {}

    def freeze(path):
        path = local_path(path)
        relative = path.resolve().relative_to(ROOT).as_posix()
        required[relative] = sha(path)
        return relative

    for name in ("run_job.py", "evaluate_images.py"):
        freeze(HERE / name)
    freeze(ROOT / "experiments/dynamic_sr_20260918/evaluate.py")
    commands, plans = [], []
    for row in selected:
        envelope = json.loads(local_path(row["index"]).read_text())
        index = envelope["methods"][row["key"]] if row.get("key") else envelope
        assert index["method"] == row["method"] and index["scene"] == row["scene"]
        assert index.get("original_LR", envelope.get("original_LR")) is True
        assert index.get("no_proposed_modules", envelope.get("no_proposed_modules")) is True
        assert index["status"] in ("complete_native_training_and_300_HR_PNGs", "completed", "completed_returned_unified_PNG_metrics")
        # Existing common scores are retained, never overwritten/reselected.
        assert "unified_metrics" not in index, "Endpoint already has shared metrics"
        assert row["scene"] in SCENES
        manifest = local_path(index["manifest"]["path"])
        assert sha(manifest) == index["manifest"]["sha256"]
        freeze(manifest)
        render = index.get("renders") or index["rendered_pngs"]
        directory = local_path(render["directory"])
        pattern = render["pattern"]
        assert {x.name for x in directory.glob("*.png")} == {pattern.format(frame=f) for f in range(300)}
        for frame in range(300):
            freeze(directory / pattern.format(frame=frame))
        metrics = Path("output/dynamic_lr_baselines_20261020/shared_A100_metrics") / row["method"] / row["scene"]
        assert not (ROOT / metrics).exists(), f"Evaluation output already exists: {metrics}"
        commands.append(dict(name=f"metrics-{row['method']}-{row['scene']}", argv=[
            "{ROOT}/.venv/bin/python", "-u", "{ROOT}/" + str(REL / "evaluate_images.py"),
            "--manifest", "{ROOT}/" + str(manifest.relative_to(ROOT)),
            "--render-dir", "{ROOT}/" + str(directory.relative_to(ROOT)), "--pattern", pattern,
            "--method", row["method"], "--out", "{ROOT}/" + str(metrics)]))
        plans.append(dict(method=row["method"], scene=row["scene"], metrics=str(metrics / "summary.json"),
                          source_index_sha256=sha(local_path(row["index"]))))
    receipt = json.loads((HERE / "a100_input_verified.json").read_text())
    assert receipt["status"] == "verified"
    output = Path("output/dynamic_lr_baselines_20261020/shared_eval_jobs") / a.job_id
    assert not (ROOT / output).exists()
    job = dict(job_id=a.job_id, method="shared_image_metrics_only", scene="explicit_returned_endpoints",
               original_LR=True, no_proposed_modules=True, gpu_uuid=GPU_UUIDS[a.gpu_index],
               out=str(output), input_receipt=str(REL / "input_verified.json"),
               input_inventory_sha256=receipt["inventory_sha256"],
               env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", PYTHONNOUSERSITE="1"),
               required_files=[dict(path=k, sha256=v) for k, v in sorted(required.items())],
               commands=commands, return_paths=[str(output)] + [str(Path(x["metrics"]).parent) for x in plans])
    destination = HERE / "jobs" / (a.job_id + ".json")
    with destination.open("x") as stream:
        stream.write(json.dumps(job, indent=2) + "\n")
    plan_path = destination.with_suffix(".plan.json")
    with plan_path.open("x") as stream:
        stream.write(json.dumps(dict(job_sha256=sha(destination), gpu_index=a.gpu_index, endpoints=plans), indent=2) + "\n")
    print(json.dumps(dict(status="prepared_not_submitted", job=str(destination), plan=str(plan_path),
                          endpoints=len(plans), frozen_files=len(required))))


if __name__ == "__main__":
    main()
