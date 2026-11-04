"""Claude-only receipt construction from real fixed-service exports.

This utility neither grants admission nor observes process exit. It binds actual
service evidence already obtained by Claude. Never run with fabricated exports.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *
from acceptance_new import pointer, verify_service_exit
from make_rows import build, command, contexts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("admission", "exit"))
    p.add_argument("--context", choices=tuple(c["key"] for c in contexts()), required=True)
    p.add_argument("--raw-service-export", required=True)
    p.add_argument("--project-id-pointer", default="/project_id")
    p.add_argument("--job-id-pointer", default="/job_id")
    p.add_argument("--unit-pointer", default="/unit")
    p.add_argument("--gpu-uuid-pointer", default="/gpu_uuid")
    p.add_argument("--status-pointer", default="/status")
    p.add_argument("--exit-code-pointer", default="/exit_code")
    a = p.parse_args()
    raw = read(a.raw_service_export)
    fields = {key: getattr(a, key + "_pointer") for key in
              ("project_id", "job_id", "unit", "gpu_uuid", "status", "exit_code")}
    project = pointer(raw, fields["project_id"])
    job = pointer(raw, fields["job_id"])
    unit = pointer(raw, fields["unit"])
    gpu = pointer(raw, fields["gpu_uuid"])
    status = pointer(raw, fields["status"])
    require(project == "4dsr" and isinstance(job, str) and job
            and unit.endswith(".service") and "/" not in unit
            and isinstance(gpu, str) and gpu.startswith("GPU-"), "Actual project/job/unit/GPU export required")
    key = a.context
    plan = OUT / f"plans/{key}/registered_plan.json"
    admission_path = OUT / f"owner_inputs/{key}/admission.json"
    tasks = build()[0]
    task = next(t for t in tasks if t["name"] == f"sched_{key}_cuda")
    if a.mode == "admission":
        require(status in ("admitted", "queued", "ready", "running"),
                "Service has not admitted this job")
        value = dict(schema="fixed_service_overlap_admission_v1", project_id=project,
                     job_id=job, unit=unit, gpu_uuid=gpu, admitted=True, interpreter=TRAINPY,
                     env_id="mml_bw", plan=entry(plan), runtime_root=str(ROOT),
                     producer_source=entry(FP / "full_held_parent_overlap.py"),
                     manager_source=entry(CODE / "overlap_new.py"),
                     resource_adapter=entry(CODE / "overlap_new.py"),
                     physical_gpu_lock=str(OUT / "locks" / (gpu + ".coarse-overlap.lock")),
                     producer_command=command(task),
                     admission_evidence=entry(a.raw_service_export),
                     raw_fields={k: fields[k] for k in ("project_id", "job_id", "unit", "gpu_uuid", "status")},
                     raw_admitted_status=status)
        out = admission_path
    else:
        admission = read(admission_path)
        code = pointer(raw, fields["exit_code"])
        require(status in ("succeeded", "completed", "success")
                and type(code) is int and code == 0, "Actual successful terminal Exit0 required")
        run = OUT / f"overlap/{key}/run_v1"
        value = dict(schema="fixed_service_overlap_exit_v1", project_id=project, job_id=job,
                     unit=unit, gpu_uuid=gpu, interpreter=TRAINPY,
                     exited_normally=True, exit_code=code,
                     producer_complete=entry(run / "complete.json"),
                     all_output_validation=entry(run / "all_output_validation.json"),
                     admission=entry(admission_path), producer_command=admission["producer_command"],
                     raw_service_result=entry(a.raw_service_export), raw_fields=fields,
                     raw_success_status=status)
        out = OUT / f"owner_inputs/{key}/service_exit.json"
    write_new(out, value)
    if a.mode == "exit":
        verify_service_exit(out, bound(value["producer_complete"]), bound(value["all_output_validation"]))
    no_torch()
    print(json.dumps(dict(receipt=entry(out), status="bound_actual_service_export"), ensure_ascii=False))


if __name__ == "__main__":
    main()
