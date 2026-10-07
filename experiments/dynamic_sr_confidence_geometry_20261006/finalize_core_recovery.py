"""Guard recovered core consolidation with real exit events and full receipts.

Run with system Python. No training entry point is exposed. Numerical and GPU
evaluation commands are dispatched only through the unchanged original
finalize_core.run, after both completed workers and all twelve endpoints pass.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from cg_common import ROOT, HERE, OUT, read, sha, write
import finalize_core as original

METHODS = ("C1", "Jperm", "B2perm", "R", "G", "RG")
STEPS = [("extra_worker.py", ("--include-modules",)), ("summarize.py", ()),
         ("final_integrity.py", ()), ("execution_index.py", ("--out", OUT / "accounting_final_core", "--verify-checkpoints")),
         ("figures.py", ())]


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def wait_pid_event(pid):
    """Wait in the kernel, never poll files or infer completion from idle GPUs."""
    pid = int(pid)
    if pid <= 0 or pid == os.getpid():
        raise ValueError("Dependency must be another positive process ID")
    started = time.monotonic()
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return dict(pid=pid, event="already_exited_before_pidfd_open", seconds=time.monotonic() - started)
    try:
        process = Path(f"/proc/{pid}")
        try:
            cmdline = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            starttime = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            cmdline, starttime = None, None
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)
    return dict(pid=pid, event="pidfd_exit_received", process_cmdline=cmdline,
                process_starttime_ticks=starttime, seconds=time.monotonic() - started)


def resolve_checkpoint(entry, project_root):
    path = Path(entry["path"])
    return path if path.is_absolute() else project_root / path


def validate_readiness(run_root=OUT, project_root=ROOT):
    run_root, project_root = Path(run_root), Path(project_root)
    # Inner worker.main may write batch_completed before the outer controller
    # validates B2perm and every repeat-2 extra receipt. Require that outer
    # controller's final success, bound through its source and registration.
    recovery_state_path = run_root / "core_recovery_state.json"
    recovery_state = read(recovery_state_path)
    if recovery_state.get("status") != "batch_completed":
        raise ValueError(f"Outer recovery controller has not completed successfully: {recovery_state.get('status')}")
    registration_path = run_root / "core_recovery_registration.json"
    registration = read(registration_path)
    controller = project_root / "experiments/dynamic_sr_confidence_geometry_20261006/recover_core.py"
    expected_source = identity(controller)
    for claimed in [registration.get("source", {}), recovery_state.get("source", {})]:
        if (resolve_checkpoint(claimed, project_root).resolve() != controller.resolve() or
                claimed.get("sha256") != expected_source["sha256"]):
            raise ValueError("Outer recovery source identity mismatch")
    if (registration.get("status") != "registered_CPU_checked_recovery_only" or
            registration.get("never_train_B2perm") is not True or
            registration.get("methods_to_train") != ["R", "G", "RG"] or
            recovery_state.get("B2perm_retrained") is not False or recovery_state.get("new_training_budget") != 18000 or
            set(recovery_state.get("methods", [])) != set(METHODS) or len(recovery_state["methods"]) != 6):
        raise ValueError("Outer recovery completion does not cover the fixed six methods/B2 preservation")
    verification_path = run_root / "remote_recovery_verification.json"
    verification_claim = recovery_state["verification"]
    if (resolve_checkpoint(verification_claim, project_root).resolve() != verification_path.resolve() or
            verification_claim["sha256"] != sha(verification_path)):
        raise ValueError("Outer recovery verification receipt mismatch")
    verification = read(verification_path)
    reg_claim = verification["registration"]
    source_claim = verification["source"]
    if (verification.get("status") != "passed_frozen_recovery_inventory" or
            resolve_checkpoint(reg_claim, project_root).resolve() != registration_path.resolve() or
            reg_claim["sha256"] != sha(registration_path) or
            resolve_checkpoint(source_claim, project_root).resolve() != controller.resolve() or
            source_claim["sha256"] != expected_source["sha256"]):
        raise ValueError("Outer recovery verification/source/registration chain mismatch")
    workers = []
    for filename in ["worker_local_r1.json", "worker_a100-train_r2.json"]:
        path = run_root / filename
        receipt = read(path)
        if receipt.get("status") != "batch_completed":
            raise ValueError(f"Worker has not completed successfully: {filename}: {receipt.get('status')}")
        workers.append(identity(path))
    protocol_path = run_root / "protocol.json"
    protocol = read(protocol_path)
    protocol_claim = registration["protocol"]
    if (resolve_checkpoint(protocol_claim, project_root).resolve() != protocol_path.resolve() or
            protocol_claim["sha256"] != sha(protocol_path)):
        raise ValueError("Outer recovery registration belongs to another core protocol")
    expected = {(method, str(repeat)) for method in METHODS for repeat in [1, 2]}
    tasks = protocol["core_task_plan"]
    actual = {(t["method"], str(t["repeat"])) for t in tasks}
    if actual != expected or len(tasks) != 12:
        raise ValueError("Core protocol must contain exactly six methods x two suffixes")
    endpoints = []
    for method, repeat in sorted(expected):
        label = f"r{repeat}_{method}"
        directory = run_root / "runs" / label
        complete_path = directory / "complete.json"
        complete = read(complete_path)
        if (complete.get("status") != "completed_training" or complete.get("method") != method or
                str(complete.get("repeat")) != repeat or complete.get("updates") != 6000 or
                complete.get("suffix_endpoint") != 6000 or complete.get("adam_calls") != 12000 or
                complete.get("topology_unchanged") is not True):
            raise ValueError(f"Incomplete or mismatched fixed endpoint: {label}")
        rgb_per_round = 3 if method == "B2perm" else 2 if method == "Jperm" else 1
        if (complete.get("training_rgb_forwards") != 6000 * rgb_per_round or
                complete.get("moment_forwards") != (6000 if method in ["G", "RG"] else 0)):
            raise ValueError(f"Endpoint budget mismatch: {label}")
        checkpoint = resolve_checkpoint(complete["checkpoint"], project_root).resolve()
        if checkpoint != (directory / "checkpoint_12000.pt").resolve() or not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise ValueError(f"Missing or misplaced fixed endpoint checkpoint: {label}")
        cp_receipt_path = checkpoint.with_suffix(".json")
        cp = read(cp_receipt_path)
        if (cp.get("sha256") != complete["checkpoint"]["sha256"] or
                cp.get("metadata", {}).get("suffix_step") != 6000 or
                cp["metadata"].get("intervention_step") != 12000):
            raise ValueError(f"Endpoint checkpoint receipt mismatch: {label}")
        # The original final_integrity and execution_index --verify-checkpoints
        # perform complete checkpoint-byte verification after the cheap gate.
        config_path = directory / "config.json"
        config = read(config_path)
        if (config.get("protocol_sha256") != sha(protocol_path) or config.get("original_HR_training") is not False or
                config.get("method") != method or str(config.get("repeat")) != repeat):
            raise ValueError(f"Endpoint configuration identity mismatch: {label}")
        evaluation = run_root / "evaluation" / label
        evaluation_path = evaluation / "complete.json"
        evaluated = read(evaluation_path)
        if (evaluated.get("status") != "completed_evaluation" or
                evaluated.get("checkpoint", {}).get("sha256") != complete["checkpoint"]["sha256"]):
            raise ValueError(f"Missing primary endpoint evaluation: {label}")
        splits = []
        for split in ["test", "dev", "train_fixed"]:
            entry = evaluated["splits"][split]
            metric_path = resolve_checkpoint(entry, project_root).resolve()
            if metric_path != (evaluation / split / "metrics.json").resolve() or sha(metric_path) != entry["sha256"]:
                raise ValueError(f"Primary split receipt mismatch: {label}/{split}")
            metric = read(metric_path)
            if (metric.get("checkpoint_sha256") != complete["checkpoint"]["sha256"] or
                    metric.get("parameter_updates") != 0 or metric.get("split") != split):
                raise ValueError(f"Primary split identity mismatch: {label}/{split}")
            splits.append(identity(metric_path))
        endpoints.append(dict(task=label, updates=6000, training_complete=identity(complete_path),
            checkpoint_path=str(checkpoint), registered_checkpoint_sha256=complete["checkpoint"]["sha256"],
            checkpoint_receipt=identity(cp_receipt_path), config=identity(config_path),
            primary_complete=identity(evaluation_path), splits=splits))
    return dict(status="passed_recovered_core_readiness", workers=workers, endpoints=endpoints,
        outer_recovery=dict(state=identity(recovery_state_path), registration=identity(registration_path),
                            verification=identity(verification_path), source=expected_source),
        endpoint_count=12, formal_updates=72000, protocol=identity(protocol_path),
        checkpoint_byte_checks="Deferred to unchanged final_integrity and execution_index --verify-checkpoints",
        quality_threshold_used=False)


def guarded_steps(run_function, run_root=OUT, project_root=ROOT, before_steps=None):
    readiness = validate_readiness(run_root, project_root)
    if before_steps is not None:
        before_steps(readiness)
    for script, arguments in STEPS:
        run_function(script, *arguments)
    return readiness


def archive_prior_failure(output):
    archived = output / "prior_failure"
    archived.mkdir()
    state = read(OUT / "core_finalization_state.json")
    if state.get("status") != "failed":
        raise ValueError("Prior finalization must be an explicit preserved failure before recovery")
    records = []
    for filename in ["core_finalization_state.json", "core_finalization.log"]:
        source = OUT / filename
        expected = identity(source)
        destination = archived / filename
        shutil.copyfile(source, destination)
        if sha(destination) != expected["sha256"] or sha(source) != expected["sha256"]:
            raise ValueError("Prior failure changed during archival")
        destination.chmod(0o444)
        records.append(dict(original=expected, archived=identity(destination)))
    write(archived / "identity.json", dict(status="archived_original_failure_exact_bytes", records=records))
    (archived / "identity.json").chmod(0o444)
    return identity(archived / "identity.json")


def recover(args):
    with (OUT / "core_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(parents=True, exist_ok=False)
        executed, waits = [], []
        plan_path = args.out / "plan.json"
        try:
            archive = archive_prior_failure(args.out)
            sources = [identity(HERE / name) for name in [Path(__file__).name, "finalize_core.py", "cg_common.py",
                "worker_async.py", "recover_core.py", "extra_worker.py", "summarize.py", "final_integrity.py", "execution_index.py", "figures.py"]]
            plan = dict(status="registered_recovery_consolidation", recovery_controller_pid=args.recovery_pid,
                local_worker_pid=args.local_pid, prior_failure=archive, sources=sources,
                unchanged_dispatch=identity(OUT / "module_dispatch_state.json"),
                outer_recovery_registration=identity(OUT / "core_recovery_registration.json"),
                required_worker_status="batch_completed", required_endpoints=12, updates_per_endpoint=6000,
                steps=[dict(script=s, arguments=list(map(str, a))) for s, a in STEPS],
                training_dispatch_allowed=False, quality_selection_allowed=False)
            write(plan_path, plan)
            for role, pid in [("recovery_controller", args.recovery_pid), ("local_worker", args.local_pid)]:
                event = wait_pid_event(pid); event["role"] = role; waits.append(event)
                write(args.out / "dependency_events.json", dict(status="exit_events_received", events=waits))
            for entry in sources:
                if sha(entry["path"]) != entry["sha256"]:
                    raise ValueError("Frozen consolidation source changed while waiting")
            if identity(OUT / "module_dispatch_state.json") != plan["unchanged_dispatch"]:
                raise ValueError("Original dispatch evidence changed")
            if identity(OUT / "core_recovery_registration.json") != plan["outer_recovery_registration"]:
                raise ValueError("Outer recovery registration changed while waiting")

            def accepted(readiness):
                write(args.out / "readiness.json", readiness)
                write(OUT / "core_finalization_state.json", dict(status="consolidating_complete_core",
                    started_unix=time.time(), recovery_plan=identity(plan_path), recovery_dependencies=waits,
                    original_failure_preserved=archive))

            def run_step(script, *arguments):
                original.run(script, *arguments)
                executed.append(script)
                write(args.out / "steps.json", dict(status="running", completed_steps=executed))

            readiness = guarded_steps(run_step, before_steps=accepted)
            completion = dict(status="passed_recovery_consolidation", plan=identity(plan_path),
                readiness=identity(args.out / "readiness.json"), dependency_events=identity(args.out / "dependency_events.json"),
                completed_steps=executed, original_failure_preserved=archive, source_sha256=sha(__file__))
            write(args.out / "complete.json", completion)
            write(OUT / "core_finalization_state.json", dict(status="core_complete_pending_research_and_visual_review",
                final_integrity=identity(OUT / "final_integrity.json"), quality_summary=identity(OUT / "quality_summary.json"),
                accounting=identity(OUT / "accounting_final_core/execution_index.json"), figures=identity(OUT / "figures/index.json"),
                recovery=identity(args.out / "complete.json"), original_failure_preserved=archive,
                recovered_endpoints=readiness["endpoint_count"], conditional_phase13_automatically_started=False,
                chat_notification_claimed=False))
            print(json.dumps(dict(status="core_complete_pending_research_and_visual_review", recovery=str(args.out))))
        except BaseException:
            failure = dict(status="failed", traceback=traceback.format_exc(), completed_steps=executed,
                dependency_events=waits, training_dispatched=False, source_sha256=sha(__file__))
            write(args.out / "failed.json", failure)
            write(OUT / "core_finalization_state.json", dict(status="failed", traceback=failure["traceback"],
                recovery_failure=identity(args.out / "failed.json"), completed_steps=executed,
                prior_failure_archive=str(args.out / "prior_failure")))
            raise


def cpu_self_check(output):
    """Synthetic receipts and real CPU pidfd event; no real outputs are mutated."""
    with tempfile.TemporaryDirectory(prefix="cg_finalize_recovery_") as temporary:
        project = Path(temporary); run = project / "output"; run.mkdir()
        protocol = dict(core_task_plan=[dict(method=m, repeat=str(r)) for m in METHODS for r in [1, 2]])
        write(run / "protocol.json", protocol)
        controller = project / "experiments/dynamic_sr_confidence_geometry_20261006/recover_core.py"
        controller.parent.mkdir(parents=True); controller.write_bytes(b"SYNTHETIC_CPU_CONTROLLER_IDENTITY")
        registration_path = run / "core_recovery_registration.json"
        write(registration_path, dict(status="registered_CPU_checked_recovery_only", source=identity(controller),
            never_train_B2perm=True, methods_to_train=["R", "G", "RG"], protocol=identity(run / "protocol.json")))
        verification_path = run / "remote_recovery_verification.json"
        write(verification_path, dict(status="passed_frozen_recovery_inventory", source=identity(controller), registration=identity(registration_path)))
        recovery_state_path = run / "core_recovery_state.json"
        completed_outer = dict(status="batch_completed", methods=list(METHODS), B2perm_retrained=False,
            new_training_budget=18000, source=identity(controller), verification=identity(verification_path))
        write(recovery_state_path, completed_outer)
        for name in ["worker_local_r1.json", "worker_a100-train_r2.json"]:
            write(run / name, dict(status="batch_completed"))
        for method in METHODS:
            for repeat in [1, 2]:
                label = f"r{repeat}_{method}"; directory = run / "runs" / label; directory.mkdir(parents=True)
                checkpoint = directory / "checkpoint_12000.pt"; checkpoint.write_bytes(b"SYNTHETIC_CPU_ONLY")
                cp = identity(checkpoint)
                write(checkpoint.with_suffix(".json"), dict(sha256=cp["sha256"], metadata=dict(suffix_step=6000, intervention_step=12000)))
                write(directory / "complete.json", dict(status="completed_training", method=method, repeat=str(repeat),
                    updates=6000, suffix_endpoint=6000, adam_calls=12000, topology_unchanged=True, checkpoint=cp,
                    training_rgb_forwards=6000 * (3 if method == "B2perm" else 2 if method == "Jperm" else 1),
                    moment_forwards=6000 if method in ["G", "RG"] else 0))
                write(directory / "config.json", dict(method=method, repeat=str(repeat), protocol_sha256=sha(run / "protocol.json"), original_HR_training=False))
                splits = {}
                for split in ["test", "dev", "train_fixed"]:
                    path = run / "evaluation" / label / split / "metrics.json"
                    write(path, dict(checkpoint_sha256=cp["sha256"], parameter_updates=0, split=split)); splits[split] = identity(path)
                write(run / "evaluation" / label / "complete.json", dict(status="completed_evaluation", checkpoint=cp, splits=splits))
        calls = []
        def mock_run(script, *arguments):
            calls.append(script)
        # All twelve endpoints and inner workers can be complete while the
        # outer B2/extra verification still fails. This must remain blocked.
        write(recovery_state_path, dict(completed_outer, status="stopped_real_failure"))
        try:
            guarded_steps(mock_run, run, project)
        except ValueError:
            outer_failure_refused = not calls
        else:
            outer_failure_refused = False
        assert outer_failure_refused
        write(recovery_state_path, completed_outer)
        failed_worker = run / "worker_a100-train_r2.json"
        write(failed_worker, dict(status="failed"))
        try:
            guarded_steps(mock_run, run, project)
        except ValueError:
            failed_worker_refused = not calls
        else:
            failed_worker_refused = False
        assert failed_worker_refused
        write(failed_worker, dict(status="batch_completed"))
        missing = run / "runs/r2_G/complete.json"; saved = missing.read_bytes(); missing.unlink()
        try:
            guarded_steps(mock_run, run, project)
        except FileNotFoundError:
            missing_endpoint_refused = not calls
        else:
            missing_endpoint_refused = False
        assert missing_endpoint_refused
        missing.write_bytes(saved)
        accepted = guarded_steps(mock_run, run, project)
        assert accepted["endpoint_count"] == 12 and calls == [s for s, _ in STEPS]
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.15)"])
        try:
            event = wait_pid_event(child.pid)
            assert event["event"] == "pidfd_exit_received" and child.wait() == 0
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
        output.mkdir(parents=True, exist_ok=False)
        result = dict(status="passed_CPU_recovery_guards", source=identity(__file__), failed_worker_refused_before_steps=failed_worker_refused,
            outer_failed_with_inner_workers_and_12_endpoints_completed_refused_before_steps=outer_failure_refused,
            outer_completion_source_registration_verification_bound=True,
            missing_endpoint_refused_before_steps=missing_endpoint_refused, all_12_mock_endpoints_allowed=True,
            exact_mock_step_order=calls, real_cpu_dependency_event=event, actual_pipeline_steps=0,
            actual_gpu_dispatches=0, actual_training_dispatches=0, real_outputs_mutated=False)
        write(output / "cpu_checks.json", result)
        print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-pid", type=int)
    parser.add_argument("--local-pid", type=int, default=300903)
    parser.add_argument("--cpu-self-check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_self_check:
        cpu_self_check(args.out)
    else:
        if args.recovery_pid is None or args.recovery_pid <= 0 or args.local_pid <= 0:
            parser.error("Positive --recovery-pid and --local-pid are required")
        recover(args)


if __name__ == "__main__":
    main()
