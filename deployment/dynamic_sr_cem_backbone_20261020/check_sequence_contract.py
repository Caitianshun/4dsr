"""CPU-only synthetic tests of run_sequence; never invokes a production plan."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("run_sequence.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing existing QA result")
    results = []
    with tempfile.TemporaryDirectory(prefix="4dsr_sequence_cpu_qa_") as folder:
        base = Path(folder)
        runner = base / "runner.py"
        runner.write_text("import importlib.util, pathlib, sys\n"
            "spec=importlib.util.spec_from_file_location('sequence',sys.argv[1])\n"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
            "m.ROOT=pathlib.Path(sys.argv[2]);m.SNAPSHOT_SECONDS=float(sys.argv[3])\n"
            "sys.argv=[sys.argv[1],'--plan',sys.argv[4]]\n"
            "sys.exit(m.main())\n")

        def case(name, plan, *, files=None, action=None, expected=0, snapshot=3600):
            root = base / name
            root.mkdir()
            for path, value in (files or {}).items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
            plan_path = root / "input_plan.json"
            plan_path.write_text(json.dumps(plan))
            before = time.monotonic()
            process = subprocess.Popen([sys.executable, str(runner), str(SOURCE), str(root), str(snapshot), str(plan_path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            thread = None
            if action:
                thread = threading.Thread(target=action, args=(root, process), daemon=True)
                thread.start()
            stdout, stderr = process.communicate(timeout=10)
            if thread:
                thread.join(timeout=2)
                assert not thread.is_alive()
            seconds = time.monotonic() - before
            assert process.returncode == expected, (name, process.returncode, stdout, stderr)
            out = root / plan["outdir"]
            record = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
            item = dict(name=name, passed=True, returncode=process.returncode, elapsed_seconds=seconds,
                        sequence_status=record.get("status") if record else "rejected_before_output_creation")
            results.append(item)
            return root, record, seconds

        wait = dict(type="wait_json", path="events/a/b/result.json", expect={"status":"collected", "remote_terminal.outcome":"success"}, timeoutseconds=4)
        marker_command = dict(type="command", argv=[sys.executable, "-c",
            "import os,pathlib,json;pathlib.Path('marker.json').write_text(json.dumps({'cwd':os.getcwd(),'CVD':os.environ.get('CUDA_VISIBLE_DEVICES')}))"])
        ready = dict(status="collected", remote_terminal=dict(outcome="success"))
        root, record, _ = case("existing_completion", dict(outdir="run", steps=[wait, marker_command]), files={wait["path"]: ready})
        marker = json.loads((root / "marker.json").read_text())
        assert marker == dict(cwd=str(root), CVD="") and len(record["steps"]) == 2
        assert record["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()

        def atomic_event(root, _process):
            time.sleep(0.15)
            target = root / wait["path"]
            target.parent.mkdir(parents=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(ready))
            temporary.replace(target)

        _, record, seconds = case("missing_parents_atomic_event", dict(outdir="run", steps=[wait]), action=atomic_event)
        assert seconds < 2 and record["steps"][0]["filesystem_events"] > 0

        root, record, seconds = case("collected_remote_failure", dict(outdir="run", steps=[wait, marker_command]),
            files={wait["path"]: dict(status="collected", remote_terminal=dict(outcome="failure"))}, expected=1)
        assert seconds < 1 and not (root / "marker.json").exists()
        assert record["steps"][0]["observed_failure"] == {"remote_terminal.outcome":"failure"}

        _, record, seconds = case("failure_cannot_be_expected_success", dict(outdir="run", steps=[dict(wait, expect={"status":"failed"})]),
                                  files={wait["path"]: dict(status="failed")}, expected=1)
        assert seconds < 1 and "Upstream failure" in record["error"]

        _, record, seconds = case("terminal_mismatch", dict(outdir="run", steps=[dict(wait, expect={"status":"completed", "count":3})]),
                                  files={wait["path"]: dict(status="completed", count=2)}, expected=1)
        assert seconds < 1 and "Terminal JSON" in record["error"]

        def partial_then_complete(root, _process):
            time.sleep(0.12)
            target = root / wait["path"]
            target.write_text('{"status":')
            time.sleep(0.12)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(ready))
            temporary.replace(target)

        _, record, _ = case("partial_then_atomic_complete", dict(outdir="run", steps=[wait]),
                            files={wait["path"]:dict(status="running")}, action=partial_then_complete)
        assert record["steps"][0]["json_reads"] >= 3

        # Accelerate bookkeeping ONLY in this isolated child; production remains
        # the fixed 3600-second constant. Snapshot writes must not poll JSON.
        root, record, _ = case("timeout_hourly_bookkeeping_no_json_poll", dict(outdir="run", steps=[dict(wait, timeoutseconds=.3)]),
                              files={wait["path"]:dict(status="running")}, expected=1, snapshot=.05)
        assert (root / "run/snapshot.json").exists() and record["steps"][0]["json_reads"] == 1
        assert "Timed out" in record["error"]

        root, record, _ = case("command_failure_stops_sequence", dict(outdir="run", steps=[
            dict(type="command", argv=[sys.executable,"-c","import sys;print('synthetic failure');sys.exit(7)"]), marker_command]), expected=1)
        assert len(record["steps"]) == 1 and record["steps"][0]["returncode"] == 7 and not (root / "marker.json").exists()
        assert "synthetic failure" in Path(record["steps"][0]["log"]).read_text()

        root, record, _ = case("refuse_old_output", dict(outdir="old", steps=[marker_command]), files={"old/sentinel.json":dict(keep=True)}, expected=2)
        assert record is None and json.loads((root / "old/sentinel.json").read_text()) == dict(keep=True)

        case("reject_output_escape", dict(outdir="../escape", steps=[marker_command]), expected=2)
        case("reject_wait_escape", dict(outdir="run", steps=[dict(wait,path="../escape.json")]), expected=2)
        case("reject_absolute_wait", dict(outdir="run", steps=[dict(wait,path="/tmp/result.json")]), expected=2)
        case("reject_command_cwd", dict(outdir="run", steps=[dict(marker_command,cwd="/tmp")]), expected=2)
        case("reject_invalid_timeout", dict(outdir="run", steps=[dict(wait,timeoutseconds=0)]), expected=2)

        def terminate(root, process):
            time.sleep(.2)
            process.send_signal(signal.SIGTERM)

        root, record, _ = case("signal_wait_final_status", dict(outdir="run", steps=[wait]), action=terminate, expected=1)
        assert record["error_type"] == "Stopped" and (root / "run/failure.json").exists()

        root, record, _ = case("signal_command_final_status", dict(outdir="run", steps=[dict(type="command",argv=[sys.executable,"-c","import time;time.sleep(9)"])]),
                              action=terminate, expected=1)
        assert record["error_type"] == "Stopped" and record["steps"][0]["returncode"] == -signal.SIGTERM

        # Add a file just after the first failed read and before the second watch
        # check. The queued ancestor event closes this check/watch race.
        spec = importlib.util.spec_from_file_location("sequence_contract", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        race_root = base / "read_watch_race"
        race_root.mkdir()
        module.ROOT = race_root
        original = module.FileEvents.watch_nearest
        calls = [0]

        def raced(self, target):
            result = original(self, target)
            calls[0] += 1
            if calls[0] == 2:
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps(ready))
            return result

        module.FileEvents.watch_nearest = raced
        observed = {}
        module.wait_json(wait, observed, lambda: None)
        assert observed["observed_expectations"] == {"status":"collected", "remote_terminal.outcome":"success"}
        results.append(dict(name="read_watch_race", passed=True, filesystem_events=observed["filesystem_events"]))

    result = dict(schema="CEM_area4_CPU_sequence_contract_v1", status="passed", checked_utc=datetime.now(timezone.utc).isoformat(),
                  source_path=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  test_source_path=str(Path(__file__)), test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tests=results, test_count=len(results), temporary_roots_removed=True,
                  production_sequences_started=0, ssh_calls=0, CUDA_calls=0, production_hourly_interval_seconds=3600)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], tests=len(results), source_sha256=result["source_sha256"], output=str(args.output))))


if __name__ == "__main__":
    main()
