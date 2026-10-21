"""Execute one finite, frozen CPU orchestration plan; never retry a step.

Example plan (all paths are relative to the project root)::

  {"outdir":"output/example_sequence_v1", "steps":[
    {"type":"wait_json", "path":"output/parent/complete.json",
     "expect":{"status":"completed"}, "timeoutseconds":86400},
    {"type":"command", "name":"build", "argv":["python3", "{ROOT}/builder.py"]}
  ]}

Run with a persistent supervisor if survival of terminal disconnects is needed.
JSON completion uses Linux inotify events, including atomic rename and missing
parent directories. The hourly snapshot is bookkeeping, not a JSON poll.
Commands execute without a shell, with cwd ROOT and local CUDA hidden. The
caller alone selects and freezes any remote builder/stager/broker commands.
SIGINT/SIGTERM record failure and stop only this executor's local child group;
SIGKILL, host loss and power loss cannot run a finalizer. No remote cancellation,
GPU selection, lock cleanup, automatic replay or new experiment is implemented.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_SECONDS = 3600.0
FAILURE_VALUES = {"failed", "failure", "error", "cancelled", "canceled", "sequencefailed"}
TERMINAL_VALUES = {"completed", "complete", "collected", "success", "passed", "verified"}
MISSING = object()


class SequenceError(RuntimeError):
    pass


class Stopped(SequenceError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def contained(path, *, relative=False):
    path = Path(path)
    if relative and path.is_absolute():
        raise SequenceError("Expected a root-relative path: " + str(path))
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise SequenceError("Path must remain strictly inside ROOT: " + str(path))
    return result


def dotted(value, key):
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def exact(left, right):
    # JSON booleans and numbers must not compare equal by Python coercion.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("outdir"), str):
        raise SequenceError("Plan must contain string outdir and a finite steps array")
    out = contained(plan["outdir"], relative=True)
    if out.exists() or out.is_symlink():
        raise SequenceError("Refusing existing output directory: " + str(out))
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SequenceError("steps must be a nonempty JSON array")
    for step in steps:
        if not isinstance(step, dict):
            raise SequenceError("Each step must be a JSON object")
        if "name" in step and (not isinstance(step["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", step["name"])):
            raise SequenceError("Step name must contain only letters, digits, underscore or hyphen")
        if step.get("type") == "wait_json":
            if not isinstance(step.get("path"), str):
                raise SequenceError("wait_json needs a relative string path")
            contained(step["path"], relative=True)
            expect = step.get("expect")
            if not isinstance(expect, dict) or not expect or any(not isinstance(k, str) or not k or any(not p for p in k.split(".")) for k in expect):
                raise SequenceError("wait_json needs a nonempty dotted-key expect map")
            timeout = step.get("timeoutseconds")
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise SequenceError("timeoutseconds must be positive and finite")
        elif step.get("type") == "command":
            argv = step.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or "\x00" in a for a in argv) or not argv[0]:
                raise SequenceError("command argv must be a nonempty array of strings")
            if step.get("cwd", ".") not in (".", "{ROOT}", str(ROOT)):
                raise SequenceError("Command cwd is fixed to ROOT")
        else:
            raise SequenceError("Unsupported step type: " + str(step.get("type")))
    return out


class FileEvents:
    """Keep ancestor watches so parent replacement cannot lose the target."""

    MASK = (0x00000002 | 0x00000004 | 0x00000008 | 0x00000040 |
            0x00000080 | 0x00000100 | 0x00000200 | 0x00000400 | 0x00000800)

    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]
        self.libc.inotify_init1.restype = ctypes.c_int
        self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.watches = {}

    def close(self):
        os.close(self.fd)

    def watch_nearest(self, target):
        ancestor = target.parent
        while not ancestor.is_dir():
            if ancestor == ROOT:
                raise SequenceError("ROOT disappeared during wait")
            ancestor = ancestor.parent
        ancestor = contained(ancestor) if ancestor != ROOT else ROOT
        if ancestor in self.watches.values():
            return False
        descriptor = self.libc.inotify_add_watch(self.fd, os.fsencode(ancestor), self.MASK)
        if descriptor < 0:
            code = ctypes.get_errno()
            # A concurrent mkdir/rename/delete must be rechecked before sleeping.
            if code in (errno.ENOENT, errno.ENOTDIR):
                return True
            raise OSError(code, "inotify_add_watch: " + str(ancestor))
        self.watches[descriptor] = ancestor
        return True

    def drain(self):
        count = 0
        while True:
            try:
                data = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                return count
            offset = 0
            while offset < len(data):
                descriptor, mask, _, size = struct.unpack_from("iIII", data, offset)
                offset += 16 + size
                count += 1
                if mask & 0x00008000:  # IN_IGNORED: re-add replaced ancestor.
                    self.watches.pop(descriptor, None)


def wait_json(step, result, snapshot):
    target = contained(step["path"], relative=True)
    began = time.monotonic()
    deadline = began + step["timeoutseconds"]
    next_snapshot = began + SNAPSHOT_SECONDS
    result.update(path=str(target), expect=step["expect"], json_reads=0,
                  filesystem_events=0, wait_mechanism="inotify+select; no timed JSON polling")
    watcher = FileEvents()
    need_check = True
    try:
        while True:
            if need_check:
                # Install the nearest watch BEFORE read, then install any newly
                # created deeper parent AFTER read and recheck. Events arriving
                # between read and select remain queued in this descriptor.
                target = contained(step["path"], relative=True)
                watcher.watch_nearest(target)
                try:
                    result["json_reads"] += 1
                    data = target.read_bytes()
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        raise ValueError("Wait JSON root is not an object")
                except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                    result["last_observation"] = type(error).__name__ + ": " + str(error)[:300]
                else:
                    for key in ("status", "remote_terminal.outcome"):
                        observed = dotted(value, key)
                        if isinstance(observed, str) and observed.lower() in FAILURE_VALUES:
                            result.update(observed_failure={key: observed}, observed_json_sha256=digest(data))
                            raise SequenceError("Upstream failure in " + str(target) + ": " + key + "=" + observed)
                    actual = {key: dotted(value, key) for key in step["expect"]}
                    if all(actual[key] is not MISSING and exact(actual[key], expected) for key, expected in step["expect"].items()):
                        result.update(observed_expectations=actual, observed_json_sha256=digest(data))
                        return
                    if any(isinstance(dotted(value, key), str) and dotted(value, key).lower() in TERMINAL_VALUES
                           for key in ("status", "remote_terminal.outcome")):
                        result.update(observed_json_sha256=digest(data), observed_terminal_mismatch={
                            key: "<missing>" if observed is MISSING else observed for key, observed in actual.items()})
                        raise SequenceError("Terminal JSON does not satisfy expect: " + str(target))
                    result["last_observation"] = "Valid JSON; expected fields do not yet match"
                if watcher.watch_nearest(target):
                    need_check = True
                    continue
                need_check = False
            now = time.monotonic()
            if now >= deadline:
                raise SequenceError("Timed out waiting for " + str(target))
            if now >= next_snapshot:
                snapshot()
                next_snapshot = now + SNAPSHOT_SECONDS
            readable, _, _ = select.select([watcher.fd], [], [], max(0, min(deadline, next_snapshot) - time.monotonic()))
            if readable:
                result["filesystem_events"] += watcher.drain()
                need_check = True
    finally:
        watcher.close()


def stop_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def command(step, result, out, index):
    argv = [value.replace("{ROOT}", str(ROOT)) for value in step["argv"]]
    log = out / (f"step_{index:03d}_" + step.get("name", "command") + ".log")
    result.update(argv=argv, cwd=str(ROOT), log=str(log), child_environment={"CUDA_VISIBLE_DEVICES": ""})
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = None
    with log.open("x") as stream:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            result["local_child_pid"] = process.pid
            result["returncode"] = process.wait()
        except BaseException:
            if process is not None:
                stop_child(process)
                result["returncode"] = process.returncode
            raise
    if result["returncode"] != 0:
        raise SequenceError("Command failed with rc=" + str(result["returncode"]) + "; see " + str(log))


def run(plan_path):
    plan_path = contained(plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    out = validate(plan)
    # exist_ok=False is the final race-safe refusal of an old output directory.
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    record = dict(schema="CEM_area4_finite_CPU_sequence_v1", status="running", root=str(ROOT),
                  source_path=str(Path(__file__).resolve()), source_sha256=digest(Path(__file__).read_bytes()),
                  plan_path=str(plan_path), plan_sha256=digest(plan_bytes), started_utc=utc(),
                  pid=os.getpid(), host=os.uname().nodename, current_step=None, steps=[],
                  retry_policy="none", direct_local_CUDA="hidden", remote_actions="only explicit plan commands")
    original_handlers = {}

    def save(snapshot=False):
        value = dict(record, updated_utc=utc(), elapsed_seconds=time.monotonic() - began)
        write_json(out / ("snapshot.json" if snapshot else "status.json"), value)
        return value

    def stopped(signum, _frame):
        raise Stopped("Received " + signal.Signals(signum).name)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, stopped)
        (out / "plan.json").write_bytes(plan_bytes)
        save()
        for index, step in enumerate(plan["steps"]):
            record["current_step"] = index
            result = dict(index=index, type=step["type"], name=step.get("name", f"step_{index:03d}"),
                          status="running", started_utc=utc())
            record["steps"].append(result)
            save()
            before = time.monotonic()
            try:
                if step["type"] == "wait_json":
                    wait_json(step, result, lambda: save(snapshot=True))
                else:
                    command(step, result, out, index)
                result["status"] = "completed"
            finally:
                result.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - before)
            save()
        record.update(status="completed", current_step=None, finished_utc=utc())
        value = save()
        write_json(out / "complete.json", value)
        return 0
    except BaseException as error:
        if record["steps"] and record["steps"][-1]["status"] == "running":
            record["steps"][-1]["status"] = "failed"
        record.update(status="failed", finished_utc=utc(), error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc())
        value = save()
        write_json(out / "failure.json", value)
        return 1
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args.plan)
    except BaseException as error:
        # Invalid plans are rejected before creating output; never modify an old
        # directory just to record a rejection. The caller receives nonzero.
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
