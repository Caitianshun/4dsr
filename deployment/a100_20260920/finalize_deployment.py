"""Finish this deployment once; failed attempts can retry before smoke output exists."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'deployment/a100_20260920'
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def write(state):
    tmp = DEPLOY / 'finalize_status.tmp'
    with tmp.open('w') as handle:
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(DEPLOY / 'finalize_status.json')


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def preflight():
    status_path = DEPLOY / 'finalize_status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous and previous.get('status') == 'complete':
        raise RuntimeError('Deployment already completed; refusing to replace its acceptance records')
    smoke = ROOT / 'output/a100_deployment_smoke_20260920'
    if os.path.lexists(smoke):
        raise RuntimeError(f'Smoke output already exists: {smoke}; preserve it and choose a new '
                           'smoke destination before retrying. No installation was started.')
    if previous and previous.get('status') == 'running':
        pid = previous.get('runner_pid')
        if pid is None:
            raise RuntimeError('Previous running record has no PID; determine that attempt\'s '
                               'outcome before retrying')
        started = process_start(pid)
        if started is not None and (previous.get('runner_start_ticks') is None or
                                    started == previous['runner_start_ticks']):
            raise RuntimeError(f'Previous finalizer is still alive: PID {pid}')
    return previous


def wait_exit(pid):
    # This is a dependency, not our child: interruption must not terminate it.
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            selector.select()
    finally:
        os.close(fd)


class Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


class Processes:
    def __init__(self):
        self.active = None
        self.launching = False
        self.interrupted_by = None

    def on_signal(self, signum, _frame):
        self.interrupted_by = signum
        # Ensure cleanup gets the process handle if a signal arrives in Popen.
        if not self.launching:
            raise Interrupted(signum)

    def run(self, state, stage, command):
        state.update(stage=stage, child_pid=None, command=command)
        write(state)
        print('STAGE', stage, flush=True)
        self.launching = True
        try:
            self.active = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        finally:
            self.launching = False
        if self.interrupted_by is not None:
            raise Interrupted(self.interrupted_by)
        state['child_pid'] = self.active.pid
        write(state)
        rc = self.active.wait()
        state['stage_returncodes'][stage] = rc
        write(state)
        if rc:
            raise subprocess.CalledProcessError(rc, command)
        self.active = None
        state['child_pid'] = None

    def stop(self, signum):
        if self.active is None:
            return None

        def send(value):
            try:
                os.killpg(self.active.pid, value)
            except ProcessLookupError:
                pass

        send(signum)
        try:
            # The smoke shell forwards to its sessions and waits for cleanup;
            # leave enough time for run_control's bounded child-group shutdown.
            self.active.wait(timeout=30)
        except subprocess.TimeoutExpired:
            send(signal.SIGTERM)
            try:
                self.active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                send(signal.SIGKILL)
                self.active.wait()
        send(signal.SIGKILL)
        return self.active.returncode


def finalize(wait_pid):
    # Hold the advisory lock for the whole attempt. It is released by the OS
    # even after interruption, so a failed attempt does not leave a stale lock.
    with (DEPLOY / 'finalize.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = preflight()
        except (BlockingIOError, RuntimeError) as error:
            print(f'Refusing finalization: {error}', file=sys.stderr, flush=True)
            return 2
        attempt = f'{time.time_ns()}_{os.getpid()}'
        history = DEPLOY / 'finalize_history' / attempt
        if previous or (DEPLOY / 'install_exit.json').exists():
            history.mkdir(parents=True, exist_ok=False)
            for name in ('finalize_status.json', 'install_exit.json'):
                source = DEPLOY / name
                if source.exists():
                    shutil.copy2(source, history / name)
        state = dict(status='running', stage='waiting_for_wheels', started_unix=time.time(),
                     runner_pid=os.getpid(), runner_start_ticks=process_start(os.getpid()),
                     attempt=attempt, previous_records=str(history) if history.exists() else None,
                     child_pid=None, stage_returncodes={})
        processes = Processes()
        handlers = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
        exit_code = 1
        try:
            for sig in STOP_SIGNALS:
                signal.signal(sig, processes.on_signal)
            write(state)
            if wait_pid is not None:
                wait_exit(wait_pid)
            assert len(json.loads((DEPLOY / 'prefetched_wheels.json').read_text())) == 22
            processes.run(state, 'install_environment', ['bash', str(DEPLOY / 'install_environment.sh')])
            assert json.loads((DEPLOY / 'raw_download_complete.json').read_text())['status'] == 'complete'
            processes.run(state, 'restore_images', [str(ROOT / '.venv/bin/python'),
                          str(DEPLOY / 'restore_images.py'), '--allow-regenerate'])
            processes.run(state, 'verify_all_data', ['bash', '-c',
                          'sha256sum --quiet -c deployment/a100_20260920/data_required.sha256 '
                          '> deployment/a100_20260920/data_required_check.txt 2>&1'])
            (DEPLOY / 'data_required_verified.json').write_text(json.dumps(
                dict(status='passed', files=8048, bytes=8275451297), indent=2))
            processes.run(state, 'training_and_evaluation', ['bash', str(DEPLOY / 'smoke_training.sh')])
            state.update(status='complete', stage=None)
            exit_code = 0
        except (Interrupted, KeyboardInterrupt) as error:
            for sig in STOP_SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            signum = error.signum if isinstance(error, Interrupted) else signal.SIGINT
            rc = processes.stop(signum)
            state.update(status='interrupted', signal=signal.Signals(signum).name,
                         child_returncode=rc, child_pid=None)
            exit_code = 128 + signum
        except Exception as error:
            for sig in STOP_SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            rc = processes.stop(signal.SIGTERM)
            state.update(status='failed', error=repr(error), child_returncode=rc, child_pid=None)
        finally:
            for sig in STOP_SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            state.update(finished_unix=time.time(), returncode=exit_code)
            try:
                write(state)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        print(json.dumps(state, indent=2), flush=True)
        return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wait_pid', nargs='?', type=int, help='Optional existing wheel-download PID to wait on')
    args = parser.parse_args()
    raise SystemExit(finalize(args.wait_pid))
