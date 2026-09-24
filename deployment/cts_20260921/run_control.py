#!/usr/bin/env python3
"""Run controlled training and exit-triggered evaluation under a shared CTS lock.

Arguments pass unchanged to the generic runner. A concurrent --apply sync holds
the exclusive form of this same lock, so either operation fails before overlap.
"""
from __future__ import annotations

import fcntl
import argparse
import json
import os
from pathlib import Path
import runpy
import stat
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCK_RELATIVE = 'deployment/cts_20260921/project_access.lock'
PENDING_RELATIVE = 'deployment/cts_20260921/sync_pending.json'
ACCEPTANCE_RELATIVE = 'deployment/cts_20260921/verification/acceptance_exit.json'


def acquire_shared_lock(root=ROOT):
    root = Path(root)
    lock = root / LOCK_RELATIVE
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f'Project root must be a real directory: {root}')
    for path in [lock, *lock.parents]:
        if path.is_symlink():
            raise RuntimeError(f'Refusing lock-path symlink: {path}')
        if path == root:
            break
    if not lock.parent.is_dir():
        raise RuntimeError(f'Deployment directory does not exist: {lock.parent}')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError(f'Lock must be a regular file: {lock}')
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(fd)
        raise RuntimeError('CTS source/data sync is active; retry this new job after sync completes.') from error
    except BaseException:
        os.close(fd)
        raise
    return fd


def require_ready(root=ROOT):
    """Call only while holding the shared project lock, before taking GPU 0."""
    root = Path(root)
    pending = root / PENDING_RELATIVE
    if os.path.lexists(pending):
        raise RuntimeError('CTS sync has not passed complete SHA256 verification; '
                           'rerun --apply successfully before training. Do not delete sync_pending.json manually.')
    acceptance = root / ACCEPTANCE_RELATIVE
    for path in [acceptance, *acceptance.parents]:
        if path.is_symlink():
            raise RuntimeError(f'Refusing acceptance-path symlink: {path}')
        if path == root:
            break
    try:
        value = json.loads(acceptance.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError('CTS deployment acceptance is missing or unreadable; complete deployment acceptance before training.') from error
    if value.get('returncode') != 0 or value.get('stage') != 'complete':
        raise RuntimeError('CTS deployment acceptance has not passed (requires returncode 0 and stage complete).')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', choices=('0',))
    parser.parse_known_args()
    fd = acquire_shared_lock()
    gpu_fd = None
    try:
        if '--help' not in sys.argv and '-h' not in sys.argv:
            require_ready()
            gpu_fd = os.open(ROOT / 'deployment/cts_20260921/gpu0.lock',
                             os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(gpu_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another CTS project job owns GPU 0; choose a free resource.') from error
        print(f'CTS shared project lock held by PID {os.getpid()}: {ROOT / LOCK_RELATIVE}', flush=True)
        # runpy preserves sys.argv[1:] and the original runner's signal handling,
        # child cleanup, exact exit status, and immediate evaluation semantics.
        runpy.run_path(str(ROOT / 'deployment/a100_20260920/run_control.py'), run_name='__main__')
    finally:
        if gpu_fd is not None:
            os.close(gpu_fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == '__main__':
    main()
