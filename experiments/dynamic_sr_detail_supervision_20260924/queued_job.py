#!/usr/bin/env python3
"""The detail runner with kernel-blocking cooperative GPU lease acquisition."""
import run_job

_lock = run_job.lock_file


def queued_lock_file(path, blocking=True):
    return _lock(path, blocking=True)


def main():
    run_job.lock_file = queued_lock_file
    run_job.main()


if __name__ == '__main__':
    main()
