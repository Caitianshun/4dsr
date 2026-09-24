#!/usr/bin/env python3
"""Use run_job's unchanged CLI with blocking cooperative GPU-job queues.

Only lock acquisition changes: jobs sleep in flock until the previous owner
releases the same GPU lease. After acquiring it, run_job still records and
checks nvidia-smi, rejecting unrelated compute processes. The training lease
is released before waiting for evaluation; training algorithms, arguments,
checkpoints, evaluation protocols and the frozen run_job.py are unchanged.
"""
import run_job


_original_lock_file = run_job.lock_file


def queued_lock_file(path, blocking=True):
    """Force kernel-blocking locks, including run_job's training GPU lease."""
    return _original_lock_file(path, blocking=True)


def main():
    run_job.lock_file = queued_lock_file
    run_job.main()


if __name__ == '__main__':
    main()
