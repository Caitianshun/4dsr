# Continue the original completed Desk DAG

This operator entry resumes the fixed Desk half-resolution, seed07 task table after its original service has exited. It uses a fresh orchestration directory, pipeline directory, service name, and service log. The original table and task output paths remain unchanged.

The original completion checker verifies and reuses the five completed teacher and LR tasks. It checks the original checkpoint hashes, teacher indexes, command identities, and update counts. The remaining tasks retain their original arguments and budgets. Total training is 69,000 updates including the already completed 17,000; completed work is not retrained.

An actual original terminal classification, native exit receipt, and two-reading CTS GPU0 idle receipt must be bound into the new spec. The short remote preflight checks the original five completion metadata files and their hashes, source/table identity, and the absence of the failed evaluation output directory. The original runner then checks checkpoint weights and teacher indexes once before reusing each completed task. Partial outputs, differing artifacts, another service invocation, and foreign GPU claims are preserved and rejected.

The connection is `ssh -J a100-train cts`, with `ServerAliveInterval=30` and `ServerAliveCountMax=6` only for this entry's connections and rsync transport. The original runtime library environment applies only to this entry's Python calls and user service. GPU thresholds, process rejection, UUID selection, and exclusive GPU claim remain unchanged. This entry adds no graphical-process allowance and does not stop ToDesk. The original worker retains its single two-second idle-check buffer. An SSH failure preserves the persistent remote service and artifacts; it never triggers automatic dispatch again.

The original synchronization and startup acceptance are reused. This entry runs the original persistent DAG, waits for its native service exit, then returns artifacts and performs the original headroom summary. A separate fixed local DAG-resume lock allows the independent original terminal-artifact return to finish. The GPU retains the original exclusive UUID lock.

Create the spec using the actual private receipts:

```bash
python -B deployment/resume_existing_completed_dag_20261011/resume_completed_dag.py \
  --make-spec --run-id NEW_UNIQUE_RUN --spec-out NEW_SPEC.json \
  --terminal-classification ACTUAL_TERMINAL.json \
  --resource-receipt ACTUAL_GPU_READINGS.json \
  --native-exit-receipt ACTUAL_ORIGINAL_EXIT.json

python -B deployment/resume_existing_completed_dag_20261011/resume_completed_dag.py \
  --spec NEW_SPEC.json --validate-only
```

`--make-spec` and `--validate-only` perform local binding checks without SSH or GPU execution. The operator launches `--spec NEW_SPEC.json` through a fresh persistent local service. Real acceptance receipts, measurements, logs, and checkpoint indexes remain private research artifacts. There is no automatic retry or experiment expansion.
