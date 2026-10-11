# Desk runtime environment recovery

This local adapter resumes the Desk run whose13,335 input SHA checks passed,
but whose50-step startup exited before training because Tk could not load
`libBLT.2.5.so.8.6`. It supports only CTS, `dgH_cube`, half rung and seed20261007.
The previous failure/output and the original28-task table remain unchanged.

It subclasses the existing `SyncRecovery`. It skips CPU preparation, Z5,
registration, table generation, synchronization and the old no-dispatch audit
that requires the now-existing failed smoke directory to be absent. It reuses
the previous complete SHA acceptance receipt without rehashing all remote inputs,
then calls the unchanged startup→runner→pull→summarize methods. The condition flock, GPU claims,
65GB future-output/60GB reserve guards, diagnostic50-step/three-frame boundary,
failure return and fixed final headroom analysis remain inherited.

At the exact inherited SSH command boundary it supplies:

```text
LD_LIBRARY_PATH=/home/cts/Project/4DSR/tools/runtime-libs:/home/cts/Project/4DSR/tools/cuda-12.8-minimal/lib64
```

For this fresh startup/runner `systemd-run` invocation it adds an explicit
`--setenv`; for ordinary remote Python commands it prefixes `env`. It records
the actual argv. Local CPU calls and rsync are unchanged. It does not change
SSH configuration, the user-manager environment, the virtual environment,
scientific source, training loss, sampling or budgets. The short subprocess
adapter is local to one synchronous coordinator process and is restored on exit.

The operator first verifies the actual CTS library path and import and records
that evidence in a local JSON receipt. Creating/validating the spec uses local
files only; neither command launches SSH, inference or training:

```sh
python deployment/runtime_env_resume_20261011/runtime_env_resume.py --make-spec \
  --run-id deskgames_runtime_resume_UNIQUE \
  --spec-out output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json \
  --runtime-audit output/dynamic_sr_highres_20261011/analysis/RUNTIME_AUDIT.json
python deployment/runtime_env_resume_20261011/runtime_env_resume.py \
  --spec output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json --validate-only
```

Use a unique lowercase run ID. The root operator starts `--spec` in a new
persistent local user service after runtime/GPU acceptance. No automatic retry
or scientific/input rewrite is provided. A further failure is preserved for
explicit review rather than dispatched again.
