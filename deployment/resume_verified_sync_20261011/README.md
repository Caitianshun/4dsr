# Explicit Desk additive-sync recovery

This adapter applies only to the original Desk run
`deskgames_cube_seed07_20261011t060613z`. Its six local CPU stages completed.
The additive rsync exited with socket-I/O code10 after a CTS timeout. Its23
events end at synchronization preflight; no startup/runner dispatch appears.
Its failure receipt has `remote_started=false`.

The adapter imports the frozen original supervisor. It never calls its
`run()` or any CPU preparation method, never reruns Z1/Z5, never remakes a manifest,
initialization, HR expansion, registration, teacher plan or task table. A
fresh recovery spec/output stores all new logs. The original failure,
events, partial transfer and source records are preserved.

Before recovery it checks the exact23-event sequence, all six CPU exit
codes, original failure mechanism, old accepted preflight partition, final
event020 freeze and original sync inventory. The initial
`screening_freeze.json` contains fewer inputs and is insufficient by itself.
All original science/hardware settings and the28-task table bytes remain
fixed. It validates the original prepared/decode/HR receipts and numeric
Z1/Z5 acceptance. `--make-spec` streams every local synchronization file
through SHA256. It adds its three source files and immutable old-run
evidence to the exact final freeze, and binds the operator's read-only
remote audit receipt.

Execution acquires the original condition lock, verifies that the old remote
startup/headroom units, condition GPU products and shared GPU claim are
absent, then invokes the unchanged parent `sync(inventory)`. That method
recomputes which remote members are missing, reuses equal SHA files,
rejects different existing files, transfers only missing members with
`--ignore-existing`, and checks all synchronized file hashes. Its65GB
future-output plus60GB reserve guard is preserved. Old preflight acceptance
does not count as completed synchronization. A differing partial file is
preserved and blocks recovery; it is never overwritten automatically.

After successful sync, the adapter repeats the scoped remote no-dispatch
audit and calls only the original `startup`, `runner`, `pull`, and
`summarize` methods. The startup is the original50-step/three-frame diagnostic
and cannot become a formal training parent. The new run ID changes only the
outer startup/pipeline/service/output identities; the original task table
and condition-level training outputs remain unchanged. GPU allocation,
claims, completion events, immediate evaluation/return and no-retry rules
remain in the original methods. Final `summarize()` may call only the original
`headroom.py` CPU analysis with the exact recovery-output spec/result paths
and original six endpoints/budgets. All other CPU calls are rejected.

Only the root operator runs these commands after its remote audit:

```sh
python deployment/resume_verified_sync_20261011/resume_sync.py --make-spec \
  --run-id deskgames_sync_recovery_UNIQUE \
  --spec-out output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json \
  --remote-audit output/OPERATOR_REMOTE_AUDIT.json
python deployment/resume_verified_sync_20261011/resume_sync.py \
  --spec output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json --validate-only
```

Use a lowercase unique run ID. Neither command connects to a host, calls a
GPU, or launches work. The root starts `--spec` using a fresh persistent
local user service. No automatic retry/remake occurs. If a future stage
fails after GPU dispatch, this adapter must not be reused: the new failure
and remote artifacts require a separate explicit review.

The scoped absence check is a guard against this condition's prior
dispatch. It does not claim that no unrelated GPU work ever ran on CTS.
Live GPU availability is separately checked by the unchanged parent
methods immediately before launch. Long360 is unsupported.
