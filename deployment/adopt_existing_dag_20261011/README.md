# Observe the existing Desk DAG

Use this entry to recover result collection after a local SSH waiter exits.
An SSH exit does not establish that the remote job stopped. Before activation,
obtain a fresh observation of the same original unit/InvocationID and, when
active, its MainPID/start ticks/UID/cwd/argv. The source pins the original unit
and InvocationID; a later invocation of the same unit is rejected both locally
and remotely. An old startup receipt proves identity, not current liveness.
Keep actual identity observations and acceptance results in private receipts.

The entry observes only the pinned existing Desk service. It never
calls the old `run`, `startup`, `runner` or `sync`, and never launches a GPU
service, teacher, evaluation or training. A separate CPU observer unit uses
`pidfd_open` plus blocking `select` to wait for that exact existing process.
It persists across SSH loss and writes an exit receipt. No model polling or
system package is required. The original DAG completion/failure file decides
the result; a single post-exit systemd reading is diagnostic. If the original
unit is already terminal, only a short receipt/return operation is needed.

The local observer keeps the original spec/run ID and result paths. It uses a
fresh local output for its wait/return logs, checkpoint index and final headroom
summary. The original startup-acceptance JSON is copied byte-for-byte with its
source pinned. The original condition flock protects local ownership; the
observer never takes a GPU claim or changes the running service.

The declared mutable snapshot scope contains exactly three paths: the original
pipeline teacher log, pipeline `status.json` and original service log. Only
after confirmed original exit, the entry records their SHA and renames these
snapshots into its private archive. It then uses the original additive/SHA pull.
Other local files remain in place for reuse. Old failure records, inventories
and specs remain unchanged. Different local results are never silently
overwritten.

An operator creates a new config only after actual identity confirmation:

- `schema`: `adopt_existing_Desk_DAG_v1`; a unique lowercase `observer_id`.
- `original_spec`: entry of the original output's `spec.json`.
- `identity_receipt`: entry of a fresh normalized actual observation with schema
  `existing_owned_Desk_DAG_identity_v1`; `target`, `remote_root`, original `unit`,
  `InvocationID`, `MainPID`, `process_alive`, `ActiveState`, and `ssh_jump`.
  When active, include actual `start_ticks`, `uid`, `cwd`, full `argv`.
  A terminal observation has `MainPID=0` and inactive/failed state.
- `freeze`: original freeze plus exact entries for this entry and README,
  identity receipt, original spec/startup acceptance/result inventory and both
  preserved failure records, plus
  `tmp/codex_highres_validation_20261010/resource_audit/Desk_formal_startup_actual_20261011.json`.
  Do not alter the original spec.

First validate the private config locally:

```sh
python deployment/adopt_existing_dag_20261011/adopt_existing_dag.py \
  --config PRIVATE_CONFIG.json --validate-only
```

This checks local bindings without SSH or GPU work. After actual identity
acceptance, the root launches the same entry with `--config PRIVATE_CONFIG.json`
in an independent persistent local service. The remote observer uses only CPU
process-exit waiting; the original GPU service remains untouched. A failed SSH
wait never authorizes another GPU job or an automatic retry.
