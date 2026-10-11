# Return using the accepted terminal inventory

The earlier CPU collection accepted the completed Desk DAG and its562-file
stable inventory. Its archive step stopped before changing any local file:
the live HRref training log copied at17:13 contained22 complete rows more than
the earlier inventory. The old2,639,743-byte prefix has exactly the recorded
SHA; the actual2,644,722-byte local snapshot ends at fine iteration11765.
This is a later live-copy snapshot, rather than the final14000-step log.
The private diagnostic receipt pins this prefix/row evidence; the entry does
not repeat that diagnosis.

This thin entry reuses the pinned successful terminal inventory. It does not
collect another remote inventory, reread accepted weights, dispatch a GPU,
wait for training, or repeat a scientific task. The original terminal-return
source, spec, config, output and failure records stay unchanged.

The new config pins exactly four measured local snapshots and their fixed
stat records. A snapshot whose exact measured SHA differs from the stable
inventory is renamed into the fresh private archive; an equal file remains.
The archive records both the earlier live inventory and actual copied bytes,
including the training log's exact prefix and appended-row evidence. Changed
or unknown local assets are rejected. No deletion or silent overwrite occurs.

After archival, the inherited additive missing-only transfer uses the already
accepted stable inventory and checks returned bytes. The accepted43-file stat
cache, original result paths, original startup acceptance and headroom CPU
whitelist are retained. Only the local collection output and logs are fresh.
SSH or transfer failure is preserved; no automatic retry or resubmission occurs.

```sh
python3 -B deployment/return_terminal_inventory_20261011/return_terminal_inventory.py \
  --make-config --return-id UNIQUE_RETURN_ID --config-out PRIVATE_CONFIG.json
python3 -B deployment/return_terminal_inventory_20261011/return_terminal_inventory.py \
  --config PRIVATE_CONFIG.json --validate-only
```

The operator then runs the same entry with `--config PRIVATE_CONFIG.json` in
a fresh persistent local CPU service. It continues only result transfer and
the original final analysis through the existing process-exit chain.
