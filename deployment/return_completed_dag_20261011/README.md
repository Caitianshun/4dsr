# Return the completed Desk DAG

This CPU-only entry completes result collection after the local SSH waiter
failed. It binds the unchanged original spec, formal dispatch, table, scientific
sources and the actual successful terminal receipt. The original transient
service may already be unloaded; its pinned final DAG identity and completion
files establish that all28 tasks finished. It never starts or waits for GPU
work, repeats preparation, screening, synchronization, evaluation or training.

A fresh local output holds the stable terminal inventory, transfer logs,
checkpoint index and original headroom summary. Original task and pipeline
paths stay unchanged. One terminal CPU collection verifies the small pinned
receipts and collects the completed inventory; the inherited additive transfer
then copies only missing results, rejects different immutable files and checks
returned bytes. The original startup acceptance is copied without changing it.

The accepted43-file return and checkpoint index remain provenance. Their local
assets must have size/mtime/ctime compatible with the earlier acceptance; a
later local write is rejected without rereading the old weights. Fixed
device/inode/size/mtime/ctime records are checked before and after transfer.
The remote cache reuses accepted SHA only when size and preserved mtime match.
Other files receive a real SHA, cached only while their fixed stat stays equal.

Only four snapshot paths can be archived: HRref training JSONL, its current
pipeline log, pipeline status and current service log. A file is moved into
the fresh private archive only if its current SHA matches the preserved17:13
inventory and differs from the stable final inventory. Equal files remain;
any other difference stops collection. Old specs, inventories and failure
records are preserved. No automatic retry or remote submission occurs.

Create a private config after accepting the actual terminal receipt:

```sh
python3 -B deployment/return_completed_dag_20261011/return_completed_dag.py \
  --make-config --return-id UNIQUE_RETURN_ID \
  --terminal-receipt PRIVATE_ACTUAL_RECEIPT.json --config-out PRIVATE_CONFIG.json
python3 -B deployment/return_completed_dag_20261011/return_completed_dag.py \
  --config PRIVATE_CONFIG.json --validate-only
```

The operator then runs the same entry with `--config PRIVATE_CONFIG.json` in a
fresh persistent local CPU service. The existing verified A100-to-CTS jump and
connection keepalive apply only to this process. The original headroom CPU
whitelist and six endpoint budgets are retained. Remote state is not changed;
collection failure leaves assets and diagnostic records for explicit review.
