# Local event-driven validation supervisor

`local_orchestrator.py` is an execution entry, not an active service on creation.
Only half-resolution DeskGames cube on CTS GPU0 and Long360 on A100 GPU0/GPU1,
with seed20261007, are accepted. Every card runs one GPU task at a time.

The root operator waits until fresh code is frozen, then captures a spec:

```bash
python -B experiments/dynamic_sr_highres_20261011/local_orchestrator.py \
  --make-spec --dataset long360 --supervisor-pid OWNED_Z5_FOLLOW_PID \
  --gpu-uuid A100_GPU0_UUID --gpu-uuid A100_GPU1_UUID \
  --spec-out output/dynamic_sr_highres_20261011/orchestration/spec_long360.json
```

For DeskGames use `--dataset deskgames_cube` and one CTS GPU0 UUID. If the owned
Z5 follow service has already exited and returned its complete receipt, omit
`--supervisor-pid`. The PID is bound to its start ticks and command hash, so a
reused PID cannot become a completion event. The entire fresh Python inventory,
Z1, dataset config and selected window are frozen. No GPU is queried by spec
creation or `--validate-only`.

```bash
python -B experiments/dynamic_sr_highres_20261011/local_orchestrator.py \
  --spec output/dynamic_sr_highres_20261011/orchestration/spec_long360.json \
  --validate-only
```

Root starts this command as a persistent local user service without
`--validate-only`. It then performs these ordered steps:

1. Wait once on the owned Z5 supervisor's Linux process-exit event. Check the
   returned completion, sample pixels, source code, frozen prior and metrics.
   A rejected/missing Z5 receipt writes `reject.json`; no preparation, transfer
   or training is launched.
2. Prepare only the half rung. Verify full source-video decode counts and
   packet PTS for every camera, initialize from train LR only at seed07, and
   expand privileged HR separately with exact LR round-trip hash checks.
3. Register LR, write the frozen teacher plan, and generate the six-endpoint
   screening DAG. Generator Z5 references use project-relative paths. Source
   records in manifests and receipts keep their original contents.
4. Stream SHA256 inventories in1MiB blocks. Ask the remote host which files are
   absent; refuse any differing existing file. Rsync only absent files, then
   verify every required file. No historical results, raw datasets, HOI assets,
   local CUDA extensions or environment installations are transferred. The
   pinned generator Python snapshot in an old output directory is the only
   explicit source-only exception; its original path and SHA remain unchanged.
   The four external frozen metric helper files are also transferred explicitly.
5. Check remote idle resource readings20seconds apart, disk reserve and
   explicit GPU index/UUID pairs. Validate the DAG, launch a persistent remote
   user service for50coarse LR steps and a3-frame render evaluation first:
   A100GPU1 forLong360, CTSGPU0 forDeskGames. This independent checkpoint lives
   under `smoke/` with a marker refusing formal-parent use. Success verifies the
   new scene's gradient/CUDA/render path, not convergence or image quality.
   Return its checkpoints/logs and verify all hashes before formal work.
6. Launch the formal persistent remote user service. The original
   `systemd-run --wait` command waits on native D-Bus unit-completion events;
   it does not issue a second start/restart command or poll process state.
   A nonzero exit still triggers artifact return before failure is reported.
   The runner itself
   checks gates/source identities before each task and immediately chains its
   training evaluation/diagnostics.
7. Return output metadata, logs and all checkpoints by an additive verified
   transfer. Store a local checkpoint index with local/remote paths, hardware,
   seed, condition, bytes and SHA256. Teacher PNGs remain remote; their inventory
   and acceptance receipts return locally and can support a later requested
   incremental transfer.
8. Validate six full300 evaluations and their seed/condition/checkpoint bindings,
   write a `headroom.py` spec, and save the paired empirical headroom table.
   Also summarize the four training-view prior-fit evaluations at15/150/285,
   on every training camera. Direct teacher, bicubic, raw rendered and learned
   affine-corrected metrics remain named separately. Conditional image/render
   differences are not automatically attributed to a module.

Every stage checks the unchanged namespace inventory and frozen Z1/Z5 bytes.
CPU stages have persistent logs plus immediate start/exit/failure records. The
local and remote guards retain60GB free after declared stage/transfer costs;
remote planning also reserves65GB for teacher/checkpoint work. A failure does
not resubmit a job, overwrite a partial output, extend to full resolution,
create a second seed, or launch a module. Healthy remote work is preserved if
SSH waiting fails. The service records failure immediately and attempts one
artifact return; retained remote jobs and any incomplete return are explicitly
recorded rather than declared successful.

A completed single-seed screen remains awaiting research review. Metric
subtraction does not identify a causal module gain or a mathematical upper
bound, and ROI diagnostics retain their30-frame sampling scope.
