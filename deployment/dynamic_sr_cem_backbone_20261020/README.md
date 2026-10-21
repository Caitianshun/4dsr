# CEM area4 backbone diagnosis

This experiment migrates the project's fixed area4 supervision chain to the
author's CEM-4DGS representation. It is separate from the original bicubic-LR
reconstruction benchmark. It does not claim that CEM originally proposed SR.

## Fixed scientific scope

- Cook and Cut, frames 0..299, cam00 held out. Cook has 20 training cameras;
  Cut has 19 and excludes cam04. HR is 1344x1008, area4 LR is 336x252.
- Native LR parent: fresh native frame-zero sparse COLMAP from training LR,
  original CEM stages 0->40000->total50000, original optimizer and topology.
- Six SR arms: areaL0, naive, areaT, areaTm, areaTm_w4, areaOracle. All restore
  the same scene parent and all eleven RAdam parameter groups, then execute
  exactly 6000 updates on the existing frozen three-camera schedule.
- Independent native HRref: the same LR-only initializer, original CEM stages
  from scratch using training HR. It is a privileged fitting diagnostic.
- Every endpoint is fixed in advance. No held-out score chooses checkpoints.

The SR continuation uses absolute iterations 50001..56000. Original CEM
densification has ended at 30k and CEC correction has ended before 50k.
Neither is restarted. Consequently this is a fixed-topology continuation.
The existing Wu SR chain re-densifies during its 6000 updates; Multi's teacher
is active throughout its 20k fine stage. Cross-system scores do not by
themselves isolate a representation-only causal effect.

## Execution

`initialize_cem.py` prepares and runs CPU-only native sparse initialization;
`collect_initialization.py` waits for its terminal event and returns assets.
Initialization failures are preserved in separate attempts.

`prepare_assets.py` and `verify_assets.py` create closed input inventories and
rehash all input bytes. Missing byte-identical HR may be hardlinked from the
existing original scene. Conflicting destinations are preserved and rejected.
Teachers and training HR are unavailable to the native LR parent's reader.

`build_native_job_v3.py` freezes stage1, stage2, rendering, and native indexing
and validates the broker's lowercase job-ID format. Earlier builder versions
remain as immutable provenance for the already launched parent jobs.

The isolated five-iteration HR fixture exposed a native early-save problem:
`save_ply` cannot flatten the still-uninitialized motion array at iteration5.
Its HR forward/backward startup and Torch peak are recorded, but it is a
failed serialization fixture, not a successful native endpoint.
`recover_hr_startup.py` proves that no full native stage started, preserves the
failure, and freezes a fresh HR run without that artificial early-save
fixture. It preserves the original 40k/50k schedule and native source.
Startup memory cannot establish mature-model memory or convergence.

`stage_job.py` copies only absent frozen source files and verifies remote
hashes; it never overwrites conflicting sources. `submit_remote.py` uses the
existing GPU broker, records exact-once submission intent, waits on the remote
process exit event, and returns the declared output directories. `--reattach`
only collects the same job; it never submits it again. `run_job.py` executes
the declared commands serially and stops on failure. No GPU eviction or
automatic training replay is implemented.

## Evaluation and interpretation

The shared A100 evaluator consumes all 300 cam00 RGB PNGs. Quantization is
round-to-nearest after clipping to [0,1]. Report mean per-frame PSNR, SSIM and
AlexNet LPIPS. No additional project culling, test-camera affine, or warp is
used. Wu checkpoints are independently re-rendered without posthoc culling.
CTS HRref rendering must be transferred for the same A100 evaluator.

Cam01 frames 0,150,299 provide training-fit diagnostics. SR rendering saves
both raw and observation-corrected training images. Whole images and fixed
face regions accompany the metrics. An oracle continuation that remains
blurred is insufficient to prove a backbone ceiling: examine the independent
native HRref, training fit, and native topology/optimization differences.

Generated inventories, job requests, runtime snapshots, data, checkpoints,
metrics and reports remain private research assets. Only the source and this
usage description belong in the source-only public publication.
