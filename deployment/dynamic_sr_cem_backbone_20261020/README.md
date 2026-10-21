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

`build_hr_job.py` applies that recorded startup policy to the second scene,
using a fresh HR output directory. It does not change the native schedule or
native model code. HR references run serially on CTS and retain their failed
startup attempt as separate evidence.

`stage_job.py` copies only absent frozen source files and verifies remote
hashes; it never overwrites conflicting sources. `submit_remote.py` uses the
existing GPU broker, records exact-once submission intent, waits on the remote
process exit event, and returns the declared output directories. `--reattach`
only collects the same job; it never submits it again. `run_job.py` executes
the declared commands serially and stops on failure. No GPU eviction or
automatic training replay is implemented.

`run_sequence.py --plan <root-relative-plan.json>` runs a finite sequence of
explicit commands and JSON completion dependencies. It uses file events,
stops at the first failed dependency, and never chooses an experiment, GPU,
or retry on its own. The plan, source hash, step logs and terminal state are
saved in a new sequence directory.

`collect_parent_essential.py` returns the final native checkpoint and its
identity metadata after successful remote completion. This allows the
three-step SR wiring checks to start while the independent full collector
returns all images and intermediate checkpoints. Essential collection is
not completion of full artifact return. The six scientific continuations
require review of real mature-parent wiring and memory evidence first.

## Evaluation and interpretation

The shared A100 evaluator consumes all 300 cam00 RGB PNGs. Quantization is
round-to-nearest after clipping to [0,1]. Report mean per-frame PSNR, SSIM and
AlexNet LPIPS. No additional project culling, test-camera affine, or warp is
used. Wu checkpoints are independently re-rendered without posthoc culling.
CTS HRref rendering must be transferred for the same A100 evaluator.

After the complete CTS run has returned, `returned_image_assets.py --scene
<scene> --stage` validates the 300 images and their native provenance, copies
only missing image/evaluation assets to A100, and returns a full-SHA receipt.
Use the explicit `native_HRref_v2` run when building its evaluation job. The
image transfer does not require or load the checkpoint. Full checkpoint return
is still required from the original native job collector.

Cam01 frames 0,150,299 provide training-fit diagnostics. SR rendering saves
both raw and observation-corrected training images. Whole images and fixed
face regions accompany the metrics. An oracle continuation that remains
blurred is insufficient to prove a backbone ceiling: examine the independent
native HRref, training fit, and native topology/optimization differences.

`diagnostics_images.py` consumes completed endpoints on CPU. It checks the
predeclared frames and face regions, preserves raw versus corrected training
renders, and can add the six complete Wu references. Its default gate rejects
missing endpoints. `--allow-incomplete` produces an explicitly partial preview
without a scientific completion gate. The optional temporal diagnostic is
ground-truth-relative consecutive-image difference error; it is not optical
flow compensation or a stand-alone perceptual stability verdict.

`summarize_results.py --output <new-directory>` requires all 28 endpoints:
eight CEM endpoints and six frozen Wu references for each scene. It rechecks
the common metric identities, all 300 PNGs and per-frame CSVs, then writes
the source values, paired differences, resources and interpretation limits.
Only a complete matrix receives a completeness marker. The optional partial
mode is an explicitly incomplete research record. This marker does not
constitute human visual acceptance or proof of a backbone capacity limit.

Generated inventories, job requests, runtime snapshots, data, checkpoints,
metrics and reports remain private research assets. Only the source and this
usage description belong in the source-only public publication.
