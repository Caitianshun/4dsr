# Six-scene static SR adaptation on CTS

This deployment prepares and evaluates SplatSuRe, IE-SRGS, Sequence Matters and
3DSR on the shared basic Wu 4DGaussians backend. These are disclosed dynamic
adaptations, not unmodified reproductions of the original static backbones.
See `experiments/static_sr_wu_20261022/README.md` for method-specific differences.

The fixed suite contains Cook07, Cut07, Coffee07, Flame07, Discussion07 and
VRHeadset07. Every scene uses its own 300 frames, area4 input, original
`frame / 300` timestamps, and training cameras. Camera 00 is reserved for final
evaluation. Scene-specific priors and optimization state are never transferred
between scenes.

## Execution and storage

The controller completes one scene at a time. `plan_scene.py` requires real GPU
wiring, teacher and metric acceptance receipts, full training-input identities,
and an explicit per-stage disk budget before it emits a runnable job. A missing
budget or acceptance receipt produces a blocked plan, not training permission.
Discussion07 is the initial planning scene because its raw prior volume is
smaller; all six scenes remain required.

The static teacher bundles must be complete before the three final static SR
branches run. LR reconstruction and IE internal-teacher training are prerequisite
stages. 3DSR retains four alternating teacher/reconstruction rounds, since later
teacher targets depend on the preceding trained reconstruction and cannot be
generated entirely in advance.

`run_job.py` executes stages serially under the existing resource admission and
project/GPU locks. The training entry points immediately evaluate their final
models. `submit_remote.py` waits for the remote supervisor's exit event, returns
files and directories, and fails if the job failed or declared outputs are absent.
Collecting artifacts alone does not certify their local SHA256 identities.

The user authorized retaining final metrics and compact provenance only. After
all four methods have complete 300-frame metrics and a successful terminal job,
`cleanup_scene.py verify-return` verifies the local compact package and remote
supervisor receipt. On CTS, its default dry run freezes the exact eligible
files. Only an explicit `--apply` with those receipts deletes the generated
scene/run caches, checkpoints, renders and latents. Original inputs, shared
pretrained weights, other runs, symlinks and hardlinks are excluded. Failed or
incomplete scenes are not eligible. Intermediate artifacts cannot be recovered
without rerunning the corresponding preparation/training stages.

## Entry points

- `asset_inventory.py` and `transport_inventory.py` create frozen transfer
  inventories. `sync_assets.py` performs additive, SHA256-verified transfer into
  an isolated task snapshot. It preserves Git-tracked upstream content and
  uninitialized submodule directory identities.
- `prepare_cpu_suite.py` registers all six scenes and generates CPU-only camera
  order plans. It writes a terminal success/failure receipt.
- `build_extensions_cpu.py`, together with the two experiment build helpers,
  compiles the pinned extensions on the target host. `deploy_extensions_cpu.py`
  preserves the old task binaries and validates the production loaders after
  switching. These steps do not run CUDA kernels.
- `bootstrap_remote.py` relocates the task-specific Python overlays and offline
  model caches without modifying the shared Python environment.
- `continue_acceptance.py` follows CPU-plan and transfer exit events, then runs
  the fixed teacher acceptance jobs. It never launches formal scene training or
  automatically replays a failed job.
- `plan_scene.py` constructs the formal scene pipeline and collects compact
  results. Its disk guards require measured available space before each stage.
- `cleanup_scene.py` validates returned results, prepares a closed deletion
  manifest, and applies cleanup only after all completion and identity checks.

Host-specific jobs, transfer inventories, runtime binaries, model weights,
generated priors, experiment results, and report documents are separate local
assets. Their presence and acceptance status must be checked on the execution
host; cloning the public source alone does not provide a runnable experiment.
The deployment fixes the authorized CTS host and snapshot paths in its source;
another installation must explicitly adapt these host constants and repeat
the corresponding environment and resource checks.

## Verification scope

CPU contract tests, target compilation/imports, bounded GPU wiring, actual
teacher inference, and full-scene convergence are different acceptance stages.
The wiring test deliberately uses marked, non-formal supervision fixtures to
exercise losses, gradients, optimizer updates and checkpoint restoration.
These fixtures are rejected by formal training. A short test cannot establish
method quality or replace the final PSNR, SSIM and LPIPS comparison.

Long-task hourly checks are an abnormal-state fallback. Normal evaluation and
artifact return follow process completion events. A local callback is not proof
that an application notification was sent.
