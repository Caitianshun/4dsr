# Training-camera visibility culling retrain (2026-10-09)

Lightweight entry for the next step after the 2026-10-09 zero-training checks: retrain the LR stage of the
development parents with in-training culling of Gaussians that no training camera sees (M1), with and without a
per-training-camera colour affine (M8), continue with the Bsync SR stage under the same culling, and evaluate
cam00 on all 300 frames.

- `cullretrain.py lr` replays the author fine stage (Wu 4DGaussians 843d5ac, as registered in
  `dynamic_sr_multiview_footprint_20261007/full_prefix_registered.py`) from a parent's coarse-end checkpoint
  (model, Adam, sampler and RNG), optionally culling every 1000 steps during densification and at the end,
  removing the 200k-point gate on the author low-opacity prune in the fine stage, and applying the colour affine.
  `--coarse-only` replays the unchanged author coarse stage when the coarse-end checkpoint is unavailable.
- `cullretrain.py sr` runs the registered Bsync SR stage (LR closure on view a, frozen teacher L1 on views b and c,
  `--lambda-sr`, optional `--dssim`), culling at the start and every 1000 steps.
- `cullretrain.py eval` evaluates cam00 on all 300 frames with the official legacy metric functions, optionally
  after post-hoc culling of the loaded model.
- `pipeline.py` runs a task graph as concurrent subprocesses on one GPU; several pipelines can share one task list
  through claim files (`pipeline_owned.py` runs a pre-claimed share of a list). `tasks.py` builds the validation,
  wave, control and confirmation task lists.
- Analysis: `summarize.py` and `report_tables.py` (tables from result files), `figures.py` (report figures),
  `artifact_check.py` (held-out-view artifact area from saved previews), `sh_degree_check.py` (render-time SH degree
  truncation, no updates), `affine_traindiag.py` (training-camera diagnostic with the learned colour affine applied).

Culling criterion: a Gaussian is removed when no training camera has, at any of 10 sampled times, a 3-sigma
footprint intersecting its image at a depth greater than 0.25 times that camera's near bound (5th percentile depth
of the LR-triangulated initial points). The held-out camera is never used.

The runtime loads the baseline 4DGaussians code and compiled rasterizer from a private byte-identical copy
(`third_party/4dgs_843d5ac_runtime`, not published). Data, parents, teachers and outputs are local assets and are
not part of this repository.

Results of the 2026-10-09 runs are summarised in the project report `docs/dynamic_sr_cull_retrain_2026-10-09.md`
(local, not published): culling is required to remove failures; culling during training without the colour affine
hurts held-out colour; with the affine, post-training and in-training culling give similar means; D-SSIM 0.2 improves
PSNR, SSIM and LPIPS consistently.
