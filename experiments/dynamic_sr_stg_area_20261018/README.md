# STG backbone validation under the registered area4 protocol

This adapter tests whether a different dynamic reconstruction pipeline can use
the same low-resolution observations and frozen SR teacher as the current Wu
4DGaussians and Multi4D comparisons. It does not replace the main pipeline.

The upstream is OPPO's [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians),
commit `427abfc58309a4a5213843dd673fb22c4529306c`. Use the **full** model,
native 50-frame duration, batch of two distinct cameras at one timestamp,
original optimizer schedules, temporal representation, and topology controller.
The full representation uses per-Gaussian motion, time-dependent opacity, and
nine feature channels decoded to RGB. Its results are not interchangeable with
the lite model.

## What is controlled

The registered Cut scene has 300 frames, cam00 held out, and 19 training cameras
(cam04 is absent). HR is 1344 by 1008; area4 LR is 336 by 252. The original
manifest timestamps remain frame/300. Each STG segment uses local time
(frame minus segment start)/50 at the model interface only.

| Arm | Training information | Role |
| --- | --- | --- |
| areaL0 | Training-camera area LR | No-teacher control |
| areaTm | Same LR and frozen area-matched SwinIR teacher | Legal SR pipeline |
| areaOracle | Same LR and training-camera HR | Diagnostic only |

HR renders pass through a differentiable 4 by 4 average before LR loss.
The teacher term is weighted 0.2, using 0.8 L1 and 0.2 times one minus SSIM.
The same native sampling sequence is verified by SHA256 across arms.
Test-camera pixels never enter initialization or training supervision.

Initialization reuses Multi4D's registered, training-LR-only frame-zero
triangulation. All segment seeds use those points. This differs from official
STG's per-frame SfM initialization and is an explicit adaptation. A weak result
cannot alone rule out the STG representation or reproduce its published scores.

The original full 300-frame plan uses six independent STG models per arm, whereas Wu
and Multi4D use one 300-frame model. The amended screening below retains two
50-frame windows. Report total parameters, point counts, model bytes, training
time, hardware and every segment boundary present in the selected range.
This is a comparison of complete adapted pipelines, not equal-capacity models.

## Deployment and execution

`build_adapter.py` verifies pinned upstream file hashes and generates a separate
patch. Original sources are retained. Changes adapt RGB losses and EMS error
selection to area LR, repeat LR residuals to the HR grid for native depth-driven
point insertion, make an unused MMCV import lazy, and record execution identity.
The original second-camera densification statistic is preserved and disclosed.

`deploy_runtime.py` builds ch9 and simple-knn in an isolated dependency target.
The existing Python environment is read only. For CUDA 12.8, simple-knn needs
the standard `cfloat` include for `FLT_MAX`; the deployment receipt records that
runtime-only patch. CPU import checks do not certify CUDA execution.

`stage_inputs.py` produces an explicit input inventory and SHA256 verification
receipt. Source manifests are never rewritten. Existing destination files are
not overwritten and mismatched bytes fail verification.

`prepare_plan.py` freezes executable source and stage-spec identities.
`run_pipeline.py` must run inside the existing registered remote GPU broker,
with a verified GPU UUID and input receipt. It launches one CUDA child at a
time. Successful training exits directly trigger evaluation; failure stops the
pipeline. It never clears another task's lock or replays failed training.

`run_validation.py` can run the registered smoke and formal plans sequentially
inside one broker job after the full input receipt passes. It checks the three
native checkpoint reloads and paired sampling before formal training. The GPU
lock remains held across that transition. Waiting for data or another task to
release a GPU belongs outside this job. A resource admission rejection before
the argv starts is recorded separately from a failed CUDA training run.

The initial engineering run uses 40 forward/backward batches per arm and checks
native PLY/decoder save-load render parity. Formal evaluation uses the original
25, 000 pre-update checkpoint, representing 24,999 optimizer updates. The native
learning-rate schedule remains 30, 000 steps; subsequent updates cannot affect
that already-saved checkpoint and are omitted. PLY plus decoder is a rendering
checkpoint. It cannot resume optimizer, RNG and EMS state along the same path.

Each stage specification declares `manifest`, `init`, `teacher_index`,
`upstream`, `adapter`, `python_dependencies`, `native_config`, `method`,
`segment_start`, `duration`, `seed`, `iterations`, and teacher weights.
All paths and byte identities are recorded in local execution receipts.

## Evaluation and stopping

`evaluate.py` uses the same model-independent spatial metrics as the current
Wu/Multi4D evaluator: PSNR, SSIM, AlexNet LPIPS, frequency bands, fixed temporal
variation masks, and fixed region crops. Training views are diagnostics only.
`compare.py` requires all six completed segments per arm for a 300-frame result.
It joins 294 within-segment and five cross-segment temporal pairs. Missing
outputs remain null. It supports explicit remote-to-local path mapping without
rewriting the original result files.

Face/text crops inherit preregistered coordinates. They are not semantic masks,
and a moving face can leave a fixed crop. Human blinded acceptance remains a
separate step. No metric or automatically generated crop certifies acceptance.

The user amended the scope on 2026-10-19 after inspecting development results.
Retain Cut frames 0–49 for all three arms, and 50–99 for areaL0/areaTm only. The
remaining 13 Cut stages and Cook training are deferred. The old absolute-difference
Cook gate is historical and must not launch extra training. A separate dated
decision is required for one fixed Cook teacher confirmation, supported by a
favorable quality or regional/temporal tradeoff. This is development screening,
not a full 300-frame benchmark or an independent generalization test.

`evaluate_wu.py` renders two already-trained Wu checkpoints for cam00 frames 0–99
without model updates. Its frozen registration binds the native checkpoint,
original training-camera visibility cull, complete manifest, fixed mask/ROIs,
metric sources and offline LPIPS weights. CPU-only `--preflight-only` checks do
not import torch. Actual rendering requires the registered remote broker and
exact GPU UUID. It checks old/new frame0 floating renders and per-frame global
metrics before accepting new regional and temporal results. It never applies
test-camera color or spatial corrections, silently retries, or overwrites an
existing evaluation. Registrations, data, checkpoints and caches remain local
assets and are not distributed here.

`screening.py` is CPU-only. Explicit Wu input mappings and remote path mappings
join existing results for 0–49, 50–99 and0–99, including fixed face/dynamic/static
metrics and all 99 temporal pairs. The 49→50 model-switch boundary is separate.
All model, sampling, source, mask, ROI and frame-coverage identities must pass;
missing inputs retain `incomplete` with null comparisons. A complete selected
range is named `completed_screening`, never full 300-frame. It supports the CLI options
`--manifest`, `--stg-root`, `--multi4d-root`, `--wu-evaluations`,
`--path-map OLD=NEW` and a new `--out` path. Frozen registrations and the original
technical exit status remain in local deployment receipts.

`watch_events.py` and `collect_events.py` transfer stage artifacts on inotify
completion events. The registered hourly monitor is an exception fallback;
local completion callbacks do not themselves imply a chat notification.

Upstream third-party licenses continue to apply. This repository contains the
adapter and deployment code, not datasets, weights, upstream source snapshots,
compiled libraries, or generated experiment results.
