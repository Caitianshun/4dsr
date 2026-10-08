# 4dsr: general dynamic scene super-resolution code

This repository publishes the **code and necessary text configuration** for the general dynamic scene super-resolution (SR) research in this project. It contains no datasets, pretrained weights, checkpoints, generated results, or HOI experiments.

## Code map

| Path | Role |
| --- | --- |
| `experiments/dynamic_sr_20260918`–`dynamic_sr_20260921` | N3DV and MeetRoom preparation, baseline training/evaluation, frozen SR prior, and controlled diagnostics |
| `experiments/dynamic_sr_20260923` | Temporal sharing controls and associated evaluations |
| `experiments/dynamic_sr_motion_bound_20260923` | Spatial splitting and parent-motion experiments |
| `experiments/dynamic_sr_scene_residual_20260923` | Scene residual experiment |
| `experiments/dynamic_sr_soft_motion_20260924` | Soft motion constraint experiment |
| `experiments/dynamic_sr_multi4d_20260924` | Pinned Multi4D LR/SR adapters, full-state validation, and extended Wu controls |
| `experiments/dynamic_sr_controlled_headroom_20260926` | Fixed-topology Z/U/O teacher-removal and privileged training-HR supervision controls |
| `experiments/dynamic_sr_confidence_geometry_20261006` | Registered observation permutations, image-nullspace SR supervision, LR-only depth ordering, and finite cause probes |
| `experiments/dynamic_sr_multiview_footprint_20261007` | Same-time LR multiview controls, differentiable HR footprint reconstruction, paired continuation and event-driven evaluation |
| `experiments/sr4d_20260922` | SR4D comparison adapters and evaluation |
| `experiments/dynamic_sr_surface_20260923` | Archived non-HOI ZJU single-person exploration; **not** evidence for the current full-scene route |
| `deployment/` | Host-specific setup and run scripts; paths and device identifiers describe the original machines |

The older `boundary_*`, `roi_*`, `sr_six_view_*`, `src/hoi_*`, and related code used HOI data or human/object-specific assumptions and is outside this repository.

## External dependencies and scope

The main scripts import Wu et al.'s 4DGaussians from a separate checkout. Set `FOURDSR_UPSTREAM` to that checkout; the original local default path in some scripts is specific to the development machine. The research workspace used upstream commit `843d5ac636c37e4b611242287754f3d4ed150144` with local compatibility changes, so this repository alone is **not** a turnkey reproduction of the original runs. Follow the upstream project's license and build instructions for its CUDA extensions.

Frozen single-image SR priors require the separately obtained SwinIR network and weights. The preparation scripts also used `timm==1.0.15`; its local installed copy is deliberately omitted. Other Python and CUDA dependencies are described by the deployment requirements and upstream projects. Supply the public N3DV/MeetRoom data, camera calibration, prepared manifests, and weights separately according to their own terms. No such assets are stored here.

The SR4D comparison uses an external author implementation at commit `c6466663743382d48c5f23701568113eef9e961b`; `deployment/sr4d_20260922/general_sr_source.patch` records the non-HOI code changes. The original local patch also changed the author's README to point to an old HOI experiment; that documentation hunk is deliberately omitted here. The full author-source snapshot and binary deployment archive are omitted, so the archived deployment script that expects those files needs the original local archive to run unchanged.

This is research code, including historical pilots and negative controls. The current full-scene work uses N3DV and MeetRoom. A script name or inclusion here does not imply that its method was accepted as the final approach or that a fresh clone will reproduce a paper result without the corresponding external assets and protocol.

## Updates

The publication boundary and update procedure are in [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md). Completed, checked changes to the non-HOI SR code are synced to this repository promptly from the private research workspace.

## Fixed geometry-residual prototype

`experiments/dynamic_sr_geometry_residual_20260924/` adds matched eight-term Legendre or cubic B-spline world-center residuals to previously split children after the shared 4DGaussians deformation. The coefficients share the original world-position learning-rate schedule and use the unchanged full-camera U supervision. It depends on the preceding detail-supervision and motion-refinement adapters, externally installed 4DGaussians/CUDA extensions, and private prepared manifests, frozen teacher caches and LR-parent checkpoints. This is a controlled short-window development prototype, not a full reproduction of Gaussian-Flow or SplineGS, and the source alone does not establish a quality gain.

## Multi4D and longer Wu validation

The Multi4D adapter requires a separate checkout of [the author repository](https://github.com/BatFaceWayne/Multi4D) at `c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0`, its DyNeRF configuration, and independently compiled native CUDA extensions. `prepare_upstream.py` applies explicit compiler headers and confirmed hybrid-rasterizer fixes (an unwritten backward cutoff statistic and empty transient-branch handling). `multi_data.py` also replaces an undefined singleton KNN scale with the scale of the corresponding persistent seed. These changes are disclosed adaptations; this is not an unchanged author implementation.

`build_adapter.py` generates the training adapter from the pinned source. It retains the native three-branch model, controls and losses while projecting HR renders onto observed LR targets. The optional teacher term uses frozen SR images from the same two training samples. The scripts prepare a legal frame-zero LR cloud, save/reload complete three-branch states, check gradients and phase transitions, and evaluate fixed endpoints. The Wu continuation retains the original learning-rate schedule and exact existing training prefix. Full inference-state archives include all branches, networks and fixed buffers, separately from Adam/training checkpoints; they are explicit-schema archives, not upstream PLY exports or a compressed codec.

Use an isolated environment for Multi4D, set `MULTI4D_UPSTREAM`, and supply manifests, teacher inventories, checkpoints and run specifications separately. Local service/GPU paths in the orchestration scripts describe the original workspace and must be configured for another host. Neither training launch nor engineering preflight is evidence of an SR quality gain; final comparisons require fixed-view metrics, visual review and cost analysis.

## Controlled teacher increment and supervision headroom

`experiments/dynamic_sr_controlled_headroom_20260926/` holds a fixed Wu split topology, paired LR/SR camera streams and the original continuation schedule constant. Z retains the second render and graph-connected zero backward without reading SR/HR targets. U uses the existing frozen SwinIR target with weight 0.1. O substitutes only the legal training-camera HR target at the same weight and is always labeled **privileged**; it is a diagnostic, not an LR-only method or a guaranteed upper bound. Shared point selection previously used teacher gradients, so Z removes subsequent teacher loss rather than all historical teacher information.

The entrypoints save/restore complete optimizer, sampler and RNG state, evaluate registered update endpoints, add fixed-training-view teacher RGB errors, and use process-exit events for result return and uniform evaluation. Supply the original manifests, selection/checkpoints, frozen sampling/LR records and target inventories separately. `prepare.py` validates the common start, `train.py` accepts the fixed intervention, and orchestration scripts consume workspace-specific JSON specifications. These private run assets and result artifacts are intentionally excluded. Original metric/video helpers and external Wu dependencies remain required; this code does not claim a successful scientific result.

## Paired continuation with fixed structure

`experiments/dynamic_sr_view_recovery_20260926/` audits endpoint error changes, effective Gaussian states and training-only sparse support, then compares joint continuation against an appearance-only whitelist from an existing split-topology checkpoint. The whitelist retains base/child SH and the independent time-dependent SH head while freezing geometry, opacity, shared features, fixed buffers and their inherited Adam state. It preserves the forward configuration, sampler suffix, both optimizers and the original learning-rate curve. Stored motion remains active over time.

The module includes zero-step, freeze and short save/resume checks, separate records of CUDA replay variation, fixed endpoint gates, and process-exit evaluation/visualization. It reuses the accepted floating-point metric implementation. Workspace-specific manifests, original teacher inventories, complete checkpoints, historical evaluation assets and registered protocols must be supplied separately; the public source does not contain them or establish a quality improvement. Run paths and the physical GPU identity are local execution settings, not portable defaults for another machine.


## Source-specific prior gradient controls

`experiments/dynamic_sr_prior_guidance_20260927/` keeps the original LR and regularization gradients while controlling the SR-only pathway. It compares a full joint continuation, final-center detachment from an early endpoint, final-center detachment in a shared late tail, and an SR-only appearance whitelist. Base and child posed centers are concatenated before detachment; covariance, opacity and SH stay connected. Shared parameters can still change future positions indirectly.

The experiment reuses full model/Adam/RNG/sampler checkpoints and a fixed shared prefix, with exit-triggered evaluation, a predeclared early safety gate and fixed final endpoints. Auxiliary scripts audit independent one-step gradients, unclamped alpha/depth moments, grayscale operator consistency and a fixed Depth Anything V2 Small prior. Depth inference requires the separately obtained official source and checkpoint; it is diagnostic and is not added to the RGB training objective. Masks and HR-supervised reference comparisons remain privileged diagnostics. No private manuscript, dataset, model weight or generated result is included. Local protocol and history files referenced by the controllers must be provided or explicitly adapted; these research entry points are not a turnkey benchmark.


### Bounded evidence repair and fixed-time fitting probes

`experiments/dynamic_sr_evidence_repair_20260927/` contains explicit legacy-import
pinning, named-gradient/Adam repeat comparisons, continuous-versus-restart controls,
train-LR three-view track validation, raw depth-moment audits, and two fixed-time
fitting probes. `register_protocol.py` registers the fixed thresholds and input
identities before measurements. `run_replay.py` and `run_remaining.py` chain
independent evaluation to successful training exits; `finish_receipts.py` uses a
Linux process-exit event to complete diagnostics and machine-readable receipts.

The Shared40 and Baked40 probes use the same posed point state and camera sequence,
with new zero-state Adam optimizers. They are not equal-capacity dynamic methods.
`evaluate_probes.py` and `audit_roi.py` preserve raw teacher, signed residual, HR,
and LR-reprojection metrics, and distinguish raster candidates from composited
group contributions. `audit_depth_evidence.py` requires third-view geometry,
nonnegative moment validity, actual pixel-weight coverage and sampler exposure;
a failing gate does not trigger depth training. Numerical repeat envelopes are
kept fixed even when a comparison fails.

These scripts require the already prepared generic-scene assets, upstream
4DGaussians environment, frozen teachers and complete historical checkpoints.
Paths and hashes are resolved in the local protocol. The public repository does
not contain datasets, weights, checkpoints or generated experiment results, and
cloning it alone does not reproduce these studies. Source snapshots and historical
renderer versions must remain distinct when comparing runs.


### Fixed-footprint two-teacher probe

`experiments/dynamic_sr_conflict_probe_20260927/` registers a training-LR-only
anchor pair and shared camera sequence, bakes one posed state, and compares four
600-step SH-only objectives with exactly three forwards per step. Geometry and
opacity stay fixed. Disposable copies of actual Adam updates provide isolated
and conditional attribute-transfer diagnostics; native auxiliary rendering
measures footprint-bucket alpha contributions without changing the renderer.

The controller attaches evaluation to process completion, freezes the train-only
decision before loading HR/development images, and admits at most one complete
four-arm repeat under fixed gates. `summarize.py` checks logs, checkpoints, frozen
attributes, data identities and information boundaries. A diagnostic repair can
reuse saved Adam candidates without spending additional optimizer updates.
Prepared project-specific protocols, upstream extensions, checkpoints, teachers
and track files must be supplied separately; the public source is not a turnkey
benchmark and makes no claim of a positive research result.


### Fixed-time covariance adaptation probe

`experiments/dynamic_sr_covariance_probe_20260927/` loads the indexed AB600
baked endpoint and compares 600 additional SH-only updates against SH plus
scale/rotation updates. Both arms start with empty Adam state, use identical
LR sampling, and preserve centers, opacity, point order and the native renderer.
LR-defined person/interaction ROIs are evaluation-only. Full-image residuals
are formed before cropping; HR evaluation follows a frozen training decision.

The Stage A controller conditionally admits one paired repeat and records
whether dynamic validation is eligible. It does not automatically implement
or launch Stage B; that requires a separately verified dynamic routing entry
if all Stage A gates pass. The exercised Stage A failure path ends the branch.
The engineering fixture verifies complete parameter/Adam/RNG/sampler restoration
against the project's established CUDA replay envelope. Endpoint footprint
measurements use actual native alpha contributions and explicit checkpoint
identities. Private prepared protocols, teacher data, parent checkpoints and
upstream extensions are required. Results, plots and reports stay outside this
public source repository.


### Bounded dynamic SR routing validation

`experiments/dynamic_sr_dynamic_validation_20260927/` contains the three-arm
`J_joint` / `A_sh` / `S_cov` continuation, explicit SR-gradient routing,
registered three-RNG suffix schedules, protocol-driven decisions, and an exact
19-camera by four-frame evaluator. The original data adapters, two Adam states,
renderer, and recorded learning-rate curve are retained. Full dynamic training
requires a current engineering acceptance receipt tied to the source, protocol,
and U6000 parent. A failed or absent receipt prevents launch.

The current execution stopped at its 32-round engineering budget: an ordinary
CUDA replay exceeded the registered absolute rendering error bound. No formal
three-arm endpoint or method-quality conclusion was produced. The prior static
failure remains unchanged. The supplementary replay controls retain the original
failure and numerical limits; they do not authorize retraining or threshold search.

The repository contains source only. The parent checkpoint, frozen teacher cache,
input manifest, registered protocols, and evaluation assets must be supplied
separately; the controller is specific to the documented local/A100 deployment.
A complete success-path orchestration was not executed after the failed gate.

### Paired SR attribute routing experiment

`experiments/dynamic_sr_attribute_routing_20260928/` provides an independent
`SRAttributeRouter` facade over the previously audited derivative arithmetic.
The revised experiment registers both J/A/S suffix sets unconditionally, keeps
the old numerical replay failure as a diagnostic warning, and requires a
separate research-readiness receipt for state, data, and routing correctness.
Full-image PSNR, SSIM, and LPIPS are all primary metrics; no auxiliary quality
threshold controls whether the second set executes.

Immutable attempts support reuse of a verified SR12000 endpoint or recovery
from SR9000, with separate effective and actual update budgets. Persistent
services use process-exit events for checkpoint return and uniform evaluation.
The module adds no learnable parameters or inference operations. Two suffixes
sharing U6000 are paired continuations, not independent from-scratch trials.
As with the earlier experiment, all checkpoint, input, teacher, protocol, and
evaluation assets are external to this source-only repository.

### Frozen temporal prior experiment

`experiments/dynamic_sr_temporal_prior_20260928/` replaces the offline SR target
while retaining the complete joint 4D Gaussian continuation and both Adam states.
The official BasicVSR++ REDS BI x4 c64n7 checkpoint is used with seven registered
training frames, comparing real neighbors against seven copies of the center.
Targets are quantized RGB PNGs with per-observation source and generator hashes.
The official model and MMCV deformable alignment are external dependencies;
the isolated inference environment uses PyTorch 1.12.0 cu113 and MMCV-full 1.6.0.
The wrapper resets the mirror flag and strictly loads the generator after
excluding only the restorer's scalar `step_counter` buffer.

Four paired suffix tasks use the existing U6000 state, fixed sampling and
learning-rate schedules, and one physical training GPU. Native LR and HR
checkpoints are evaluated separately on the common HR output grid; the
native-LR bicubic reference upsamples a floating-point model render.
Process-exit listeners return caches and checkpoints and start uniform
evaluation. Historical checkpoints, model downloads, prepared images,
teacher caches and experimental results are not distributed here.

### Image-prior diagnosis and selective SR supervision

`experiments/dynamic_sr_prior_diagnosis_20260929/` audits native LR, enlarged LR,
and HR image priors, additive orthonormal DCT/RGB error budgets, offline oracle
counterfactuals, native output sampling, and Gaussian ray moments. Diagnostic
HR inputs remain separate from the training input allowlist. Different Gaussian
topologies are compared through contributions and common spatial statistics,
not Gaussian indices.

The bounded training experiment transfers the view-dependent demand-selection
mechanism from [SplatSuRe](https://github.com/pranav-asthana/SplatSuRe), upstream
commit `df40abb8d92a02656268cb0a0b97acc2495b9307`, to same-time dynamic views.
This is a mechanism adaptation, not a reproduction of its complete static
pipeline. Demand maps are frozen from U6000 and use the existing SwinIR targets;
the project retains its joint two-Adam update and LR loss. A fixed calibration
matches the initial geometric SR-gradient scale using legal training inputs.
The source also contains diagnostic sampling utilities; the executed protocol
selects T only and does not enable supersampling or depth/flow supervision.

Two registered suffixes run concurrently on distinct A100 GPUs, with process-exit
checkpoint return and evaluation on one RTX 3090. Prepared protocols, upstream
4DGaussians/SplatSuRe code, DAv2/RAFT weights, legal images, frozen targets, and
complete parent states are external assets. The scripts assume the documented
research deployment; cloning this repository alone cannot reproduce the run.
The manuscript-facing reports and all generated results remain local.

### Same-observation supervision from initialization

`experiments/dynamic_sr_same_observation_20260930/` registers one independent
4DGaussians training run from the original training-LR point cloud, without
loading a reconstruction checkpoint. From the first coarse update onward,
each step renders one HR image and jointly minimizes its downsampled error to
the corresponding real LR image and its error to the corresponding frozen
SwinIR target, with SR weight 0.1. One combined backward and one Adam update
follow. The original coarse/fine deformation policy and warmup densification
schedule are retained; both stages use LR and SR supervision.

The fixed budget is 1,000 coarse plus 19,200 fine updates. Registration validates
the 19 training cameras, 60 frames and all cached targets. Training returns
complete checkpoints and legal-input audits, and the deployment controller
connects remote process completion to local evaluation. Prepared manifests,
LR point clouds, teacher caches, CUDA extensions and deployment environments
are external dependencies. Historical U6000 continuations differ in capacity,
initialization history and SR exposure, so equal update counts alone do not
establish a controlled improvement from pairing.

The optional `frequency.py` diagnostic partitions render-minus-HR residuals
into radial FFT bands at 0.125 and 0.25 cycles per HR pixel. It reports
additive band MSE, mean per-frame band PSNR and pooled error-energy shares,
and validates numerical compatibility against a separately supplied reference
script. Main comparisons retain floating renders before PNG quantization.
`frequency_operator.py` checks the actual bicubic operator: a resize residual
is not automatically in its nullspace. HR-dependent statistics are diagnostic
only, and cannot serve as LR-only training masks. The local reports discuss
possible constraints without implementing or claiming a new trained method.

### Frozen confidence and soft geometry controls

`experiments/dynamic_sr_confidence_geometry_20261006/` compares a same-image
LR/SR continuation with one or two permuted SR observations, retaining the
complete parent model, both Adam states, topology and recorded learning rates.
The R module projects image residuals through the nullspace of the actual
linear antialiased bicubic degradation before applying frozen confidence.
The G module uses relative inverse-depth ordering from legal training LR,
nonnegative area-aggregated ray moments, and gradients through the same
posed centers used by the RGB renderer. Covariance and opacity are detached
only from this auxiliary geometry path.

Offline confidence uses LR closure and valid same-time training-view support;
missing support is explicitly unknown. Conditional time/view confidence caches
require complete registered LR-only flow coverage. Source includes finite
LR-only position/SH interventions, full-state shadow updates, numerical
repeat diagnostics, metric/FFT/region evaluation and process-exit orchestration.
The image nullspace property does not guarantee that a renderer/Adam parameter
update preserves LR output. Relative depth is not metric geometry truth, and
two suffixes from one parent are not independent from-scratch trials.

Prepared manifests, frozen teachers, complete parent checkpoints, registered
run records and official Depth Anything V2 source/weights must be supplied
separately. Deployment scripts describe the original local/A100 machines.
No training quality result or completed full-scene benchmark is implied by
publishing these development controls; datasets, weights, generated metrics
and research reports remain outside this repository.

### Same-time multiview and cross-view pixel footprints

`experiments/dynamic_sr_multiview_footprint_20261007/` defines six paired
controls: the existing complete-SR baseline, same-time grouping, three-view
LR averaging (M), cross-view footprint reconstruction (X), M+X, and a simple
L1/MSE objective control. All controls retain complete frozen SR supervision.
Same-time schedules preserve each registered observation multiset within
the declared block and use three distinct training cameras.

X warps current source HR RGB using current target axial depth, then applies
the original antialiased bicubic degradation and clamp to match real target
LR. Its differentiable RGB mip pyramid approximates minification; detached
levels and complete interpolation/filter support are explicit. Frozen support
aggregates raw opacity-weighted physical moments before depth normalization.
Unknown variance has neutral weight. There is no HR-error or photometric-error
threshold for choosing training views or hiding difficult pixels. RGB follows
normal appearance/geometry gradients, while the auxiliary moment path detaches
covariance and opacity. This is a reconstruction constraint, not verified
metric depth or a pure geometry update.

`run_suite.py` preserves full model, two disjoint Adam states, RNG and registered
learning rates. Its resource checks protect foreign compute processes and bind
physical GPU UUIDs. Independent suffixes can use different machines, but all
arms within one suffix use the same training platform. A separate uniform
evaluator must be registered before deferred evaluation is used. Preparation,
calibration, diagnostics and failed work have separate counters; paired suffixes
from one parent do not constitute independent from-scratch seeds.

The source also contains full-time input preparation and legal training-LR
initialization adapters. Full-time manifests, complete SR caches and independently
trained LR prefixes are separate prerequisites. Dataset splits and project
resolution adaptation must be reported separately. Publishing these tools
does not imply a completed quality comparison or an official full benchmark.

`full_prefix.py` adapts the pinned Wu N3DV coarse3000/fine14000 training loop.
It retains stage optimizer resets, original multiview densification statistics
and the author's final fine iteration without an Adam step. Adaptations include
known-camera LR sparse initialization in place of supplied COLMAP points,
project crop/resolution, and resumable serial sampling in place of prefetched
workers. The initial adapter accepts Cook/Cut at the declared LR resolution;
other datasets require an explicit configuration adaptation. Its native parent
has no historical child optimizer, so full-scene SR refinement needs a separate
adapter rather than the short-window continuation entry.


`full_refine.py` continues the native full-time parent with one eight-group Adam
optimizer and all shared state. It uses three RGB forwards and one backward per
update; same-time controls reuse one deformation state. All methods use the same
SUM gradient, maximum radius and OR visibility statistics for original topology
operations. The declared SR-stage topology clock starts at one, while learning
rate and SH clocks continue the parent. LR pixel-unit screen statistics are reset
once at that boundary; resumed segments retain their saved statistics. These
are explicit protocol adaptations, not an unchanged original training run.

`full_support_prepare.py` requires a selected configuration, a native full parent
and its own full schedule before exporting raw HR parent moments and frozen
support. Short-window support cannot satisfy that identity. `full_evaluate.py`
accepts native full parents/refinements, checks model/optimizer/RNG immutability,
and evaluates all registered test frames from floating-point renders. It records
PSNR, SSIM, LPIPS, fixed available ROI, absolute residual-frequency energies and
test-only LR closure diagnostics. Test LR is read after rendering for evaluation
and does not supervise inference. Code publication and CPU contract checks do
not constitute native CUDA acceptance, full-scene quality results or confirmation
on unseen scenes.

`full_frozen_parent_overlap.py` exposes the frozen native parent's coarse
visibility calculation. `full_native_schedule_materializer.py` uses training
camera poses and those accepted overlap ratios to create balanced simultaneous
three-camera schedules, or to retain the exact path/SHA of an existing schedule.
These portable entries preserve the previously frozen calculation functions;
their source identities are new and require new registrations. They depend on
the registered native prefix/refinement/protocol helpers, the footprint and raw
depth-moment helpers, the pinned Wu author runtime, and separately supplied legal
LR manifests, initialized parents and exact-exit/SHA receipts. Planning uses
Python 3.10 with NumPy 1.26.4 and hides CUDA; actual moment preparation additionally
needs PyTorch, the native CUDA extensions and explicit resource ownership.
The schedule entry emits preparation artifacts only. It does not select a method
or register training. An operational manager for the new producer source must
be registered separately; historical accepted manager bytes are not a substitute.
See [the dependency, provenance and command notes](experiments/dynamic_sr_multiview_footprint_20261007/full_overlap_schedule_README.md).

The additive `full_held_*` entries bind the two separately registered confirmation
scenes to a completed development record and a single frozen Bsync/B0 configuration.
They preserve the registered native refinement computation and require actual
training-LR teacher acceptance, same-parent CUDA diagnostics, frozen-parent
overlap, schedule acceptance and per-role byte-SHA closure before dispatch.
Their evaluation and temporal interfaces retain the existing measurement functions
and require explicit post-freeze HR authorization. Persistent coordination uses
owned process exits and the physical-GPU lock; it does not infer completion from
progress logs. The operational adapters support the audited Coffee/A100 GPU1
and Flame/A100 GPU0 bindings and a uniform RTX3090 evaluator. Dataset pixels, checkpoints, acceptance
records and host-specific dependencies are not included in this source release.
See [the exact confirmation prerequisites and commands](experiments/dynamic_sr_multiview_footprint_20261007/full_held_confirmation_README_v3.md).

`dispatch_prepared_remote.py` can attach or dispatch the fixed remote paired
worker after an exact preparation exit and remote SHA validation, while immutable
assets are returned. It never publishes the local preparation gate early. The
uniform evaluator and local worker still require complete SHA-verified return.
Its worker identity matches the original continuation, preventing a second launch.

The root-controlled remote launcher requires its owned output log directory to
exist before dispatch. `observe_prepared_remote.py` retains the exact existing
worker invocation with pidfd and systemd events, reconnecting after SSH transport
failures without redispatching training. `preparation_transfer_reuse.py` seeds
only complete immutable payloads with matching path, size and SHA into a new
transfer stage. It leaves final all-file verification and completion publication
to the original return callback. Cache permissions alone are not an immutability
claim; content hashes are checked again before each reuse.
