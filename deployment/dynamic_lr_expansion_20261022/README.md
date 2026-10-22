# Six method bicubic completion preparation

This package prepares Wu 4DGS, Yang 4DGS, 4C4D, Multi4D, LocalDyGS and CEM-4DGS on the six registered full-time scenes. The selected input is the existing **bicubic antialias LR**, not area4. The current user instruction is preparation only while Claude uses the GPUs. No scheduler, watcher, remote submission or automatic start is installed.

| Method | Reusable trained scenes | Missing training | Missing train HR scores | Missing test HR scores |
| --- | ---: | ---: | ---: | ---: |
| Wu 4DGS | 6 | 0 | 6 | 4 |
| Yang 4DGS | 2 | 4 | 6 | 4 |
| 4C4D | 2 | 4 | 6 | 4 |
| Multi4D | 2 | 4 | 6 | 4 |
| LocalDyGS | 2 | 4 | 6 | 4 |
| CEM-4DGS | 2 | 4 | 6 | 4 |
| Total | 16 | 20 | 36 | 24 |

Cook07 and Cut07 have the twelve reusable common test reports. The other scenes are Coffee07, Flame07, Discussion07 and VRHeadset07. Wu already has clean native LR endpoints on all six scenes; it must not be trained again just to fill the remaining table cells.

## Data and evaluation contract

- Train only on the registered scene's own training-camera LR frames 0 through 299. Exclude cam00 pixels, all HR supervision, SR teachers, project-specific correction modules and cross-scene learned state.
- Training-view HR evaluation uses **all training cameras at frames 0, 10, ..., 290**. This fixed 30-frame subset contains 3,000 images per method across six scenes; it is not the full training image pool.
- Test-view HR evaluation uses held-out **cam00 at all 300 frames**. Twelve reports are already complete; 24 method-scene reports remain.
- The manifest time is `frame / 300`. Preserve native internal time conversions, including Yang/4C4D `frame / 30`, and each method's original schedule.
- Render uncorrected native HR RGB using each camera's `K_hr`, including the half-pixel convention. Store uint8 RGB PNG and score PSNR, RGB SSIM and AlexNet LPIPS v0.1 through the shared frozen helper. HR references are opened only by the separate scoring step after prediction.
- Preserve author-native prefilters. Do not add learned image warps, colour fitting or posthoc visibility culling. This external native protocol is separate from existing area4 P/PC training-view observation-model diagnostics.
- Verify the exact manifest, scene, method, helper, PNG convention and LPIPS state identities before reusing a report. One immutable CEM Cut report uses the recorded display name `CEM_native_original_LR`.

## CPU preparation

Run from the project root. These commands prepare metadata only; none starts training or GPU evaluation.

```bash
python3 deployment/dynamic_lr_expansion_20261022/prepare_suite.py --verify sample
python3 deployment/dynamic_lr_expansion_20261022/prepare_wu_multi.py
python3 deployment/dynamic_lr_expansion_20261022/prepare_strict4d.py
python3 deployment/dynamic_lr_expansion_20261022/prepare_local_cem.py --write-configs
python3 deployment/dynamic_lr_expansion_20261022/prepare_meetroom_adapter.py
python3 deployment/dynamic_lr_expansion_20261022/audit_local_cem_plans_cpu.py
python3 deployment/dynamic_lr_expansion_20261022/assemble_plan.py
python3 deployment/dynamic_lr_expansion_20261022/cpu_preflight.py
```

`prepare_strict4d.py` requires PyYAML. Optional Multi4D `--prepare-initialization --cpu-threads 1` requires the existing NumPy/OpenCV CPU environment and uses training LR only. The four new Multi4D frame0 SIFT initializations and original schedules have been prepared locally. No dependency installation is performed by this package.

Generated configurations, SIFT points, view plans, source/checkpoint hashes, audit notes and result ledgers remain local. They are not original third-party code and are not published as project source. A fresh public checkout does not include datasets, author installations, weights, old outputs or their local identities; obtain those dependencies separately.

These integration commands operate on the existing research checkout and its method-owned preparation/audit notes. `assemble_plan.py` and `cpu_preflight.py` deliberately require those frozen native view plans and historical endpoint registries. They are not a bootstrap installer for an empty public checkout. Individual renderer CLIs can regenerate their view plans from the matching manifest, runtime, config and fixed endpoint; `--help` lists those metadata arguments.

`output/dynamic_lr_expansion_20261022/preparation/suite_plan.json` records the 36 held method-scene tasks. `completion_ledger.json` joins all 72 native view plans and the **60 missing scoring scopes**, comprising 18,000 training-view images and 7,200 new-view images. Its argv arrays are inert metadata with unresolved environment/resource placeholders. The ledger is not an executable queue.

## Native recipes and remaining acceptance

N3DV uses the matching author scene recipes. LocalDyGS Coffee ends at 30k; Flame and MeetRoom end at 40k. CEM Coffee runs 40k then to total 50k; Flame runs 20k then to total 30k. Yang/4C4D preserve the Coffee environment map at resolution 500. Never substitute a common Cook schedule or disable that environment map.

MeetRoom adaptations are disclosed in the method plans: Yang/4C4D and CEM freeze the author's Cook recipe for both scenes; LocalDyGS VRHeadset freezes the official Discussion recipe. These are cross-dataset configuration transfers without cam00-driven tuning, not official author MeetRoom results. Preserve author initialization: Yang COLMAP dense, 4C4D posed MASt3R, LocalDyGS its native per-dataset point-cloud path, and CEM frame0 COLMAP sparse. The generic manifest triangulation is not a universal substitute.

CPU metadata and numerical camera checks do **not** validate native CUDA extensions, model loading, initialization quality, convergence or rendered images. Native full data loaders and author initialization must pass their exact input contracts before future dispatch. MeetRoom LocalDyGS/CEM additionally require a matching runtime camera/initialization acceptance receipt; an absent receipt blocks execution. Actual environment and camera-render smoke tests remain held.

`strict4d_contract.py` uses NumPy to replay the pinned author projection arithmetic for training LR, training HR and test HR cameras. `audit_meetroom_adapter_cpu.py` checks the LocalDyGS/CEM adapter with CUDA initialization blocked; it does not validate CUDA model loading. Use the existing research environments containing the required CPU libraries. `prepare_meetroom_adapter.py` emits small reviewable patches by default; only explicit `--materialize` constructs an isolated source runtime. `freeze_meetroom_initialization.py` requires completed, identity-matched LR-only native point-cloud and calibration receipts; a CPU projection check alone cannot create that acceptance.

`prepare_suite.py --verify sample` checks all required paths and registered identities and samples file contents. Full transferred-file SHA256 verification remains mandatory on the future execution host. It must not be described as completed by the sample pass.

## Future explicit resumption

All three renderers and `score_views.py` default to plan-only mode. They require explicit `--execute` for actual work; GPU work also requires a caller-admitted single `CUDA_VISIBLE_DEVICES`. This package does not claim that setting an environment variable establishes GPU ownership. Resource admission, locks, compatible method Python, short task-specific temporary paths, runtime identities and author initialization must be verified by the future caller.

Scoring requires a locally cached AlexNet backbone under the chosen `--torch-home`; it refuses an absent cache and never downloads a model. LPIPS model state must match the twelve preserved reports. A directory containing only its exact plan may be reused for execution; existing images, metric results and receipts are protected.

On later successful training exit, immediately render and score missing scopes, verify final endpoints and output hashes, and return provenance, hardware and costs to the local project. On failure, record the error and do not score an incomplete endpoint. This completion contract is recorded now; no process-exit hook is active while the work is held. Update the existing document with one combined training-view/new-view table per dataset only after metrics exist. Pending entries remain explicitly missing.
