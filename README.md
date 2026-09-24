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
