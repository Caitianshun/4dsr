# Full native parent overlap and schedules

These two portable entries expose the frozen-parent coarse visibility calculation and the corresponding CPU schedule calculation. They do not choose a candidate, train SR, evaluate image quality, register machine services, or grant GPU ownership.

`full_frozen_parent_overlap.py` preserves the native coarse-moment, projection, occlusion, pair-overlap and observer-RNG functions. It creates its own source-bound plan. `full_native_schedule_materializer.py` preserves the Hungarian assignment, pose-plus-overlap cost, seeded ordering, exposure checks and scalar acceptance functions. It outputs a schedule and statistics; the private full-stage operator remains responsible for selection, teacher identities, native training plans, actual units, storage and resource authorization.

The portable files have new SHA256 identities. They are not the original private source bytes and cannot be substituted into existing registered plans, schedules or accepted experiment receipts. A source mapping and CPU equivalence receipt are retained privately by the operator. Existing experiment artifacts remain unchanged.

## Inputs and boundaries

Supply the complete derived project manifest, the independent same-seed native LR parent, its completion and accepted-entry metadata, the pinned author checkout and full-data readiness. The protocol uses frames 0–299 and the registered project crop, not the original paper's complete benchmark. `cam00` is held out; `cam01` is a legal training camera. The producer reads no HR, teacher or image pixels. CPU validation hashes legal LR and initialization bytes; actual preparation loads the native checkpoint and computes coarse moments.

The coarse grid is exactly LR/4: 84×63 for registered N3DV LR336×252, or 80×45 for MeetRoom LR320×180. The output is a parent-model approximation. A null overlap remains unknown. Neither overlap nor camera-pair agreement proves ground-truth geometry.

Scheduling uses camera-center distance, optical-axis angle and the conservative minimum of two directed parent-overlap ratios. It writes 6000 simultaneous three-camera rows, corresponding random rows for B0, equal exposure in each column, pair frequencies, distances, angles, overlap proportions and training-camera metadata anchors. Non-Cook populations receive rotating duplicate slots; they are not reported as once-only exposure. All three cameras in each simultaneous row differ.

## CPU planning

Run in the registered Python3.10/NumPy1.26.4 metadata environment. Set `CUDA_VISIBLE_DEVICES=''` and `FOURDSR_UPSTREAM` to the actual pinned author tree. Data, weights, checkpoints and execution receipts are separate assets and are not supplied by cloning this repository.

```sh
CUDA_VISIBLE_DEVICES='' FOURDSR_UPSTREAM="$AUTHOR_ROOT" python experiments/dynamic_sr_multiview_footprint_20261007/full_frozen_parent_overlap.py --mode plan --manifest "$MANIFEST" --seed "$SEED" --parent "$PARENT" --parent-complete "$PARENT_COMPLETE" --parent-acceptance "$PARENT_ACCEPTANCE" --parent-index "$PARENT_INDEX" --readiness "$READINESS" --upstream "$AUTHOR_ROOT" --registered-plan "$NEW_COARSE_PLAN" --inventory "$NEW_INPUT_INVENTORY"
```

The producer's `prepare` mode additionally requires a fresh output, explicit GPU UUID, shared physical lock and an exact source/plan/unit/argv authorization. It performs two resource checks separated by 30 seconds and a prelaunch check. Only the allocator may register and launch it. CPU planning is not CUDA acceptance or actual preparation.

If original absolute input records belong to another workspace, explicitly set `FOURDSR_ORIGIN_ROOT` to their original root. This maps paths during execution and preserves the original record bytes. Relative paths are preferred.

## CPU schedule and exact reuse

```sh
CUDA_VISIBLE_DEVICES='' FOURDSR_UPSTREAM="$AUTHOR_ROOT" python experiments/dynamic_sr_multiview_footprint_20261007/full_native_schedule_materializer.py --manifest "$MANIFEST" --seed "$SEED" --parent "$PARENT" --parent-complete "$PARENT_COMPLETE" --coarse-acceptance "$ACCEPTED_COARSE_EXIT_AND_SHA_RECEIPT" --upstream "$AUTHOR_ROOT" --out output/dynamic_sr_multiview_footprint_20261007/full_native_schedules/new_revision_A
```

The schedule entry consumes a precise successful coarse manager exit and the immediate all-output SHA receipt. It checks every scalar frame file, populations, ratios, source identities, native model/Adam/RNG invariants and actual costs. It reuses the prior acceptance of raw NPZ packets and does not hash or decode them again. The original exact observer and typed closure manager sources must be supplied through their immutable receipt references; host-specific operational replacements need a separately reviewed metadata revision.

For revision B, supply `--reuse-schedule ORIGINAL_A_SCHEDULE` and optionally `--assignment-acceptance ACCEPTED_ASSIGNMENT_RECEIPT`, with a new `--out`. The schedule, pose, prepared overlap and statistics retain their exact original path/SHA identities. New-path copies are not equivalent identities. The result still declares `GPU_dispatch_allowed=false` and `native_training_plan_registered=false`; final assignment acceptance does not grant dispatch.

The schedule entry intentionally does not read the private short-window result policy, teacher caches, ROI assets or machine registrations, and does not emit native training/evaluation commands. It is a scientific preparation entry, not a replacement for the operator's complete training→evaluation→return chain.
