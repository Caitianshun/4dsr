# Shared existing SwinIR inputs for SplatSuRe and IE-SRGS

This is an explicitly labelled adaptation on Wu-4DGS. SplatSuRe and IE-SRGS
receive the same existing area4-finetuned SwinIR RGB8 PNGs. Sequence Matters
retains its PSRT teacher; 3DSR retains its four diffusion/3D-feedback rounds.
The frozen implementations in `experiments/static_sr_wu_20261022` are unchanged.

The imported model is **SwinIR-M x4 fine-tuned on the recorded external area4
dataset**, not the original DF2K checkpoint and not StableSR. Its source
inventory, weights, fine-tuning summary, base weights and external dataset
index retain their original identities. A later change to the historical
fine-tuning script is disclosed as a hash mismatch; the old recorded hash is
not rewritten or presented as the current source.

## Import and SplatSuRe bundle

Run these CPU commands from the project root with Pillow and PyTorch available.
Every destination must be new. No command below performs GPU inference.

```bash
python experiments/static_sr_shared_teacher_20261024/shared_cache.py \
  --manifest MANIFEST --teacher-index ORIGINAL_TEACHER_INDEX \
  --tag sr_swinir_x4_areaft --finetune-summary ORIGINAL_FT_SUMMARY \
  --output NEW_SHARED_CACHE

python experiments/static_sr_shared_teacher_20261024/build_splatsure_bundle.py config \
  --base-config REGISTERED_SPLATSURE_CONFIG --shared-cache NEW_SHARED_CACHE \
  --out NEW_SHARED_CONFIG

python experiments/static_sr_shared_teacher_20261024/build_splatsure_bundle.py bundle \
  --config NEW_SHARED_CONFIG --shared-cache NEW_SHARED_CACHE \
  --weights-receipt GENUINE_FULL_CONTRIBUTION_RECEIPT \
  --lr-completion COMPLETED_FORMAL_LR_STAGE --out NEW_BUNDLE
```

The importer verifies every training LR/SR SHA and RGB8 PNG size. Keys and time
come from the legal training manifest; cam00, missing/duplicate keys, path
substitution and symlinks are rejected. It reads neither held-out pixels nor
training HR pixels. `verify_cache()` reconstructs the registration from the
original inventory and files rather than trusting its status field.

Both methods use `float32(CHW RGB) / 255`, with no resize, w4 displacement or
colour correction. SplatSuRe keeps its original LR loss and contribution-based
selective SR loss. The completed scene-local LR checkpoint is used only to
produce contribution maps with the official rasterizer and simultaneous
training cameras. Final SR starts fresh. The generated configuration changes
only `teacher_variant`, `label`, and `information.shared_teacher`.

The bundle uses the existing `static_sr_wu_prior_bundle_v1` interface and passes
the unchanged trainer's preflight. The new producer is recorded explicitly;
the areaft cache is never labelled as an author StableSR inference receipt.

## IE-SRGS shared external RGB

`prepare_ie_shared.py external` makes a CPU plan by default. `--run` explicitly
loads only official Depth Anything V2-Large, then predicts depth from the same
PNG. No SwinIR model is loaded again. `--fixture-key cam01/0000` creates a
separate diagnostic schema which cannot be merged as a formal full cache.

```bash
python experiments/static_sr_shared_teacher_20261024/prepare_ie_shared.py external \
  --config REGISTERED_IE_CONFIG --shared-cache NEW_SHARED_CACHE --out NEW_DEPTH_DIR
python experiments/static_sr_shared_teacher_20261024/prepare_ie_shared.py merge \
  --external-index COMPLETE_SHARED_EXTERNAL_INDEX \
  --internal-index COMPLETE_ORIGINAL_IE_INTERNAL_INDEX --out NEW_MERGED_JSON
```

IE's internal Wu+Mip teacher, three same-time view sampling, fusion losses,
depth losses and schedules remain unchanged. The external RGB precision is
explicitly RGB8 PNG, whereas the author producer originally retains float SR
RGB. Report this adaptation and the additional depth prior in comparisons.

## Remote SplatSuRe continuation

`run_splatsure_shared.py` only plans or executes an already admitted remote
job. It never submits, transfers, deletes or retries. It rejects PRO6000 under
the current resource instruction. A new task-local runtime-profile JSON must
contain actual verified values:

- `schema: shared_splatsure_remote_runtime_v1`, `status: CPU_imports_verified`,
  `broker: remote`, `project_id: 4dsr`.
- Exact executing `host`, task `root`, shared `broker_project_root`, selected
  `gpu_uuid` and single `visible_device`.
- `base_python`, optional `base_launcher`, and `launcher_arguments` (empty
  without a launcher). The interpreter path retains its venv symlink.
- `required_files`, each with actual `path`, `sha256` and `bytes`, including
  the interpreter, launcher if used, selected environment files and real CPU
  import receipt. `CPU_acceptance` identifies that receipt.

Create the profile on the intended host after its environment audit. Re-import
the original teacher index there into a **new** shared cache record and derive
a new shared config; this registers current importer paths without rewriting
the original index/manifest. The copied LR checkpoint, registered configuration
and completion retain their original bytes. Source configuration and Wu
configuration must match.

The original LR completion contains a source-host absolute checkpoint path.
Do not replace that path inside the original JSON or create a fake local
`stage_completed` receipt. Register its copied bytes with an explicit portable
import instead:

```bash
python experiments/static_sr_shared_teacher_20261024/build_splatsure_bundle.py import-lr \
  --original-completion COPIED_ORIGINAL_COMPLETE_JSON \
  --origin-completion-path /absolute/source/host/path/to/complete.json \
  --imported-checkpoint COPIED_ORIGINAL_CHECKPOINT \
  --imported-config COPIED_ORIGINAL_REGISTERED_CONFIG \
  --config NEW_SHARED_CONFIG --out NEW_PORTABLE_LR_IMPORT
```

`shared_splatsure_portable_LR_import_v1` preserves original completion/config/
checkpoint identities and explicitly maps only the imported file locations.
Origin paths are provenance and are never opened on the destination host.
Config/checkpoint SHA equality is mandatory. Imported byte sizes are recorded;
an original byte-size field is also compared when it exists. The original
checkpoint receipt has no byte-size field, so the importer does not invent one
inside that original identity. CPU inspection of the copied checkpoint verifies
the original LR stage, 30000 iterations, 29999 updates, scene, manifest and Wu
configuration. The final bundle retains the original completion identity and
the import mapping. The runtime freezes both the mapping and its copied source
JSON/config, in addition to the imported checkpoint.

```bash
python experiments/static_sr_shared_teacher_20261024/run_splatsure_shared.py plan \
  --config NEW_SHARED_CONFIG --shared-cache NEW_SHARED_CACHE \
  --lr-completion NEW_PORTABLE_LR_IMPORT --run-id NEW_RUN_ID \
  --runtime-profile ACTUAL_REMOTE_PROFILE \
  --wiring-receipt ACTUAL_SAME_GPU_WIRING_RECEIPT \
  --metric-receipt ACTUAL_SAME_GPU_METRIC_RECEIPT --out NEW_PLAN
```

The plan emits `broker_entry_argv`. Submit that exact command through the
existing remote supervisor. Execution checks its context, host, GPU, boot/PID
identity, ancestry, project, job ID, working directory and exact request argv.
It obtains no additional lock and does not replace resource admission.

The four serial stages are contribution maps, shared bundle, old CPU preflight,
then fresh SR30000 with immediate full 300-frame cam00 PSNR/SSIM/LPIPS evaluation.
Capacity guards reserve 20 GiB plus remaining output estimates. A 2 GiB
checkpoint file cap is a **failure boundary**, not a promise of checkpoint size
or permission to truncate a successful model, prune points or shorten training.
Any failure preserves logs and stops; an existing run directory is not replayed.
`pipeline/complete.json` precedes successful terminal status publication.

Old StableSR generated PNGs, shards and teacher-output receipts are not
dependencies of this pipeline. Source, environment, model and PNG identities
are rechecked before each child. The full scene's contribution generation,
training and evaluation still require real GPU acceptance and execution;
CPU tests do not demonstrate reconstruction quality.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  experiments.static_sr_shared_teacher_20261024.test_shared_cache \
  experiments.static_sr_shared_teacher_20261024.test_prepare_ie_shared \
  experiments.static_sr_shared_teacher_20261024.test_run_splatsure_shared -v
```
