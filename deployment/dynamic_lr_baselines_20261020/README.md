# Original LR dynamic reconstruction comparison

This runner compares six dynamic Gaussian reconstruction methods on the full
Cook and Cut sequences. It uses the original registered bicubic antialias LR
images. It does not use the subsequent area4 input variant or proposed SR,
colour, displacement, or visibility modules.

## Frozen inputs and evaluation

- Each scene contains 300 frames. Cook has 20 training cameras; Cut has 19
  because cam04 is absent. Camera cam00 is held out throughout training and
  initialization.
- Training images are 336 by 252. Held-out references are 1344 by 1008.
- Time is the original frame index divided by 300. Data adapters preserve the
  calibrated focal lengths, principal points, poses, and crop conventions.
- The selected inventory contains training LR images, held-out HR references,
  manifests, and explicitly identified LR initialization files. The references
  are used only after reconstruction, for scoring and visual inspection.
- Native initialization may differ between methods. The author-supported
  COLMAP, learned geometric prior, or LR triangulation procedure is disclosed
  in each method's record; it is not silently labelled a common initialization.
- Every method produces all 300 held-out HR RGB PNGs. The main table uses the
  same `evaluate_images.py` and A100 base environment for all PNGs. Historical
  float-render metrics remain separate diagnostics.
- PSNR is the mean of full-image per-frame PSNR; SSIM uses an 11 by 11 Gaussian
  window and excludes the outer five pixels; LPIPS uses AlexNet version 0.1.
  These definitions are inherited from the frozen project evaluator.

Run `prepare_protocol.py` to verify the local manifests and build the complete
input inventory. After incremental synchronization, run
`verify_remote_inputs.py` on each destination. A verified receipt is required
before a GPU job starts. Generated inventories, outputs, environment copies,
third-party runtime copies, and deployment logs are local assets.

## Execution and recovery

`submit_remote.py` submits an explicit GPU job through the registered
`claude-training` broker. `run_job.py` checks frozen identities and runs the
registered commands serially. A successful training exit starts rendering and
evaluation immediately. An unsuccessful stage records failure and stops.

The local wrapper waits on filesystem events, verifies transport submission,
waits for the remote terminal state, and returns the registered output paths.
Remote supervisors survive SSH disconnection. Reattach to an existing job
after a transport failure; never submit a second copy without first proving
the original task's terminal state. Retain the original failure record and use
a new job identity for a justified repair.

For an existing submission, use the same arguments with `--reattach`. This
mode checks the saved job hash, host, and complete broker request, then only
waits and collects. It never submits a GPU task. Dispatch watches tolerate the
broker's temporary directory permission changes by watching a readable parent
and its permission events; they do not change those permissions.

The resource allocation starts with A100 GPU0 for Yang 4DGS and 4C4D. A100
GPU1 first finishes Wu/Multi4D, then prepares Cut's native COLMAP initialization
and runs Cut CEM-4DGS. CTS GPU0 prepares Cook's initialization, runs Cook
LocalDyGS and CEM-4DGS, then runs Cut LocalDyGS after the returned initialization.
Pending independent jobs may move to an explicitly released GPU; an unstarted
job gets a new frozen identity and only one submitting controller.
Each GPU runs serial stages. Immediate CTS scores are provisional; the same
returned PNGs are evaluated on A100 for the main table. Broker admission and
project locks protect existing work. This runner does not use the 5090 GPU.

Completed original-LR Wu checkpoints can be reused after their inputs,
training settings, absence of proposed modules, and hashes are verified.
Their historical training hardware remains part of the result record.

## Method adapters

See `methods/wu_multi/README.md`, `methods/strict4d/`, and
`methods/localdygs_cem/README.md` for pinned upstream versions, native schedules,
initialization details, compatibility fixes, and calibrated camera interfaces.
The external repositories and compiled dependencies are installed separately.
This source directory alone is not a complete experiment package.

## Returned-result validation

`gather_results.py --registry <explicit-runs.json> --out <comparison-dir>`
accepts an explicit list of the twelve fixed method/scene endpoints. It checks
returned checkpoint hashes, the complete 300-image set, each image's evaluated
hash, the HR reference hashes, and the per-frame arithmetic means. All main
table entries must share the same A100 evaluation environment, evaluator,
SSIM definition, and LPIPS weights. Add `--require-complete` for the final
table; missing entries remain pending before that point. No checkpoint or run
is selected by held-out scores.

For returned methods without common scores, `make_shared_eval_job.py` freezes
an explicit render-only job from their registry entries. It includes the
hashes of all 300 PNGs per endpoint and emits a separate evaluation plan.
The caller must synchronize the listed files, acquire an available A100 GPU
through the broker, and collect the results before updating the registry.
Preparing the job does not submit it or claim its metrics are complete.
Use `--scene` to select only a newly returned scene when another scene from
the same method has already received common scores.

`make_visuals.py` uses preselected cam00 frames 0, 150, and 299 and fixed face
regions. Whole images are displayed at half size, and face crops are enlarged
with nearest-neighbor interpolation. The source renders remain available at
their original resolution; display resizing does not affect the metrics.

`collect_one.py` is the persistent local wrapper for a single explicit job.
It writes a successful receipt only after the remote broker reports success
and collection exits successfully. It never retries. The separate hourly
observer reads progress and failure state; normal rendering, evaluation and
collection still start from process-exit events.

`make_report.py` requires all twelve validated endpoints and recorded inspection
of the twelve fixed visual panels before writing the final DOCX/Markdown.
DOCX rendering and inspection remain a separate delivery gate.
