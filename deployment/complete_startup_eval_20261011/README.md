# Complete the existing Desk startup evaluation

The previous Desk runtime recovery completed its50-step coarse diagnostic and
saved checkpoint SHA `0fc0729727f8d20f109561f407e0dd3a9e830b3b166a36a70a648423ad3d9edf`.
Its idle check failed before eval3 launched. The failure log lacks the exact
memory/utilization reading; delayed exit cleanup or a transient reading is a
possible cause. The original failure is preserved.

This adapter references that checkpoint in place and performs only the original
`entry.py eval --cull --frames 0 150 299`. It does not copy/retrain the checkpoint,
rerun50steps, redo CPU/Z5/sync, remake the original28-task table, or change the
frozen scientific namespace, losses, sampling or budgets. Its fresh startup unit
is a persistent owned `python -c` worker; no remote installation is needed.

The worker keeps the original GPU UUID claim,1024MiB/10% idle limits and rejection
of foreign compute processes. Before each original idle check it sleeps2seconds
once to allow resource readings to settle. It does not retry a failed check or
lower a threshold. The same small worker wraps the unchanged formal runner so
its task boundaries receive the same buffer. Both worker and adapter identities
are recorded. The inherited CTS library environment applies only to these owned
services/remote Python calls, without changing SSH or the user manager.

Evaluation exits trigger immediate CSV/summary checks for exactly cam00 frames
0/150/299 and finite PSNR/SSIM/LPIPS. The original evaluator also emits its normal
sampled training-view diagnostics; this remains a three-frame startup diagnostic,
not a300-frame result. Successful acceptance records the original50step receipt
and new eval3 receipt separately, then calls the original runner/pull/summarize.
Condition flock, GPU claims, disk reserves, failure returns and the fixed final
headroom-analysis whitelist remain inherited. Existing13,335-file full-SHA sync
acceptance is reused, without another remote full-input hash scan or transfer.

The root operator creates a new local spec and validates its actual artifact
bindings, then dispatches it through a new persistent local service:

```sh
python deployment/complete_startup_eval_20261011/complete_startup_eval.py --make-spec \
  --run-id deskgames_eval3_resume_UNIQUE \
  --spec-out output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json
python deployment/complete_startup_eval_20261011/complete_startup_eval.py \
  --spec output/dynamic_sr_highres_20261011/orchestration/FRESH_SPEC.json --validate-only
```

Use a unique lowercase run ID. These two commands only inspect local existing
evidence; they do not connect to CTS, import CUDA or launch evaluation/training.
Execution is restricted to CTS Desk half seed20261007 and this exact checkpoint.
No automatic retry is provided. The diagnostic checkpoint remains ineligible as
the formal LR parent.
