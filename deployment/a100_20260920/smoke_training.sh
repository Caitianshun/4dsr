#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/3DGS/4dsr
source deployment/a100_20260920/activate_a100.sh
OUT=output/a100_deployment_smoke_20260920
test ! -e "$OUT"
mkdir -p "$OUT"
EVALUATION_PID=
TRAINING_PID=
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ "$rc" -ne 0 ]; then
    for pid in "$EVALUATION_PID" "$TRAINING_PID"; do
      if [ -n "$pid" ]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      fi
    done
    wait 2>/dev/null || true
  fi
  printf '{"returncode":%s}\n' "$rc" > deployment/a100_20260920/training_smoke_exit.json
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CUDA_VISIBLE_DEVICES=0 python experiments/dynamic_sr_20260918/run_experiment.py \
  --task warmup --observation integrated_lr --coarse-steps 2 --fine-steps 2 \
  --manifest data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json \
  --out "$OUT/warmup" > "$OUT/warmup.log" 2>&1
CUDA_VISIBLE_DEVICES=0 setsid python experiments/dynamic_sr_20260918/evaluate.py \
  --manifest data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json \
  --checkpoint checkpoints/bootstrap/cook_spinach/checkpoint_integrated.pt \
  --out "$OUT/cook_parent_evaluation" --no-video > "$OUT/cook_parent_evaluation.log" 2>&1 &
EVALUATION_PID=$!
setsid python deployment/a100_20260920/run_control.py --gpu 1 \
  --manifest data/dynamic_sr/meetroom_prepared/discussion/manifest.json \
  --checkpoint checkpoints/bootstrap/meetroom_discussion/checkpoint_integrated.pt \
  --out "$OUT/meetroom_sr20" --teacher sr --weight 0.1 --steps 20 \
  --milestones 20 --prior-cameras cam02,cam04,cam08,cam12 \
  > "$OUT/meetroom_runner.log" 2>&1 &
TRAINING_PID=$!
# Both tasks are awaited by their exact child exit, without status polling.
set +e
wait "$EVALUATION_PID"; EVAL_RC=$?
wait "$TRAINING_PID"; TRAIN_RC=$?
set -e
test "$EVAL_RC" = 0
test "$TRAIN_RC" = 0
python - <<'PY'
import json,math
from pathlib import Path
root=Path.cwd(); d=root/'deployment/a100_20260920'; out=root/'output/a100_deployment_smoke_20260920'
reference=json.loads((d/'local_parent_reference.json').read_text())
metrics=json.loads((out/'cook_parent_evaluation/metrics.json').read_text())
assert metrics['checkpoint_sha256']==reference['checkpoint_sha256']
assert metrics['manifest_sha256']==reference['manifest_sha256']
keys=['psnr_mean','ssim_mean','lpips_alex_mean']
comparison={k:dict(local=reference['aggregate']['full'][k],a100=metrics['aggregate']['full'][k],
                   delta=metrics['aggregate']['full'][k]-reference['aggregate']['full'][k]) for k in keys}
assert all(math.isfinite(v['a100']) for v in comparison.values())
tolerances={'psnr_mean':1e-3,'ssim_mean':1e-5,'lpips_alex_mean':1e-4}
for key, values in comparison.items():
    values['absolute_tolerance']=tolerances[key]
    values['passed']=abs(values['delta']) <= tolerances[key]
assert all(v['passed'] for v in comparison.values()), comparison
job=json.loads((out/'meetroom_sr20_job/status.json').read_text())
assert job['status']=='complete'
meeting=json.loads((out/'meetroom_sr20_evaluation/metrics.json').read_text())
report=dict(status='passed',scope='Deployment smoke, not evidence of method improvements or convergence.',
            warmup='2 coarse + 2 fine iterations on GPU0 from training-LR initialization',
            sr_fit='20 iterations on GPU1 from the unchanged MeetRoom common parent, followed immediately by full evaluation',
            frames_per_evaluation=60,parent_comparison=comparison,
            meetroom_smoke_metrics=meeting['aggregate']['full'],
            original_evaluation_version=reference['version'],remote_evaluation_version=metrics['version'])
(d/'training_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
