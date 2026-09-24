#!/usr/bin/env bash
# Optional first argument selects a new output directory. GPU work is sequential.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VERIFY="$ROOT/deployment/cts_20260921/verification"
OUT="${1:-output/cts_deployment_smoke_20260921}"
if [ "$#" -gt 1 ]; then
  printf 'Usage: %s [new-output-directory]\n' "$0" >&2
  exit 2
fi
mkdir -p "$VERIFY"
for target in "$OUT" "$VERIFY/training_smoke.json" "$VERIFY/training_smoke_exit.json"; do
  if [ -e "$target" ] || [ -L "$target" ]; then
    printf 'Refusing to overwrite existing smoke artifact: %s\n' "$target" >&2
    exit 2
  fi
done
STATUS_PYTHON="$(command -v python3)"
mkdir -p "$(dirname -- "$OUT")"
mkdir -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
ACTIVE_PID=
STAGE=activation
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ "$rc" -ne 0 ] && [ -n "$ACTIVE_PID" ]; then
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  "$STATUS_PYTHON" - "$VERIFY/training_smoke_exit.json" "$rc" "$STAGE" "$OUT" "$STARTED" <<'PY'
import json, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, code, stage, out, started = sys.argv[1:]
record = dict(status='passed' if int(code) == 0 else 'failed', returncode=int(code),
              stage=stage, output=out, host=socket.gethostname(), gpu=0,
              started_at=started, finished_at=datetime.now(timezone.utc).isoformat())
with Path(path).open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\n')
PY
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source deployment/cts_20260921/activate_cts.sh
export CUDA_VISIBLE_DEVICES=0
run_stage() {
  STAGE="$1"
  local logfile="$2"
  shift 2
  setsid "$@" > "$logfile" 2>&1 &
  ACTIVE_PID=$!
  # Exact child-exit wait, without polling. No second GPU stage is concurrent.
  wait "$ACTIVE_PID"
  ACTIVE_PID=
}
run_stage warmup "$OUT/warmup.log" \
  python experiments/dynamic_sr_20260918/run_experiment.py \
  --task warmup --observation integrated_lr --coarse-steps 2 --fine-steps 2 \
  --manifest data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json \
  --out "$OUT/warmup"
run_stage meetroom_sr_and_evaluation "$OUT/meetroom_runner.log" \
  python deployment/a100_20260920/run_control.py --gpu 0 \
  --manifest data/dynamic_sr/meetroom_prepared/discussion/manifest.json \
  --checkpoint checkpoints/bootstrap/meetroom_discussion/checkpoint_integrated.pt \
  --out "$OUT/meetroom_sr20" --teacher sr --weight 0.1 --steps 20 \
  --milestones 20 --prior-cameras cam02,cam04,cam08,cam12
run_stage cook_parent_evaluation "$OUT/cook_parent_evaluation.log" \
  python experiments/dynamic_sr_20260918/evaluate.py \
  --manifest data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json \
  --checkpoint checkpoints/bootstrap/cook_spinach/checkpoint_integrated.pt \
  --out "$OUT/cook_parent_evaluation" --no-video
STAGE=integrity_and_metric_comparison
python - "$OUT" "$VERIFY" <<'PY'
import hashlib, json, math, socket, sys
from pathlib import Path
root = Path.cwd()
out, verify = map(Path, sys.argv[1:])
reference_path = root / 'deployment/a100_20260920/local_parent_reference.json'
reference = json.loads(reference_path.read_text())
metrics = json.loads((out / 'cook_parent_evaluation/metrics.json').read_text())
assert metrics['checkpoint_sha256'] == reference['checkpoint_sha256']
assert metrics['manifest_sha256'] == reference['manifest_sha256']
assert len(metrics['rows']) == 60, len(metrics['rows'])
keys = ['psnr_mean', 'ssim_mean', 'lpips_alex_mean']
comparison = {k: dict(local=reference['aggregate']['full'][k], cts=metrics['aggregate']['full'][k],
                     delta=metrics['aggregate']['full'][k] - reference['aggregate']['full'][k])
              for k in keys}
assert all(math.isfinite(v['cts']) for v in comparison.values())
tolerances = {'psnr_mean': 1e-3, 'ssim_mean': 1e-5, 'lpips_alex_mean': 1e-4}
for key, values in comparison.items():
    values['absolute_tolerance'] = tolerances[key]
    values['passed'] = abs(values['delta']) <= tolerances[key]
job = json.loads((out / 'meetroom_sr20_job/status.json').read_text())
assert job['status'] == 'complete', job
meeting = json.loads((out / 'meetroom_sr20_evaluation/metrics.json').read_text())
assert len(meeting['rows']) == 60, len(meeting['rows'])
assert all(math.isfinite(meeting['aggregate']['full'][k]) for k in keys)
report = dict(status='passed' if all(v['passed'] for v in comparison.values()) else 'failed',
              host=socket.gethostname(), project_root=str(root), output=str(out), gpu=0,
              scope='Deployment smoke, not evidence of method improvements or convergence.',
              warmup='2 coarse + 2 fine iterations from training-LR initialization',
              sr_fit='20 iterations from the unchanged MeetRoom common parent, immediately '
                     'followed by full evaluation; all stages run sequentially on GPU0',
              frames_per_evaluation=dict(cook=len(metrics['rows']), meetroom=len(meeting['rows'])),
              parent_comparison=comparison, meetroom_smoke_metrics=meeting['aggregate']['full'],
              reference_path=str(reference_path),
              reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              checkpoint_sha256=metrics['checkpoint_sha256'], manifest_sha256=metrics['manifest_sha256'],
              original_evaluation_version=reference['version'], remote_evaluation_version=metrics['version'])
with (verify / 'training_smoke.json').open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(report, indent=2))
assert report['status'] == 'passed', comparison
PY
STAGE=complete
