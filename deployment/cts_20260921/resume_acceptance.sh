#!/usr/bin/env bash
set -euo pipefail
cd /home/cts/Project/4DSR
VERIFY=deployment/cts_20260921/verification
ARCHIVE="$VERIFY/failure_missing_tkinter"
test ! -e "$ARCHIVE"
mkdir "$ARCHIVE"
mv "$VERIFY/acceptance_exit.json" "$VERIFY/training_smoke_exit.json" "$ARCHIVE/"
exec 9>deployment/cts_20260921/gpu0.lock
flock -n 9
STAGE=runtime_additions
trap 'rc=$?; printf "{\"returncode\":%s,\"stage\":\"%s\"}\n" "$rc" "$STAGE" > "$VERIFY/acceptance_exit.json"' EXIT
source activate_cts.sh
python deployment/cts_20260921/install_runtime_additions.py
python -c 'import tkinter; import sys; sys.path.insert(0,"experiments/dynamic_sr_20260918"); import common; print("General model import passed")'
python - <<'PY'
import subprocess
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not apps,apps
PY
export CUDA_VISIBLE_DEVICES=0
STAGE=training_and_evaluation
bash deployment/cts_20260921/smoke_training.sh output/cts_deployment_smoke_20260921_v2 > deployment/cts_20260921/logs/training_smoke_v2.log 2>&1
STAGE=video
python deployment/cts_20260921/smoke_video.py > deployment/cts_20260921/logs/video_smoke.log 2>&1
STAGE=complete
