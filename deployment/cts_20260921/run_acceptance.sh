#!/usr/bin/env bash
# Run as a persistent cts user service after data and environment verification.
set -euo pipefail
cd /home/cts/Project/4DSR
VERIFY=deployment/cts_20260921/verification
mkdir -p "$VERIFY" deployment/cts_20260921/logs
test ! -e "$VERIFY/acceptance_exit.json"
exec 9>deployment/cts_20260921/gpu0.lock
flock -n 9
STAGE=resource_preflight
trap 'rc=$?; printf "{\"returncode\":%s,\"stage\":\"%s\"}\n" "$rc" "$STAGE" > "$VERIFY/acceptance_exit.json"' EXIT
resource_check() {
  python3 - "$1" <<'PY'
import csv,datetime,json,os,socket,subprocess,sys
from pathlib import Path
def smi(*args): return subprocess.check_output(['nvidia-smi', *args], text=True).strip()
gpu=smi('--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits')
apps=smi('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits')
rows=list(csv.reader(gpu.splitlines()))
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),
            gpu=gpu,compute_processes=apps,load=os.getloadavg(),owner='general dynamic-SR cts deployment acceptance')
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
assert len(rows)==1 and '3090' in rows[0][2], record
assert not apps and int(rows[0][3])<10 and int(rows[0][4])<2048, record
PY
}
resource_check "$VERIFY/resources_before_install.json"
STAGE=install
bash deployment/cts_20260921/install_environment.sh > deployment/cts_20260921/logs/install.log 2>&1
source deployment/cts_20260921/activate_cts.sh
python deployment/cts_20260921/restore_console_scripts.py
resource_check "$VERIFY/resources_before_gpu.json"
export CUDA_VISIBLE_DEVICES=0
STAGE=environment_smoke
python deployment/cts_20260921/smoke_environment.py > deployment/cts_20260921/logs/environment_smoke.log 2>&1
STAGE=training_and_evaluation
bash deployment/cts_20260921/smoke_training.sh > deployment/cts_20260921/logs/training_smoke.log 2>&1
STAGE=video
python deployment/cts_20260921/smoke_video.py > deployment/cts_20260921/logs/video_smoke.log 2>&1
STAGE=complete
