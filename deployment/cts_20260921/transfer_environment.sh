#!/usr/bin/env bash
# Run locally. Read only the dedicated general dynamic-SR environment on A100.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trap 'rc=$?; printf "{\"returncode\":%s}\n" "$rc" > "$TASK_DIR/verification/environment_transfer_exit.json"' EXIT
ssh cts 'cd /home/cts/Project/4DSR && test ! -e .venv && python3 -m venv --without-pip .venv && mkdir -p deployment/cts_20260921/verification tools .cache/torch'
ssh a100-train 'cd /home/ubuntu/3DGS/4dsr && tar --exclude="__pycache__" --exclude="*.pyc" -cf - .venv/lib/python3.10/site-packages tools/cuda-12.8-minimal .cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth' \
  | gzip -1 \
  | ssh cts 'cd /home/cts/Project/4DSR && gzip -d | tar -xf -'
ssh a100-train 'cd /home/ubuntu/3DGS/4dsr && .venv/bin/python -m pip list --format=json' > "$TASK_DIR/verification/a100_source_packages.json"
ssh cts '/home/cts/Project/4DSR/.venv/bin/python -m pip list --format=json' > "$TASK_DIR/verification/cts_transferred_packages.json"
