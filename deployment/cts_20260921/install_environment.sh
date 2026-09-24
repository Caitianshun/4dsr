#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p deployment/cts_20260921/verification deployment/cts_20260921/logs
trap 'rc=$?; printf "{\"returncode\":%s}\n" "$rc" > deployment/cts_20260921/verification/install_exit.json' EXIT
source deployment/cts_20260921/activate_cts.sh
python deployment/cts_20260921/install_runtime_additions.py
# Recreate the native build entry point without downloading an unchanged wheel.
ln -sfn "$(python -c 'import ninja; print(ninja.BIN_DIR + "/ninja")')" .venv/bin/ninja
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  -e vendor/4dgs/submodules/depth-diff-gaussian-rasterization \
  -e vendor/4dgs/submodules/simple-knn
mkdir -p tools/ffmpeg/bin
ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" tools/ffmpeg/bin/ffmpeg
printf 'source /home/cts/Project/4DSR/deployment/cts_20260921/activate_cts.sh\n' > activate_cts.sh
python -m pip check > deployment/cts_20260921/verification/pip_check.txt
python -m pip freeze > deployment/cts_20260921/verification/requirements_frozen.txt
python -m pip list --format=json > deployment/cts_20260921/verification/packages.json
