#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/3DGS/4dsr
trap 'rc=$?; printf "{\"returncode\":%s}\n" "$rc" > deployment/a100_20260920/install_exit.json' EXIT
# The host's existing local HTTP proxy is used only for package acquisition.
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="127.0.0.1,localhost,.tsinghua.edu.cn"
export no_proxy="$NO_PROXY"
UV=/home/ubuntu/.local/bin/uv
PY=/home/ubuntu/3DGS/4dsr/.venv/bin/python
if compgen -G 'deployment/wheelhouse/*.whl' > /dev/null; then
  "$UV" pip install --python "$PY" --no-deps deployment/wheelhouse/*.whl \
    > deployment/logs/local_wheels_install.log 2>&1
fi
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl' \
  'https://download.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl' \
  > deployment/logs/torch_install_proxy.log 2>&1
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  setuptools==80.9.0 wheel pip ninja==1.11.1.4 > deployment/logs/build_requirements.log 2>&1
# MMCV 1.6 imports pkg_resources at build time; use the pinned setuptools.
"$UV" pip install --python "$PY" --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r deployment/a100_20260920/requirements.txt > deployment/logs/dependencies_install.log 2>&1
mkdir -p tools/ffmpeg/bin
ln -sfn "$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" tools/ffmpeg/bin/ffmpeg
source deployment/a100_20260920/activate_a100.sh
"$UV" pip install --python "$PY" --no-build-isolation \
  -e vendor/4dgs/submodules/depth-diff-gaussian-rasterization \
  -e vendor/4dgs/submodules/simple-knn > deployment/logs/extensions_build.log 2>&1
"$UV" pip check --python "$PY" > deployment/a100_20260920/pip_check.txt 2>&1
"$UV" pip freeze --python "$PY" > deployment/a100_20260920/requirements_frozen.txt
"$PY" deployment/a100_20260920/smoke_environment.py > deployment/logs/environment_smoke.log 2>&1
printf 'Environment installation and smoke test completed\n'
