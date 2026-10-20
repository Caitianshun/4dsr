#!/usr/bin/env bash
# Build only an isolated environment. CUDA compile is CPU work; no CUDA model
# or training is executed. Root supplies the correct toolkit and GPU arch.
set -euo pipefail
export PYTHONNOUSERSITE=1
base_python=$1
runtime=$2
venv=$3
method=$4
test -n "${CUDA_HOME:?Set private CUDA toolkit directory}"
test -n "${TORCH_CUDA_ARCH_LIST:?Set 8.0 for A100 or 8.6 for CTS}"
if [ -e "$venv" ]; then
  test -f "$venv/pyvenv.cfg"
  test -x "$venv/bin/python"
else
  "$base_python" -m venv --without-pip --system-site-packages "$venv"
fi
base_sites=$("$base_python" -c 'import site; print("\n".join(site.getsitepackages()))')
target_site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$base_sites" > "$target_site/registered_base_environment.pth"
"$venv/bin/python" -m pip install --no-deps imagesize omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 kornia==0.6.12 torchmetrics==0.11.4 plyfile tensorboard ninja
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
compiler_headers=$("$base_python" -c 'import site;from pathlib import Path;print(":".join(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/include") if p.is_dir()))')
export CPATH="$compiler_headers${CPATH:+:$CPATH}"
"$venv/bin/python" -m pip install --no-build-isolation --no-deps "$runtime/simple-knn"
"$venv/bin/python" -m pip install --no-build-isolation --no-deps "$runtime/pointops2"
if [ "$method" = 4c4d ]; then
  "$venv/bin/python" -m pip install --no-build-isolation --no-deps "$runtime/diff-gaussian-rasterization"
  "$venv/bin/python" -m pip install --no-build-isolation --no-deps "$runtime/fused-ssim-main"
fi
"$venv/bin/python" -c 'import torch, torchvision, cv2, scipy, matplotlib, PIL, plyfile, omegaconf, kornia, imagesize, torchmetrics; from simple_knn._C import distCUDA2; print("environment imports passed", torch.__version__)'
