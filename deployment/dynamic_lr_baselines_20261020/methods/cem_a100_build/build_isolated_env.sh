#!/usr/bin/env bash
# Build only in a new leaf environment. This script never starts training.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo 'usage: build_isolated_env.sh {localdygs|cem} BASE_PYTHON VENV UPSTREAM CUDA_HOME' >&2
  exit 2
fi
method=$1
base_python=$2
venv_dir=$3
upstream_dir=$4
cuda_root=$5
[[ "$method" == localdygs || "$method" == cem ]] || exit 2
[[ -x "$base_python" && -x "$cuda_root/bin/nvcc" && -d "$upstream_dir/.git" ]] || exit 2
base_site=$($base_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
# The deployed Python may omit Debian ensurepip. Borrow base pip read-only
# through the .pth below instead of requiring a system package installation.
if [[ -e "$venv_dir" ]]; then
  [[ "${FOURDSR_ENV_RESUME:-0}" == 1 && -x "$venv_dir/bin/python" ]] || { echo "Refusing existing environment without explicit build resume: $venv_dir" >&2; exit 2; }
else
  "$base_python" -m venv --without-pip "$venv_dir"
fi
python="$venv_dir/bin/python"
leaf_site=$($python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$base_site" > "$leaf_site/remote_base_torch_readonly.pth"
if [[ "${FOURDSR_CUDA_OVERLAY:-0}" == 1 ]]; then
  "$base_python" "$(dirname "$0")/prepare_cuda_overlay.py" --compiler-root "$cuda_root" --base-site "$base_site" --output "$venv_dir/cuda-toolkit-overlay"
  cuda_root="$venv_dir/cuda-toolkit-overlay"
fi
export CUDA_HOME="$cuda_root"
export PATH="$venv_dir/bin:$cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$cuda_root/include:${CPATH:-}"
export LIBRARY_PATH="$cuda_root/lib64:${LIBRARY_PATH:-}"
export CUDAToolkit_ROOT="$cuda_root"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80;86}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PIP_REQUIRE_VIRTUALENV=true
mkdir -p "$venv_dir/build_sources"
"$python" -m pip install setuptools wheel ninja packaging
if ! command -v cmake >/dev/null; then
  "$python" -m pip install cmake
fi
if [[ "$method" == cem ]] && ! "$python" -c 'import subprocess; from packaging.version import Version; assert Version(subprocess.check_output(["cmake","--version"],text=True).split()[2]) >= Version("3.24")'; then
  "$python" -m pip install 'cmake>=3.24,<4'
fi
"$python" -c 'import torch; assert torch.__version__.startswith("2.7.1"), torch.__version__; print(torch.__version__, torch.version.cuda)'

if [[ "$method" == localdygs ]]; then
  "$python" -m pip install 'numpy<2' pillow scipy tqdm plyfile matplotlib colorama 'mmcv==1.6.0' einops wandb lpips laspy torchmetrics jaxtyping pytorch-msssim 'imageio[ffmpeg]' opencv-python tensorboard open3d
  if [[ -n "${TORCH_SCATTER_WHEEL:-}" ]]; then
    [[ "$(sha256sum "$TORCH_SCATTER_WHEEL" | cut -d ' ' -f 1)" == "3a20acd4777bc1d996756df5bb0c2ac42440fd434315e37464715567699b4bfb" ]] || exit 2
    "$python" -m pip install --no-deps "$TORCH_SCATTER_WHEEL"
  else
    "$python" -m pip install --no-build-isolation --no-deps torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  fi
  # Upstream calls this version 1.7 but does not publish a v1.7 Git tag.
  # This verified commit declares VERSION 1.7 and uses C++17 for modern torch.
  if [[ "${FOURDSR_DEPS_PREVERIFIED:-0}" != 1 ]]; then
    git init "$venv_dir/build_sources/tiny-cuda-nn"
    git -C "$venv_dir/build_sources/tiny-cuda-nn" remote add origin https://github.com/NVlabs/tiny-cuda-nn.git
    git -C "$venv_dir/build_sources/tiny-cuda-nn" fetch --depth 1 origin 3294b150084266f24a04009d47dcb530d0a8fb03
    git -C "$venv_dir/build_sources/tiny-cuda-nn" checkout --detach FETCH_HEAD
    git -C "$venv_dir/build_sources/tiny-cuda-nn" submodule update --init --recursive --depth 1
  fi
  [[ "$(git -C "$venv_dir/build_sources/tiny-cuda-nn" rev-parse HEAD)" == 3294b150084266f24a04009d47dcb530d0a8fb03 ]]
  if [[ "${FOURDSR_REUSE_INSTALLED_TCNN:-0}" == 1 ]]; then
    "$python" -c 'from importlib.metadata import version; assert version("tinycudann") == "1.7"'
  else
    "$python" -m pip install --no-build-isolation --no-deps "$venv_dir/build_sources/tiny-cuda-nn/bindings/torch"
  fi
  "$python" "$(dirname "$0")/prepare_simple_knn_build.py" --source "$upstream_dir/submodules/simple-knn" --output "$venv_dir/build_sources/simple-knn-cfloat"
  "$python" -m pip install --no-build-isolation --no-deps "$upstream_dir/submodules/diff-gaussian-rasterization" "$venv_dir/build_sources/simple-knn-cfloat"
  # tinycudann's top-level import queries a real CUDA device. Verify its
  # installed module/binary here; its GPU import runs only inside the broker.
  "$python" -c 'import torch, torch_scatter, diff_gaussian_rasterization, simple_knn, importlib.util, pathlib; assert importlib.util.find_spec("tinycudann"); import tinycudann_bindings; assert list(pathlib.Path(tinycudann_bindings.__path__[0]).glob("*_C*.so")); print("LocalDyGS CPU import/binary acceptance passed; tinycudann GPU import deferred to broker; no CUDA allocation performed")'
else
  # GUI-only packages from the broad upstream requirements are not imported by
  # train/render. Install the actual headless runtime plus native build tools.
  "$python" -m pip install 'numpy<2' pillow scipy tqdm plyfile matplotlib joblib natsort opencv-python kornia scikit-image scikit-learn lpips tensorboard fvcore iopath swig
  if [[ "${FOURDSR_DEPS_PREVERIFIED:-0}" != 1 ]]; then
    git clone --branch v0.7.9 --depth 1 https://github.com/facebookresearch/pytorch3d.git "$venv_dir/build_sources/pytorch3d"
  fi
  [[ "$(git -C "$venv_dir/build_sources/pytorch3d" rev-parse HEAD)" == 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba ]]
  FORCE_CUDA=1 "$python" -m pip install --no-build-isolation --no-deps "$venv_dir/build_sources/pytorch3d"
  if [[ "${FOURDSR_DEPS_PREVERIFIED:-0}" != 1 ]]; then
    git clone --branch v1.12.0 --depth 1 https://github.com/facebookresearch/faiss.git "$venv_dir/build_sources/faiss"
  fi
  [[ "$(git -C "$venv_dir/build_sources/faiss" rev-parse HEAD)" == e8234e563f1ecef5f036e83c3cfee366d3f1fbca ]]
  cmake -S "$venv_dir/build_sources/faiss" -B "$venv_dir/build_sources/faiss/build" \
    -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_C_API=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${FAISS_CUDA_ARCHITECTURES:-86}" -DPython_EXECUTABLE="$python" \
    -DSWIG_EXECUTABLE="$(command -v swig)"
  cmake --build "$venv_dir/build_sources/faiss/build" --target swigfaiss --parallel "$MAX_JOBS"
  "$python" -m pip install --no-build-isolation --no-deps "$venv_dir/build_sources/faiss/build/faiss/python"
  "$python" -m pip install --no-build-isolation --no-deps "$upstream_dir/submodules/diff_gaussian_rasterization_df" "$upstream_dir/submodules/simple-knn"
  "$python" -c 'import torch, faiss, pytorch3d, diff_gaussian_rasterization_df, simple_knn; assert hasattr(faiss,"StandardGpuResources"); assert hasattr(faiss,"GpuIndexFlatL2"); print("CEM import acceptance passed; no CUDA allocation performed")'
fi
"$python" -m pip freeze > "$venv_dir/installed_packages.txt"
git -C "$upstream_dir" rev-parse HEAD > "$venv_dir/upstream_commit.txt"
"$cuda_root/bin/nvcc" --version > "$venv_dir/compiler.txt"
echo "Environment built: $venv_dir"
