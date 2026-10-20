#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
project=$1
source_dir=$2
venv="$project/envs/dynamic_lr_baselines_20261020/mast3r"
base_python="$project/.venv/bin/python"
if [ ! -e "$venv" ]; then
  "$base_python" -m venv --without-pip --system-site-packages "$venv"
fi
base_sites=$("$base_python" -c 'import site;print("\n".join(site.getsitepackages()))')
target_site=$("$venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$base_sites" > "$target_site/registered_base_environment.pth"
# Pin Torch family and NumPy to the read-only base. Dependencies install only
# in this separate MASt3R preprocessing environment.
"$base_python" -c 'import importlib.metadata as m;from pathlib import Path;Path("'$venv'/base_constraints.txt").write_text("\n".join(f"{n}=={m.version(n)}" for n in ["torch","torchvision","numpy"]))'
# ASMK's original NumPy distutils compiler imports distutils.msvccompiler.
# Setuptools 74+ removed that compatibility module. Pin only this venv's
# build frontend; leave the shared base and original ASMK source unchanged.
"$venv/bin/python" -m pip install setuptools==69.5.1
"$venv/bin/python" -m pip install -c "$venv/base_constraints.txt" gradio==4.44.1 huggingface-hub==0.29.3 timm==0.9.16 einops roma==1.5.0 faiss-cpu==1.10.0 open3d==0.18.0 trimesh pyglet==1.5.29 pyaml cython scikit-learn
"$venv/bin/python" -m pip install --no-build-isolation --no-deps "$source_dir/mast3r/asmk"
# MAtCha's COLMAP reader imports focal2fov/fov2focal from a module which also
# imports the pure-Python Transform3d class. MASt3R-SfM never invokes any
# PyTorch3D native operator. Install the author's 0.7.4 transform dependency
# with the official no-extension build option; do not replace its functions.
pytorch3d_source="$project/third_party/mast3r_dependencies_20261020/pytorch3d"
test "$(git -C "$pytorch3d_source" rev-parse HEAD)" = 297020a4b1d7492190cb4a909cafbd2c81a12cb5
"$venv/bin/python" -m pip install fvcore==0.1.5.post20221221 iopath==0.1.10
PYTORCH3D_NO_EXTENSION=1 "$venv/bin/python" -m pip install --no-build-isolation --no-deps "$pytorch3d_source"
cd "$source_dir"
export CUDA_VISIBLE_DEVICES=""
"$venv/bin/python" mast3r/run_mast3r.py --help
"$venv/bin/python" -m pip freeze > "$venv/installed_packages.txt"
