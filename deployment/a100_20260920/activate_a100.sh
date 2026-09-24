# Source this file in Bash; this environment is specific to the A100 host.
export FOURDSR_ROOT="${FOURDSR_ROOT:-/home/ubuntu/3DGS/4dsr}"
source "$FOURDSR_ROOT/.venv/bin/activate"
export FOURDSR_UPSTREAM="$FOURDSR_ROOT/vendor/4dgs"
export FOURDSR_SWINIR_NETWORK="$FOURDSR_ROOT/vendor/swinir/network_swinir.py"
export FOURDSR_SWINIR_CHECKPOINT="$FOURDSR_ROOT/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
export CUDA_HOME="$FOURDSR_ROOT/tools/cuda-12.8-minimal"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_HOME="$FOURDSR_ROOT/.cache/torch"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="4"
export MPLBACKEND="Agg"
export PYTHONFAULTHANDLER="1"
export FOURDSR_FFMPEG="$FOURDSR_ROOT/tools/ffmpeg/bin/ffmpeg"
# Select physical GPU 0 or 1 with CUDA_VISIBLE_DEVICES for each run.
# No default GPU is forced, so callers can choose either device explicitly.
