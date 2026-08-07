#!/usr/bin/env bash
# Installs everything needed to serve a model, then downloads it and starts
# the server. Follows the setup in
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# verbatim: uv, a Python 3.12 venv, `uv pip install vllm`, and the CUDA
# toolkit that vLLM's JIT needs at runtime.
#
# NOTHING IS BUILT FROM SOURCE. This script used to clone vLLM and DeepGEMM
# and compile CUDA kernels for 30-90 minutes, to work around SM120 aborts
# ("Unknown SF transformation" / "Unsupported architecture") in the DeepGEMM
# revision the old fork pinned. vLLM 0.26.0 vendors DeepGEMM kernels that
# handle SM120 - a startup log line reading
#   Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM
# is that fix working. So the fork, the DeepGEMM checkout, the wheel cache,
# TORCH_CUDA_ARCH_LIST/MAX_JOBS/NVCC_THREADS and the corrupt-dist-info
# cleanup are all gone.
#
# vLLM lives in a venv rather than being installed system-wide with
# --break-system-packages, which is what the guide does and also avoids the
# class of breakage this project hit repeatedly: a stale system tree
# shadowing a fresh install (the quantization/inc/ vs inc.py ImportError),
# and half-removed dist-info directories poisoning later installs.
#
# Safe to rerun: each step is skipped when already satisfied, and anything
# holding GPU memory is killed before the server starts.
#
# Usage:
#   bash setup.sh                              # DeepSeek V4 Flash (default)
#   CONFIG_FILE=qwen_common.sh bash setup.sh   # a different model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Which model to serve. Defaults to the DeepSeek config so existing
# invocations keep working.
CONFIG_FILE="${CONFIG_FILE:-ds_common.sh}"
echo "==> Using config: $CONFIG_FILE"
# shellcheck source=./ds_common.sh
source "$SCRIPT_DIR/$CONFIG_FILE"

mkdir -p "$LOG_DIR"

echo "==> Architecture: $(uname -m) (expecting x86_64)"
if [ "$(uname -m)" != "x86_64" ]; then
    echo "!! This script assumes x86_64." >&2
    exit 1
fi

# Before anything imports vllm, not just before the server starts: importing
# the package runs platform detection that queries CUDA, and a previous
# run's workers still holding the GPUs can make that block indefinitely -
# which looks exactly like the script hanging on `vllm --version`.
runpod_kill_gpu_holders

# ---------------------------------------------------------------------------
# CUDA toolkit
# ---------------------------------------------------------------------------
# Needed at RUNTIME, not for building anything: vLLM JIT-compiles FlashInfer,
# DeepGEMM and Triton kernels on first use and needs nvcc to do it. The
# driver alone is not enough.
if command -v nvcc >/dev/null 2>&1 || [ -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "==> CUDA toolkit already present:"
    "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)" --version | tail -2
else
    echo "==> Installing CUDA toolkit $CUDA_TOOLKIT_VERSION"
    # Derive the repo path from this image rather than hardcoding one: the
    # guide's URL names its own Ubuntu release, and using it on a different
    # release installs packages built against the wrong glibc.
    distro_id="$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')"
    keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro_id}/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "    repo: $distro_id"
    tmp_deb="$(mktemp --suffix=.deb)"
    if ! curl -fsSL -o "$tmp_deb" "$keyring_url"; then
        echo "!! No NVIDIA CUDA repo for '$distro_id' at $keyring_url" >&2
        echo "!! Check https://developer.download.nvidia.com/compute/cuda/repos/ for the right name," >&2
        echo "!! or install the toolkit however this image expects." >&2
        exit 1
    fi
    dpkg -i "$tmp_deb"
    rm -f "$tmp_deb"
    apt-get update -qq
    apt-get install -y -qq "cuda-toolkit-${CUDA_TOOLKIT_VERSION//./-}"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# uv + venv + vLLM
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# The installer drops uv in ~/.local/bin, which a non-login shell may not
# have on PATH. The guide re-execs the shell for this; sourcing the env file
# it writes achieves the same thing without restarting the script.
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
uv --version

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at $VENV_DIR (Python $VENV_PYTHON)"
    mkdir -p "$(dirname "$VENV_DIR")"
    # --managed-python: uv fetches its own interpreter rather than using
    # whatever the image ships, so the version is the same on every pod.
    # --seed: pre-installs pip/setuptools, which some vLLM extras expect.
    uv venv --python "$VENV_PYTHON" --seed --managed-python "$VENV_DIR"
else
    echo "==> Reusing existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Bounded for the same reason as the version check below: this import is
# not cheap and can block on a busy GPU. A timeout is treated as "not
# installed", which at worst reinstalls something already present.
if timeout 300 python -c "import vllm" 2>/dev/null && [ "$FORCE_REINSTALL_VLLM" != "true" ]; then
    echo "==> vLLM already installed: $(timeout 300 python -c 'import vllm; print(vllm.__version__)' || echo unknown)"
else
    echo "==> Installing vLLM${VLLM_VERSION:+==$VLLM_VERSION}"
    # --torch-backend=auto lets uv pick the torch build matching the driver
    # on this machine; pinning a CUDA-suffixed wheel by hand is what
    # previously left the wrong torch installed and produced
    # "SM 12.x requires CUDA >= 12.9" despite a correct-looking install.
    uv pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}" --torch-backend=auto
fi

echo "==> vLLM version:"
# Bounded: this imports the entire package (torch included) and probes the
# GPU, which is slow on a cold cache and has hung outright when the devices
# were still busy. A timeout here is informative, not fatal - the server
# launch below is the real test.
if ! timeout 300 vllm --version; then
    echo "!! 'vllm --version' did not finish in 300s." >&2
    echo "!! Usually means the GPUs are still busy - check \`nvidia-smi\`. Continuing anyway." >&2
fi

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if ! command -v hf >/dev/null 2>&1; then
    echo "==> Installing Hugging Face CLI"
    uv pip install huggingface_hub
fi

mkdir -p "$HF_HOME"
export HF_HOME

# Checkpoints here run 167-230 GiB and /workspace on RunPod is a network
# volume whose usable quota can sit well below the size the dashboard shows -
# check before spending an hour downloading into a wall.
echo "==> Free space on \$HF_HOME's volume:"
df -h "$HF_HOME"

echo "==> Downloading $MODEL_REPO"
hf download "$MODEL_REPO" 2>&1 | tee "$LOG_DIR/download.log"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# GPUs were already cleared at the top of this script.
runpod_check_gpu_topology

start_vllm
wait_for_health

runpod_print_summary
echo "    venv       : ${VENV_DIR}  (activate it to run vllm by hand)"
echo "    HF cache   : ${HF_HOME}"
echo "    next time  : bash start.sh  - skips install and download entirely"
