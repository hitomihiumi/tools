#!/usr/bin/env bash
# Builds vLLM from source (main branch, for GLM-4.7-Flash's glm4_moe_lite
# architecture support, which historically only lands there before a
# stable release), targeting CUDA 13 + Blackwell (SM 12.x/sm_120)
# explicitly via TORCH_CUDA_ARCH_LIST, then downloads the two model
# checkpoints (GLM 4.7 AWQ 4-bit + GLM 4.7 Flash FP8) and launches both as
# vLLM OpenAI-compatible servers.
#
# Assumes the pod image already has Python and CUDA 13 (x86_64) installed
# - everything else (build tooling, torch, vLLM itself, HF download
# tooling) is installed by this script. It does NOT set up the LLM-Hell
# backend itself (that runs on a separate VPS, not here).
#
# Use this script for a genuinely fresh pod, or when the vLLM build itself
# needs redoing (FORCE_REBUILD_VLLM=true, or bumping VLLM_GIT_REF). If
# vLLM is already installed and both checkpoints are already fully
# downloaded, `runpod_start.sh` skips straight to launching the servers -
# no reason to re-run the build/download machinery just to restart them.
#
# The built vLLM wheel is cached under VLLM_WHEEL_DIR (default:
# /workspace/vllm-wheels, i.e. on the pod's persistent volume if one is
# attached) - a rerun reuses it instead of rebuilding from scratch, which
# otherwise takes a long time (compiling vLLM's CUDA kernels for a single
# target architecture is still commonly 30-90+ minutes depending on CPU
# core count). Set FORCE_REBUILD_VLLM=true in runpod_common.sh to force a
# fresh build anyway (e.g. after bumping VLLM_GIT_REF).
#
# Safe to rerun: kills anything holding GPU memory (by PID, from
# nvidia-smi directly - not just a `vllm serve` command-line pattern
# match, which misses vLLM's renamed EngineCore/Worker subprocesses and
# was observed to leave a GPU stuck full of orphaned memory across
# reruns) before starting new servers.
#
# Default GPU split assumes 5x GPUs: 4 for the planner (GLM 4.7, tensor-
# parallel-size must divide its vocab_size so 3 GPUs doesn't work), 1 for
# the flash executor (GLM 4.7 Flash). Edit the CONFIG block in
# runpod_common.sh if your pod's GPU count/topology differs.
#
# Usage: bash runpod_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runpod_common.sh
source "$SCRIPT_DIR/common.sh"

mkdir -p "$LOG_DIR" "$VLLM_WHEEL_DIR"

runpod_kill_gpu_holders

echo "==> System CUDA toolkit on this pod:"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
"$CUDA_HOME/bin/nvcc" --version || { echo "!! nvcc not found at $CUDA_HOME/bin/nvcc - is CUDA 13 actually installed here?" >&2; exit 1; }
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "==> Architecture: $(uname -m) (expecting x86_64)"
if [ "$(uname -m)" != "x86_64" ]; then
    echo "!! This script assumes x86_64 - torch/vLLM build steps below are not set up for other architectures." >&2
    exit 1
fi

runpod_check_gpu_overlap
runpod_check_gpu_topology "planner" "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE"
runpod_check_gpu_topology "flash" "$FLASH_GPUS" "$FLASH_TP_SIZE" "$FLASH_PP_SIZE"

# ---------------------------------------------------------------------------
# Build (or reuse a cached build of) vLLM from source
# ---------------------------------------------------------------------------

existing_wheel="$(ls "$VLLM_WHEEL_DIR"/vllm-*.whl 2>/dev/null | head -1 || true)"

if [ -n "$existing_wheel" ] && [ "$FORCE_REBUILD_VLLM" != "true" ]; then
    echo "==> Reusing cached vLLM wheel: $existing_wheel"
else
    echo "==> Building vLLM from source (ref: $VLLM_GIT_REF, arch: $TORCH_CUDA_ARCH_LIST) - this can take a long time on a cold cache"

    echo "==> Installing build tooling"
    apt-get update -qq
    # cmake/ninja also come from requirements/build/cuda.txt below (pip
    # packages that bundle working binaries), but build-essential
    # (gcc/g++/make) and git aren't available as pip packages, and ccache
    # speeds up a future rebuild against the same object files.
    apt-get install -y -qq git build-essential ccache curl >/dev/null

    # setuptools-rust (in requirements/build/cuda.txt below) needs an
    # actual Rust toolchain present - rustup rather than a distro package
    # so the version is new enough regardless of which base image this is.
    if ! command -v cargo >/dev/null 2>&1; then
        echo "==> Installing Rust toolchain (needed by setuptools-rust)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi

    pip install -q -U uv

    if [ -d "$VLLM_SRC_DIR/.git" ]; then
        echo "==> Updating existing vLLM checkout"
        git -C "$VLLM_SRC_DIR" fetch --depth 1 origin "$VLLM_GIT_REF"
        git -C "$VLLM_SRC_DIR" checkout FETCH_HEAD
    else
        echo "==> Cloning vLLM ($VLLM_GIT_REF)"
        git clone --branch "$VLLM_GIT_REF" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
    fi
    vllm_commit="$(git -C "$VLLM_SRC_DIR" rev-parse --short HEAD)"
    echo "==> Building commit $vllm_commit"

    # --break-system-packages: fine on a throwaway pod rebuilt from the
    # image rather than hand-maintained; do not use this on a machine you
    # upgrade in place. --index-strategy unsafe-best-match: without it, uv
    # can silently prefer a plain-PyPI torch build over the cu130 one on
    # --extra-index-url even when a specific version is named - this
    # requirements file pins the exact torch version (2.13.0, checked
    # against the actual file at build time, not hardcoded here) vLLM's
    # build expects, and we need that exact version resolved from the
    # cu130 index, not whatever plain PyPI happens to have.
    uv pip install --system --break-system-packages --index-strategy unsafe-best-match \
        -r "$VLLM_SRC_DIR/requirements/build/cuda.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu130

    export TORCH_CUDA_ARCH_LIST
    export MAX_JOBS
    export NVCC_THREADS
    export VLLM_TARGET_DEVICE=cuda

    (
        cd "$VLLM_SRC_DIR"
        python3 -m build --wheel --no-isolation -o "$VLLM_WHEEL_DIR"
    ) 2>&1 | tee "$LOG_DIR/vllm-build.log"

    # No renaming needed: setuptools-scm already bakes the commit into the
    # wheel's own version segment (e.g. "vllm-0.1.dev1+g30b4e7f47-...whl"),
    # which is where a "+something" is actually valid in a wheel filename
    # - appending our own after the platform tag instead produced an
    # invalid filename that `pip`/`uv` correctly refused to install.
    existing_wheel="$(ls "$VLLM_WHEEL_DIR"/vllm-*.whl 2>/dev/null | head -1 || true)"
    if [ -z "$existing_wheel" ]; then
        echo "!! Build finished but no wheel found in $VLLM_WHEEL_DIR - check $LOG_DIR/vllm-build.log" >&2
        exit 1
    fi
    echo "==> Built and cached: $existing_wheel (commit $vllm_commit)"
fi

echo "==> Installing vLLM from wheel: $existing_wheel"
pip install -q -U uv
uv pip install --system --break-system-packages --force-reinstall "$existing_wheel"

echo "==> vLLM version now installed:"
vllm --version

echo "==> Installing Hugging Face download tooling"
# No "[cli]" extra: huggingface_hub 1.x folded the `hf` CLI into the base
# package and no longer has an extra by that name (uv just warns and
# installs anyway if you ask for it, but there's no reason to keep asking).
uv pip install --system --break-system-packages -U huggingface_hub hf_transfer

mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

# huggingface_hub now downloads through hf-xet (its new storage backend) by
# default whenever it's installed, in preference to hf_transfer/plain HTTP -
# and hf-xet is known to fail reconstructing large files (>15GB, which
# every shard of these checkpoints exceeds) with errors like "File
# reconstruction error: ... receiver dropped" or "Background writer channel
# closed" (https://github.com/huggingface/xet-core/issues/763). Setting
# HF_HUB_DISABLE_XET=1 is not reliable by itself (a huggingface_hub bug
# still routes through xet regardless: https://github.com/huggingface/
# huggingface_hub/issues/3266) - actually uninstalling the package is the
# only fix confirmed to work, forcing a fall back to plain HTTP/hf_transfer.
uv pip uninstall --system hf_xet 2>/dev/null || true
export HF_HUB_DISABLE_XET=0

# One-time migration: earlier runs before HF_HOME pointed at /workspace may
# have left partial/complete downloads under the default root-disk cache -
# harmless to leave, but they're dead weight eating the ~30GB root disk for
# no benefit once every download goes through $HF_HOME instead, so reclaim
# the space if any of that leftover state is still there.
for stale_dir in \
    "$HOME/.cache/huggingface/hub/models--${PLANNER_MODEL_REPO/\//--}" \
    "$HOME/.cache/huggingface/hub/models--${FLASH_MODEL_REPO/\//--}"; do
    if [ -d "$stale_dir" ]; then
        echo "==> Removing stale root-disk HF cache entry: $stale_dir"
        rm -rf "$stale_dir"
    fi
done

echo "==> Downloading $PLANNER_MODEL_REPO"
hf download "$PLANNER_MODEL_REPO" 2>&1 | tee "$LOG_DIR/download-planner.log"

echo "==> Downloading $FLASH_MODEL_REPO"
hf download "$FLASH_MODEL_REPO" 2>&1 | tee "$LOG_DIR/download-flash.log"

start_vllm "planner" "$PLANNER_MODEL_REPO" "$PLANNER_SERVED_NAME" "$PLANNER_PORT" \
    "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE" "$PLANNER_QUANTIZATION" "$PLANNER_KV_CACHE_DTYPE" "$PLANNER_MAX_MODEL_LEN" ""

start_vllm "flash" "$FLASH_MODEL_REPO" "$FLASH_SERVED_NAME" "$FLASH_PORT" \
    "$FLASH_GPUS" "$FLASH_TP_SIZE" "$FLASH_PP_SIZE" "$FLASH_QUANTIZATION" "$FLASH_KV_CACHE_DTYPE" "$FLASH_MAX_MODEL_LEN" "$FLASH_ENFORCE_EAGER"

wait_for_health "planner" "$PLANNER_PORT"
wait_for_health "flash" "$FLASH_PORT"

runpod_print_summary
echo "    vLLM wheel cache: ${VLLM_WHEEL_DIR} (reused on the next run unless FORCE_REBUILD_VLLM=true)"
echo "    HF model cache  : ${HF_HOME} (reused on the next run - or use start.sh to skip straight to launching)"
