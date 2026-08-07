#!/usr/bin/env bash
# Launches the server WITHOUT installing or downloading anything - use this
# once setup.sh has run on a pod. It is what you want after a pod restart:
# the server runs as a plain nohup'd background process, so restarting the
# pod kills it while leaving the venv and the checkpoint on /workspace
# intact.
#
# Fails fast with a clear message rather than trying to fix anything itself
# if the venv, vLLM or the checkpoint is missing - run setup.sh in that case.
#
# Usage:
#   bash start.sh                              # DeepSeek V4 Flash (default)
#   CONFIG_FILE=qwen_common.sh bash start.sh   # a different model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-ds_common.sh}"
echo "==> Using config: $CONFIG_FILE"
# shellcheck source=./ds_common.sh
source "$SCRIPT_DIR/$CONFIG_FILE"

mkdir -p "$LOG_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "!! No venv at $VENV_DIR - run setup.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! command -v vllm >/dev/null 2>&1; then
    echo "!! 'vllm' not found in $VENV_DIR - run setup.sh first." >&2
    exit 1
fi

# Clear the GPUs BEFORE printing the version: `vllm --version` imports the
# whole package, and vLLM's platform detection queries CUDA during import.
# With a previous run's workers still holding the devices that query can
# block indefinitely, which reads as the script hanging on the version line.
runpod_kill_gpu_holders

echo "==> vLLM version:"
if ! timeout 300 vllm --version; then
    echo "!! 'vllm --version' did not finish in 300s - continuing anyway." >&2
fi

# nvcc is a runtime dependency here, not a build one: vLLM JIT-compiles
# FlashInfer/DeepGEMM/Triton kernels on first use. Warn rather than fail -
# a pod that has already served this model once has its JIT cache warm and
# may not touch nvcc again.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "!! nvcc not on PATH - JIT kernel compilation will fail on a cold cache." >&2
    echo "!! Run setup.sh to install the CUDA toolkit if startup errors out." >&2
fi

# The repo counts as present if it has a non-empty snapshot directory under
# $HF_HOME. Not a byte-for-byte check - that is what `hf download`'s own
# resume logic is for - just enough to tell "never downloaded" from "ready".
echo "==> Checking the checkpoint is present in \$HF_HOME ($HF_HOME)"
export HF_HOME
model_dir="$HF_HOME/hub/models--${MODEL_REPO/\//--}"
# -type f alone is wrong here: the HF cache stores snapshot entries as
# SYMLINKS into ../../blobs/, and -type f does not match symlinks - so a
# fully downloaded model was reported as missing. Accept either, since all
# we need to know is whether anything was ever fetched.
if [ ! -d "$model_dir/snapshots" ] ||
   [ -z "$(find "$model_dir/snapshots" -mindepth 2 \( -type f -o -type l \) -print -quit 2>/dev/null)" ]; then
    echo "!! Not downloaded: $MODEL_REPO (looked in $model_dir)" >&2
    echo "!! Run setup.sh first." >&2
    exit 1
fi
echo "==> Checkpoint present"

# GPUs were already cleared above, before the vllm import.
runpod_check_gpu_topology

start_vllm
wait_for_health

runpod_print_summary
