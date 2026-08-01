#!/usr/bin/env bash
# Launches the planner + flash vLLM servers WITHOUT touching the build or
# download steps - use this instead of runpod_setup.sh once vLLM is
# already installed and both checkpoints are already fully downloaded
# (e.g. restarting after a pod reboot, or after killing the servers to
# free GPU memory for something else). Re-running the full setup script
# every time re-verifies/re-fetches things that are already sitting on
# disk for no benefit, and on a slow connection or a flaky HF backend
# that's wasted time at best.
#
# Fails fast with a clear message (rather than trying to download
# anything itself) if vLLM isn't installed or a checkpoint isn't fully
# present - run setup.sh first in that case.
#
# Usage: bash runpod_start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runpod_common.sh
source "$SCRIPT_DIR/common.sh"

mkdir -p "$LOG_DIR"

echo "==> Checking vLLM is installed"
if ! command -v vllm >/dev/null 2>&1; then
    echo "!! 'vllm' command not found - run setup.sh first to build/install it." >&2
    exit 1
fi
vllm --version

# A repo is "present" if it has at least one non-empty snapshot directory
# under $HF_HOME - this isn't a byte-for-byte completeness check (that's
# what `hf download`'s own resume logic is for), just a fast, good-enough
# signal that a full `hf download` already ran here before. Missing means
# it definitely hasn't; present-but-partial (interrupted mid-download) is
# rare enough, and cheap enough to catch via the health-check failing
# below, that it's not worth this script re-implementing hf_hub's own
# manifest verification.
_model_present() {
    local repo="$1"
    local model_dir="$HF_HOME/hub/models--${repo/\//--}"
    [ -d "$model_dir/snapshots" ] && [ -n "$(find "$model_dir/snapshots" -mindepth 2 -type f -print -quit 2>/dev/null)" ]
}

echo "==> Checking both checkpoints are already downloaded to \$HF_HOME ($HF_HOME)"
missing=""
_model_present "$PLANNER_MODEL_REPO" || missing="$missing $PLANNER_MODEL_REPO"
_model_present "$FLASH_MODEL_REPO" || missing="$missing $FLASH_MODEL_REPO"
if [ -n "$missing" ]; then
    echo "!! Not downloaded yet:$missing" >&2
    echo "!! Run setup.sh first - it builds vLLM (if needed) and downloads both checkpoints." >&2
    exit 1
fi
echo "==> Both checkpoints present"

runpod_kill_gpu_holders
runpod_check_gpu_overlap
runpod_check_gpu_topology "planner" "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE"
runpod_check_gpu_topology "flash" "$FLASH_GPUS" "$FLASH_TP_SIZE" "$FLASH_PP_SIZE"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

start_vllm "planner" "$PLANNER_MODEL_REPO" "$PLANNER_SERVED_NAME" "$PLANNER_PORT" \
    "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE" "$PLANNER_QUANTIZATION" "$PLANNER_KV_CACHE_DTYPE" "$PLANNER_MAX_MODEL_LEN" ""

start_vllm "flash" "$FLASH_MODEL_REPO" "$FLASH_SERVED_NAME" "$FLASH_PORT" \
    "$FLASH_GPUS" "$FLASH_TP_SIZE" "$FLASH_PP_SIZE" "$FLASH_QUANTIZATION" "$FLASH_KV_CACHE_DTYPE" "$FLASH_MAX_MODEL_LEN" "$FLASH_ENFORCE_EAGER"

wait_for_health "planner" "$PLANNER_PORT"
wait_for_health "flash" "$FLASH_PORT"

runpod_print_summary
