#!/usr/bin/env bash
# Launches the vLLM server WITHOUT touching the build or download steps -
# use this instead of setup.sh once vLLM is already installed and the
# checkpoint is already fully downloaded (e.g. restarting after a pod
# reboot, or after killing the server to free GPU memory for something
# else). Re-running the full setup script every time re-verifies/re-fetches
# things that are already sitting on disk for no benefit, and on a slow
# connection or a flaky HF backend that's wasted time at best.
#
# This is the script to reach for after a RunPod restart: the servers run as
# plain nohup'd background processes, not a service, so a pod restart kills
# them while leaving /workspace (and therefore the checkpoint) intact.
#
# Fails fast with a clear message (rather than trying to download anything
# itself) if vLLM isn't installed or the checkpoint isn't present - run
# setup.sh first in that case.
#
# Usage: bash start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/ds_common.sh"

mkdir -p "$LOG_DIR"

echo "==> Checking vLLM is installed"
if ! command -v vllm >/dev/null 2>&1; then
    echo "!! 'vllm' command not found - run setup.sh first to build/install it." >&2
    exit 1
fi
vllm --version

# The repo counts as "present" if it has at least one non-empty snapshot
# directory under $HF_HOME - not a byte-for-byte completeness check (that's
# what `hf download`'s own resume logic is for), just a fast, good-enough
# signal that a full `hf download` already ran here before. Missing means it
# definitely hasn't; present-but-partial (interrupted mid-download) is rare
# enough, and cheap enough to catch via the health check failing below, that
# it isn't worth re-implementing hf_hub's own manifest verification here.
echo "==> Checking the checkpoint is already downloaded to \$HF_HOME ($HF_HOME)"
model_dir="$HF_HOME/hub/models--${MODEL_REPO/\//--}"
if [ ! -d "$model_dir/snapshots" ] || [ -z "$(find "$model_dir/snapshots" -mindepth 2 -type f -print -quit 2>/dev/null)" ]; then
    echo "!! Not downloaded yet: $MODEL_REPO (looked in $model_dir)" >&2
    echo "!! Run setup.sh first - it builds vLLM (if needed) and downloads the checkpoint." >&2
    exit 1
fi
echo "==> Checkpoint present"

runpod_kill_gpu_holders
runpod_check_gpu_topology

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

start_vllm
wait_for_health

runpod_print_summary
