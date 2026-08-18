#!/usr/bin/env bash
# Standalone: installs everything, downloads DeepSeek V4 Flash, runs it.
# No other files needed. 4 GPUs, no CPU offloading.
#
# Follows https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# (that report is 2 GPUs; this is the same recipe with data-parallel 4).
#
# Usage:  bash run_deepseek.sh
# Runs in the foreground. To background it:  nohup bash run_deepseek.sh &

set -euo pipefail

MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
SERVED_NAME="deepseek-v4-flash"
PORT=8000
GPUS="0,1,2,3"
GPU_COUNT=4
MAX_MODEL_LEN=524288

# Which torch build to install. "auto" resolves it from the driver, which is
# right when it works; override when it does not - the step below says so
# explicitly rather than leaving you to guess.
#   TORCH_BACKEND=cu129 bash run_deepseek.sh
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

# On /workspace, not $HOME: a pod's home directory is on the ephemeral root
# overlay and is wiped on restart. The venv is several GB and the checkpoint
# is ~167 GiB - neither is worth re-fetching every time.
export HF_HOME=/workspace/hf-cache
export VLLM_CACHE_ROOT=/workspace/vllm-cache
VENV=/workspace/serving/.venv

echo "=============================================================="
echo " 1/6  Open-file limit"
echo "=============================================================="
# THIS IS THE ONE THAT BIT US. Data-parallel opens a coordinator, 4 API
# servers, 4 engine cores and several ZMQ sockets per rank. At the default
# 1024 the coordinator dies while creating sockets - before it can report
# anything - and the only error you get is:
#   RuntimeError: DP Coordinator process failed to report ZMQ addresses
#   within timeout=120 seconds during startup
ulimit -n 65536 2>/dev/null || true
echo "open files: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
if [ "$(ulimit -Sn)" != "unlimited" ] && [ "$(ulimit -Sn)" -lt 8192 ]; then
    echo ""
    echo "!! Only $(ulimit -Sn) file descriptors, and the hard limit blocks raising it."
    echo "!! Data-parallel will almost certainly fail here. Recreate the pod with"
    echo "!!   --ulimit nofile=65536"
    echo "!! (or the equivalent in the RunPod template) and run this again."
    echo ""
    read -r -p "Continue anyway? [y/N] " reply
    [ "$reply" = "y" ] || exit 1
fi

echo "=============================================================="
echo " 2/6  Clearing GPUs"
echo "=============================================================="
# nvidia-smi is authoritative - killing "vllm serve" alone misses the renamed
# children (EngineCore_DP*, ApiServer_*, DPCoordinator, VLLM::*), and a
# leftover one holds the ZMQ addresses the next run needs.
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | xargs -r kill -9 || true
for pat in "vllm serve" "VLLM::" "EngineCore" "ApiServer" "DPCoordinator"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo "=============================================================="
echo " 3/6  CUDA toolkit"
echo "=============================================================="
# Needed at RUNTIME, not to build anything: vLLM JIT-compiles FlashInfer,
# DeepGEMM and Triton kernels on first use and calls nvcc to do it.
for f in /etc/apt/sources.list.d/*; do case "$f" in *cuda-ubuntu2404-x86_64.list) continue;; esac; grep -ql "nvidia.com/compute/cuda" "$f" 2>/dev/null && mv "$f" "$f.disabled" && echo "disabled $f"; done; sed -i '\|nvidia\.com/compute/cuda|s|^|#|' /etc/apt/sources.list; apt-get update -qq && echo "apt OK"
distro="$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')"
echo "installing cuda-toolkit-13-3 for $distro"
curl -fsSL -o /tmp/cuda-keyring.deb \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-3

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=============================================================="
echo " 4/6  uv + venv + vLLM"
echo "=============================================================="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# The installer writes to ~/.local/bin, which a non-login shell may not have
# on PATH; the guide re-execs the shell, sourcing this env file is equivalent.
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
uv --version

if [ ! -d "$VENV" ]; then
    mkdir -p "$(dirname "$VENV")"
    uv venv --python 3.12 --seed --managed-python "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Whether this venv can actually run on THIS GPU - not merely whether vLLM
# is importable.
#
# `import vllm` alone was the old condition, and it is not evidence. The venv
# lives on /workspace so it survives pod restarts and pod moves, and a torch
# built without kernels for the current card persists happily inside it. The
# import succeeds, the install is skipped, and the failure surfaces much
# later as the first kernel launch on the device:
#
#   torch.AcceleratorError: CUDA error: no kernel image is available
#
# which arrives at `torch.zeros(1, device=...)` inside nccl setup and reads
# like a distributed-comms problem rather than a packaging one.
#
# The architecture is read off the device rather than hardcoded, so this
# stays correct on whatever card the pod happens to have. Any failure -
# torch absent, vllm absent, CUDA unavailable - means "not ready", so the
# same condition covers the empty-venv case without erroring under `set -e`.
gpu_ready() {
    python - <<'PY' 2>/dev/null
import sys
try:
    import torch
    import vllm  # noqa: F401
except Exception:
    sys.exit(1)
try:
    major, minor = torch.cuda.get_device_capability()
except Exception:
    sys.exit(1)
sys.exit(0 if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else 1)
PY
}

if ! gpu_ready; then
    # --torch-backend=auto picks the torch build matching this driver.
    # Hand-picking a CUDA-suffixed wheel is what previously left the wrong
    # torch installed and produced "SM 12.x requires CUDA >= 12.9".
    uv pip install --reinstall vllm --torch-backend="$TORCH_BACKEND"

    # Verify rather than assume. Without this the script proceeds to spend
    # twenty minutes loading a 167 GiB checkpoint before discovering that
    # the very first kernel launch cannot run.
    if ! gpu_ready; then
        echo ""
        echo "!! torch still has no kernels for this GPU after installing with"
        echo "!!   --torch-backend=$TORCH_BACKEND"
        python -c "import torch; print('   installed:', torch.__version__, 'cuda', torch.version.cuda); print('   arch list:', torch.cuda.get_arch_list()); print('   this GPU :', 'sm_%d%d' % torch.cuda.get_device_capability())" || true
        echo "!! Re-run with an explicit backend, e.g.  TORCH_BACKEND=cu129 bash $0"
        echo "!! (sm_120 / Blackwell needs a CUDA >= 12.9 build.)"
        exit 1
    fi
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

echo "=============================================================="
echo " 5/6  Model"
echo "=============================================================="
mkdir -p "$HF_HOME"
command -v hf >/dev/null 2>&1 || uv pip install huggingface_hub
df -h "$HF_HOME" | tail -1
# Resumes and no-ops if already complete.
hf download "$MODEL"

echo "=============================================================="
echo " 6/6  Starting vLLM"
echo "=============================================================="
export OMP_NUM_THREADS=8
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export CUDA_VISIBLE_DEVICES="$GPUS"

# Notes on the flag set, all learned the hard way:
#   --data-parallel-size + --enable-expert-parallel
#       Not tensor-parallel. For a MoE this shards experts across ranks
#       instead of splitting every matmul, which is what makes the report's
#       278 tok/s possible on PCIe hardware with no NVLink.
#   --kv-cache-dtype fp8
#       Mandatory, not a memory choice: DeepSeek V4's sparse-MLA attention
#       only implements an fp8 cache layout and asserts during layer
#       construction on anything else.
#   --disable-custom-all-reduce
#       vLLM's custom all-reduce assumes fast peer-to-peer links.
#   no --host
#       Deliberately absent. The default binds fine for an SSH tunnel
#       (which connects to 127.0.0.1:PORT inside the pod).
#   no --kv-offloading-*
#       No CPU offloading, as asked. It would spill KV cache to host RAM
#       over PCIe on every token.
#
# If startup fails, the parsers on the next line are the first thing to
# drop - they are the only flags here the linked report does not use.
exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice \
    --data-parallel-size "$GPU_COUNT" \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --port "$PORT"
