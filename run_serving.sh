#!/usr/bin/env bash
# Standalone: installs everything and serves BOTH models on one pod.
#
#   DeepSeek V4 Flash  - GPUs 0-3, port 8000, the answer engine
#   Qwen3-VL 8B FP8    - GPU 4,    port 8001, reads PDF pages that have no
#                                             text layer
#
# DeepSeek follows https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# (that report is 2 GPUs; this is the same recipe with data-parallel 4).
#
# Usage:
#   bash run_serving.sh              # both, stays in the foreground
#   SERVE=deepseek bash run_serving.sh
#   SERVE=qwen     bash run_serving.sh
#
# To background it:  nohup bash run_serving.sh &
# Logs go to /workspace/logs regardless, so a backgrounded run is still
# readable.

set -euo pipefail

# --- what to serve ----------------------------------------------------------

SERVE="${SERVE:-both}"
case "$SERVE" in
    both|deepseek|qwen) ;;
    *)
        # Caught here rather than at the end, where an empty server list would
        # surface as an unbound-variable error from `wait`.
        echo "SERVE must be one of: both, deepseek, qwen (got: $SERVE)" >&2
        exit 1
        ;;
esac

DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_NAME="deepseek-v4-flash"
DEEPSEEK_PORT=8000
DEEPSEEK_GPUS="0,1,2,3"
DEEPSEEK_DP=4
DEEPSEEK_MAX_LEN=524288

QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"
QWEN_NAME="qwen3-vl"
QWEN_PORT=8001
QWEN_GPUS="4"
QWEN_MAX_LEN=32768
# One page of a PDF is one image. Four covers a spread plus context without
# letting a single request eat the whole context window.
QWEN_MAX_IMAGES=4

# --- pins -------------------------------------------------------------------
#
# vLLM is pinned, and that is not caution for its own sake. Unpinned, uv
# resolved 0.27.1, whose vendored DeepGEMM aborts loading this checkpoint with
#
#   Assertion error (deepgemm-src/csrc/apis/layout.hpp:60): Unknown SF transformation
#
# after downloading 167 GiB. 0.26.0 is the version the linked report ran, and
# DeepGEMM ships *inside* vLLM, so its version is not separately choosable.
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"

# "auto" resolves the torch build from the driver. It is right when uv is
# recent enough: uv 0.9.0 did not know CUDA 13 and quietly installed
# torch+cu126, whose kernels stop at sm_90 - two generations below these
# cards. The uv self-update below is what makes "auto" trustworthy again.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

# On /workspace, not $HOME: a pod's home directory is on the ephemeral root
# overlay and is wiped on restart. The venv is several GB and the DeepSeek
# checkpoint is ~167 GiB - neither is worth re-fetching every time.
export HF_HOME=/workspace/hf-cache
export VLLM_CACHE_ROOT=/workspace/vllm-cache
VENV=/workspace/serving/.venv
LOGS=/workspace/logs

serve_deepseek() { [ "$SERVE" = "both" ] || [ "$SERVE" = "deepseek" ]; }
serve_qwen()     { [ "$SERVE" = "both" ] || [ "$SERVE" = "qwen" ]; }

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
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Serving Qwen on GPU 4 requires GPU 4 to exist. Failing here with a clear
# line is better than vLLM failing later with an ordinal error.
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if serve_qwen && [ "$gpu_count" -lt 5 ]; then
    echo ""
    echo "!! Qwen is configured for GPU $QWEN_GPUS but this pod has $gpu_count card(s)."
    echo "!! Either add the card, or run just the answer model:"
    echo "!!   SERVE=deepseek bash $0"
    exit 1
fi

echo "=============================================================="
echo " 3/6  CUDA toolkit"
echo "=============================================================="
# Needed at RUNTIME, not to build anything: vLLM JIT-compiles FlashInfer,
# DeepGEMM and Triton kernels on first use and calls nvcc to do it. This is
# separate from the CUDA a torch wheel is built against - installing the
# toolkit does nothing for the kernels inside torch.
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
# uv itself is installed once and then never updated, which is how a uv old
# enough to be unaware of CUDA 13 survived on this pod and installed a torch
# that could not run on the cards. Cheap to keep current.
uv self update 2>/dev/null || true
uv --version

if [ ! -d "$VENV" ]; then
    mkdir -p "$(dirname "$VENV")"
    uv venv --python 3.12 --seed --managed-python "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Whether this venv is one we can actually serve from - which is two
# questions, and the old script asked neither.
#
#   1. Is vLLM the PINNED version? The venv lives on /workspace and survives
#      pod restarts, so an unpinned install from an earlier run persists
#      happily. `import vllm` succeeding says nothing about which vllm.
#   2. Does torch have kernels for THIS card? A torch built without them
#      imports fine and fails much later, at the first kernel launch:
#
#        torch.AcceleratorError: CUDA error: no kernel image is available
#
#      which arrives inside nccl setup and reads like a networking problem
#      rather than a packaging one.
#
# The architecture is read off the device rather than hardcoded, so this
# stays right on whatever card the pod has. Any failure - torch absent, vllm
# absent, CUDA unavailable - means "not ready", so the same condition covers
# the empty-venv case without erroring under `set -e`.
venv_ready() {
    VLLM_WANT="$VLLM_VERSION" python - <<'PY' 2>/dev/null
import os
import sys

try:
    import torch
    import vllm
except Exception:
    sys.exit(1)

if vllm.__version__ != os.environ["VLLM_WANT"]:
    sys.exit(1)

try:
    major, minor = torch.cuda.get_device_capability()
except Exception:
    sys.exit(1)

sys.exit(0 if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else 1)
PY
}

if ! venv_ready; then
    echo "installing vllm==$VLLM_VERSION (torch backend: $TORCH_BACKEND)"
    uv pip install --reinstall "vllm==$VLLM_VERSION" --torch-backend="$TORCH_BACKEND"

    # Verify rather than assume. Without this the script goes on to spend
    # twenty minutes loading a 167 GiB checkpoint before discovering that the
    # very first kernel launch cannot run.
    if ! venv_ready; then
        echo ""
        echo "!! The venv is still not usable after installing."
        python - <<'PY' || true
import torch
import vllm
print("   vllm     :", vllm.__version__)
print("   torch    :", torch.__version__, "cuda", torch.version.cuda)
print("   arch list:", torch.cuda.get_arch_list())
print("   this GPU :", "sm_%d%d" % torch.cuda.get_device_capability())
PY
        echo "!! If the arch list has no sm_ entry for this GPU, re-run with an"
        echo "!! explicit backend:  TORCH_BACKEND=cu130 bash $0"
        echo "!! (sm_120 / Blackwell needs a CUDA 13.x build.)"
        exit 1
    fi
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

echo "=============================================================="
echo " 5/6  Models"
echo "=============================================================="
mkdir -p "$HF_HOME" "$LOGS"
command -v hf >/dev/null 2>&1 || uv pip install huggingface_hub
df -h "$HF_HOME" | tail -1
# Resumes and no-ops if already complete.
serve_deepseek && hf download "$DEEPSEEK_MODEL"
serve_qwen     && hf download "$QWEN_MODEL"

echo "=============================================================="
echo " 6/6  Starting vLLM"
echo "=============================================================="
export OMP_NUM_THREADS=8
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# CUDA_VISIBLE_DEVICES is set per launch rather than exported, which is the
# whole point of running two servers here: each must see only its own cards,
# or they fight over the same memory and the second one hangs at the nccl
# banner with no error at all.

pids=()

# Both children die with the script. Without this, Ctrl-C leaves a server
# holding its GPUs and the next run's "clearing GPUs" step has to kill it.
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Waits for a server to answer, and dies early if its process is already gone
# - a crashed launch should be reported in seconds, not after the timeout.
wait_ready() {
    local name="$1" port="$2" pid="$3" log="$4" waited=0
    while [ "$waited" -lt "${READY_TIMEOUT:-3600}" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "!! $name exited during startup. Last lines of $log:"
            tail -n 30 "$log"
            return 1
        fi
        if curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            echo "$name ready on :$port after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "!! $name did not become ready within ${READY_TIMEOUT:-3600}s - see $log"
    return 1
}

if serve_deepseek; then
    # Notes on the flag set, all learned the hard way:
    #   --data-parallel-size + --enable-expert-parallel
    #       Not tensor-parallel. For a MoE this shards experts across ranks
    #       instead of splitting every matmul, which is what makes the
    #       report's 278 tok/s possible on PCIe hardware with no NVLink.
    #   --kv-cache-dtype fp8
    #       Mandatory, not a memory choice: DeepSeek V4's sparse-MLA attention
    #       only implements an fp8 cache layout and asserts during layer
    #       construction on anything else.
    #   --disable-custom-all-reduce
    #       vLLM's custom all-reduce assumes fast peer-to-peer links.
    #   no --host
    #       Deliberately absent. The default binds fine for an SSH tunnel
    #       (which connects to 127.0.0.1:PORT inside the pod).
    #
    # If startup fails, the parsers on the next line are the first thing to
    # drop - they are the only flags here the linked report does not use.
    echo "starting DeepSeek on GPUs $DEEPSEEK_GPUS -> :$DEEPSEEK_PORT"
    CUDA_VISIBLE_DEVICES="$DEEPSEEK_GPUS" \
    vllm serve "$DEEPSEEK_MODEL" \
        --served-model-name "$DEEPSEEK_NAME" \
        --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice \
        --data-parallel-size "$DEEPSEEK_DP" \
        --enable-expert-parallel \
        --disable-custom-all-reduce \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --trust-remote-code \
        --max-model-len "$DEEPSEEK_MAX_LEN" \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.95 \
        --port "$DEEPSEEK_PORT" \
        > "$LOGS/deepseek.log" 2>&1 &
    deepseek_pid=$!
    pids+=("$deepseek_pid")
fi

if serve_qwen; then
    # Qwen is the transcriber, not a second answerer: it turns a PDF page into
    # text and that text goes into DeepSeek's prompt like any other snippet.
    # So it is sized for short work, not for long conversations.
    #
    #   --limit-mm-per-prompt
    #       An image is expensive in tokens. Without a bound, one request with
    #       a whole document attached can exceed the context window and fail
    #       at generation time rather than at request time.
    #   --gpu-memory-utilization 0.90
    #       Lower than DeepSeek's 0.95: an 8B FP8 model has room to spare on a
    #       96 GB card, and leaving headroom means image preprocessing spikes
    #       do not OOM the server.
    #   no --reasoning-parser
    #       Describing a page is not a reasoning task, and an Instruct model
    #       asked to think about a diagram mostly produces preamble.
    echo "starting Qwen3-VL on GPU $QWEN_GPUS -> :$QWEN_PORT"
    CUDA_VISIBLE_DEVICES="$QWEN_GPUS" \
    vllm serve "$QWEN_MODEL" \
        --served-model-name "$QWEN_NAME" \
        --trust-remote-code \
        --max-model-len "$QWEN_MAX_LEN" \
        --limit-mm-per-prompt "{\"image\": $QWEN_MAX_IMAGES}" \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.90 \
        --port "$QWEN_PORT" \
        > "$LOGS/qwen.log" 2>&1 &
    qwen_pid=$!
    pids+=("$qwen_pid")
fi

echo ""
echo "logs: $LOGS/*.log   (tail -f them while this waits)"
echo ""

# Qwen first: it is the smaller model and reports in a minute or two, so a
# mistake in its flags surfaces long before DeepSeek has finished loading.
serve_qwen     && wait_ready "Qwen3-VL"  "$QWEN_PORT"     "$qwen_pid"     "$LOGS/qwen.log"
serve_deepseek && wait_ready "DeepSeek"  "$DEEPSEEK_PORT" "$deepseek_pid" "$LOGS/deepseek.log"

echo ""
echo "=============================================================="
serve_deepseek && echo " answer model : $DEEPSEEK_NAME  http://127.0.0.1:$DEEPSEEK_PORT/v1"
serve_qwen     && echo " vision model : $QWEN_NAME      http://127.0.0.1:$QWEN_PORT/v1"
echo ""
echo " Tunnel both from your workstation:"
serve_deepseek && echo "   ssh -N -L 0.0.0.0:8010:127.0.0.1:$DEEPSEEK_PORT root@<host> -p <port>"
serve_qwen     && echo "   ssh -N -L 0.0.0.0:8011:127.0.0.1:$QWEN_PORT root@<host> -p <port>"
echo "=============================================================="

# Foreground for as long as the servers live. If either dies, this returns and
# the trap takes the other one down with it, rather than leaving half a stack
# running and a tunnel pointing at nothing.
wait -n "${pids[@]}"
echo "!! a server exited - shutting the other one down"
