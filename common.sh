# Shared CONFIG and functions for runpod_setup.sh (build + download + launch)
# and runpod_start.sh (launch only, skips build/download if already done).
# Sourced, not executed directly - has no shebang/set -euo pipefail of its
# own on purpose, it inherits whatever the caller already set.
#
# Kept in one place deliberately: several of today's failures came from the
# two scripts (or two edits of the same script) drifting out of sync on
# GPU/port assignments - a single source of truth for CONFIG removes that
# whole class of bug.

# ---------------------------------------------------------------------------
# CONFIG - edit before running if your setup differs
# ---------------------------------------------------------------------------

# --- vLLM build ---
VLLM_GIT_REF="main"                    # branch/tag/commit to build
VLLM_SRC_DIR="/workspace/vllm-src"     # kept around so a rebuild is an update+recompile, not a full re-clone
VLLM_WHEEL_DIR="/workspace/vllm-wheels" # built .whl is cached here across pod restarts
# Hugging Face's default cache location (~/.cache/huggingface) lives on the
# pod's root overlay filesystem, which on this pod's image is only ~30GB
# total - nowhere near enough for either checkpoint, let alone both. Model
# downloads MUST be redirected onto /workspace's persistent volume (same
# reasoning as VLLM_WHEEL_DIR above), or they silently eat the root disk
# until something fails with "Not enough free disk space" mid-load - which
# is exactly what happened to the flash server (GPU 4 sat idle because
# flash never actually finished loading, not because of a GPU/TP problem).
HF_HOME="/workspace/hf-cache"
FORCE_REBUILD_VLLM="false"             # set true to rebuild even if a cached wheel exists
TORCH_CUDA_ARCH_LIST="12.0"            # Blackwell (RTX PRO 6000) - add more (e.g. "8.9;9.0;12.0") if targeting other GPUs too
MAX_JOBS="$(nproc)"                    # parallel compile jobs - lower this if the build OOMs on system RAM (not GPU memory)
NVCC_THREADS=4                         # threads per individual nvcc invocation

# Swapped from QuantTrio/GLM-4.7-AWQ to rule out a bad/misbehaving
# checkpoint as a contributor to an earlier hang, independent of the
# FLASH_GPUS overlap bug fixed below - different quantizer/calibration
# (AWQ v1.0, group-size 32 vs QuantTrio's), same AWQ 4-bit format and
# comparable weight size, so no other CONFIG changes needed.
PLANNER_MODEL_REPO="cyankiwi/GLM-4.7-AWQ-4bit"
PLANNER_SERVED_NAME="glm-4.7"
PLANNER_PORT=8000
# tensor-parallel-size must divide the model's vocab size (151552 for
# GLM 4.7 - vLLM shards the vocab embedding layer evenly across the TP
# group). 151552 = 2^12 * 37, so only powers of 2 (1/2/4/8/...) or
# multiples of 37 are valid - neither 3 nor 6 is, and either fails at
# model-load time with "AssertionError: 151552 is not divisible by N",
# not at startup/config time, so this only shows up once you've already
# waited through download + partial load.
#
# On an 8-GPU pod (6 for the planner, 2 for flash): TP=6 is invalid per
# the above, so the planner's 6 GPUs are split TP=2 x PP=3 instead
# (tensor-parallel-size x pipeline-parallel-size must multiply out to
# PLANNER_GPUS' count - runpod_check_gpu_topology below asserts this).
# Pipeline parallelism has its own overhead (bubble time between stages)
# that plain TP doesn't, and this is the first time this script has used
# it - if it misbehaves, the fallback in the plan-review discussion was
# TP=4 with 2 of the 6 GPUs left idle (proven, no PP involved).
PLANNER_GPUS="0,1,2,3,4,5"
PLANNER_TP_SIZE=2
PLANNER_PP_SIZE=3
# vLLM refuses to start if this doesn't match what's actually in the
# checkpoint's own config: cyankiwi's AWQ-4bit repack is serialized in the
# compressed-tensors container format (a common way checkpoints package
# an AWQ-style quant these days), not vLLM's plain "awq" quant method -
# forcing "awq" here failed fast with "Quantization method specified in
# the model config (compressed-tensors) does not match the quantization
# method specified in the `quantization` argument (awq)". Leaving this
# empty lets vLLM read the real method from the checkpoint instead of
# guessing from the repo name, same as FLASH_QUANTIZATION below.
PLANNER_QUANTIZATION=""
PLANNER_KV_CACHE_DTYPE="fp8"

FLASH_MODEL_REPO="unsloth/GLM-4.7-Flash-FP8-Dynamic"
FLASH_SERVED_NAME="glm-4.7-flash"
# NOT 8001: RunPod's own nginx (part of this pod's base image, serving its
# web terminal/other template services) already owns 8001, 3001, 7861,
# 8081, 9091 and 7270 - vLLM binding to any of those fails immediately
# with "OSError: [Errno 98] Address already in use" (confirmed via
# `ss -ltnp` on the pod: nginx pid 110 holds all of the above). This
# doesn't need to be one of RunPod's exposed/proxied ports since we reach
# it over a direct SSH tunnel (`ssh -L`), not RunPod's HTTP proxy - it
# just needs to be a port nothing else on the pod is listening on.
FLASH_PORT=8002
# Must not overlap PLANNER_GPUS - planner and flash launch concurrently
# in the background (see start_vllm below), and two vLLM processes racing
# to grab the same physical GPU for their own NCCL/CUDA context is exactly
# what produced a silent hang at "vLLM is using nccl==..." for 17+ minutes
# with zero further log output on a prior run (planner was on 0,1,2,3
# while this was still "3" from an older 4-GPU config - a leftover from
# resizing the pod that runpod_check_gpu_overlap below now catches
# immediately instead of hanging through the full HEALTH_TIMEOUT_SECONDS).
#
# TP=2 (both of flash's 2 GPUs, no pipeline parallelism needed here - 2
# divides 151552 cleanly). This doubles flash's old 1-GPU allocation,
# which is exactly the trigger condition the FLASH_MAX_MODEL_LEN and
# FLASH_ENFORCE_EAGER comments below call out for revisiting those - not
# changed automatically here since neither has been verified on this
# specific 2-GPU shape yet, but worth trying once this launches cleanly.
FLASH_GPUS="6,7"
FLASH_TP_SIZE=2
FLASH_PP_SIZE=1
FLASH_QUANTIZATION=""  # weights are already FP8 in the checkpoint - vLLM auto-detects
# --kv-cache-dtype fp8 is known to make this specific checkpoint loop/repeat
# garbage output on vLLM (https://huggingface.co/unsloth/GLM-4.7-Flash-FP8-Dynamic/discussions/2,
# unresolved as of writing, and unrelated to the CUDA version - it's a
# checkpoint/kernel issue). Leave at "auto" unless you've verified fp8
# works on your vLLM build with a short manual test.
FLASH_KV_CACHE_DTYPE="auto"
# The actual weights are only ~31 GiB on disk, but the process OOMed at
# ~94.5 GiB on a 96 GiB card even at a modest max-model-len - the
# unaccounted ~60 GiB is very likely CUDA graph capture (vLLM profiles a
# max-batch forward pass and captures graphs across many batch-size
# buckets) plus chunked-prefill's own working set, not the weights
# themselves. --enforce-eager skips graph capture entirely, trading some
# throughput for a guaranteed lower memory footprint - set to "" to
# re-enable graphs once you've confirmed there's headroom to spare
# (e.g. after giving flash 2 GPUs instead of 1).
FLASH_ENFORCE_EAGER=""

# Separate max-model-len per server, not a shared one: the planner has 4
# GPUs (384 GiB) to spread weights + KV cache across, flash has 1 (96 GiB)
# - the same 200k-token context that fits fine on the planner OOMs flash
# during KV cache allocation, since its weights + CUDA-graph capture
# buffers alone already use nearly the whole card. Raise
# FLASH_MAX_MODEL_LEN only after giving flash more GPUs (increase
# FLASH_TP_SIZE and FLASH_GPUS, taking GPUs away from the planner) -
# lowering GPU_MEM_UTILIZATION won't fix it, there's just not enough room
# left after the weights on a single card at 200k context.
PLANNER_MAX_MODEL_LEN=80000
FLASH_MAX_MODEL_LEN=132000
GPU_MEM_UTILIZATION=0.90
# The FIRST run on a given pod needs much longer than model loading alone:
# FlashInfer JIT-compiles/downloads its kernel cubins for this GPU arch on
# first use, and there are thousands of them. Subsequent runs reuse
# ~/.cache/flashinfer and are much faster - lower this once you've
# confirmed a cold start actually completes. This is separate from (and
# on top of) the vLLM build time in runpod_setup.sh.
HEALTH_TIMEOUT_SECONDS=2700

# Escape hatch: if a server's log shows FlashInfer/PTX/JIT compile errors,
# set this to "FLASH_ATTN" (or "XFORMERS") and rerun - vLLM reads this as
# an env var, not a CLI flag.
VLLM_ATTENTION_BACKEND_OVERRIDE=""

# RunPod's inter-GPU interconnect is virtualized and commonly lacks working
# P2P/NVLink or InfiniBand, which can make NCCL's transport negotiation
# hang indefinitely at startup (log gets stuck on "vLLM is using
# nccl==X.Y.Z" with no further output) instead of falling back cleanly.
# Non-empty enables the workaround (NCCL falls back to a slower but
# working transport immediately instead of hanging while negotiating one
# that isn't there); set to "" only after confirming P2P/IB actually work
# on this pod (e.g. `nvidia-smi topo -m` shows real NVLink/PIX links).
NCCL_P2P_DISABLE_WORKAROUND="1"
NCCL_IB_DISABLE_WORKAROUND="1"

LOG_DIR="/var/log/vllm"

# ---------------------------------------------------------------------------
# Shared functions
# ---------------------------------------------------------------------------

runpod_kill_gpu_holders() {
    echo "==> Killing any process currently holding GPU memory"
    # Authoritative source, not a command-line guess: ask nvidia-smi
    # directly for every PID with an active GPU context and kill all of
    # them. This is what actually matters - a previous run's `vllm serve`
    # (the API server) getting killed does NOT kill its spawned
    # EngineCore/Worker subprocesses, since vLLM renames those via
    # setproctitle to "VLLM::EngineCore" / "VLLM::Worker" rather than
    # inheriting the "vllm serve ..." command line, so a pattern-matching
    # `pkill -f "vllm serve"` alone leaves them orphaned, silently holding
    # the GPU's memory across every rerun.
    local gpu_pids
    gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)"
    if [ -n "$gpu_pids" ]; then
        echo "$gpu_pids" | while read -r pid; do
            echo "   killing pid $pid (holding GPU memory)"
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    # Belt-and-suspenders for anything nvidia-smi didn't catch (e.g. a
    # process mid-startup, not yet holding a CUDA context).
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    for name in planner flash; do
        if [ -f "$LOG_DIR/$name.pid" ]; then
            local old_pid
            old_pid="$(cat "$LOG_DIR/$name.pid")"
            kill -9 "$old_pid" 2>/dev/null || true
            rm -f "$LOG_DIR/$name.pid"
        fi
    done
    sleep 5

    echo "==> GPUs visible on this pod:"
    nvidia-smi -L || { echo "!! nvidia-smi not found - is this actually a GPU pod?" >&2; exit 1; }

    echo "==> GPU memory after cleanup (should be ~0 used on every line):"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    local still_held
    still_held="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [ -n "$still_held" ]; then
        echo "!! GPU memory is still held after the kill step:" >&2
        echo "$still_held" >&2
        echo "!! Starting new servers now would likely OOM again - investigate before continuing." >&2
        exit 1
    fi
}

runpod_check_gpu_overlap() {
    echo "==> Checking PLANNER_GPUS/FLASH_GPUS don't overlap"
    # planner and flash launch concurrently (both start_vllm calls
    # background themselves and return immediately) - if they share a GPU
    # index, two processes race to initialize NCCL/CUDA on the same
    # physical device, which was observed to hang silently at "vLLM is
    # using nccl==..." for the full HEALTH_TIMEOUT_SECONDS rather than
    # failing fast. Catch that here, in seconds, instead of after a
    # 45-minute wait.
    local overlap
    overlap="$(comm -12 <(echo "$PLANNER_GPUS" | tr ',' '\n' | sort -u) <(echo "$FLASH_GPUS" | tr ',' '\n' | sort -u) || true)"
    if [ -n "$overlap" ]; then
        echo "!! PLANNER_GPUS ($PLANNER_GPUS) and FLASH_GPUS ($FLASH_GPUS) share GPU index/indices: $(echo "$overlap" | tr '\n' ' ')" >&2
        echo "!! Fix the CONFIG block above - each GPU must belong to exactly one server." >&2
        exit 1
    fi
}

runpod_check_gpu_topology() {
    # tensor-parallel-size x pipeline-parallel-size must equal the number
    # of GPUs handed to that server, or vLLM either errors immediately
    # (TP too high for CUDA_VISIBLE_DEVICES) or silently leaves GPUs idle
    # (TP*PP too low) - catch a mismatch here, in seconds, rather than
    # after a 45-minute wait for the wrong outcome.
    local name="$1" gpus="$2" tp="$3" pp="$4"
    local gpu_count
    gpu_count="$(echo "$gpus" | tr ',' '\n' | grep -c .)"
    if [ "$((tp * pp))" -ne "$gpu_count" ]; then
        echo "!! $name: TP=$tp x PP=$pp = $((tp * pp)), but ${name^^}_GPUS (\"$gpus\") lists $gpu_count GPU(s)." >&2
        echo "!! Fix the CONFIG block above - tensor-parallel-size x pipeline-parallel-size must equal the GPU count." >&2
        exit 1
    fi
}

start_vllm() {
    local name="$1" model="$2" served_name="$3" port="$4" gpus="$5" tp="$6" pp="$7" quant="$8" kv_dtype="$9" max_len="${10}"
    local enforce_eager="${11}"

    echo "==> Starting $name ($model) on GPUs [$gpus], port $port, max-model-len $max_len"
    # A prefix env-assignment like `NAME=value cmd` is only recognized by
    # bash when "NAME=" is literal text in the source - substituting the
    # whole "NAME=value" via `${VAR:+NAME=value}` does NOT work (bash
    # parses assignment-word shape before expansion, not after), and
    # produces a bare word bash then tries to run as a command, e.g. "line
    # N: NCCL_P2P_DISABLE=1: command not found". Keeping "NAME=" literal
    # and only substituting the value (empty string when the workaround
    # var is unset, which NCCL/vLLM both treat the same as "not set")
    # sidesteps that entirely.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND_OVERRIDE" \
    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_WORKAROUND:+1}" \
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE_WORKAROUND:+1}" \
    nohup vllm serve "$model" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        ${quant:+--quantization "$quant"} \
        --kv-cache-dtype "$kv_dtype" \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --max-model-len "$max_len" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
        ${enforce_eager:+--enforce-eager} \
        --host 0.0.0.0 \
        --port "$port" \
        > "$LOG_DIR/$name.log" 2>&1 &

    echo $! > "$LOG_DIR/$name.pid"
}

_diagnose_known_failures() {
    local logfile="$1"
    # Patterns are deliberately specific (not bare words like "error" or
    # "not supported") - those match unrelated log noise.
    if grep -qiE "libnvptxcompiler|ptxas fatal|PTX JIT (failed|error)" "$logfile"; then
        echo "   -> looks like a FlashInfer/PTX-JIT compile failure." >&2
        echo "      Try setting VLLM_ATTENTION_BACKEND_OVERRIDE=\"FLASH_ATTN\" at the top of common.sh and rerun." >&2
    fi
    if grep -qiE "unrecognized model type|Model architectures .* are not supported|glm4_moe_lite" "$logfile"; then
        echo "   -> looks like GLM-4.7-Flash's architecture isn't recognized by this vLLM build." >&2
        echo "      Unexpected for a main-branch build - check VLLM_GIT_REF and the build log at $LOG_DIR/vllm-build.log." >&2
    fi
    if grep -qiE "CUDA out of memory|OutOfMemoryError" "$logfile"; then
        echo "   -> GPU ran out of memory (weights + CUDA-graph buffers left no room for KV cache)." >&2
        echo "      Lower PLANNER_MAX_MODEL_LEN/FLASH_MAX_MODEL_LEN, or give that server more GPUs" >&2
        echo "      (raise its *_TP_SIZE and *_GPUS), rather than GPU_MEM_UTILIZATION - the crash is" >&2
        echo "      real physical VRAM pressure, not vLLM being too conservative about a soft limit." >&2
    fi
    if grep -qiE "no kernel image is available|CUDA error: no kernel image" "$logfile"; then
        echo "   -> the build didn't produce a kernel for this GPU's architecture." >&2
        echo "      Check TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\" matches this GPU, then set" >&2
        echo "      FORCE_REBUILD_VLLM=true and rerun (in setup.sh - this script doesn't rebuild)." >&2
    fi
    if grep -qiE "Address already in use" "$logfile"; then
        echo "   -> something else on the pod already owns this port (RunPod's own nginx commonly" >&2
        echo "      reserves 8001/3001/7861/8081/9091/7270 for its web terminal/template services)." >&2
        echo "      Check with \`ss -ltnp\` on the pod and pick a free PLANNER_PORT/FLASH_PORT." >&2
    fi
    if grep -qiE "Not enough free disk space" "$logfile"; then
        echo "   -> ran out of disk space downloading/loading the model." >&2
        echo "      Check HF_HOME actually points at /workspace (df -h /), not the pod's small root disk." >&2
    fi
}

wait_for_health() {
    local name="$1" port="$2"
    local waited=0

    echo "==> Waiting for $name to become healthy (timeout ${HEALTH_TIMEOUT_SECONDS}s)"
    until curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null; then
            echo "!! $name process died during startup - last 50 log lines:" >&2
            tail -n 50 "$LOG_DIR/$name.log" >&2
            _diagnose_known_failures "$LOG_DIR/$name.log"
            exit 1
        fi
        if [ "$waited" -ge "$HEALTH_TIMEOUT_SECONDS" ]; then
            echo "!! $name did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s - last 50 log lines:" >&2
            tail -n 50 "$LOG_DIR/$name.log" >&2
            _diagnose_known_failures "$LOG_DIR/$name.log"
            exit 1
        fi
        if [ $((waited % 60)) -eq 0 ] && [ "$waited" -gt 0 ]; then
            echo "   ... still waiting on $name (${waited}s elapsed) - last log line:"
            tail -n 1 "$LOG_DIR/$name.log"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "==> $name is healthy"
}

runpod_print_summary() {
    cat <<EOF

==> Both endpoints are up and reporting /metrics:
    planner : http://0.0.0.0:${PLANNER_PORT}/v1  (served-model-name: ${PLANNER_SERVED_NAME})
    flash   : http://0.0.0.0:${FLASH_PORT}/v1  (served-model-name: ${FLASH_SERVED_NAME})
    logs    : ${LOG_DIR}/{planner,flash}.log
    pids    : ${LOG_DIR}/{planner,flash}.pid

Point LLM-Hell's model_endpoints at these two via 'manage.py add-endpoint'
using this pod's public host/port, and add both as Prometheus scrape
targets in prometheus/prometheus.yml (metrics_path: /metrics).
EOF
}
