# Shared CONFIG and functions for setup.sh (build + download + launch) and
# start.sh (launch only, skips build/download if already done).
# Sourced, not executed directly - has no shebang/set -euo pipefail of its
# own on purpose, it inherits whatever the caller already set.
#
# Kept in one place deliberately: several past failures came from the two
# scripts drifting out of sync on GPU/port assignments - a single source of
# truth for CONFIG removes that whole class of bug.
#
# Serves ONE model (DeepSeek V4 Flash). The earlier two-server planner+flash
# split is gone: the second model was dropped, and the GLM checkpoints it
# used are no longer referenced anywhere here.

# ---------------------------------------------------------------------------
# CONFIG - edit before running if your setup differs
# ---------------------------------------------------------------------------

# --- vLLM build ---
# NOT vllm-project/vllm: upstream main has no SM120 code path for DeepSeek
# V4's sparse MLA attention, and its support_deep_gemm() wrongly returns
# True on SM120, so a mainline build crashes at load with "Unknown SF
# transformation" (vllm#47436) and, once that's worked around, runs the
# sparse attention at roughly 5 tok/s. This fork carries the SM120 Triton
# sparse-MLA fallback plus the VLLM_TRITON_MLA_SPARSE_* knobs below, none
# of which exist upstream (verified against upstream envs.py - setting them
# on a mainline build is a silent no-op). Its HEAD is specifically about
# RTX PRO 6000 SM120 throughput. Reported result: ~30-35 tok/s vs ~5.
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/28
#
# Switching this REQUIRES a full rebuild: set FORCE_REBUILD_VLLM="true"
# once, since the cached wheel in VLLM_WHEEL_DIR was built from mainline
# and nothing here can tell the two apart by filename.
VLLM_GIT_REPO="https://github.com/jasl/vllm.git"
VLLM_GIT_REF="ds4-sm120-preview"       # branch/tag/commit to build

# DeepGEMM must be built from nv_dev, NOT the commit the fork pins by
# default. cmake/external_projects/deepgemm.cmake falls back to
# GIT_TAG 891d57b4... unless DEEPGEMM_SRC_DIR is exported, and that pin
# handles only arch_major 9 (Hopper) and 10 (datacenter Blackwell), so on
# SM120 it aborts twice over:
#   layout.hpp:60          "Unknown SF transformation"   (during weight load)
#   hyperconnection.hpp:56 "Unsupported architecture"    (during profiling)
# nv_dev adds an `arch_major == 12` branch to both, backed by a dedicated
# sm120_tf32_hc_prenorm_gemm kernel. Verified by diffing those two headers
# between the branches - the failing line numbers match the pinned version
# exactly. Cloned --recursive on purpose: the cmake above compiles against
# third-party/cutlass and third-party/fmt, which are submodules.
DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="nv_dev"
DEEPGEMM_SRC_DIR="/workspace/deepgemm-src"
VLLM_SRC_DIR="/workspace/vllm-src"     # kept around so a rebuild is an update+recompile, not a full re-clone
VLLM_WHEEL_DIR="/workspace/vllm-wheels" # built .whl is cached here across pod restarts
# Hugging Face's default cache location (~/.cache/huggingface) lives on the
# pod's root overlay filesystem, which on this pod's image is only ~30GB
# total - nowhere near enough for a 167GB checkpoint. Model downloads MUST
# be redirected onto /workspace's persistent volume (same reasoning as
# VLLM_WHEEL_DIR above), or they silently eat the root disk until something
# fails with "Not enough free disk space" mid-load.
HF_HOME="/workspace/hf-cache"
FORCE_REBUILD_VLLM="false"             # set true to rebuild even if a cached wheel exists
TORCH_CUDA_ARCH_LIST="12.0"            # Blackwell (RTX PRO 6000) - add more (e.g. "8.9;9.0;12.0") if targeting other GPUs too
MAX_JOBS="$(nproc)"                    # parallel compile jobs - lower this if the build OOMs on system RAM (not GPU memory)
NVCC_THREADS=4                         # threads per individual nvcc invocation

# --- Model ---
# DeepSeek ships V4-Flash natively in FP8 (config.json carries
# quantization_config {quant_method: fp8, fmt: e4m3, activation_scheme:
# dynamic, weight_block_size: [128,128]}), so there is no separate "-FP8"
# repo to look for and nothing to convert - this base repo IS the FP8 build,
# 166,886,535,336 bytes across 48 safetensors shards. unsloth's copy is
# byte-for-byte identical to deepseek-ai's own; either works.
#
# Explicitly NOT the "-GGUF" repo: that one carries only llama.cpp GGUF
# quants (IQ1..Q8, BF16 - no FP8 at all, 1.28TB for the full set), and
# vLLM's GGUF path does not meaningfully support MoE models like this one.
MODEL_REPO="unsloth/DeepSeek-V4-Flash-0731"
SERVED_NAME="deepseek-v4-flash"
# NOT 8001/3001/7861/8081/9091/7270: RunPod's own nginx (part of the pod's
# base image, serving its web terminal and template services) already owns
# all of those, and vLLM binding to one fails instantly with
# "OSError: [Errno 98] Address already in use". Verify with `ss -ltnp` on
# the pod before changing this.
PORT=8000

# tensor-parallel-size must divide the model's vocab size - vLLM shards the
# vocab embedding evenly across the TP group and asserts on a remainder at
# model-load time (not at config time, so a bad value only surfaces after
# you've already waited through the download and a partial load).
# DeepSeek V4's vocab_size is 129280 = 2^8 * 5 * 101, so 1/2/4/5/8/10/16...
# are valid and 3/6/7 are NOT. TP=8 across all 8 GPUs: 129280/8 = 16160.
#
# The official model card instead recommends data-parallel + expert-parallel
# for this MoE (`--data-parallel-size 4 --enable-expert-parallel
# --moe-backend deep_gemm_mega_moe`), which is likely faster but is a
# substantially different (and here unproven) topology. Plain TP=8 is one
# process, one set of failure modes, and known-valid for this vocab size -
# switch to the DP/EP recipe as a throughput upgrade once this is stable.
GPUS="0,1,2,3,4,5,6,7"
TP_SIZE=8
PP_SIZE=1
# Left empty on purpose: vLLM reads the real quant method out of the
# checkpoint's own config. Forcing a value that disagrees with the
# checkpoint makes it refuse to start ("Quantization method specified in
# the model config (X) does not match the quantization method specified in
# the `quantization` argument (Y)").
QUANTIZATION=""

# MUST be fp8 - this is not a tuning knob for this model. DeepSeek V4 uses
# a sparse-MLA attention whose "fp8_ds_mla" cache layout only implements an
# fp8 cache, and vLLM asserts on it while building each decoder layer:
#   AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
#   got auto
# (vllm/models/deepseek_v4/attention.py, _resolve_dsv4_kv_cache_dtype).
# "auto" was tried here and fails at load_model() - before KV cache is even
# allocated - so there is no un-quantized-cache option to fall back to. The
# official model card's --kv-cache-dtype fp8 is a requirement, not advice.
# Upside: an fp8 cache is roughly half the size, which is what makes a very
# large MAX_MODEL_LEN below plausible at all.
KV_CACHE_DTYPE="fp8"

# The model's own ceiling: config.json has max_position_embeddings =
# 1048576 (1M), reached via YaRN rope_scaling factor 16 over a base
# 65536-token window. Weights take ~167GB of the 8x96GB = 768GB total,
# leaving ~600GB for KV cache - and DeepSeek's MLA attention keeps
# per-token cache small, so 1M is plausible here rather than merely
# nominal. If startup dies during KV cache allocation, THIS is the number
# to lower (halving it halves the cache), not GPU_MEM_UTILIZATION.
MAX_MODEL_LEN=524288
# Recommended by the official model card for this model specifically;
# larger blocks cut paging overhead on very long contexts.
BLOCK_SIZE=256

# No deepseek_v4 parser exists in vLLM yet - these are the newest DeepSeek
# ones available (deepseek_v31 tool-calling / deepseek_v3 reasoning, both
# documented against DeepSeek-V3.1) and are unverified against V4's actual
# output format. If opencode's tool calls come back as plain text instead
# of structured tool_calls, this pairing is the first thing to suspect:
# check with `manage.py probe-endpoint`, and try clearing them (vLLM then
# emits raw text and the proxy's json_protocol fallback takes over).
TOOL_CALL_PARSER="deepseek_v4"
REASONING_PARSER="deepseek_v4"
TOKENIZER_MODE="deepseek_v4"  # vLLM's tokenizer-mode is required for DeepSeek's rope scaling to work correctly

# 0.93 is the value from the SM120 recipe that benchmarked at 30-35 tok/s
# on this exact GPU. The extra 3% goes to KV cache, so it helps concurrency
# more than single-stream latency - drop back to 0.90 if startup gets tight
# on memory after other changes.
GPU_MEM_UTILIZATION=0.93
# CUDA graph capture profiles a max-batch forward pass across many
# batch-size buckets, which costs far more memory than the weights alone
# suggest. Set to "true" to pass --enforce-eager and skip capture entirely,
# trading throughput for a lower, more predictable footprint - worth trying
# first if startup OOMs at a context length you expect to fit.
ENFORCE_EAGER=""

# The FIRST run on a given pod needs much longer than model loading alone:
# FlashInfer JIT-compiles/downloads its kernel cubins for this GPU arch on
# first use, and there are thousands of them. Subsequent runs reuse
# ~/.cache/flashinfer and are much faster - lower this once you've
# confirmed a cold start actually completes. This is separate from (and on
# top of) the vLLM build time in setup.sh.
HEALTH_TIMEOUT_SECONDS=2700

# Escape hatch: if the server's log shows FlashInfer/PTX/JIT compile
# errors, set this to "FLASH_ATTN" (or "XFORMERS") and rerun - vLLM reads
# this as an env var, not a CLI flag.
VLLM_ATTENTION_BACKEND_OVERRIDE=""

# Now OFF (empty) - this was costing throughput, not buying safety.
#
# Non-empty sets NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1, which forces every
# inter-GPU transfer through host memory (GPU -> PCIe -> CPU RAM -> PCIe ->
# GPU) instead of direct peer-to-peer. At TP=8 every decoder layer does an
# all-reduce across all eight GPUs, so that detour is applied thousands of
# times per generated token - it is the most expensive setting in this file
# by a wide margin.
#
# It was added for a startup hang on an EARLIER, differently-shaped pod
# (log frozen at "vLLM is using nccl==X.Y.Z" with no further output) and
# then carried forward unexamined. If that hang comes back, set these to
# "1" again - it is a restart-only change, no rebuild - but check
# `nvidia-smi topo -m` first: if the GPUs report NV#/PIX/PXB links, P2P is
# real and worth keeping, and only SYS everywhere would mean disabling it
# costs nothing.
NCCL_P2P_DISABLE_WORKAROUND=""
# Kept ON: InfiniBand is for MULTI-NODE traffic, and everything here runs on
# a single pod, so disabling it costs nothing on this topology while still
# skipping an IB probe that has no hardware to find. Unlike P2P above, this
# one was never the expensive half of the workaround.
NCCL_IB_DISABLE_WORKAROUND="1"

# DeepGEMM (DeepSeek's FP8 GEMM library, vendored into vLLM and enabled by
# default) aborts while loading this model's block-scaled FP8 weights on
# this GPU. Its scale-factor layout transform
# (deepgemm-src/csrc/apis/layout.hpp) only implements arch_major 9 (Hopper)
# and 10 (datacenter Blackwell); RTX PRO 6000 is arch_major 12 (SM120),
# which falls through to DG_HOST_UNREACHABLE and surfaces as
#   RuntimeError: Assertion error (...layout.hpp:60): Unknown SF transformation
# on every worker. Known upstream bug, not a misconfiguration here:
# https://github.com/vllm-project/vllm/issues/47436
# Non-empty disables DeepGEMM so vLLM falls back to a backend that does have
# SM120 kernels (Triton/CUTLASS).
#
# Now EMPTY (DeepGEMM enabled), because DEEPGEMM_GIT_REF=nv_dev above
# actually implements SM120 - disabling was only ever a workaround for the
# fork's broken default pin, and it never fully worked anyway: parts of the
# V4 path call DeepGEMM regardless of these env vars, which is why
# hyperconnection.hpp still aborted with the flag set to "1". Set it back to
# "1" if DeepGEMM errors reappear; that's a restart-only change, no rebuild.
DISABLE_DEEP_GEMM_WORKAROUND=""

# SM120 sparse-MLA tuning. These exist ONLY in the SM120 fork pinned as
# VLLM_GIT_REPO above - upstream vLLM has no VLLM_TRITON_MLA_SPARSE* vars at
# all, so on a mainline build every one of these is silently ignored.
# Without them the sparse attention path falls back to something that
# benchmarked at ~5 tok/s on 8x RTX PRO 6000; with them, ~30-35 tok/s.
# ALLOW_CUDAGRAPH is reported as the single most important one.
# The two chunk sizes are deliberately smaller than the fork's own defaults
# (512/256) - these are the values reported as working on this exact GPU.
# Set to "" to leave any of them at the fork's default.
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/28
TRITON_MLA_SPARSE="1"
TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="256"
TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="128"
TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH="1"

LOG_DIR="/var/log/vllm"
SERVER_NAME="deepseek"   # basename for $LOG_DIR/<name>.log and .pid

# ---------------------------------------------------------------------------
# Shared functions
# ---------------------------------------------------------------------------

runpod_kill_gpu_holders() {
    echo "==> Killing any process currently holding GPU memory"
    # Authoritative source, not a command-line guess: ask nvidia-smi
    # directly for every PID with an active GPU context and kill all of
    # them. This is what actually matters - killing a previous run's
    # `vllm serve` (the API server) does NOT kill its spawned
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
    if [ -f "$LOG_DIR/$SERVER_NAME.pid" ]; then
        kill -9 "$(cat "$LOG_DIR/$SERVER_NAME.pid")" 2>/dev/null || true
        rm -f "$LOG_DIR/$SERVER_NAME.pid"
    fi
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
        echo "!! Starting the server now would likely OOM - investigate before continuing." >&2
        exit 1
    fi
}

runpod_clean_corrupt_dist_info() {
    # A *.dist-info directory missing METADATA reports its version as None.
    # `python -m build` walks installed distributions while checking build
    # dependencies and feeds that None straight into packaging's Version(),
    # which dies with:
    #   TypeError: 'NoneType' object is not iterable
    # - a failure that names neither the package nor the file responsible,
    # and looks like it's about the package being built rather than an
    # unrelated leftover. uv's recurring "Failed to uninstall package at
    # ... due to missing `RECORD` file" warnings are the same corruption
    # showing up earlier and being tolerated.
    #
    # These are remnants of interrupted installs. Deleting them is safe:
    # a dist-info carries only metadata, and one this damaged already fails
    # to describe whatever it once owned.
    echo "==> Checking for corrupt dist-info directories"
    local site_dir di found=0
    for site_dir in /usr/local/lib/python3.12/dist-packages /usr/lib/python3/dist-packages; do
        [ -d "$site_dir" ] || continue
        for di in "$site_dir"/*.dist-info; do
            [ -d "$di" ] || continue
            if [ ! -f "$di/METADATA" ]; then
                echo "   removing (no METADATA): $di"
                rm -rf "$di"
                found=$((found + 1))
            fi
        done
    done
    [ "$found" -eq 0 ] && echo "   none found"
    return 0
}

runpod_check_gpu_topology() {
    # tensor-parallel-size x pipeline-parallel-size must equal the number of
    # GPUs handed to the server, or vLLM either errors immediately (TP too
    # high for CUDA_VISIBLE_DEVICES) or silently leaves GPUs idle (TP*PP too
    # low) - catch a mismatch here, in seconds, rather than after a long
    # wait for the wrong outcome.
    echo "==> Checking GPU topology (TP x PP vs GPU count)"
    local gpu_count
    gpu_count="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
    if [ "$((TP_SIZE * PP_SIZE))" -ne "$gpu_count" ]; then
        echo "!! TP=$TP_SIZE x PP=$PP_SIZE = $((TP_SIZE * PP_SIZE)), but GPUS (\"$GPUS\") lists $gpu_count GPU(s)." >&2
        echo "!! Fix the CONFIG block above - tensor-parallel-size x pipeline-parallel-size must equal the GPU count." >&2
        exit 1
    fi
}

start_vllm() {
    echo "==> Starting $SERVER_NAME ($MODEL_REPO) on GPUs [$GPUS], port $PORT, max-model-len $MAX_MODEL_LEN"
    
    # Exported in an if-block rather than as `VAR="${WORKAROUND:+0}"` prefix
    # assignments like the NCCL ones below: vLLM reads these two through
    # int(os.getenv(...)), so handing it an empty string (what :+ expands to
    # when the workaround is off) raises ValueError instead of meaning
    # "unset". They must be either "0" or genuinely absent.
    if [ -n "$DISABLE_DEEP_GEMM_WORKAROUND" ]; then
        echo "    (DeepGEMM disabled - SM120 workaround, see vllm#47436)"
        export VLLM_USE_DEEP_GEMM=0
        export VLLM_MOE_USE_DEEP_GEMM=0
    fi

    # Same export-in-an-if reasoning as above: these are parsed as int/bool
    # from the environment, so an empty string is not a safe "unset".
    [ -n "$TRITON_MLA_SPARSE" ] && export VLLM_TRITON_MLA_SPARSE="$TRITON_MLA_SPARSE"
    [ -n "$TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE" ] && export VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="$TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
    [ -n "$TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE" ] && export VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="$TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"
    [ -n "$TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH" ] && export VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH="$TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"
    if [ -n "$TRITON_MLA_SPARSE" ]; then
        echo "    (Triton sparse-MLA enabled - requires the SM120 fork, no-op on upstream vLLM)"
    fi

    # A prefix env-assignment like `NAME=value cmd` is only recognized by
    # bash when "NAME=" is literal text in the source - substituting the
    # whole "NAME=value" via `${VAR:+NAME=value}` does NOT work (bash parses
    # assignment-word shape before expansion, not after), and produces a
    # bare word bash then tries to run as a command, e.g.
    # "line N: NCCL_P2P_DISABLE=1: command not found". Keeping "NAME="
    # literal and only substituting the value (empty string when the
    # workaround var is unset, which NCCL/vLLM both treat the same as "not
    # set") sidesteps that entirely.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPUS" \
    VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND_OVERRIDE" \
    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_WORKAROUND:+1}" \
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE_WORKAROUND:+1}" \
    nohup vllm serve "$MODEL_REPO" \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        ${QUANTIZATION:+--quantization "$QUANTIZATION"} \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --block-size "$BLOCK_SIZE" \
        ${TOOL_CALL_PARSER:+--tool-call-parser "$TOOL_CALL_PARSER"} \
        ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
        ${TOOL_CALL_PARSER:+--enable-auto-tool-choice} \
        --tokenizer-mode "$TOKENIZER_MODE" \
        --trust-remote-code \
        --max-model-len "$MAX_MODEL_LEN" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
        ${ENFORCE_EAGER:+--enforce-eager} \
        --host 0.0.0.0 \
        --port "$PORT" \
        > "$LOG_DIR/$SERVER_NAME.log" 2>&1 &

    echo $! > "$LOG_DIR/$SERVER_NAME.pid"
}

# vLLM's own final traceback is nearly useless on startup failures: the
# API server process just reports "Engine core initialization failed. See
# root cause above.", wrapped in ~40 lines of asyncio/contextlib frames.
# The error that actually matters is raised in the separate EngineCore
# process and printed well ABOVE that, so a plain `tail` of the log shows
# only the wrapper and hides the cause. Pull the real lines out explicitly.
_print_root_cause() {
    local logfile="$1"
    local hits
    # Match any "SomethingError:" / "SomethingException:" rather than an
    # enumerated list - the list previously missed ImportError outright and
    # reported "no obvious error line found" on a log whose failure was a
    # plain ImportError. Also catches bare "ERROR" log lines.
    hits="$(grep -nE "^(ERROR|.*[A-Za-z_]+(Error|Exception):)" "$logfile" \
            | grep -viE "Engine core initialization failed|Worker failed with error" | tail -n 15 || true)"
    if [ -n "$hits" ]; then
        echo "" >&2
        echo "!! Root cause candidates (first real errors in the log, above the asyncio wrapper):" >&2
        echo "$hits" | sed 's/^/     /' >&2
        echo "" >&2
        echo "   Full log: $logfile" >&2
    else
        echo "" >&2
        echo "!! No obvious error line found - read the whole log: $logfile" >&2
    fi
}

_diagnose_known_failures() {
    local logfile="$1"
    # Patterns are deliberately specific (not bare words like "error" or
    # "not supported") - those match unrelated log noise.
     if grep -qiE "cannot import name .* from 'vllm|ImportError: cannot import name" "$logfile"; then
        echo "   -> the installed vllm tree is mixing files from two different builds (typically the" >&2
        echo "      SM120 fork installed over a previous upstream install - the fork is based on an" >&2
        echo "      older vLLM, so symbols moved). Reinstalling alone doesn't fix it: files the new" >&2
        echo "      wheel doesn't overwrite survive. Remove the package directory outright, then" >&2
        echo "      rerun ds_setup.sh (which now does this automatically):" >&2
        echo "        uv pip uninstall --system vllm; rm -rf /usr/local/lib/python3.12/dist-packages/vllm" >&2
    fi
    if grep -qiE "'NoneType' object is not iterable|missing \`RECORD\` file" "$logfile"; then
        echo "   -> some installed *.dist-info is corrupt (no METADATA), so its version reads as None" >&2
        echo "      and packaging's Version() raises TypeError while build checks dependencies. The" >&2
        echo "      uv warnings about \"Failed to uninstall ... missing RECORD file\" name the culprit." >&2
        echo "      ds_setup.sh clears these automatically now; to do it by hand, delete the offending" >&2
        echo "      *.dist-info under /usr/local/lib/python3.12/dist-packages and rerun." >&2
    fi
    if grep -qiE "version of None already set|is shallow and may cause errors" "$logfile"; then
        echo "   -> setuptools-scm couldn't derive a version from this shallow, tagless checkout." >&2
        echo "      ds_setup.sh pins SETUPTOOLS_SCM_PRETEND_VERSION for that; if it persists, delete" >&2
        echo "      $VLLM_SRC_DIR/vllm.egg-info and rerun with FORCE_REBUILD_VLLM=\"true\"." >&2
    fi
    if grep -qiE "Unknown SF transformation|Unsupported architecture|deepgemm-src" "$logfile"; then
        echo "   -> DeepGEMM was built without SM120 (RTX PRO 6000) support. The revision matters:" >&2
        echo "      the vLLM fork's default pin only handles arch_major 9 and 10, and aborts in" >&2
        echo "      layout.hpp (weight load) or hyperconnection.hpp (memory profiling)." >&2
        echo "      Check DEEPGEMM_GIT_REF is \"nv_dev\" (currently \"$DEEPGEMM_GIT_REF\") and that" >&2
        echo "      $DEEPGEMM_SRC_DIR was actually used, then rebuild with FORCE_REBUILD_VLLM=\"true\"." >&2
        echo "      Note DISABLE_DEEP_GEMM_WORKAROUND does NOT avoid this - parts of the DeepSeek V4" >&2
        echo "      path call DeepGEMM regardless of that env var." >&2
    fi
    if grep -qiE "larger than the maximum number of tokens|can be stored in KV cache|decrease max_model_len|To serve at least one request" "$logfile"; then

        echo "   -> the KV cache can't hold even ONE request at MAX_MODEL_LEN=$MAX_MODEL_LEN." >&2
        echo "      This is the model's 1M-token ceiling, which needs far more cache than these GPUs" >&2
        echo "      have left after ~167GB of weights. vLLM prints the exact GB it needs vs. has in the" >&2
        echo "      line above - halve MAX_MODEL_LEN until it fits (262144 and 131072 are sane steps)." >&2
        echo "      Raising GPU_MEM_UTILIZATION barely helps; the gap here is usually large." >&2
    fi
    if grep -qiE "libnvptxcompiler|ptxas fatal|PTX JIT (failed|error)" "$logfile"; then
        echo "   -> looks like a FlashInfer/PTX-JIT compile failure." >&2
        echo "      Try setting VLLM_ATTENTION_BACKEND_OVERRIDE=\"FLASH_ATTN\" at the top of common.sh and rerun." >&2
    fi
    if grep -qiE "unrecognized model type|Model architectures .* are not supported" "$logfile"; then
        echo "   -> this vLLM build may not know DeepseekV4ForCausalLM." >&2
        echo "      It is supported on main - check VLLM_GIT_REF and the build log at $LOG_DIR/vllm-build.log," >&2
        echo "      then set FORCE_REBUILD_VLLM=true to rebuild against a newer commit." >&2
    fi
    if grep -qiE "invalid choice|unrecognized arguments" "$logfile"; then
        echo "   -> vLLM rejected a CLI flag. Most likely TOOL_CALL_PARSER/REASONING_PARSER:" >&2
        echo "      there is no deepseek_v4 parser, and the v3-era names in common.sh are the" >&2
        echo "      closest available. Clear both to fall back to plain text output." >&2
    fi
    if grep -qiE "is not divisible by" "$logfile"; then
        echo "   -> TP_SIZE doesn't divide the model's vocab size (129280 = 2^8 * 5 * 101)." >&2
        echo "      Valid values are 1/2/4/5/8/10/16...; 3, 6 and 7 are not." >&2
    fi
    if grep -qiE "CUDA out of memory|OutOfMemoryError" "$logfile"; then
        echo "   -> GPU ran out of memory (weights + CUDA-graph buffers left no room for KV cache)." >&2
        echo "      Lower MAX_MODEL_LEN (currently $MAX_MODEL_LEN - it's the model's 1M ceiling), or set" >&2
        echo "      ENFORCE_EAGER=\"true\" to skip CUDA graph capture. Raising GPU_MEM_UTILIZATION won't" >&2
        echo "      help - the crash is real physical VRAM pressure, not a too-conservative soft limit." >&2
    fi
    if grep -qiE "no kernel image is available|CUDA error: no kernel image" "$logfile"; then
        echo "   -> the build didn't produce a kernel for this GPU's architecture." >&2
        echo "      Check TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\" matches this GPU, then set" >&2
        echo "      FORCE_REBUILD_VLLM=true and rerun (in setup.sh - start.sh doesn't rebuild)." >&2
    fi
    if grep -qiE "Address already in use" "$logfile"; then
        echo "   -> something else on the pod already owns port $PORT (RunPod's own nginx commonly" >&2
        echo "      reserves 8001/3001/7861/8081/9091/7270 for its web terminal/template services)." >&2
        echo "      Check with \`ss -ltnp\` on the pod and pick a free PORT." >&2
    fi
    if grep -qiE "Not enough free disk space|Disk quota exceeded" "$logfile"; then
        echo "   -> ran out of disk space downloading/loading the model (it needs ~167GB)." >&2
        echo "      Check HF_HOME points at /workspace and that the volume's quota actually allows it:" >&2
        echo "      \`df -h /workspace\` and \`du -sh /workspace/*\`." >&2
    fi
}

wait_for_health() {
    local waited=0

    echo "==> Waiting for $SERVER_NAME to become healthy (timeout ${HEALTH_TIMEOUT_SECONDS}s)"
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$(cat "$LOG_DIR/$SERVER_NAME.pid")" 2>/dev/null; then
            echo "!! $SERVER_NAME process died during startup - last 50 log lines:" >&2
            tail -n 30 "$LOG_DIR/$SERVER_NAME.log" >&2
            _print_root_cause "$LOG_DIR/$SERVER_NAME.log"
            _diagnose_known_failures "$LOG_DIR/$SERVER_NAME.log"
            exit 1
        fi
        if [ "$waited" -ge "$HEALTH_TIMEOUT_SECONDS" ]; then
            echo "!! $SERVER_NAME did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s - last 50 log lines:" >&2
            tail -n 30 "$LOG_DIR/$SERVER_NAME.log" >&2
            _print_root_cause "$LOG_DIR/$SERVER_NAME.log"
            _diagnose_known_failures "$LOG_DIR/$SERVER_NAME.log"
            exit 1
        fi
        if [ $((waited % 60)) -eq 0 ] && [ "$waited" -gt 0 ]; then
            echo "   ... still waiting on $SERVER_NAME (${waited}s elapsed) - last log line:"
            tail -n 1 "$LOG_DIR/$SERVER_NAME.log"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "==> $SERVER_NAME is healthy"
}

runpod_print_summary() {
    cat <<EOF

==> Endpoint is up and reporting /metrics:
    url   : http://0.0.0.0:${PORT}/v1  (served-model-name: ${SERVED_NAME})
    model : ${MODEL_REPO}  (native FP8, KV cache: ${KV_CACHE_DTYPE}, ctx: ${MAX_MODEL_LEN})
    logs  : ${LOG_DIR}/${SERVER_NAME}.log
    pid   : ${LOG_DIR}/${SERVER_NAME}.pid

Register it with LLM-Hell via 'manage.py add-endpoint' using the tunnel
address this pod is reachable on, and add it as a Prometheus scrape target
in prometheus/prometheus.yml (metrics_path: /metrics).

The model card recommends temperature 1.0 / top_p 0.95 for agentic use -
those are client-side sampling params, so set them in opencode, not here.
EOF
}
