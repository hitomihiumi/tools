# Shared functions for the vLLM launcher scripts. Sourced by a model config
# (ds_common.sh, qwen_common.sh, ...), which defines the variables these
# read - GPUS, TP_SIZE, PORT, MODEL_REPO, LOG_DIR, SERVER_NAME and so on.
#
# Split out of ds_common.sh when a second model was added: the functions are
# identical for every model, the config is not, and keeping one copy is the
# whole point (drifting duplicates of this logic caused several past
# failures). Nothing here reads config at source time, only when called, so
# the sourcing order between this file and the config doesn't matter.

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

    # vLLM's torch.compile cache (VLLM_CACHE_ROOT, default ~/.cache/vllm)
    # and Triton's temp-file directory (TMPDIR, default /tmp) both land on
    # the small root-disk overlay unless redirected - the exact same trap
    # HF_HOME exists to avoid for model weights, just for a different set
    # of libraries that don't share that env var. A compile at this
    # max-model-len (hundreds of CUDA graph capture sizes) genuinely needs
    # several GB of scratch space, which the ~30GB root disk does not
    # reliably have once the OS image and installed packages are
    # accounted for - confirmed by "OSError: No space left on device"
    # under /root/.cache/vllm/torch_compile_cache and /tmp/*.ptx on a run
    # that had HF_HOME set correctly and still failed here. Both are
    # placed next to HF_HOME so they land on the same large volume.
    local cache_root
    cache_root="$(dirname "$HF_HOME")/vllm-cache"
    mkdir -p "$cache_root/tmp"
    export VLLM_CACHE_ROOT="$cache_root"
    export TMPDIR="$cache_root/tmp"

    # Exported in an if-block rather than as `VAR="${WORKAROUND:+0}"` prefix
    # assignments like the NCCL ones below: vLLM reads these two through
    # int(os.getenv(...)), so handing it an empty string (what :+ expands to
    # when the workaround is off) raises ValueError instead of meaning
    # "unset". They must be either "0" or genuinely absent.
    if [ -n "${DISABLE_DEEP_GEMM_WORKAROUND:-}" ]; then
        echo "    (DeepGEMM disabled - SM120 workaround, see vllm#47436)"
        export VLLM_USE_DEEP_GEMM=0
        export VLLM_MOE_USE_DEEP_GEMM=0
    fi

    # Same export-in-an-if reasoning as above: these are parsed as int/bool
    # from the environment, so an empty string is not a safe "unset".
    [ -n "${TRITON_MLA_SPARSE:-}" ] && export VLLM_TRITON_MLA_SPARSE="${TRITON_MLA_SPARSE:-}"
    [ -n "${TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE:-}" ] && export VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="${TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE:-}"
    [ -n "${TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE:-}" ] && export VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="${TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE:-}"
    [ -n "${TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH:-}" ] && export VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH="${TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH:-}"
    if [ -n "${TRITON_MLA_SPARSE:-}" ]; then
        echo "    (Triton sparse-MLA enabled - requires the SM120 fork, no-op on upstream vLLM)"
    fi

    # Free-form "NAME=VALUE NAME=VALUE" from the config, exported one at a
    # time. Deliberately not passed as a prefix assignment: those must be
    # literal text in the source to be recognised as assignments at all,
    # which is the same trap the NCCL_* lines below document.
    if [ -n "${NCCL_EXTRA_ENV:-}" ]; then
        for _kv in $NCCL_EXTRA_ENV; do
            export "${_kv?}"
            echo "    (env: $_kv)"
        done
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
        ${QUANTIZATION:+--quantization "${QUANTIZATION:-}"} \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --block-size "$BLOCK_SIZE" \
        ${SPECULATIVE_CONFIG:+--speculative-config "${SPECULATIVE_CONFIG:-}"} \
        ${TOOL_CALL_PARSER:+--tool-call-parser "${TOOL_CALL_PARSER:-}"} \
        ${REASONING_PARSER:+--reasoning-parser "${REASONING_PARSER:-}"} \
        ${TOOL_CALL_PARSER:+--enable-auto-tool-choice} \
        --tokenizer-mode "$TOKENIZER_MODE" \
        --trust-remote-code \
        --max-model-len "$MAX_MODEL_LEN" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-}"} \
        ${MAX_NUM_SEQS:+--max-num-seqs "${MAX_NUM_SEQS:-}"} \
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
    # Every "edit the config" hint below has to name the config actually in
    # use, not a hardcoded ds_common.sh - these messages are read by someone
    # already confused about why their server died, and pointing them at the
    # wrong model's file is worse than saying nothing.
    local cfg="${CONFIG_FILE:-the model config}"
    # Patterns are deliberately specific (not bare words like "error" or
    # "not supported") - those match unrelated log noise.
     if grep -qiE "cannot import name .* from 'vllm|ImportError: cannot import name" "$logfile"; then
        echo "   -> the installed vllm tree is mixing files from two different builds (typically the" >&2
        echo "      SM120 fork installed over a previous upstream install - the fork is based on an" >&2
        echo "      older vLLM, so symbols moved). Reinstalling alone doesn't fix it: files the new" >&2
        echo "      wheel doesn't overwrite survive. Remove the package directory outright, then" >&2
        echo "      rerun setup.sh (which now does this automatically):" >&2
        echo "        uv pip uninstall --system vllm; rm -rf /usr/local/lib/python3.12/dist-packages/vllm" >&2
    fi
    if grep -qiE "'NoneType' object is not iterable|missing \`RECORD\` file" "$logfile"; then
        echo "   -> some installed *.dist-info is corrupt (no METADATA), so its version reads as None" >&2
        echo "      and packaging's Version() raises TypeError while build checks dependencies. The" >&2
        echo "      uv warnings about \"Failed to uninstall ... missing RECORD file\" name the culprit." >&2
        echo "      setup.sh clears these automatically now; to do it by hand, delete the offending" >&2
        echo "      *.dist-info under /usr/local/lib/python3.12/dist-packages and rerun." >&2
    fi
    if grep -qiE "version of None already set|is shallow and may cause errors" "$logfile"; then
        echo "   -> setuptools-scm couldn't derive a version from this shallow, tagless checkout." >&2
        echo "      setup.sh pins SETUPTOOLS_SCM_PRETEND_VERSION for that; if it persists, delete" >&2
        echo "      $VLLM_SRC_DIR/vllm.egg-info and rerun with FORCE_REBUILD_VLLM=\"true\"." >&2
    fi
    if grep -qiE "KeyError: 'model\.layers\.[0-9]+\.mtp_block|mtp_block\." "$logfile"; then
        echo "   -> speculative MTP is on, but this vLLM build's MTP loader expects tensor names" >&2
        echo "      (model.layers.N.mtp_block.*) that the checkpoint doesn't use. This bit DeepSeek V4" >&2
        echo "      (whose shards name them mtp.0.hc_*) and is not fixable by config - speculative" >&2
        echo "      decoding is an optimisation, so set SPECULATIVE_CONFIG=\"\" in $cfg and rerun." >&2
    fi
    if [ "${USE_DEEPGEMM:-true}" = "true" ] && grep -qiE "Unknown SF transformation|Unsupported architecture|deepgemm-src" "$logfile"; then
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
        echo "      There is no room left for cache after ~${MODEL_DISK_GIB:-?} GiB of weights across these GPUs." >&2
        echo "      vLLM prints the exact GB it needs vs. has in the line above - halve MAX_MODEL_LEN" >&2
        echo "      in $cfg until it fits (262144, then 131072, are sane steps)." >&2
        echo "      Raising GPU_MEM_UTILIZATION barely helps; the gap here is usually large." >&2
        if [ "${KV_CACHE_DTYPE:-auto}" = "auto" ]; then
            echo "      KV_CACHE_DTYPE is \"auto\" (unquantized) - setting \"fp8_e4m3\" roughly halves the" >&2
            echo "      cache and may be the cheaper fix than cutting context, if the model allows it." >&2
        fi
    fi
    if grep -qiE "libnvptxcompiler|ptxas fatal|PTX JIT (failed|error)" "$logfile"; then
        echo "   -> looks like a FlashInfer/PTX-JIT compile failure." >&2
        echo "      Try setting VLLM_ATTENTION_BACKEND_OVERRIDE=\"FLASH_ATTN\" in $cfg and rerun." >&2
    fi
    if grep -qiE "unrecognized model type|Model architectures .* are not supported" "$logfile"; then
        echo "   -> this vLLM build may not know the architecture ${MODEL_ARCH:-this checkpoint declares}." >&2
        echo "      Check VLLM_GIT_REPO/VLLM_GIT_REF in $cfg (currently $VLLM_GIT_REF) and the build" >&2
        echo "      log at $LOG_DIR/vllm-build.log, then set FORCE_REBUILD_VLLM=true to rebuild" >&2
        echo "      against a newer commit. Support for recent architectures usually lands on" >&2
        echo "      upstream main first, so a fork pinned for another model can simply be too old." >&2
    fi
    if grep -qiE "invalid choice|unrecognized arguments" "$logfile"; then
        echo "   -> vLLM rejected a CLI flag. Most likely TOOL_CALL_PARSER (\"${TOOL_CALL_PARSER:-}\") or" >&2
        echo "      REASONING_PARSER (\"${REASONING_PARSER:-}\") naming a parser this build doesn't" >&2
        echo "      register. Clear both in $cfg to fall back to plain text output; the error line" >&2
        echo "      above lists the choices this build actually accepts." >&2
    fi
    if grep -qiE "is not divisible by" "$logfile"; then
        echo "   -> TP_SIZE=$TP_SIZE doesn't divide the model's vocab size (${MODEL_VOCAB_SIZE:-see the error above})." >&2
        echo "      vLLM shards the vocab embedding evenly across the TP group, so only divisors work." >&2
        echo "      Change TP_SIZE (and GPUS/PP_SIZE to match) in $cfg." >&2
    fi
    if grep -qiE "CUDA out of memory|OutOfMemoryError" "$logfile"; then
        echo "   -> GPU ran out of memory (weights + CUDA-graph buffers left no room for KV cache)." >&2
        echo "      Lower MAX_MODEL_LEN (currently $MAX_MODEL_LEN) in $cfg, or set ENFORCE_EAGER=\"true\"" >&2
        echo "      to skip CUDA graph capture. Raising GPU_MEM_UTILIZATION won't help - the crash is" >&2
        echo "      real physical VRAM pressure, not a too-conservative soft limit." >&2
        if [ -n "${SPECULATIVE_CONFIG:-}" ]; then
            echo "      SPECULATIVE_CONFIG is set - MTP keeps its own draft-model weights and KV cache" >&2
            echo "      resident, so clearing it is another way to buy back memory." >&2
        fi
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
        echo "   -> ran out of disk space downloading/loading the model (it needs ~${MODEL_DISK_GIB:-?} GiB)." >&2
        echo "      Check HF_HOME points at /workspace and that the volume's quota actually allows it:" >&2
        echo "      \`df -h $HF_HOME\` and \`du -sh $HF_HOME/hub/*\`." >&2
    fi
    if grep -qiE "No space left on device.*(torch_compile_cache|\.cache/vllm|\.cache/triton|/tmp/tmp)" "$logfile"; then
        echo "   -> the small root-disk overlay filled up, NOT \$HF_HOME's volume - this is vLLM's" >&2
        echo "      torch.compile cache or Triton's temp files, which default to \$HOME/.cache and" >&2
        echo "      /tmp respectively and don't follow HF_HOME. start_vllm() in lib_vllm.sh already" >&2
        echo "      redirects both (VLLM_CACHE_ROOT, TMPDIR) to a vllm-cache/ dir next to HF_HOME -" >&2
        echo "      if this still fires, check nothing upstream of start_vllm() unset those, and" >&2
        echo "      \`df -h /\` to confirm the root disk (not /workspace) is what's actually full." >&2
    fi
}

wait_for_health() {
    local waited=0
    local last_line="" current_line="" stalled=0 stall_warned=0

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
            current_line="$(tail -n 1 "$LOG_DIR/$SERVER_NAME.log")"
            echo "   ... still waiting on $SERVER_NAME (${waited}s elapsed) - last log line:"
            echo "$current_line"

            # A live process writing nothing new is the signature of a hang,
            # not of slow work: every genuinely slow phase here (shard
            # loading, FlashInfer JIT) keeps emitting progress. Say so after
            # STALL_WARN_SECONDS rather than letting HEALTH_TIMEOUT_SECONDS
            # (45 min by default) elapse in silence before anyone finds out.
            if [ "$current_line" = "$last_line" ]; then
                stalled=$((stalled + 60))
            else
                stalled=0
                last_line="$current_line"
            fi
            if [ "$stalled" -ge "$STALL_WARN_SECONDS" ] && [ "$stall_warned" -eq 0 ]; then
                stall_warned=1
                echo "" >&2
                echo "!! $SERVER_NAME has produced no new log output for ${stalled}s while still running." >&2
                if printf '%s' "$current_line" | grep -qiE "using nccl==|pynccl"; then
                    echo "!! Stuck on NCCL init. On this pod's virtualized interconnect, P2P negotiation" >&2
                    echo "!! stalls instead of failing over. Ctrl-C now rather than waiting out the" >&2
                    echo "!! timeout, then in ${CONFIG_FILE:-the model config} try, in order (no rebuild needed):" >&2
                    echo "!!   1. NCCL_EXTRA_ENV=\"NCCL_P2P_LEVEL=2\"      (currently \"${NCCL_EXTRA_ENV:-}\")" >&2
                    echo "!!   2. NCCL_P2P_DISABLE_WORKAROUND=\"1\"        (currently \"${NCCL_P2P_DISABLE_WORKAROUND:-}\")" >&2
                    echo "!! Step 2 routes every all-reduce through host memory, which at TP=$TP_SIZE costs" >&2
                    echo "!! real throughput on every layer of every token - hence the ordering." >&2
                else
                    echo "!! Not a known signature - inspect $LOG_DIR/$SERVER_NAME.log directly." >&2
                fi
                echo "" >&2
            fi
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
    model : ${MODEL_REPO}  (${WEIGHTS_DESC:-weights}, KV cache: ${KV_CACHE_DTYPE}, ctx: ${MAX_MODEL_LEN})
    logs  : ${LOG_DIR}/${SERVER_NAME}.log
    pid   : ${LOG_DIR}/${SERVER_NAME}.pid

Register it with LLM-Hell via 'manage.py add-endpoint' using the tunnel
address this pod is reachable on, and add it as a Prometheus scrape target
in prometheus/prometheus.yml (metrics_path: /metrics).

${SAMPLING_HINT:-Check the model card for its recommended sampling params.} These are
client-side settings, so apply them in opencode, not here.
EOF
}
