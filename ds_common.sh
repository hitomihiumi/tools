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
MODEL_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
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
GPUS="0,1,2,3"
TP_SIZE=4
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

# Multi-token prediction (self-speculative decoding). The checkpoint ships
# the MTP head itself - config.json has num_nextn_predict_layers=1 and
# mtp_num_hidden_layers=1, and the weight index lists ~4700 mtp.* tensors -
# so no separate draft model is downloaded or configured: with "model"
# omitted, vLLM reuses the target checkpoint as its own drafter. The fork
# maps deepseek_v4 -> deepseek_mtp/DeepSeekV4MTPModel explicitly.
#
# "method" must be spelled out. Left unset it defaults to "draft_model",
# which is not what we want here. num_speculative_tokens=1 matches the
# single MTP layer this checkpoint provides; more would have to be drafted
# autoregressively from that one head, which lowers acceptance.
#
# NOT the official model card's '{"method":"dspark",...}': "dspark" is not
# among this fork's accepted methods (ngram, medusa, mlp_speculator,
# draft_model, suffix, eagle/eagle3/mtp variants, ngram_gpu) and would be
# rejected at startup.
#
# Set to "" to disable speculation entirely. Speculative decoding is a
# latency win only while draft tokens are accepted - if throughput drops
# instead of rising, check the acceptance rate in the server log before
# tuning anything else.
# DISABLED - tried and does not work with this checkpoint + this vLLM build.
#
# The weights are genuinely there (config.json: num_nextn_predict_layers=1,
# mtp_num_hidden_layers=1; ~4700 mtp.* tensors in the index), but vLLM's V4
# MTP loader looks for a different naming scheme than the release uses:
#   KeyError: 'model.layers.43.mtp_block.main_norm.weight'
# while both deepseek-ai's and unsloth's checkpoints name those tensors
# mtp.0.hc_attn_base / mtp.0.hc_ffn_base / ... - checked both indexes, and
# "mtp_block" appears in neither, so this is not an unsloth packaging quirk
# and switching to the original repo would fail identically. Nothing here
# can bridge that; it needs a vLLM build whose loader matches the release.
#
# Re-test after a vLLM upgrade by restoring:
#   SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
# (method must be spelled out - unset defaults to "draft_model"; and the
# official card's "dspark" method does not exist in this fork at all.)
SPECULATIVE_CONFIG=""

# All three deepseek_v4 values are real and verified against vLLM's source
# (an earlier comment here claimed no deepseek_v4 parser existed - that was
# read off documentation that lagged the code):
#   reasoning-parser deepseek_v4 -> DeepSeekV4ParserReasoningAdapter,
#     registered in vllm/reasoning/__init__.py
#   tokenizer-mode   deepseek_v4 -> listed in TokenizerMode alongside auto,
#     hf, slow, mistral, deepseek_v32
#
# What the reasoning parser does: it splits the model's thinking out of
# `content` into a separate `reasoning_content` field on the response. It
# cannot be done downstream - this proxy forwards upstream bytes verbatim
# and only reads a copy for metrics, so if vLLM leaves the thinking inline,
# it stays inline all the way to the client.
#
# Two distinct failure modes if thinking still shows up in the message:
#   - the flag never reached the server -> check "non-default args" in the
#     startup log, which echoes what vLLM actually parsed;
#   - vLLM did split it out, but the client ignores `reasoning_content`
#     (it is a vLLM/DeepSeek extension, not part of the OpenAI schema, and
#     opencode's openai-compatible provider need not render it). Tell the
#     two apart with a raw curl: if the JSON has a populated
#     reasoning_content, the server side is doing its job.
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
# How long the log may go completely unchanged before wait_for_health warns
# that this looks like a hang rather than slow progress. Must stay well
# under HEALTH_TIMEOUT_SECONDS to be worth anything.
STALL_WARN_SECONDS=240

# Escape hatch: if the server's log shows FlashInfer/PTX/JIT compile
# errors, set this to "FLASH_ATTN" (or "XFORMERS") and rerun - vLLM reads
# this as an env var, not a CLI flag.
VLLM_ATTENTION_BACKEND_OVERRIDE=""

# Back ON after testing both ways - this is NOT dead weight, it is load-
# bearing on this hardware.
#
# Non-empty sets NCCL_P2P_DISABLE=1, which forces inter-GPU transfers
# through host memory (GPU -> PCIe -> CPU RAM -> PCIe -> GPU) instead of
# direct peer-to-peer. That is genuinely expensive: every decoder layer
# all-reduces across the whole TP group, thousands of times per generated
# token, so it was worth trying to remove for throughput.
#
# It was removed, and startup hung immediately: the log froze at
#   (Worker pid=...) INFO [pynccl.py:111] vLLM is using nccl==2.28.9
# with no further output for minutes - the exact symptom this workaround
# was originally added for. RunPod's inter-GPU interconnect is virtualized,
# and NCCL's P2P negotiation does not fail cleanly here, it stalls forever.
#
# So: leave this at "1" unless `nvidia-smi topo -m` on the pod shows real
# NV#/PIX/PXB links between the GPUs. If it shows SYS everywhere, P2P was
# never available and disabling it costs nothing anyway - the throughput
# ceiling is elsewhere.
NCCL_P2P_DISABLE_WORKAROUND="1"
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
# Shared functions (runpod_kill_gpu_holders, start_vllm, wait_for_health, ...)
# ---------------------------------------------------------------------------
# shellcheck source=./lib_vllm.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_vllm.sh"
