# CONFIG for DeepSeek V4 Flash on 4x RTX PRO 6000 Blackwell (SM120).
# Sourced, not executed. Shares every function with the other model configs
# via lib_vllm.sh - only the values below differ.
#
#   bash setup.sh    # install + download + launch (this is the default config)
#   bash start.sh    # launch only
#
# Rewritten to follow
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# - a report of this model running on RTX 6000 Pro hardware at 278 tok/s
# output throughput (16K in / 2K out). Two things there differ sharply from
# what this file used to do, and both are the point of the rewrite:
#
#   1. Stock vLLM, no fork. That report runs 0.26.0, "includes vendored
#      DeepGEMM kernels" - so the jasl/vllm SM120 fork and the separate
#      DeepGEMM nv_dev checkout this config used to require are both gone.
#   2. Data-parallel + expert-parallel instead of tensor-parallel. For a
#      MoE this shards experts across ranks rather than splitting every
#      matmul, which cuts the per-token all-reduce traffic that dominated
#      the old TP topology on a PCIe box with no NVLink.
#
# Scaled from the report's 2 GPUs to 4: DP 2 -> 4, everything else kept.

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# --- Install ---
# Nothing is built from source. vLLM comes from PyPI into a venv, exactly as
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# does it; 0.26.0+ vendors the DeepGEMM kernels that made the old SM120
# source build necessary.
#
# Empty VLLM_VERSION means latest. Pin it (e.g. "0.26.0") if a future
# release regresses - that is a one-word change here instead of the fork
# surgery this used to need.
VLLM_VERSION=""
FORCE_REINSTALL_VLLM="false"   # true to reinstall into an existing venv
# The venv lives on /workspace, NOT in ~/serving as the guide has it: a
# RunPod pod's home directory is on the ephemeral root overlay and is wiped
# on restart, which would mean reinstalling vLLM and torch (several GB)
# every single time. Same reasoning as HF_HOME below.
VENV_DIR="/workspace/serving/.venv"
VENV_PYTHON="3.12"
# vLLM JIT-compiles FlashInfer/DeepGEMM/Triton kernels on first use and
# needs nvcc for it, so the toolkit is a runtime dependency even though
# nothing here is compiled ahead of time.
CUDA_TOOLKIT_VERSION="13.3"
# HF's default cache is on that same small root overlay - a 167 GiB
# checkpoint must land on /workspace or the download dies partway with
# "Not enough free disk space".
HF_HOME="/workspace/hf-cache"

# --- Model ---
# Natively FP8 (config.json carries quantization_config quant_method fp8),
# so there is nothing to convert and no --quantization to pass.
MODEL_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
SERVED_NAME="deepseek-v4-flash"
# NOT 8001/3001/7861/8081/9091/7270 - RunPod's own nginx owns those on this
# image, and vLLM binding to one fails instantly with EADDRINUSE.
PORT=8000

# Data-parallel + expert-parallel, NOT tensor-parallel. Each of the four
# ranks gets one GPU (TP x PP x DP must equal the GPU count, which
# runpod_check_gpu_topology asserts); attention is replicated per rank and
# the MoE experts are sharded across all four by EP.
#
# This also retires the vocab-divisibility constraint that shaped the old
# config: vocab sharding is a tensor-parallel concern, and at TP=1 there is
# no remainder to assert on.
GPUS="0,1,2,3"
TP_SIZE=1
PP_SIZE=1
DP_SIZE=4
EXPERT_PARALLEL="1"
# From the report. vLLM's custom all-reduce kernel assumes fast peer-to-peer
# links; on a PCIe box without NVLink it is at best no help and at worst the
# source of the startup stalls this project hit repeatedly under TP.
# Disabling it falls back to NCCL's own path.
DISABLE_CUSTOM_ALL_REDUCE="1"

QUANTIZATION=""

# MUST be fp8, and not because of memory: DeepSeek V4's sparse-MLA
# attention only implements an fp8 cache layout and asserts during layer
# construction otherwise -
#   AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache
# That fires at load_model(), before any cache is allocated, so there is no
# un-quantized option to fall back to. The report passes fp8 as well.
KV_CACHE_DTYPE="fp8"

# The largest context the linked report reached WITHOUT CPU offloading, on
# two GPUs. Four ranks split the experts twice as finely, leaving more VRAM
# per GPU for cache, so this should have headroom rather than be tight.
# If startup dies allocating KV cache, halve this rather than raising
# GPU_MEM_UTILIZATION.
MAX_MODEL_LEN=343296
BLOCK_SIZE=256

# Deliberately NOT enabling --kv-offloading-size/--kv-offloading-backend,
# which the report uses to push context beyond the number above. Offloading
# spills KV cache to host RAM over PCIe, and paying that on every token is
# the wrong trade here.
# (Both flags do exist in vLLM if that changes: --kv-offloading-size takes
# GiB, --kv-offloading-backend takes native|lmcache.)

TOOL_CALL_PARSER="deepseek_v4"
REASONING_PARSER="deepseek_v4"
TOKENIZER_MODE="deepseek_v4"

# MTP stays off. It was configured correctly once and still failed: this
# vLLM's MTP loader expects tensors named model.layers.N.mtp_block.*, while
# both the deepseek-ai and unsloth checkpoints name them mtp.0.hc_* -
#   KeyError: 'model.layers.43.mtp_block.main_norm.weight'
# Not fixable from config. Re-test after a vLLM upgrade with:
#   SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
SPECULATIVE_CONFIG=""

# 0.95 per the report, up from the 0.90 this config used. Worth watching:
# it leaves little headroom for allocator fragmentation, so if startup OOMs
# late (during cache allocation rather than weight loading) this is the
# first thing to walk back.
GPU_MEM_UTILIZATION=0.95
ENFORCE_EAGER=""

# Batching limits from the report. max-num-seqs 32 is deliberately modest -
# this is a very large model and each concurrent sequence costs real KV
# cache at a 343K context.
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=32

# First start on a pod JIT-compiles FlashInfer and DeepGEMM kernels, which
# the report calls out explicitly. VLLM_ENGINE_READY_TIMEOUT_S below raises
# vLLM's own internal patience to match; this one is how long the script
# waits before giving up.
HEALTH_TIMEOUT_SECONDS=2700
STALL_WARN_SECONDS=240

VLLM_ATTENTION_BACKEND_OVERRIDE=""

# Left off: the report does not disable P2P, and --disable-custom-all-reduce
# above addresses the same PCIe-topology problem more precisely. If startup
# hangs at "vLLM is using nccl==..." (wait_for_health warns after
# STALL_WARN_SECONDS), escalate in this order:
#   1. EXTRA_ENV="... NCCL_P2P_LEVEL=SYS"    (what the Qwen config uses)
#   2. NCCL_P2P_DISABLE_WORKAROUND="1"       (blunt: routes every transfer
#                                             through host memory)
NCCL_P2P_DISABLE_WORKAROUND=""
# InfiniBand is a multi-node concern and there is none here, so skipping the
# probe costs nothing.
NCCL_IB_DISABLE_WORKAROUND="1"

# The Triton sparse-MLA knobs that used to be set here only exist in the
# SM120 fork - on upstream vLLM they are silently ignored, so they are gone
# rather than left to look meaningful.

# Applied verbatim as NAME=VALUE pairs to the server process.
#   OMP_NUM_THREADS               from the report; caps a pool that
#                                 otherwise oversubscribes the CPU
#   VLLM_ENGINE_READY_TIMEOUT_S   from the report; without it vLLM can give
#                                 up on its own engine during first-run JIT
#   VLLM_CACHE_ROOT               defaults to ~/.cache/vllm on the pod's
#                                 ephemeral root disk, so torch.compile
#                                 output is discarded on every pod restart
#                                 and the multi-minute compile repeats
EXTRA_ENV="OMP_NUM_THREADS=8 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_CACHE_ROOT=/workspace/vllm-cache"

LOG_DIR="/var/log/vllm"
SERVER_NAME="deepseek"   # basename for $LOG_DIR/<name>.log and .pid

# ---------------------------------------------------------------------------
# Shared functions (runpod_kill_gpu_holders, start_vllm, wait_for_health, ...)
# ---------------------------------------------------------------------------
# shellcheck source=./lib_vllm.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_vllm.sh"
