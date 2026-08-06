# CONFIG for Qwen3.5-397B-A17B (AWQ INT4) on 4x RTX PRO 6000 Blackwell.
# Sourced, not executed. Shares every function with the DeepSeek config via
# lib_vllm.sh - only the values below differ.
#
# Use it in place of ds_common.sh:
#   CONFIG_FILE=qwen_common.sh bash ds_setup.sh    # download + launch
#   CONFIG_FILE=qwen_common.sh bash ds_start.sh    # launch only
#
# Values follow https://github.com/local-inference-lab/rtx6kpro
# (models/qwen35-397b.md), a guide written against this exact GPU. Where a
# flag in that guide is SGLang-only it has been dropped rather than passed
# to vLLM - noted individually below.

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# --- vLLM build ---
# Upstream, NOT the jasl SM120 fork the DeepSeek config pins: that fork
# exists for DeepSeek V4's sparse-MLA attention, which this model doesn't
# use, and it trails upstream by enough that Qwen3.5 support is a gamble.
# If the pod image already ships a vLLM that knows `qwen3_5_moe`, don't
# build at all - use start.sh, which never touches the build path.
VLLM_GIT_REPO="https://github.com/vllm-project/vllm.git"
VLLM_GIT_REF="main"
# This model has no DeepGEMM dependency, so the shared build step skips the
# clone/submodule/SM120-assert work entirely. That assert is a hard `exit 1`
# on a DeepGEMM revision without an arch_major == 12 branch - failing the
# Qwen build on a check for a library it never calls is pure downside, and
# the clone alone (recursive, with cutlass) is minutes of nothing.
USE_DEEPGEMM="false"
# Still read by the shared build/diagnostic code even when the above is
# "false", so keep them pointing somewhere harmless rather than unset.
DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="nv_dev"
DEEPGEMM_SRC_DIR="/workspace/deepgemm-src"
VLLM_SRC_DIR="/workspace/vllm-src"
VLLM_WHEEL_DIR="/workspace/vllm-wheels"
HF_HOME="/workspace/hf-cache"
# ~228 GiB of weights (244,394,630,034 bytes) - noticeably more than the
# guide's "~200GB" estimate, and more than DeepSeek V4's 167 GB. setup.sh
# asserts this much is actually free on $HF_HOME's volume before starting
# the download: RunPod network volumes have a usable quota well below the
# size shown in the dashboard, and finding that out 200 GiB in costs hours.
MODEL_DISK_GIB=228
# Which backend `hf download` uses. huggingface_hub 1.x (what this pod's
# image ships) dropped hf_transfer outright - HF_HUB_ENABLE_HF_TRANSFER now
# only prints a deprecation warning and is ignored - so the real choice is
# xet or a single-connection plain HTTP stream. 228 GiB over plain HTTP is
# most of a day, so: xet.
#
# The DeepSeek config sets this to "false" because hf-xet used to fail
# reconstructing >15 GB files ("File reconstruction error: ... receiver
# dropped", "Background writer channel closed" -
# https://github.com/huggingface/xet-core/issues/763). If that resurfaces
# here, flip this to "false" and rerun - `hf download` resumes, it does not
# restart. Note the two states are not symmetric: "true" INSTALLS hf_xet,
# "false" UNINSTALLS it, because HF_HUB_DISABLE_XET=1 alone was not
# reliable (https://github.com/huggingface/huggingface_hub/issues/3266).
HF_USE_XET="true"
FORCE_REBUILD_VLLM="false"
TORCH_CUDA_ARCH_LIST="12.0"
MAX_JOBS="$(nproc)"
NVCC_THREADS=4

# --- Model ---
# AWQ INT4, deliberately not the NVFP4 build the guide's own launch
# commands use: its comparison table puts AWQ ahead on both axes at once -
# KLD 0.024 vs 0.035 (higher fidelity) and 15-38% faster across every
# concurrency level, ~152 tok/s single-stream with MTP against NVFP4's 180
# claimed for a different config. So the sample commands there need the
# model id swapped, which is what this does.
MODEL_REPO="QuantTrio/Qwen3.5-397B-A17B-AWQ"
SERVED_NAME="qwen3.5"
PORT=8000
# Used only by the shared failure diagnostics in lib_vllm.sh, so that a
# startup crash names THIS model's numbers instead of DeepSeek's. Getting
# these wrong costs nothing at runtime but sends you down the wrong path
# when something does break.
MODEL_ARCH="qwen3_5_moe"
MODEL_VOCAB_SIZE="248320 = 2^9 * 5 * 97"
WEIGHTS_DESC="AWQ INT4"

# vocab_size is 248320 = 2^9 * 5 * 97, so 1/2/4/5/8/... divide it and
# 3/6/7 do not - vLLM shards the vocab embedding across the TP group and
# asserts on a remainder at model-load time, long after startup.
# TP=4 across all four GPUs: 248320/4 = 62080.
GPUS="0,1,2,3"
TP_SIZE=4
PP_SIZE=1
# Left empty on purpose. The guide is explicit that the quant method is
# auto-detected from the checkpoint ("do NOT add --quantization"), and
# forcing a value that disagrees with config.json makes vLLM refuse to
# start outright.
QUANTIZATION=""

# No KV-cache quantization, as requested. Unlike DeepSeek V4 - whose
# sparse-MLA layout hard-asserts on anything but fp8 - this model has no
# such constraint, and the guide's own vLLM command likewise passes no
# --kv-cache-dtype. Setting "fp8_e4m3" would roughly halve cache memory at
# some accuracy cost if context ever needs to grow.
KV_CACHE_DTYPE="auto"

# The checkpoint's native ceiling (config.json: max_position_embeddings =
# 262144). The guide reaches 524288 only by patching config.json with a
# YaRN rope_parameters block (factor 2.0 over the 262144 base) in a
# separate directory - that is a checkpoint edit, not a flag, so it is out
# of scope here. Raising this number alone would not work.
MAX_MODEL_LEN=262144
# vLLM's default (16). The 256 in the DeepSeek config came from that
# model's card specifically; nothing in this guide asks for it, and a large
# block size mainly pays off at million-token contexts.
BLOCK_SIZE=16

# Both verified present in vLLM's registries, not just its docs:
# qwen3_coder is registered in vllm/tool_parsers/__init__.py (the
# tool-calling doc page lists only qwen3_xml and lags the code), and qwen3
# is registered in vllm/reasoning/__init__.py.
TOOL_CALL_PARSER="qwen3_coder"
REASONING_PARSER="qwen3"
# "auto" - the deepseek_v4 tokenizer mode in the other config is specific
# to that model family and would be rejected here.
TOKENIZER_MODE="auto"

# The guide calls MTP=2 "the sweet spot", and its Known Issues section
# blames higher values for CUDA memory-access errors under load: "Reduce
# MTP to 2-3 tokens; higher values show instability". The checkpoint does
# carry the MTP layers this needs.
#
# Worth watching after the DeepSeek experience, where MTP was configured
# correctly and still failed because that vLLM build's loader expected
# different tensor names. If startup dies with a KeyError naming an MTP
# tensor, set this to "" - speculative decoding is an optimisation, not a
# requirement.
SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}'

GPU_MEM_UTILIZATION=0.9
ENFORCE_EAGER=""

# No 45-minute first-run JIT here (that was FlashInfer warming up for
# DeepSeek's sparse attention); the guide budgets ~5 minutes of kernel JIT
# on top of the download.
HEALTH_TIMEOUT_SECONDS=1800
STALL_WARN_SECONDS=240

VLLM_ATTENTION_BACKEND_OVERRIDE=""

# NOT NCCL_P2P_DISABLE=1. These GPUs talk over PCIe Gen5 with no NVLink,
# and the guide's standard setting for that topology is
# NCCL_P2P_LEVEL=SYS (see NCCL_EXTRA_ENV below) - it keeps P2P working at
# system level instead of switching it off wholesale. Its Known Issues
# section reaches for the same lever on deadlock: "Change NCCL_P2P_LEVEL=2
# or disable P2P negotiation entirely", with full disabling as the last
# resort rather than the default.
#
# That matters because disabling P2P outright routes every all-reduce
# through host memory, and at TP=4 that happens on every layer of every
# token. If startup hangs at "vLLM is using nccl==..." (wait_for_health
# warns after STALL_WARN_SECONDS), fall back in this order:
#   1. NCCL_EXTRA_ENV="NCCL_P2P_LEVEL=2"
#   2. NCCL_P2P_DISABLE_WORKAROUND="1"   (the DeepSeek config's setting)
NCCL_P2P_DISABLE_WORKAROUND=""
NCCL_IB_DISABLE_WORKAROUND="1"

# Extra environment for the server process, applied verbatim as NAME=VALUE
# pairs. NCCL_P2P_LEVEL=SYS is the guide's headline setting for this
# hardware; SAFETENSORS_FAST_GPU speeds up loading ~228 GiB of weights;
# OMP_NUM_THREADS caps a thread pool that otherwise oversubscribes the CPU.
NCCL_EXTRA_ENV="NCCL_P2P_LEVEL=SYS SAFETENSORS_FAST_GPU=1 OMP_NUM_THREADS=8"

# Batching limits from the guide's vLLM command. They shape concurrency
# rather than single-stream latency, which is where its 1,551 tok/s at 64
# concurrent users comes from.
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=128

LOG_DIR="/var/log/vllm"
# Distinct from "deepseek" so the two models' logs and pidfiles never
# collide when switching between configs on the same pod.
SERVER_NAME="qwen35"

# ---------------------------------------------------------------------------
# Shared functions
# ---------------------------------------------------------------------------
# shellcheck source=./lib_vllm.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_vllm.sh"
