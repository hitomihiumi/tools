#!/usr/bin/env bash
# Builds vLLM from source (main branch, where DeepseekV4ForCausalLM support
# lives), targeting CUDA 13 + Blackwell (SM 12.x/sm_120) explicitly via
# TORCH_CUDA_ARCH_LIST, then downloads the DeepSeek V4 Flash checkpoint
# (~167GB, natively FP8) and launches it as a vLLM OpenAI-compatible server.
#
# Assumes the pod image already has Python and CUDA 13 (x86_64) installed
# - everything else (build tooling, torch, vLLM itself, HF download
# tooling) is installed by this script. It does NOT set up the LLM-Hell
# backend itself (that runs on a separate VPS, not here).
#
# Use this script for a genuinely fresh pod, or when the vLLM build itself
# needs redoing (FORCE_REBUILD_VLLM=true, or bumping VLLM_GIT_REF). If vLLM
# is already installed and the checkpoint is already fully downloaded,
# `start.sh` skips straight to launching the server - no reason to re-run
# the build/download machinery just to restart it.
#
# The built vLLM wheel is cached under VLLM_WHEEL_DIR (default:
# /workspace/vllm-wheels, i.e. on the pod's persistent volume if one is
# attached) - a rerun reuses it instead of rebuilding from scratch, which
# otherwise takes a long time (compiling vLLM's CUDA kernels for a single
# target architecture is still commonly 30-90+ minutes depending on CPU
# core count). Set FORCE_REBUILD_VLLM=true in common.sh to force a fresh
# build anyway (e.g. after bumping VLLM_GIT_REF).
#
# Safe to rerun: kills anything holding GPU memory (by PID, from nvidia-smi
# directly - not just a `vllm serve` command-line pattern match, which
# misses vLLM's renamed EngineCore/Worker subprocesses and was observed to
# leave a GPU stuck full of orphaned memory across reruns) before starting
# the new server.
#
# Defaults assume 8 GPUs, all given to the one model at TP=8. Edit the
# CONFIG block in common.sh if your pod's GPU count/topology differs.
#
# Usage: bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./ds_common.sh
source "$SCRIPT_DIR/ds_common.sh"

mkdir -p "$LOG_DIR" "$VLLM_WHEEL_DIR"

runpod_kill_gpu_holders

echo "==> System CUDA toolkit on this pod:"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
"$CUDA_HOME/bin/nvcc" --version || { echo "!! nvcc not found at $CUDA_HOME/bin/nvcc - is CUDA 13 actually installed here?" >&2; exit 1; }
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "==> Architecture: $(uname -m) (expecting x86_64)"
if [ "$(uname -m)" != "x86_64" ]; then
    echo "!! This script assumes x86_64 - torch/vLLM build steps below are not set up for other architectures." >&2
    exit 1
fi

runpod_check_gpu_topology
runpod_clean_corrupt_dist_info

# ---------------------------------------------------------------------------
# Build (or reuse a cached build of) vLLM from source
# ---------------------------------------------------------------------------

# -t (newest first), not plain ls: wheels from different sources pile up in
# this directory and their names sort in an order that has nothing to do
# with which one is current. A mainline "vllm-0.1.dev1+g30b4e7f47..." sorts
# BEFORE a freshly built fork "vllm-0.1.dev1+ge4923b2ea...", so plain `ls`
# would keep installing the stale mainline build after a successful rebuild
# - looking like the rebuild silently did nothing.
existing_wheel="$(ls -t "$VLLM_WHEEL_DIR"/vllm-*.whl 2>/dev/null | head -1 || true)"

if [ -n "$existing_wheel" ] && [ "$FORCE_REBUILD_VLLM" != "true" ]; then
    echo "==> Reusing cached vLLM wheel: $existing_wheel"
else
    echo "==> Building vLLM from source (ref: $VLLM_GIT_REF, arch: $TORCH_CUDA_ARCH_LIST) - this can take a long time on a cold cache"

    echo "==> Installing build tooling"
    apt-get update -qq
    # cmake/ninja also come from requirements/build/cuda.txt below (pip
    # packages that bundle working binaries), but build-essential
    # (gcc/g++/make) and git aren't available as pip packages, and ccache
    # speeds up a future rebuild against the same object files.
    apt-get install -y -qq git build-essential ccache curl >/dev/null

    # setuptools-rust (in requirements/build/cuda.txt below) needs an
    # actual Rust toolchain present - rustup rather than a distro package
    # so the version is new enough regardless of which base image this is.
    if ! command -v cargo >/dev/null 2>&1; then
        echo "==> Installing Rust toolchain (needed by setuptools-rust)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi

    pip install -q -U uv

    if [ -d "$VLLM_SRC_DIR/.git" ]; then
        # An existing checkout may point at a different repo than the one now
        # configured (e.g. this switched from vllm-project/vllm to the SM120
        # fork). Fetching a fork-only branch from the old origin fails with a
        # bare "couldn't find remote ref", which reads like a network error
        # rather than the stale-remote problem it actually is - so repoint
        # origin first and let the fetch below work either way.
        current_origin="$(git -C "$VLLM_SRC_DIR" remote get-url origin 2>/dev/null || true)"
        if [ "$current_origin" != "$VLLM_GIT_REPO" ]; then
            echo "==> Repointing existing checkout: $current_origin -> $VLLM_GIT_REPO"
            git -C "$VLLM_SRC_DIR" remote set-url origin "$VLLM_GIT_REPO"
        fi
        echo "==> Updating existing vLLM checkout ($VLLM_GIT_REF)"
        git -C "$VLLM_SRC_DIR" fetch --depth 1 origin "$VLLM_GIT_REF"
        git -C "$VLLM_SRC_DIR" checkout FETCH_HEAD
        # `git checkout` only removes files git TRACKS. Anything left over
        # from a previous build or a different upstream - __pycache__, stray
        # .py files, build/ output - survives and gets packaged into the
        # wheel by setuptools, which globs the source tree rather than
        # consulting git. That is how an upstream-only
        # quantization/inc/ package directory ended up inside a wheel built
        # from a fork that ships quantization/inc.py instead, shadowing it
        # at import time (Python prefers a package dir over a module file)
        # and producing an ImportError no amount of cleaning site-packages
        # could fix - the wheel itself carried both.
        echo "==> Cleaning untracked leftovers from the checkout"
        git -C "$VLLM_SRC_DIR" clean -xfd
    else
        echo "==> Cloning vLLM from $VLLM_GIT_REPO ($VLLM_GIT_REF)"
        git clone --branch "$VLLM_GIT_REF" --depth 1 "$VLLM_GIT_REPO" "$VLLM_SRC_DIR"
    fi
    vllm_commit="$(git -C "$VLLM_SRC_DIR" rev-parse --short HEAD)"
    echo "==> Building commit $vllm_commit"

    # DeepGEMM from nv_dev (see DEEPGEMM_* in ds_common.sh for why the
    # fork's default pin is unusable on SM120). --recursive is required:
    # the build compiles against its cutlass/fmt submodules, and a
    # non-recursive clone fails later with missing headers rather than
    # anything that names the real problem.
    if [ -d "$DEEPGEMM_SRC_DIR/.git" ]; then
        echo "==> Updating existing DeepGEMM checkout ($DEEPGEMM_GIT_REF)"
        git -C "$DEEPGEMM_SRC_DIR" remote set-url origin "$DEEPGEMM_GIT_REPO"
        git -C "$DEEPGEMM_SRC_DIR" fetch --depth 1 origin "$DEEPGEMM_GIT_REF"
        git -C "$DEEPGEMM_SRC_DIR" checkout FETCH_HEAD
        git -C "$DEEPGEMM_SRC_DIR" submodule update --init --recursive --depth 1
    else
        echo "==> Cloning DeepGEMM from $DEEPGEMM_GIT_REPO ($DEEPGEMM_GIT_REF)"
        git clone --recursive --branch "$DEEPGEMM_GIT_REF" --depth 1 \
            "$DEEPGEMM_GIT_REPO" "$DEEPGEMM_SRC_DIR"
    fi
    export DEEPGEMM_SRC_DIR
    echo "==> DeepGEMM: $(git -C "$DEEPGEMM_SRC_DIR" rev-parse --short HEAD) from $DEEPGEMM_SRC_DIR"

    # Fail loudly here rather than 40 minutes into a compile: if these two
    # architecture branches are missing, this is the wrong DeepGEMM revision
    # and the build would reproduce the exact asserts we're fixing.
    if ! grep -q "arch_major == 12" "$DEEPGEMM_SRC_DIR/csrc/apis/hyperconnection.hpp"; then
        echo "!! $DEEPGEMM_SRC_DIR/csrc/apis/hyperconnection.hpp has no arch_major == 12 branch." >&2
        echo "!! This DeepGEMM revision lacks SM120 support - check DEEPGEMM_GIT_REF (want: nv_dev)." >&2
        exit 1
    fi

    # --break-system-packages: fine on a throwaway pod rebuilt from the
    # image rather than hand-maintained; do not use this on a machine you
    # upgrade in place. --index-strategy unsafe-best-match: without it, uv
    # can silently prefer a plain-PyPI torch build over the cu130 one on
    # --extra-index-url even when a specific version is named - this
    # requirements file pins the exact torch version (2.13.0, checked
    # against the actual file at build time, not hardcoded here) vLLM's
    # build expects, and we need that exact version resolved from the
    # cu130 index, not whatever plain PyPI happens to have.
    uv pip install --system --break-system-packages --index-strategy unsafe-best-match \
        -r "$VLLM_SRC_DIR/requirements/build/cuda.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu130

    export TORCH_CUDA_ARCH_LIST
    export MAX_JOBS
    export NVCC_THREADS
    export VLLM_TARGET_DEVICE=cuda

    # setuptools-scm derives the version from git tags, and this checkout is
    # deliberately shallow (--depth 1) so it contains none. On upstream that
    # still fell back to a synthetic "0.1.dev1+g<sha>"; on a fork branch it
    # instead yields no version at all, writes "Version: None" into
    # vllm.egg-info, and then `build` crashes resolving dependencies:
    #   TypeError: 'NoneType' object is not iterable   (packaging/version.py)
    # preceded by 'is shallow and may cause errors' / 'version of None
    # already set'. Pinning the version explicitly sidesteps git inference
    # entirely - cheaper than un-shallowing vLLM's very large history just to
    # recover a version string we're synthesising anyway. Same shape as the
    # upstream fallback, so the wheel still names its commit.
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.1.dev1+g${vllm_commit}"
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.1.dev1+g${vllm_commit}"
    echo "==> Pinning build version to $SETUPTOOLS_SCM_PRETEND_VERSION (shallow checkout has no tags)"

    # A previous failed build can leave egg-info carrying that "Version:
    # None", which poisons the dependency check again even once the pretend
    # version is set - it's regenerated from scratch below either way.
    rm -rf "$VLLM_SRC_DIR"/vllm.egg-info

    (
        cd "$VLLM_SRC_DIR"
        python3 -m build --wheel --no-isolation -o "$VLLM_WHEEL_DIR"
    ) 2>&1 | tee "$LOG_DIR/vllm-build.log"

    # No renaming needed: setuptools-scm already bakes the commit into the
    # wheel's own version segment (e.g. "vllm-0.1.dev1+g30b4e7f47-...whl"),
    # which is where a "+something" is actually valid in a wheel filename
    # - appending our own after the platform tag instead produced an
    # invalid filename that `pip`/`uv` correctly refused to install.
    existing_wheel="$(ls -t "$VLLM_WHEEL_DIR"/vllm-*.whl 2>/dev/null | head -1 || true)"
    if [ -z "$existing_wheel" ]; then
        echo "!! Build finished but no wheel found in $VLLM_WHEEL_DIR - check $LOG_DIR/vllm-build.log" >&2
        exit 1
    fi
    echo "==> Built and cached: $existing_wheel (commit $vllm_commit)"
fi

echo "==> Installing vLLM from wheel: $existing_wheel"
pip install -q -U uv

# Uninstall + physically clear the package directory before installing,
# rather than relying on --force-reinstall alone. Installing the SM120 fork
# over a previous upstream install left a genuinely mixed tree: a stale
# upstream quantization/inc/inc.py importing a symbol the fork's older
# fused_moe package doesn't export, which surfaces at startup as
#   ImportError: cannot import name 'RoutedExperts' from
#   vllm.model_executor.layers.fused_moe
# Any file the new wheel doesn't happen to overwrite survives, so the only
# reliable fix is to remove the old tree outright. Safe: this directory is
# owned entirely by the wheel installed on the next line.
# --break-system-packages is required on uninstall too, not just install:
# without it this fails with PEP 668 "externally managed" and, because the
# failure was previously swallowed, silently left the old package in place.
# Errors are no longer sent to /dev/null - only a genuinely-absent package
# should be tolerated, and `|| true` covers that.
uv pip uninstall --system --break-system-packages vllm || true

# The dist-info must go too, not just the package directory. uv treats an
# existing dist-info as proof the package is installed: deleting only
# vllm/ and reinstalling made it skip vllm entirely ("Installed 1 package"
# naming an unrelated dependency), leaving metadata with no code behind it.
for site_dir in /usr/local/lib/python3.12/dist-packages /usr/lib/python3/dist-packages; do
    for leftover in "$site_dir/vllm" "$site_dir"/vllm-*.dist-info; do
        if [ -e "$leftover" ]; then
            echo "==> Removing leftover: $leftover"
            rm -rf "$leftover"
        fi
    done
done

uv pip install --system --break-system-packages --force-reinstall "$existing_wheel"

echo "==> vLLM version now installed:"
vllm --version

# `vllm --version` doesn't touch the quantization registry, so it happily
# passes on an installation that dies seconds later at server startup.
# Import that registry explicitly here: it is what surfaced the stale-tree
# breakage (upstream ships quantization/inc/ as a package DIRECTORY whose
# inc.py imports RoutedExperts; this fork ships quantization/inc.py as a
# module FILE that doesn't. Python prefers the directory, so a leftover
# upstream inc/ shadows the fork's inc.py and raises ImportError). Catch it
# here, right after install, instead of after the model has loaded.
echo "==> Verifying the installed tree imports cleanly"
# get_quantization_config() must be CALLED, not merely imported: the
# `from .inc import INCConfig` that breaks lives inside the function body,
# so importing the symbol alone runs none of it and this check passed
# happily on a tree that then died at server startup.
if ! python3 -c "
from vllm.model_executor.layers.quantization import get_quantization_config
get_quantization_config('fp8')
" 2>&1; then
    echo "!! The installed vllm tree is inconsistent - most likely files from a previous," >&2
    echo "!! different vLLM build shadowing this one. Clear it completely and rerun:" >&2
    echo "!!   uv pip uninstall --system vllm" >&2
    echo "!!   rm -rf /usr/local/lib/python3.12/dist-packages/vllm" >&2
    exit 1
fi

echo "==> Installing Hugging Face download tooling"
# No "[cli]" extra: huggingface_hub 1.x folded the `hf` CLI into the base
# package and no longer has an extra by that name (uv just warns and
# installs anyway if you ask for it, but there's no reason to keep asking).
uv pip install --system --break-system-packages -U huggingface_hub hf_transfer

mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

# huggingface_hub now downloads through hf-xet (its new storage backend) by
# default whenever it's installed, in preference to hf_transfer/plain HTTP -
# and hf-xet is known to fail reconstructing large files (>15GB, which
# every shard of these checkpoints exceeds) with errors like "File
# reconstruction error: ... receiver dropped" or "Background writer channel
# closed" (https://github.com/huggingface/xet-core/issues/763). Setting
# HF_HUB_DISABLE_XET=1 is not reliable by itself (a huggingface_hub bug
# still routes through xet regardless: https://github.com/huggingface/
# huggingface_hub/issues/3266) - actually uninstalling the package is the
# only fix confirmed to work, forcing a fall back to plain HTTP/hf_transfer.
uv pip uninstall --system hf_xet 2>/dev/null || true
export HF_HUB_DISABLE_XET=1

# One-time migration: earlier runs before HF_HOME pointed at /workspace may
# have left partial/complete downloads under the default root-disk cache -
# harmless to leave, but they're dead weight eating the ~30GB root disk for
# no benefit once every download goes through $HF_HOME instead, so reclaim
# the space if any of that leftover state is still there.
stale_dir="$HOME/.cache/huggingface/hub/models--${MODEL_REPO/\//--}"
if [ -d "$stale_dir" ]; then
    echo "==> Removing stale root-disk HF cache entry: $stale_dir"
    rm -rf "$stale_dir"
fi

# ~167GB, and /workspace on RunPod is a network volume whose usable quota
# can be well below the size shown in the dashboard - check before spending
# an hour downloading into a wall.
echo "==> Free space on \$HF_HOME's volume before download:"
df -h "$HF_HOME"

echo "==> Downloading $MODEL_REPO (~167GB)"
hf download "$MODEL_REPO" 2>&1 | tee "$LOG_DIR/download.log"

start_vllm
wait_for_health

runpod_print_summary
echo "    vLLM wheel cache: ${VLLM_WHEEL_DIR} (reused on the next run unless FORCE_REBUILD_VLLM=true)"
echo "    HF model cache  : ${HF_HOME} (reused on the next run - or use start.sh to skip straight to launching)"
