#!/usr/bin/env bash
# LoRA fine-tuning of Muse Glimmer on CBVD-5, end to end, on one A100.
#
#   environment -> dataset -> model -> base-model control -> train -> eval -> compare
#
# Every step is skipped when its output already exists, and training resumes
# from its last checkpoint, so after a crash or a pod restart just run it again.
#
# Installs what it needs by itself:
#   system tools   curl, git, tmux, unzip           (apt / dnf / yum)
#   CUDA toolkit   matched to the driver, <= 12.8   (NVIDIA's repo; CUDA_WANT=skip to leave it)
#   Python         uv, Python 3.12 and a venv in $WORK/lora/.venv
#   torch          the build for that same CUDA version
#   the rest       lora/requirements.txt (transformers, peft, accelerate, ...)
# Only the NVIDIA driver must already be there.
#
# Usage (the cowbench folder must be this script's parent):
#   bash cowbench/lora/run_lora.sh           start in a background tmux session "lora"
#
# Then, from this or any other terminal / SSH connection:
#   bash cowbench/lora/run_lora.sh attach    watch it live (detach: Ctrl-b, then d)
#   bash cowbench/lora/run_lora.sh log       follow the log file (Ctrl-c stops watching only)
#   bash cowbench/lora/run_lora.sh status    one screen: stage, progress, loss, GPU
#   bash cowbench/lora/run_lora.sh stop      stop the run (restarting resumes it)
#
# The run lives in tmux, not in your shell, so closing the terminal or losing
# SSH does not stop it. Everything it prints also goes to
# $WORK/lora-runs/<run>/log.txt, with $WORK/lora-runs/current.log pointing at it.
#
# Knobs, all optional:
#   WORK=/workspace        where the venv, model cache, dataset and runs go
#                          (default /workspace if writable, else ~/lora-work)
#   WIDTH=896              frame width fed to the model (train and eval)
#   EPOCHS=1               passes over the training cows
#   TRAIN_LIMIT=0          cap on training cows, 0 = all 22 478 (e.g. 6000 for a quick proof)
#   BATCH=4 ACCUM=4        per-step batch and gradient accumulation (effective 16)
#   LR=1e-4 RANK=16        LoRA learning rate and rank
#   EVAL_BASE=1            also score the untouched model the same way, as the control
#   PRECISION=auto         bf16 on an 80 GB card, qlora on a 40 GB one
#   HF_TOKEN=...           only for qlora: the BF16 repo may be gated
#   CUDA_WANT=12.8         newest CUDA toolkit to install, or "skip"
#   TORCH_BACKEND=cu128    force a torch build instead of matching CUDA
#   RUN_NAME=...           default: lora_w<WIDTH>[_n<TRAIN_LIMIT>] - stable, so a rerun resumes it

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(dirname "$HERE")"

# /workspace is the persistent volume on RunPod-style machines. Elsewhere it may
# not exist or not be writable, and then the home directory is used instead.
if [ -z "${WORK:-}" ]; then
    if mkdir -p /workspace 2>/dev/null && [ -w /workspace ]; then WORK=/workspace
    else WORK="$HOME/lora-work"; fi
fi
WIDTH="${WIDTH:-896}"
EPOCHS="${EPOCHS:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-4}"
RANK="${RANK:-16}"
EVAL_BASE="${EVAL_BASE:-1}"
PRECISION="${PRECISION:-auto}"
# No date in the default name: a run restarted after midnight must find its
# own checkpoints, not start a fresh directory.
if [ "$TRAIN_LIMIT" != "0" ]; then RUN_NAME="${RUN_NAME:-lora_w${WIDTH}_n${TRAIN_LIMIT}}"; fi
RUN_NAME="${RUN_NAME:-lora_w${WIDTH}}"

MODEL="RedHatAI/Muse-Glimmer-30B-FP8-block"
MODEL_BF16="meta-models/Muse-Glimmer-30B"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"

# On the persistent volume, not $HOME: a pod's home is wiped on restart, and
# neither the 34 GB checkpoint nor the 12 GB dataset is worth fetching twice.
export HF_HOME="$WORK/hf-cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
VENV="$WORK/lora/.venv"
DATA="$WORK/cbvd5"
OUT="$WORK/lora-runs/$RUN_NAME"
RUNS="$WORK/lora-runs"
SESSION=lora
LOG="$OUT/log.txt"
STAGE_FILE="$RUNS/current.stage"

# ------------------------------------------------------------ system packages
# Everything the script shells out to, installed if missing. The NVIDIA driver
# is the one thing that cannot be installed from here: on a rented GPU machine
# it comes from the host, and without it there is no GPU to train on.
SYS_TOOLS=(curl git tmux unzip)
ensure_system() {
    local missing=() t
    for t in "${SYS_TOOLS[@]}"; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
    [ -d /etc/ssl/certs ] || missing+=(ca-certificates)
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "installing system packages: ${missing[*]}"
        local SUDO=""
        if [ "$(id -u)" -ne 0 ]; then
            if command -v sudo >/dev/null 2>&1; then SUDO="sudo"
            else echo "!! need root or sudo to install: ${missing[*]}"; exit 1; fi
        fi
        if command -v apt-get >/dev/null 2>&1; then
            export DEBIAN_FRONTEND=noninteractive
            $SUDO apt-get update -qq
            $SUDO apt-get install -y -qq --no-install-recommends "${missing[@]}" ca-certificates
        elif command -v dnf >/dev/null 2>&1; then
            $SUDO dnf install -y -q "${missing[@]}" ca-certificates
        elif command -v yum >/dev/null 2>&1; then
            $SUDO yum install -y -q "${missing[@]}" ca-certificates
        else
            echo "!! no apt-get, dnf or yum here - install by hand: ${missing[*]}"; exit 1
        fi
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "!! nvidia-smi is missing or cannot see a GPU. The NVIDIA driver comes from"
        echo "!! the machine image / host and cannot be installed by this script."
        exit 1
    fi
}

# ------------------------------------------------------------ watching
case "${1:-}" in
    attach)
        exec tmux attach -t "$SESSION" ;;
    log)
        # -F, not -f: keeps following across a restart that re-creates the file.
        exec tail -n 100 -F "$RUNS/current.log" ;;
    status)
        echo "stage : $(cat "$STAGE_FILE" 2>/dev/null || echo 'not started')"
        if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux  : session '$SESSION' running"
        else echo "tmux  : no session - finished, failed or not started"; fi
        if [ -f "$RUNS/current.log" ]; then
            # tqdm redraws its bar with \r; the last redraw is the current state.
            recent="$(tail -c 50000 "$RUNS/current.log" | tr '\r' '\n')"
            bar="$(printf '%s\n' "$recent" | grep -E '[0-9]+/[0-9]+ \[' | tail -1 || true)"
            [ -n "$bar" ] && echo "bar   : $bar"
            loss="$(tr '\r' '\n' < "$RUNS/current.log" | grep -oE "\{'loss'[^}]*\}" | tail -1 || true)"
            [ -n "$loss" ] && echo "loss  : $loss"
            echo "--- last lines ---"
            printf '%s\n' "$recent" | grep -v '^[[:space:]]*$' | tail -8
        fi
        echo "--- GPU ---"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
        exit 0 ;;
    stop)
        tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
        sleep 5
        tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "no session '$SESSION'"
        exit 0 ;;
    "") ;;
    *)
        echo "usage: $0 [attach|log|status|stop]"; exit 2 ;;
esac

# ------------------------------------------------------------ into tmux
# Started from a plain shell: re-launch this same script inside a detached
# tmux session and return at once. The knobs are passed explicitly - a tmux
# server that is already running would not see this shell's environment.
if [ -z "${TMUX:-}" ] && [ -z "${LORA_IN_TMUX:-}" ]; then
    ensure_system
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "A run is already going in tmux session '$SESSION'. Watch it with:"
        echo "  bash $0 attach      or      bash $0 status"
        exit 1
    fi
    knobs=""
    for v in WORK WIDTH EPOCHS TRAIN_LIMIT BATCH ACCUM LR RANK EVAL_BASE PRECISION RUN_NAME KEEP_ZIP HF_TOKEN CUDA_WANT TORCH_BACKEND; do
        [ -n "${!v:-}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    self="$(printf '%q' "$HERE/$(basename "${BASH_SOURCE[0]}")")"
    # The shell stays open after the script ends, so an attach after a failure
    # still shows the error instead of a vanished session.
    tmux new-session -d -s "$SESSION" -x 200 -y 50         "env LORA_IN_TMUX=1 $knobs bash $self; echo; echo \"[run_lora.sh exited with code \$?]\"; exec bash"
    echo "Started in tmux session '$SESSION' (run: $RUN_NAME)."
    echo
    echo "  watch live :  bash $0 attach     (leave with Ctrl-b, then d - the run keeps going)"
    echo "  log file   :  bash $0 log        ($LOG)"
    echo "  summary    :  bash $0 status"
    echo "  stop       :  bash $0 stop"
    exit 0
fi

mkdir -p "$WORK/lora" "$OUT"
ln -sfn "$LOG" "$RUNS/current.log"
# Everything below goes to the screen and to the log file. Python is told not
# to buffer, or the log would lag minutes behind what is actually happening.
exec > >(tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
echo "#### run_lora.sh started $(date '+%Y-%m-%d %H:%M:%S')  run=$RUN_NAME"
trap 'echo "$(date "+%H:%M:%S")  run_lora.sh exited with code $?" >> "$STAGE_FILE"' EXIT

step() {
    echo; echo "=============================================================="
    echo " $*   [$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "=============================================================="
    echo "$(date '+%H:%M:%S')  $RUN_NAME  $*" > "$STAGE_FILE"
}

step "1/9  System packages, GPU and disk"
ensure_system   # again: a run started inside an existing tmux skipped the check above
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
gpu_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "$used_mib" -gt 2000 ]; then
    echo "!! ${used_mib} MiB of GPU memory is already in use (a vLLM server?)."
    echo "!! BF16 LoRA needs ~62 GiB of the card to itself. Stop the other process first."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi
if [ "$PRECISION" = "auto" ]; then
    if [ "$gpu_mib" -ge 71680 ]; then PRECISION=bf16; else PRECISION=qlora; fi
fi
echo "precision: $PRECISION"
df -h "$WORK" | tail -1
free_gb="$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')"
# model 34 GB + dataset zip ~12 GB + unpacked keyframes 3 GB + venv ~8 GB + checkpoints
need_gb=65
[ "$PRECISION" = "qlora" ] && need_gb=90   # the BF16 weights are 60 GB on their own
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "!! ${free_gb} GB free on $WORK, about ${need_gb} GB needed. Continuing, but expect a full disk."
fi

step "2/9  CUDA toolkit"
# torch wheels carry their own CUDA runtime, so training itself would run
# without this. The toolkit (nvcc, headers, libs) is installed anyway so that
# anything compiling kernels on first use - Triton, bitsandbytes, a
# flash-attn build - finds a CUDA that matches the driver, and so torch below
# can be pinned to the same CUDA version instead of guessed.
#
# The version follows the driver: a toolkit newer than the driver cannot run
# on it. CUDA_WANT caps it (default 12.8, a line every current torch ships
# for); set CUDA_WANT=skip to leave CUDA alone.
CUDA_WANT="${CUDA_WANT:-12.8}"
driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
echo "driver supports CUDA up to: ${driver_cuda:-unknown}"

ver_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }

cuda_installed_version() {
    local nvcc
    nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    [ -x "$nvcc" ] || return 1
    "$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

# Duplicate NVIDIA apt sources (an old apt-key entry plus the keyring's own,
# Signed-By) make every apt command fail. Keep the keyring's file only.
cuda_fix_apt_sources() {
    local distro="$1" canonical="/etc/apt/sources.list.d/cuda-${distro}-x86_64.list" f
    while IFS= read -r f; do
        [ "$f" = "$canonical" ] && continue
        if [ "$f" = "/etc/apt/sources.list" ]; then
            $SUDO sed -i '\|developer\.download\.nvidia\.com/compute/cuda|s|^|#|' "$f"
        else
            $SUDO mv "$f" "$f.disabled"
        fi
        echo "  disabled duplicate NVIDIA source: $f"
    done < <(grep -rl "developer\.download\.nvidia\.com/compute/cuda" \
                 /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | grep -v '\.disabled$' || true)
}

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
CUDA_VER=""
if [ "$CUDA_WANT" = "skip" ]; then
    echo "CUDA_WANT=skip - not touching the CUDA toolkit"
else
    target="$CUDA_WANT"
    if [ -n "$driver_cuda" ] && ! ver_le "$target" "$driver_cuda"; then target="$driver_cuda"; fi
    current="$(cuda_installed_version || true)"
    if [ -n "$current" ] && [ "${current%%.*}" = "${target%%.*}" ] && ver_le "$current" "${driver_cuda:-99}"; then
        echo "CUDA toolkit $current already installed and usable with this driver"
        CUDA_VER="$current"
    else
        echo "installing CUDA toolkit $target (found: ${current:-none})"
        . /etc/os-release
        arch="$(uname -m)"; [ "$arch" = "aarch64" ] && arch=sbsa
        if command -v apt-get >/dev/null 2>&1; then
            distro="${ID}${VERSION_ID//./}"          # ubuntu2204, debian12, ...
            cuda_fix_apt_sources "$distro"
            if curl -fsSL -o /tmp/cuda-keyring.deb \
                "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb" \
               && $SUDO dpkg -i /tmp/cuda-keyring.deb \
               && $SUDO apt-get update -qq \
               && $SUDO apt-get install -y -qq --no-install-recommends "cuda-toolkit-${target//./-}"; then
                CUDA_VER="$target"
            fi
        elif command -v dnf >/dev/null 2>&1; then
            distro="rhel${VERSION_ID%%.*}"
            if $SUDO dnf config-manager --add-repo \
                "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-${distro}.repo" \
               && $SUDO dnf install -y -q "cuda-toolkit-${target//./-}"; then
                CUDA_VER="$target"
            fi
        fi
        if [ -z "$CUDA_VER" ]; then
            # Not fatal: torch brings its own runtime and nothing in the
            # default path compiles CUDA code. Say so and carry on.
            echo "!! could not install CUDA toolkit $target here (no NVIDIA repo for ${ID:-?} ${VERSION_ID:-?}?)."
            echo "!! Continuing: torch ships its own CUDA runtime, training does not need nvcc."
            CUDA_VER="${current:-$target}"
        fi
    fi
    # A fresh install can leave the generic symlink missing, or still pointing
    # at an older toolkit that was already on the machine.
    if [ -d "/usr/local/cuda-${CUDA_VER}" ] \
       && [ "$(readlink -f /usr/local/cuda 2>/dev/null)" != "$(readlink -f "/usr/local/cuda-${CUDA_VER}")" ]; then
        $SUDO ln -sfn "/usr/local/cuda-${CUDA_VER}" /usr/local/cuda
    fi
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
    if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -2; fi
fi

# The torch build to install: the newest CUDA line torch publishes that is not
# newer than the toolkit (or, without one, the driver).
torch_cuda="${CUDA_VER:-${driver_cuda:-12.8}}"
TORCH_BACKEND="${TORCH_BACKEND:-}"
if [ -z "$TORCH_BACKEND" ]; then
    for c in 13.0 12.9 12.8 12.6; do
        if ver_le "$c" "$torch_cuda"; then TORCH_BACKEND="cu${c//./}"; break; fi
    done
    # Below 12.6 current torch is no longer built (cu121/cu124 stop at old
    # releases transformers 5 does not support). Within CUDA 12, a cu126 wheel
    # still runs on an older 12.x driver - minor-version compatibility - and an
    # A100 needs no PTX JIT that would break it, since every build carries
    # sm_80 kernels.
    if [ -z "$TORCH_BACKEND" ] && [ "${torch_cuda%%.*}" = "12" ]; then TORCH_BACKEND=cu126; fi
    TORCH_BACKEND="${TORCH_BACKEND:-auto}"
fi
echo "torch build: $TORCH_BACKEND"

step "3/9  Python environment"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
# An old uv already on the machine is how the Blackwell pod ended up with a
# torch that could not run on its GPU; update it rather than trust it.
uv self update >/dev/null 2>&1 || true
uv --version

if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 --seed --managed-python "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The question is not "does torch import" but "can this venv build Muse
# Glimmer": the architecture is new (its config was written by
# transformers 5.15.0.dev0), and an older transformers imports fine and only
# fails at from_pretrained with "model type muse_glimmer not recognized".
env_ready() {
    python - <<'PY' 2>/dev/null
import sys
try:
    import torch, peft, accelerate, safetensors, PIL, requests, jinja2  # noqa: F401
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
except Exception:
    sys.exit(1)
if "muse_glimmer" not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
    sys.exit(1)
major, minor = torch.cuda.get_device_capability()
sys.exit(0 if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else 1)
PY
}

if ! env_ready; then
    # torch from the CUDA line picked in step 2; the PyTorch index directly if
    # this uv does not know that backend name.
    uv pip install torch torchvision --torch-backend="$TORCH_BACKEND" \
        || uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_BACKEND" \
        || uv pip install torch torchvision --torch-backend=auto
    # If no released transformers is new enough yet, install the rest now and
    # take transformers from main just below.
    uv pip install -r "$HERE/requirements.txt" \
        || uv pip install -r <(grep -v '^transformers' "$HERE/requirements.txt")
    if ! env_ready; then
        echo "  no released transformers knows muse_glimmer - installing from main"
        uv pip install "git+https://github.com/huggingface/transformers.git"
    fi
    if ! env_ready; then
        echo "!! The environment is still not usable:"
        python - <<'PY' || true
import torch, transformers
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as M
print("   torch       :", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())
print("   sees the GPU:", torch.cuda.is_available(), "(False: torch's CUDA is newer than the driver - try TORCH_BACKEND=cu124)")
print("   this GPU    :", "sm_%d%d" % torch.cuda.get_device_capability())
print("   transformers:", transformers.__version__, "| knows muse_glimmer:", "muse_glimmer" in M)
PY
        exit 1
    fi
fi
if [ "$PRECISION" = "qlora" ] && ! python -c "import bitsandbytes" 2>/dev/null; then
    uv pip install bitsandbytes
fi
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

step "4/9  Dataset"
if [ ! -f "$DATA/annotations/ava_train_v2.1.csv" ] || [ ! -d "$DATA/labelframes" ]; then
    ZIP="$WORK/cbvd-5cow-behavior-video-dataset.zip"
    if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
        echo "downloading CBVD-5 (~12 GB)"
        curl -L --fail --retry 5 -C - -o "$ZIP" "$DATASET_URL" || curl -L --fail --retry 5 -o "$ZIP" "$DATASET_URL"
    fi
    # Only what training reads: the annotations and the keyframes. The mp4s
    # and the rawframes are two thirds of the archive and are not used.
    prefix="$(unzip -Z1 "$ZIP" | grep -m1 'annotations/ava_train_v2.1.csv$' | sed 's|annotations/ava_train_v2.1.csv$||')"
    echo "archive prefix: '${prefix}'"
    mkdir -p "$DATA"
    unzip -q -o "$ZIP" "${prefix}annotations/*" "${prefix}labelframes/*" -d "$DATA/_x"
    rm -rf "$DATA/annotations" "$DATA/labelframes"
    mv "$DATA/_x/${prefix}annotations" "$DATA/annotations"
    mv "$DATA/_x/${prefix}labelframes" "$DATA/labelframes"
    rm -rf "$DATA/_x"
    [ "${KEEP_ZIP:-0}" = "1" ] || rm -f "$ZIP"
fi
echo "keyframes: $(find "$DATA/labelframes" -name '*.jpg' | wc -l)"

step "5/9  Model"
if [ "$PRECISION" = "qlora" ]; then
    hf download "$MODEL_BF16"
    hf download "$MODEL" --include "*.json" "*.jinja"   # the processor and chat template
    BASE_ARG=(--base "$MODEL_BF16")
else
    hf download "$MODEL"
    BASE_ARG=()
fi

COMMON=(--root "$DATA" --out "$OUT" --width "$WIDTH" --precision "$PRECISION" "${BASE_ARG[@]}")
cd "$BENCH"

step "6/9  Training data check"
python lora/train_lora.py data "${COMMON[@]}" --train-limit "$TRAIN_LIMIT"

step "7/9  Control: the untouched model, same width, no reasoning"
# The zero-shot runs in runs/ were made with reasoning on and at 1280/1920 px.
# This arm changes only what the LoRA arm changes apart from the training -
# width and no reasoning - so base-vs-LoRA isolates what the training bought.
# Not fatal: a failure here should not cost the night's training.
if [ "$EVAL_BASE" = "1" ]; then
    python lora/train_lora.py eval "${COMMON[@]}" --adapter none \
        || echo "!! base-model eval failed; continuing to training"
fi

step "8/9  Training"
if [ -f "$OUT/adapter/adapter_config.json" ] && [ -f "$OUT/train_meta.json" ]; then
    echo "adapter already trained: $OUT/adapter"
else
    python lora/train_lora.py train "${COMMON[@]}" --train-limit "$TRAIN_LIMIT" \
        --epochs "$EPOCHS" --batch "$BATCH" --accum "$ACCUM" --lr "$LR" --rank "$RANK" \
        --alpha "$((RANK * 2))"
fi

step "9/9  Evaluation on all 2532 val cows"
python lora/train_lora.py eval "${COMMON[@]}" --adapter "$OUT/adapter"

ZS="$BENCH/runs/2026-09-25_val-full_w1920_f1"
for arm in eval-base eval-lora; do
    [ -s "$OUT/$arm/results.jsonl" ] || continue
    python cowbench.py --out "$OUT/$arm" score
    python cowbench.py --out "$OUT/$arm" score --vote
    python cowbench.py --out "$OUT/$arm" report
done
python cowbench.py compare --a "$ZS" --b "$OUT/eval-lora" \
    --label-a "zero-shot 1920px, reasoning" --label-b "LoRA ${WIDTH}px" \
    --output "$OUT/eval-lora/compare_vs_zeroshot1920.md"
if [ -s "$OUT/eval-base/results.jsonl" ]; then
    python cowbench.py compare --a "$OUT/eval-base" --b "$OUT/eval-lora" \
        --label-a "base ${WIDTH}px, no reasoning" --label-b "LoRA ${WIDTH}px" \
        --output "$OUT/eval-lora/compare_vs_base.md"
fi

echo
echo "Done. Everything is in $OUT:"
echo "  adapter/                      the LoRA weights (~0.4 GB)"
echo "  train_meta.json               what was trained, how long, loss before and after"
echo "  eval-lora/report.md           the bench report for the fine-tuned model"
echo "  eval-lora/compare_*.md        paired comparisons against the zero-shot runs"
echo "Copy it back into cowbench/runs/ to keep it with the rest."
