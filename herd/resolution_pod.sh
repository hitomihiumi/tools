#!/usr/bin/env bash
# Which frame width (herd.py resolution) on a pod: the barn's labelled frames at Full HD down to 800 px -
# detector, posture / activity, identity, and the cameras one machine keeps up with at each width.
#
#   1. WORK/models/: what docker/models/ holds - its archives (unpacked here as the Docker build does), or the
#      image's /models copied out (docker cp <container>:/models ...; a models_unpacked/ folder inside is moved up)
#   2. WORK/data/: barn.toml, the boxes folder (frames.json and frames/), the labels, a recording of a camera
#   3. /models -> WORK/models, /data -> WORK/data: barn.toml's paths as they are
#   4. herd.py resolution -> WORK/data/OUT/ (resolution.md, resolution.json, the log) and
#      WORK/export/herd_resolution_<date>.tgz
#
#   bash herd/resolution_pod.sh          start in tmux "herd-resolution"
#   bash herd/resolution_pod.sh log      follow it
#   bash herd/resolution_pod.sh stop
#
# Knobs: WORK=/workspace  CONFIG=barn.toml  BOXES=boxes_first10  LABELS=labels_first10.json  (all in WORK/data)
#        VIDEO=rec_first10/cam1_....mp4 (in WORK/data; default: the first .mp4 / .mkv under WORK/data/rec*)
#        WIDTHS=1920,1600,1280,1024,960,800  FRAMES=0 (at most this many labelled frames; 0: all)  OUT=resolution
#        CORES= (the barn machine's CPU cores for the estimate; default: this pod's)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
M="$WORK/models"
D="$WORK/data"
OUTD="$D/${OUT:-resolution}"          # OUT stays the knob as given (passed on to the tmux run)
LOG="$OUTD/resolution_log.txt"
SESSION=herd-resolution

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_RES_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[resolution finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$OUTD"
    knobs=""
    for v in WORK CONFIG BOXES LABELS VIDEO WIDTHS FRAMES OUT CORES; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_RES_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/resolution_pod.sh"); echo; echo '[resolution finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$OUTD"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "!! $(basename "$0") stopped at line $LINENO: $BASH_COMMAND (exit $?)"' ERR   # never silently
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$WORK/herd/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"

CONFIG="$D/${CONFIG:-barn.toml}"
BOXES="$D/${BOXES:-boxes_first10}"
LABELS="$D/${LABELS:-labels_first10.json}"
if [ -n "${VIDEO:-}" ]; then
    VIDEO="$D/$VIDEO"
else
    VIDEO="$(find "$D" -maxdepth 2 -path "$D/rec*" \( -name '*.mp4' -o -name '*.mkv' \) 2>/dev/null | sort | head -1 || true)"
fi
missing=""
[ -f "$CONFIG" ] || missing+="  $CONFIG (docker/data/barn.toml)\n"
[ -f "$BOXES/frames.json" ] || missing+="  $BOXES/frames.json (docker/data/boxes_first10/, with its frames/)\n"
[ -d "$BOXES/frames" ] || missing+="  $BOXES/frames/ (the labelled pictures)\n"
[ -f "$LABELS" ] || missing+="  $LABELS (the labels.json the box page saved)\n"
[ -z "$missing" ] || { printf "!! not in %s:\n%b" "$D" "$missing"; exit 1; }
[ -n "$VIDEO" ] && [ -f "$VIDEO" ] || { echo "(no recording - the decode is not timed; VIDEO=rec_.../cam1_....mp4)"; VIDEO=""; }

step "Models in $M"
mkdir -p "$M"
for nest in "$M"/models_unpacked "$M"/models; do         # a copied folder that landed one level too deep
    [ -d "$nest" ] || continue
    echo "moving $(basename "$nest")/* up into $M"
    (shopt -s dotglob; for x in "$nest"/*; do
        n="$(basename "$x")"
        if [ -e "$M/$n" ]; then echo "  $n: already in $M, kept"; else mv "$x" "$M/"; fi
    done)
    rmdir "$nest" 2>/dev/null || echo "  $nest not empty: left"
done
for a in "$M"/*.tgz "$M"/*.tar.gz; do                    # archives, unpacked as the Docker build does
    [ -f "$a" ] || continue
    mark="$M/.unpacked_$(basename "$a")"
    [ -f "$mark" ] && continue
    gzip -t "$a" 2>/dev/null || { echo "!! $(basename "$a") is cut short (its upload not finished?): skipped"; continue; }
    echo "unpacking $(basename "$a")"
    tar -xzf "$a" -C "$M"
    touch "$mark"
done
for p in /models /data; do
    t="$M"; [ "$p" = /data ] && t="$D"
    if [ -L "$p" ] || [ ! -e "$p" ]; then
        ln -sfn "$t" "$p"
    elif [ "$(readlink -f "$p")" != "$(readlink -f "$t")" ]; then
        echo "!! $p is a folder of its own on this pod: barn.toml's $p/... paths would not reach $t"; exit 1
    fi
done
echo "/models -> $M, /data -> $D"
python3 - "$CONFIG" <<'PYEOF'
import os, re, sys
gone = []
for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    for p in re.findall(r'"(/models/[^"]+)"', line.split("#", 1)[0]):
        if not os.path.exists(p):
            gone.append(f"  line {n}: {p}")
if gone:
    sys.exit("!! barn.toml names models not in /models - upload them (docker/models, or the image's /models):\n"
             + "\n".join(gone) + "\n   what is there: ls -R /models | head -50")
print("every /models/... path in barn.toml is here")
PYEOF
export HF_HOME="$M/hf-cache" TZ="${TZ:-Europe/Kyiv}"     # as in the Docker image
if [ -d "$M/hf-cache" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # DINOv2 from the copy, not downloaded again
fi

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
if grep -rqs '"kind": *"rfdetr"' "$M" --include=det_train_meta.json; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "CPU cores: $(nproc); RAM: $(free -g | awk '/Mem:/ {print $2 " GB, " $7 " GB free"}')"

step "herd.py resolution: widths ${WIDTHS:-1920,1600,1280,1024,960,800}"
args=(--config "$CONFIG" --boxes "$BOXES" --labels "$LABELS" --widths "${WIDTHS:-1920,1600,1280,1024,960,800}"
      --frames "${FRAMES:-0}" --out "$OUTD")
[ -n "$VIDEO" ] && args+=(--video "$VIDEO")
[ -n "${CORES:-}" ] && args+=(--cores "$CORES")
"$PY" "$HERE/herd.py" resolution "${args[@]}"

step "Export"
mkdir -p "$WORK/export"
EXP="$WORK/export/herd_resolution_$(date +%Y%m%d_%H%M).tgz"
tar -czf "$EXP" -C "$(dirname "$OUTD")" "$(basename "$OUTD")"
echo "-> $EXP   (scp it back: resolution.md, resolution.json, the log)"
cat "$OUTD/resolution.md" 2>/dev/null || true
