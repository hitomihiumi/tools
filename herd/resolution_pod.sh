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
#        CROPS=224,native,168,112 (crop sizes compared: a side for every cow, or native - the box's own pixels)
#        CROP_WIDTHS=1920,1280 (the frame widths they are compared at; "" to skip)  ONLY_CROPS=1 (WIDTHS=1920 only)
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
    log)                                                # the run started last, whatever its OUT
        [ -z "${OUT+x}" ] && [ -f "$D/.resolution_last_log" ] && LOG="$(cat "$D/.resolution_last_log")"
        if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "(running in tmux '$SESSION'; log: $LOG)"; else echo "(not running now; log: $LOG)"; fi
        if [ ! -f "$LOG" ]; then
            echo "no log yet - the tmux screen:"; tmux capture-pane -p -t "=$SESSION:" -S -60 2>/dev/null || echo "  (no tmux session)"; exit 0
        fi
        exec tail -n 100 -F "$LOG" ;;
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
    for v in WORK CONFIG BOXES LABELS VIDEO WIDTHS FRAMES OUT CORES CROPS CROP_WIDTHS ONLY_CROPS; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_RES_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/resolution_pod.sh"); echo; echo '[resolution finished]'; exec bash"
    echo "$LOG" > "$D/.resolution_last_log"
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

source "$HERE/pod_setup.sh"

step "herd.py resolution: widths ${WIDTHS:-1920,1600,1280,1024,960,800}"
[ "${ONLY_CROPS:-0}" = "1" ] && WIDTHS="${WIDTHS:-1920}"
args=(--config "$CONFIG" --boxes "$BOXES" --labels "$LABELS" --widths "${WIDTHS:-1920,1600,1280,1024,960,800}"
      --frames "${FRAMES:-0}" --out "$OUTD")
[ -n "$VIDEO" ] && args+=(--video "$VIDEO")
[ -n "${CORES:-}" ] && args+=(--cores "$CORES")
args+=(--crops "${CROPS-224,native,168,112}" --crop-widths "${CROP_WIDTHS-1920,1280}")
"$PY" "$HERE/herd.py" resolution "${args[@]}"

step "Export"
mkdir -p "$WORK/export"
EXP="$WORK/export/herd_resolution_$(date +%Y%m%d_%H%M).tgz"
tar -czf "$EXP" -C "$(dirname "$OUTD")" "$(basename "$OUTD")"
echo "-> $EXP   (scp it back: resolution.md, resolution.json, the log)"
cat "$OUTD/resolution.md" 2>/dev/null || true
