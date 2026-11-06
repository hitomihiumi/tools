#!/usr/bin/env bash
# How many cameras one GPU keeps up with, live (herd.py stress): the barn's recordings played as N cameras at once
# through the barn's code (a tick a second, a 7 s burst a minute each), at each frame width - more cameras until
# it no longer keeps up - and one table of it all.
#
#   1. WORK/models/, WORK/data/barn.toml as for resolution_pod.sh; WORK/data/VIDEOS/ the recordings to play
#   2. for each width: the recordings re-encoded at it once (ffmpeg -> WORK/data/stress_clips_<w>/: a camera
#      sending that size; RESIZE=1 plays the Full HD files and lets the pipeline shrink them instead), barn.toml
#      with [sampling] frame_width = w
#   3. for each camera count (CAMERAS, rising): herd.py stress --videos ... --cameras N; a count that does not keep
#      up ends that width
#   4. WORK/data/OUT/stress.md (the table), each run's stress_herd_<N>cam.json / .md, the log;
#      WORK/export/herd_stress_<date>.tgz
#
#   bash herd/stress_pod.sh          start in tmux "herd-stress"
#   bash herd/stress_pod.sh log      follow it
#   bash herd/stress_pod.sh stop
#
# Knobs: WORK=/workspace  CONFIG=barn.toml  VIDEOS=rec_first10  WIDTHS="1920 1600 1280 1024 960 800"
#        CAMERAS="5 8 12 16 20 24" (each width after the first starts at the most cameras the one before kept up with:
#        a narrower frame is no heavier; ALL=1 tries every count at every width)
#        DURATION=240 (s measured)  WARMUP=60  RESIZE=0  OUT=stress  CLIP_S=0 (seconds of each recording kept, 0: all)
#        BURST_FPS= (frames a second a burst encodes, e.g. 12.5; default the config's)  BURST_EVERY= (s, e.g. 120)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
M="$WORK/models"
D="$WORK/data"
OUTD="$D/${OUT:-stress}"
LOG="$OUTD/stress_log.txt"
SESSION=herd-stress

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_STRESS_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[stress finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$OUTD"
    knobs=""
    for v in WORK CONFIG VIDEOS WIDTHS CAMERAS DURATION WARMUP RESIZE OUT CLIP_S BURST_FPS BURST_EVERY ALL; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_STRESS_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/stress_pod.sh"); echo; echo '[stress finished]'; exec bash"
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
VIDEOS="$D/${VIDEOS:-rec_first10}"
WIDTHS="${WIDTHS:-1920 1600 1280 1024 960 800}"
CAMERAS="${CAMERAS:-5 8 12 16 20 24}"
[ -f "$CONFIG" ] || { echo "!! no $CONFIG (docker/data/barn.toml)"; exit 1; }
[ -n "$(find "$VIDEOS" -maxdepth 1 \( -name '*.mp4' -o -name '*.mkv' \) 2>/dev/null | head -1)" ] \
    || { echo "!! no recordings (.mp4, .mkv) in $VIDEOS (VIDEOS=rec_...)"; exit 1; }

source "$HERE/pod_setup.sh"

best=0                                                     # the most cameras the width before kept up with
for w in $WIDTHS; do
    if [ "${RESIZE:-0}" = "1" ]; then
        clips="$VIDEOS"
    else
        clips="$D/stress_clips_$w"
        if [ ! -f "$clips/.done" ]; then
            step "The recordings at $w px (a camera sending that size)"
            rm -rf "$clips" && mkdir -p "$clips"
            for v in "$VIDEOS"/*.mp4 "$VIDEOS"/*.mkv; do
                [ -f "$v" ] || continue
                cut=(); [ "${CLIP_S:-0}" != "0" ] && cut=(-t "$CLIP_S")
                ffmpeg -loglevel error -y "${cut[@]}" -i "$v" -vf "scale=$w:-2" -c:v libx264 -preset veryfast -crf 23 -an \
                    "$clips/$(basename "${v%.*}").mp4"
                echo "  $(basename "$v")"
            done
            touch "$clips/.done"
        fi
    fi
    cfg="$OUTD/barn_$w.toml"
    python3 - "$CONFIG" "$cfg" "$w" "${BURST_FPS:-}" "${BURST_EVERY:-}" <<'PYEOF'
import re, sys
src, dst, w, bfps, bevery = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = re.sub(r'://[^/"@\s]+@', "://", text)                 # no camera passwords in the copy
def put(text, key, value):
    pat = rf'^{key}\s*=\s*[^#\n]*'
    if re.search(pat, text, flags=re.M):
        return re.sub(pat, f"{key} = {value}    ", text, count=1, flags=re.M)
    return re.sub(r'^\[sampling\][^\n]*\n', lambda m: m.group(0) + f"{key} = {value}\n", text, count=1, flags=re.M)
text = put(text, "frame_width", w)
if bfps:
    text = put(text, "burst_fps", bfps)
if bevery:
    text = put(text, "burst_every_s", bevery)
open(dst, "w", encoding="utf-8").write(text)
print(f"[sampling] frame_width = {w}" + (f", burst_fps = {bfps}" if bfps else "") + (f", burst_every_s = {bevery}" if bevery else ""))
PYEOF
    first=1
    kept=0
    for n in $CAMERAS; do
        if [ "${ALL:-0}" != "1" ] && [ "$n" -lt "$best" ]; then
            continue                                           # the wider frame kept up with more: so does this one
        fi
        out="$OUTD/w${w}_${n}cam"
        if [ -f "$out/stress_herd_${n}cam.json" ]; then
            step "$w px, $n cameras: done already ($out)"
            ok="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('live',{}).get('keeps_up'))" "$out/stress_herd_${n}cam.json")"
        else
            step "$w px, $n cameras live (${DURATION:-240} s measured after ${WARMUP:-60} s)"
            alone=0; [ "$first" = 1 ] && alone=3                   # the cost of a burst alone, once a width
            set +e
            "$PY" "$HERE/herd.py" stress --config "$cfg" --videos "$clips" --cameras "$n" --duration "${DURATION:-240}" \
                --warmup "${WARMUP:-60}" --alone "$alone" --out "$out"
            rc=$?
            set -e
            ok="$([ $rc = 0 ] && echo True || echo False)"
        fi
        first=0
        if [ "$ok" != "True" ]; then
            echo "$w px: $n cameras do not keep up - no more cameras at this width"
            break
        fi
        kept=$n
    done
    echo "$w px: keeps up with $kept cameras (of those tried: $CAMERAS)"
    [ "$kept" -gt "$best" ] && best=$kept
done

step "The table"
"$PY" - "$OUTD" <<'PYEOF'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "w*_*cam", "stress_herd_*cam.json"))):
    w = int(os.path.basename(os.path.dirname(p))[1:].split("_")[0])
    L = json.load(open(p)).get("live") or {}
    if not L:
        continue
    lag = L.get("tick_lag_s", {})
    rows.append((w, L["cameras"], L["keeps_up"], L.get("gpu_busy_share"), lag.get("p99"), L.get("bursts_done"),
                 L.get("bursts_due"), (L.get("burst_late_s") or {}).get("p90"), L.get("cpu_cores_used"),
                 L.get("torch_peak_mib"), L.get("cameras_max_estimate")))
rows.sort()
lines = ["| width | cameras | keeps up | GPU busy | tick lag p99, s | bursts done | burst late p90, s | CPU cores used | "
         "GPU memory, MiB | cameras by GPU (est.) |", "|---|---|---|---|---|---|---|---|---|---|"]
f = lambda x, d=2: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
for w, n, ok, busy, lag, bd, bdue, late, cpu, mem, est in rows:
    lines.append(f"| {w} | {n} | {'yes' if ok else 'NO'} | {f(busy and busy * 100, 0)}% | {f(lag)} | {bd}/{bdue} | {f(late, 1)} | "
                 f"{f(cpu, 1)} | {f(mem, 0)} | {f(est)} |")
text = "\n".join(lines)
open(os.path.join(out, "stress.md"), "w").write(text + "\n")
print(text)
PYEOF

step "Export"
mkdir -p "$WORK/export"
EXP="$WORK/export/herd_stress_$(date +%Y%m%d_%H%M).tgz"
tar -czf "$EXP" -C "$(dirname "$OUTD")" --exclude='*.sqlite*' --exclude='gallery' "$(basename "$OUTD")"
echo "-> $EXP   (scp it back: stress.md, each run's json, the log)"
