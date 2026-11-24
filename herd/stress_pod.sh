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
#        DURATION=240 (s measured)  WARMUP=150 (s: every camera through two bursts - CUDA tunes for each new
#        batch size on the first ones, and the lag then would count against it)  NO_MAX=2 (failed counts in a row
#        that end a width: one may fail on a stray second)  RESIZE=0  OUT=stress  CLIP_S=0 (seconds of each recording kept, 0: all)
#        BURST_FPS= (frames a second a burst encodes, e.g. 12.5; default the config's)  BURST_EVERY= (s, e.g. 120)
#        MAX_LAG=2 (s: a second's rows written at most this late, p99, to keep up; the store and the counts can
#        take more - the 10 s ring buffer is the hard limit - but live cameras meet a second late, past ~1 s, apart)
#        BURST_CROP= (px of a burst's crops, a multiple of 14, e.g. 168; default the config's [model] burst_crop)
#        TRACK_FPS= (detector passes a second for the tracker, e.g. 1; default the config's [sampling] track_fps)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
M="$WORK/models"
D="$WORK/data"
OUTD="$D/${OUT:-stress}"
LOG="$OUTD/stress_log.txt"
SESSION=herd-stress

case "${1:-}" in
    log)                                                # the run started last, whatever its OUT
        [ -z "${OUT+x}" ] && [ -f "$D/.stress_last_log" ] && LOG="$(cat "$D/.stress_last_log")"
        if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "(running in tmux '$SESSION'; log: $LOG)"; else echo "(not running now; log: $LOG)"; fi
        if [ ! -f "$LOG" ]; then
            echo "no log yet - the tmux screen:"; tmux capture-pane -p -t "=$SESSION:" -S -60 2>/dev/null || echo "  (no tmux session)"; exit 0
        fi
        exec tail -n 100 -F "$LOG" ;;
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
    for v in WORK CONFIG VIDEOS WIDTHS CAMERAS DURATION WARMUP RESIZE OUT CLIP_S BURST_FPS BURST_EVERY TRACK_FPS BURST_CROP ALL NO_MAX MAX_LAG; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_STRESS_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/stress_pod.sh"); echo; echo '[stress finished]'; exec bash"
    echo "$LOG" > "$D/.stress_last_log"
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
CAMERAS="${CAMERAS:-2 3 4 5 6 8 10 12 16 20}"
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
    python3 - "$CONFIG" "$cfg" "$w" "${BURST_FPS:-}" "${BURST_EVERY:-}" "${TRACK_FPS:-}" "${BURST_CROP:-}" <<'PYEOF'
import re, sys
src, dst, w, bfps, bevery, tfps, bcrop = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = re.sub(r'://[^/"@\s]+@', "://", text)                 # no camera passwords in the copy
def put(text, key, value, section="sampling"):
    pat = rf'^{key}\s*=\s*[^#\n]*'
    if re.search(pat, text, flags=re.M):
        return re.sub(pat, f"{key} = {value}    ", text, count=1, flags=re.M)
    head = rf'^\[{section}\][^\n]*\n'
    if not re.search(head, text, flags=re.M):
        return text + f"\n[{section}]\n{key} = {value}\n"
    return re.sub(head, lambda m: m.group(0) + f"{key} = {value}\n", text, count=1, flags=re.M)
text = put(text, "frame_width", w)
if bfps:
    text = put(text, "burst_fps", bfps)
if bevery:
    text = put(text, "burst_every_s", bevery)
if tfps:
    text = put(text, "track_fps", tfps)
if bcrop:
    text = put(text, "burst_crop", bcrop, "model")
open(dst, "w", encoding="utf-8").write(text)
print(f"[sampling] frame_width = {w}" + (f", burst_fps = {bfps}" if bfps else "") + (f", burst_every_s = {bevery}" if bevery else "")
      + (f", track_fps = {tfps}" if tfps else "") + (f"; [model] burst_crop = {bcrop}" if bcrop else ""))
PYEOF
    first=1
    kept=0
    fails=0
    for n in $CAMERAS; do
        if [ "${ALL:-0}" != "1" ] && [ "$n" -lt "$best" ]; then
            continue                                           # the wider frame kept up with more: so does this one
        fi
        out="$OUTD/w${w}_${n}cam"
        if [ -f "$out/stress_herd_${n}cam.json" ]; then
            step "$w px, $n cameras: done already ($out)"
            ok="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('live',{}).get('keeps_up'))" "$out/stress_herd_${n}cam.json")"
        else
            step "$w px, $n cameras live (${DURATION:-240} s measured after ${WARMUP:-150} s)"
            alone=0; [ "$first" = 1 ] && alone=3                   # the cost of a burst alone, once a width
            rc=0                                               # exit 1: it did not keep up - an answer, not an error
            "$PY" "$HERE/herd.py" stress --config "$cfg" --videos "$clips" --cameras "$n" --duration "${DURATION:-240}" \
                --warmup "${WARMUP:-150}" --alone "$alone" --max-lag "${MAX_LAG:-2}" --out "$out" || rc=$?
            ok="$([ $rc = 0 ] && echo True || echo False)"
        fi
        first=0
        if [ "$ok" != "True" ]; then
            fails=$((fails + 1))
            if [ "$fails" -ge "${NO_MAX:-2}" ]; then
                echo "$w px: $n cameras do not keep up either ($fails in a row) - no more cameras at this width"
                break
            fi
            echo "$w px: $n cameras do not keep up - one more count to be sure"
            continue
        fi
        fails=0
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
                 L.get("torch_peak_mib"), L.get("cameras_max_estimate"), L.get("gpu_busy_by_kind") or {},
                 max([v for k, v in (L.get("gpu_wait_max_s_by_kind") or {}).items()
                      if k in ("second_detect", "second_encode", "heads", "track_detect")] or [None]) if L.get("gpu_wait_max_s_by_kind") else None))
rows.sort()
lines = ["| width | cameras | keeps up | GPU busy | tick lag p99, s | bursts done | burst late p90, s | CPU cores used | "
         "GPU memory, MiB | cameras by GPU (est.) | a tick's longest wait for the GPU, s | GPU by part |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
f = lambda x, d=2: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
for w, n, ok, busy, lag, bd, bdue, late, cpu, mem, est, kinds, wait in rows:
    parts = ", ".join(f"{k} {v * 100:.0f}%" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]) if v >= 0.02)
    lines.append(f"| {w} | {n} | {'yes' if ok else 'NO'} | {f(busy and busy * 100, 0)}% | {f(lag)} | {bd}/{bdue} | {f(late, 1)} | "
                 f"{f(cpu, 1)} | {f(mem, 0)} | {f(est)} | {f(wait)} | {parts} |")
text = "\n".join(lines)
open(os.path.join(out, "stress.md"), "w").write(text + "\n")
print(text)
PYEOF

step "Export"
mkdir -p "$WORK/export"
EXP="$WORK/export/herd_stress_$(date +%Y%m%d_%H%M).tgz"
tar -czf "$EXP" -C "$(dirname "$OUTD")" --exclude='*.sqlite*' --exclude='gallery' "$(basename "$OUTD")"
echo "-> $EXP   (scp it back: stress.md, each run's json, the log)"
