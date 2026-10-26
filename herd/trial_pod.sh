#!/usr/bin/env bash
# The barn's rehearsal (herd.py trial) on a pod, once for every identity model, each scored on your
# labelled boxes: which identity model is best on your own cows (IDF1), not on the public sets.
#
#   1. WORK/models/: the archives docker/models/ holds (herd_models_*.tgz, herd_rumination_*.tgz, the
#      barn detector's) and a herd_reid_*.tgz of each identity model to compare; unpacked there as the
#      Docker build unpacks them into /models. Identity models this pod trained (WORK/reid/id_model_all*/,
#      reid_pod.sh) are taken as they are.
#   2. WORK/data/: what docker/data/ holds for a trial - barn.toml (the RTSP passwords out: the trial plays
#      files), the recordings (rec_fixed/), boxes/ (frames.json) and labels.json
#   3. /models -> WORK/models, /data -> WORK/data: barn.toml's /models/... and /data/... paths as they are
#   4. for every identity model: barn.toml with its [model] identity, `herd.py trial ... --boxes --labels`
#      -> WORK/data/trial_<model>/ (score.json, trial.json); one that has its score.json is not run again
#   5. a table - IDF1 of every model, all cameras and each - and WORK/export/herd_trials_<date>.tgz
#      (score.json, trial.json, the second pass's herd.sqlite of each, the log, the table; no barn.toml)
#
#   bash herd/trial_pod.sh          start in tmux "herd-trial"
#   bash herd/trial_pod.sh log      follow it
#   bash herd/trial_pod.sh stop
#
# Knobs: WORK=/workspace  CONFIG=barn.toml  VIDEOS=rec_fixed  BOXES=boxes  LABELS=labels.json (all in WORK/data)
#        IDENTITIES="<identity.pt> ..." (default: every WORK/models/reid/*/identity.pt)  AGAIN=1 (run every
#        model again)  SHOW=1 (keep each trial's show_<cam>/ pages and videos in the export too)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
H="$WORK/herd"
M="$WORK/models"
D="$WORK/data"
LOG="$D/trial_log.txt"
SESSION=herd-trial

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_TRIAL_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[trial finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$D"
    knobs=""
    for v in WORK CONFIG VIDEOS BOXES LABELS IDENTITIES AGAIN SHOW; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_TRIAL_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/trial_pod.sh"); echo; echo '[trial finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$D"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "!! $(basename "$0") stopped at line $LINENO: $BASH_COMMAND (exit $?)"' ERR   # never silently
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$H/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$M/hf-cache" TZ="${TZ:-Europe/Kyiv}"           # as in the Docker image

CONFIG="$D/${CONFIG:-barn.toml}"
VIDEOS="$D/${VIDEOS:-rec_fixed}"
BOXES="$D/${BOXES:-boxes}"
LABELS="$D/${LABELS:-labels.json}"
missing=""
[ -f "$CONFIG" ] || missing+="  $CONFIG (docker/data/barn.toml, the RTSP passwords out)\n"
[ -d "$VIDEOS" ] || missing+="  $VIDEOS/ (the recordings the trial plays: docker/data/rec_fixed)\n"
[ -f "$BOXES/frames.json" ] || missing+="  $BOXES/frames.json (docker/data/boxes/)\n"
[ -f "$LABELS" ] || missing+="  $LABELS (docker/data/labels.json)\n"
[ -z "$missing" ] || { printf "!! not in %s:\n%b" "$D" "$missing"; exit 1; }

step "Models: the archives in $M, unpacked as the Docker build does"
mkdir -p "$M"
for a in "$M"/*.tgz "$M"/*.tar.gz; do
    [ -f "$a" ] || continue
    mark="$M/.unpacked_$(basename "$a")"
    [ -f "$mark" ] && continue
    echo "unpacking $(basename "$a")"
    tar -xzf "$a" -C "$M"
    touch "$mark"
done
for d in "$WORK"/reid/id_model_all*/; do                    # identity models trained on this pod
    [ -f "$d/identity.pt" ] || continue
    n="$(basename "$d")"
    [ -e "$M/reid/$n" ] || { mkdir -p "$M/reid"; ln -s "${d%/}" "$M/reid/$n"; echo "this pod's $n"; }
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

if [ -z "${IDENTITIES:-}" ]; then
    IDENTITIES="$(ls "$M"/reid/*/identity.pt 2>/dev/null || true)"
fi
[ -n "$IDENTITIES" ] || { echo "!! no identity model: upload a herd_reid_*.tgz of each to $M (or IDENTITIES=...)"; exit 1; }
for f in $IDENTITIES; do
    [ -f "$f" ] || { echo "!! $f: no such file"; exit 1; }
    echo "identity model $(basename "$(dirname "$f")"): $(python3 -c "import json,sys; e=json.load(open(sys.argv[1])); print(e.get('loss', {}))" "$(dirname "$f")/eval.json" 2>/dev/null || echo '(no eval.json)')"
done

# every /models/... and /data/... file barn.toml names (a model, the detector) must be here
python3 - "$CONFIG" <<'PYEOF'
import os, re, sys
gone = []
for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    line = line.split("#", 1)[0]
    for p in re.findall(r'"(/models/[^"]+)"', line):
        if not os.path.exists(p):
            gone.append(f"  line {n}: {p}")
if gone:
    sys.exit("!! barn.toml names models not in /models - upload the archive that has each (docker/models/):\n"
             + "\n".join(gone))
PYEOF

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
if grep -qs '"kind": *"rfdetr"' "$M"/herd/*/best/det_train_meta.json; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

names=()
for f in $IDENTITIES; do
    n="$(basename "$(dirname "$f")")"
    names+=("$n")
    out="$D/trial_$n"
    if [ -f "$out/score.json" ] && [ "${AGAIN:-0}" != "1" ]; then
        step "$n: scored already ($out/score.json; AGAIN=1 runs it again)"
        continue
    fi
    step "Trial with the identity model $n"
    rm -rf "$out"
    mkdir -p "$out"
    cfg="$out/barn.toml"
    python3 - "$CONFIG" "$cfg" "$(readlink -f "$f")" <<'PYEOF'
import re, sys
src, dst, ident = sys.argv[1:]
text = open(src, encoding="utf-8").read()
m = re.search(r'^identity = "(.*?)"', text, flags=re.M)     # [model] identity (commented lines stay)
if not m:
    sys.exit(f"!! {src} has no identity = \"...\" line ([model])")
text = text[:m.start()] + f'identity = "{ident}"' + text[m.end():]
text = re.sub(r'://[^/"@\s]+@', "://", text)                 # no camera passwords in the copy (it plays files)
open(dst, "w", encoding="utf-8").write(text)
print(f"[model] identity = \"{ident}\"   (was \"{m.group(1)}\")")
PYEOF
    "$PY" "$HERE/herd.py" trial --config "$cfg" --videos "$VIDEOS" --out "$out" --boxes "$BOXES" --labels "$LABELS"
done

step "Every identity model on your labels"
python3 - "$D" "${names[@]}" <<'PYEOF'
import json, os, sys
d, names = sys.argv[1], sys.argv[2:]
rows, table = [], {}
for n in names:
    p = os.path.join(d, f"trial_{n}", "score.json")
    if not os.path.exists(p):
        print(f"  {n}: no score.json (the trial stopped: see the log)")
        continue
    sc = json.load(open(p))
    a = sc["all"]
    cams = {c: r["idf1"] for c, r in sorted(sc["cameras"].items())}
    table[n] = {"idf1": a["idf1"], "idp": a["idp"], "idr": a["idr"], "cameras": cams, "its_cows": sc["its_cows"],
                "your_cows": sc["your_cows"], "merged": len(sc["merged"]), "split": len(sc["split"])}
    rows.append(n)
if rows:
    cams = sorted({c for n in rows for c in table[n]["cameras"]})
    w = max(len(n) for n in rows)
    print(f"  {'model':<{w}}  IDF1    IDP     IDR    " + "  ".join(f"{c:>6}" for c in cams) + "   its/your cows  merged split")
    for n in sorted(rows, key=lambda n: -table[n]["idf1"]):
        t = table[n]
        print(f"  {n:<{w}}  {t['idf1']:6.1%}  {t['idp']:6.1%}  {t['idr']:6.1%} "
              + "  ".join(f"{t['cameras'].get(c, 0):6.1%}" for c in cams)
              + f"   {t['its_cows']:>4}/{t['your_cows']:<4}      {t['merged']:>4}  {t['split']:>4}")
    json.dump(table, open(os.path.join(d, "trials.json"), "w"), indent=2)
PYEOF

mkdir -p "$WORK/export"
T="$WORK/export/herd_trials_$(date +%Y%m%d_%H%M).tgz"
parts=("data/trial_log.txt")
[ -f "$D/trials.json" ] && parts+=("data/trials.json")
for n in "${names[@]}"; do
    for p in score.json trial.json pass2/herd.sqlite same_cow.html; do
        [ -e "$D/trial_$n/$p" ] && parts+=("data/trial_$n/$p")
    done
    if [ "${SHOW:-0}" = "1" ]; then
        for s in "$D/trial_$n"/show_*; do [ -e "$s" ] && parts+=("${s#"$WORK"/}"); done
    fi
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "results: $T   (scp -P <port> root@<ip>:$T .)"
