#!/usr/bin/env bash
# The barn's rehearsal (herd.py trial) twice on a pod - barn.toml as it is, and with settings changed (by default
# the lighter load: one detector pass a second, half a burst's frames, a burst's crops 168 px) - each scored on your
# labels, and the two side by side: who is who (IDF1), posture, activity, rumination. Does the lighter load cost
# anything?
#
#   1. WORK/models/, WORK/data/barn.toml as for resolution_pod.sh; WORK/data/VIDEOS/ the recordings, BOXES/ and
#      LABELS (the box page's labels.json: ids, posture, activity)
#   2. WORK/data/OUT/base/ and OUT/changed/: herd.py trial ... --boxes --labels (a trial with its score.json is not
#      played again)
#   3. WORK/data/OUT/compare.md, the log; WORK/export/herd_trial_ab_<date>.tgz (the scores, trial.json, the
#      second pass's herd.sqlite of each)
#
#   bash herd/trial_ab_pod.sh          start in tmux "herd-trial-ab"
#   bash herd/trial_ab_pod.sh log      follow it
#   bash herd/trial_ab_pod.sh stop
#
# Knobs: WORK=/workspace  CONFIG=barn.toml  VIDEOS=rec_first10  BOXES=boxes_first10  LABELS=labels_first10.json
#        CHANGE="sampling.track_fps=1 sampling.burst_fps=12.5 model.burst_crop=168" (section.key=value, space apart)
#        OUT=trial_ab  AGAIN=1 (play both again)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
M="$WORK/models"
D="$WORK/data"
OUTD="$D/${OUT:-trial_ab}"
LOG="$OUTD/trial_ab_log.txt"
SESSION=herd-trial-ab

case "${1:-}" in
    log)
        [ -z "${OUT+x}" ] && [ -f "$D/.trial_ab_last_log" ] && LOG="$(cat "$D/.trial_ab_last_log")"
        if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "(running in tmux '$SESSION'; log: $LOG)"; else echo "(not running now; log: $LOG)"; fi
        if [ ! -f "$LOG" ]; then
            echo "no log yet - the tmux screen:"; tmux capture-pane -p -t "=$SESSION:" -S -60 2>/dev/null || echo "  (no tmux session)"; exit 0
        fi
        exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_AB_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[trial ab finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$OUTD"
    knobs=""
    for v in WORK CONFIG VIDEOS BOXES LABELS CHANGE OUT AGAIN; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_AB_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/trial_ab_pod.sh"); echo; echo '[trial ab finished]'; exec bash"
    echo "$LOG" > "$D/.trial_ab_last_log"
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
BOXES="$D/${BOXES:-boxes_first10}"
LABELS="$D/${LABELS:-labels_first10.json}"
CHANGE="${CHANGE:-sampling.track_fps=1 sampling.burst_fps=12.5 model.burst_crop=168}"
missing=""
[ -f "$CONFIG" ] || missing+="  $CONFIG\n"
[ -d "$VIDEOS" ] || missing+="  $VIDEOS/ (the recordings)\n"
[ -f "$BOXES/frames.json" ] || missing+="  $BOXES/frames.json\n"
[ -f "$LABELS" ] || missing+="  $LABELS\n"
[ -z "$missing" ] || { printf "!! not in %s:\n%b" "$D" "$missing"; exit 1; }

source "$HERE/pod_setup.sh"

for v in base changed; do
    out="$OUTD/$v"
    mkdir -p "$out"
    python3 - "$CONFIG" "$out/barn.toml" "$([ $v = changed ] && echo "$CHANGE" || echo "")" <<'PYEOF'
import re, sys
src, dst, change = sys.argv[1], sys.argv[2], sys.argv[3].split()
text = re.sub(r'://[^/"@\s]+@', "://", open(src, encoding="utf-8").read())    # no camera passwords in the copy
def put(text, section, key, value):
    # the key inside its own section, set where it is or added at the section's top
    m = re.search(rf'^\[{re.escape(section)}\][^\n]*\n', text, flags=re.M)
    if not m:
        return text + f"\n[{section}]\n{key} = {value}\n"
    end = re.search(r'^\[', text[m.end():], flags=re.M)
    body_end = m.end() + (end.start() if end else len(text) - m.end())
    body = text[m.end():body_end]
    pat = rf'^{re.escape(key)}\s*=\s*[^#\n]*'
    body = re.sub(pat, f"{key} = {value}    ", body, count=1, flags=re.M) if re.search(pat, body, flags=re.M) \
        else f"{key} = {value}\n" + body
    return text[:m.end()] + body + text[body_end:]
for c in change:
    k, val = c.split("=", 1)
    section, key = k.rsplit(".", 1)
    text = put(text, section, key, val)
    print(f"[{section}] {key} = {val}")
open(dst, "w", encoding="utf-8").write(text)
PYEOF
    if [ -f "$out/score.json" ] && [ "${AGAIN:-0}" != "1" ]; then
        step "$v: played and scored already ($out; AGAIN=1 plays it again)"
        continue
    fi
    step "Trial: $v${CHANGE:+$([ $v = changed ] && echo " ($CHANGE)")}"
    "$PY" "$HERE/herd.py" trial --config "$out/barn.toml" --videos "$VIDEOS" --out "$out" --boxes "$BOXES" --labels "$LABELS" \
        || echo "!! $v: the trial stopped (above)"
done

step "Side by side"
"$PY" - "$OUTD" "$CHANGE" <<'PYEOF'
import json, os, sqlite3, sys
out, change = sys.argv[1], sys.argv[2]
cols = {}
for v in ("base", "changed"):
    p = os.path.join(out, v, "score.json")
    if not os.path.exists(p):
        continue
    sc = json.load(open(p))
    a = sc["all"]
    r = {"IDF1 (all)": a["idf1"]}
    for c, x in sorted(sc["cameras"].items()):
        r[f"IDF1 {c}"] = x["idf1"]
    r["boxes found with a cow"] = a["found_with_a_cow"] / max(1, a["your_boxes"])
    r["its cows / yours"] = f"{sc['its_cows']} / {sc['your_cows']}"
    for k, keys in (("posture", ("share",)), ("activity", ("share",)), ("rumination", ("share",))):
        x = sc.get(k) or {}
        if x.get("boxes"):
            r[f"{k} right (your boxes)"] = x["share"]
    pc = (sc.get("posture") or {}).get("confusion", {}).get("lying", {})
    if pc:
        r["lying found"] = pc.get("lying", 0) / max(1, sum(pc.values()))
    db = os.path.join(out, v, "pass2", "herd.sqlite")
    if os.path.exists(db):
        c = sqlite3.connect(db)
        n, rum = c.execute("SELECT COUNT(*), AVG(ruminating) FROM bursts WHERE ruminating IS NOT NULL").fetchone()
        r["bursts (pass 2)"] = n
        if rum is not None:
            r["bursts saying ruminating"] = rum
        lying = c.execute("SELECT AVG(posture = 'lying') FROM seconds").fetchone()[0]
        if lying is not None:
            r["cow-seconds lying"] = lying
    cols[v] = r
keys = list(dict.fromkeys(k for r in cols.values() for k in r))
fmt = lambda x: f"{x:.1%}" if isinstance(x, float) else str(x)
lines = [f"changed: {change}", "", "| | base | changed |", "|---|---|---|"]
for k in keys:
    lines.append(f"| {k} | " + " | ".join(fmt(cols.get(v, {}).get(k, "-")) for v in ("base", "changed")) + " |")
text = "\n".join(lines)
open(os.path.join(out, "compare.md"), "w").write(text + "\n")
print(text)
PYEOF

step "Export"
mkdir -p "$WORK/export"
EXP="$WORK/export/herd_trial_ab_$(date +%Y%m%d_%H%M).tgz"
(cd "$(dirname "$OUTD")" && tar -czf "$EXP" "$(basename "$OUTD")/compare.md" "$(basename "$OUTD")/trial_ab_log.txt" \
    $(for v in base changed; do for f in score.json trial.json barn.toml pass2/herd.sqlite; do
        [ -f "$(basename "$OUTD")/$v/$f" ] && echo "$(basename "$OUTD")/$v/$f"; done; done))
echo "-> $EXP"
