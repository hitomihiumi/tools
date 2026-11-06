# The pod's set-up shared by resolution_pod.sh and stress_pod.sh (sourced, not run): the models laid out as the
# Docker image has them, /models and /data linked, every /models/... path of barn.toml checked, the Python
# environment (run_pod.sh step 1), ffmpeg, rfdetr. Needs WORK, M, D, CONFIG, HERE, PY and step() set.

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

