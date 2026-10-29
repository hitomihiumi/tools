# herd in Docker Desktop

One image holds the code, its Python environment and the models. Nothing is installed on the computer
besides Docker Desktop. Training stays on a GPU pod (`herd/reid_pod.sh`, `herd/run_pod.sh`). The
image runs everything that happens in the barn:

- the barn live;
- recording the cameras;
- the two dataset tools: naming the cows, and correcting the detector's boxes;
- reports, `show-ids`, `show-zone` and the stress test.

## 1. Docker Desktop

- **Windows** (an NVIDIA card): install a current NVIDIA driver for Windows, then Docker Desktop with
  the WSL 2 backend (its default). Nothing else is needed for the GPU: Docker Desktop passes the card to
  containers.
- **Mac**, or a computer without an NVIDIA card: Docker Desktop alone; use `compose.cpu.yaml` (below).
  Everything runs on the CPU: fine for the labelling pages and a few recordings, too slow for a live
  barn.
- Docker Desktop → Settings → Resources: give it memory. A live barn wants ~1.5 GB a camera plus ~4 GB;
  on Windows the limit is in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=24GB`).

## 2. Put the models in

Copy the archives into `docker/models/` (the folder this file is in):

| archive | needed | made by |
|---|---|---|
| `herd_models_*.tgz` | yes | `herd/export_pod.sh` on the pod: the herd model, the detectors, DINOv2 |
| `herd_reid_*.tgz` | no | `herd/reid_pod.sh`: the identity model (`reid/id_model_all/`) |
| `herd_rumination_*.tgz` | no | the rumination model |

## 3. Build the image

In a terminal (PowerShell on Windows), in this `docker/` folder:

```bash
docker compose build                                 # an NVIDIA card (RTX 20-40)
TORCH_BACKEND=cu128 docker compose build             # RTX 50 series (PowerShell: $env:TORCH_BACKEND="cu128"; docker compose build)
docker compose -f compose.cpu.yaml build             # no NVIDIA card
```

The build unpacks the models into the image and loads each of them once. The finished image runs
without the internet. The build stops with a message when `docker/models/` has no
`herd_models_*.tgz`. The image is large (the CUDA build is around 10 GB).

Check it:

```bash
docker compose run --rm herd models
```

It lists the models in the image and the GPU. `GPU: none` with an NVIDIA card means the driver or
Docker Desktop does not pass the card: update both, restart Docker Desktop.

With `compose.cpu.yaml`, put `-f compose.cpu.yaml` after `docker compose` in every command below.

## 4. The data folder

`docker/data/` on the computer is `/data` in the container. Everything the commands read or write
lives there:

```
docker/data/barn.toml         the config
docker/data/rec/              the recordings (cam1_2026-10-20_06-00-00.mp4 ...)
docker/data/barn/             the live barn: herd.sqlite, gallery/, names.json
docker/data/barn_video/       recordings played for naming the cows (tracks.html)
docker/data/boxes/            frames for the detector (boxes.html)
```

The config, with the image's model paths filled in:

```bash
docker compose run --rm herd init                    # -> docker/data/barn.toml
                                                     # (again on an existing one: fills in model paths
                                                     #  that point at nothing, e.g. after a new build)
```

Then edit `docker/data/barn.toml`:

- `[farm] timezone_offset_hours`;
- the `[[cameras]]`: `id`, the `url` (`rtsp://...`), the `mask`.

See `herd/SETUP.md` sections 3–4. Paths in it are container paths (`/models/...`, `/data/...`).
Docker Desktop's network reaches cameras on the farm's network as the computer does.

**The zones, drawn on the cameras' pictures.** With recordings in `docker/data/rec/` (named as in
section 5):

```bash
docker compose run --rm herd zones page --config /data/barn.toml --videos /data/rec --cameras cam1 cam2 cam3 cam4 \
    --out /data/zones
```

(`--cameras`: the camera ids, also those not in `barn.toml` yet; the recordings are found by them.)

1. Open `docker/data/zones/zones.html`. It shows each camera's picture with the cows the detector
   found, coloured by the zone rule.
2. Draw:
   - `exclude`: neighbouring pens, the yard, cows that are not this pen's;
   - `mask`: the floor this camera counts;
   - the `url`.
3. **Save cameras.toml** and move the file to `docker/data/`.
4. Apply it:

   ```bash
   docker compose run --rm herd zones apply --config /data/barn.toml --cameras /data/cameras.toml
   ```

**Zones, floor calibration and its check in one page (studio).** Once the cameras are in `barn.toml`
(their `url`, e.g. from the zones page above), everything else can be edited in one place and saved
straight into `barn.toml`:

```bash
docker compose run --rm -p 8765:8765 herd studio --config /data/barn.toml --videos /data/rec_fixed
```

Open <http://localhost:8765> in a browser (Ctrl+C in the terminal stops it). `-p 8765:8765` lets the
browser reach it. Every camera's picture at one moment of the recordings (`--at`; the left panel can move
it), one, two or four cameras at once (keys `1` `2` `4`) or the **Plan** (`P`): the floor from above, the
cameras' pictures carried onto it, their masks and the spots. Tools (keys in brackets):

- **Mask / Exclude / Tracking** (`M` `E` `T`): click round an area, a click on the first corner or Enter
  closes it; drag a corner, double-click an edge for a corner more, right-click a corner to remove it;
- **Floor** (`O`): the floors themselves (a pen is floor A, the next pen B, an alley C - "+ New floor"),
  the main way to calibrate. Click a floor's corners round its edge on one camera (A1, A2, ...), then the
  same corners on the others: the next corner that camera has not got is picked (ringed where the planes
  put it, once there are planes); `N` skips one it does not see, Esc to add new corners there instead. A
  corner the camera does not see (the near ones are often under it) is no problem: pick an **edge** (the
  buttons A1–A2, ... on the right, or a click on the outline) and click a few points along it on the
  pictures - where the fence or kerb meets the floor; a double-click on the outline puts a corner between
  two. Measured sizes put the plan on the real floor: for 4 corners "a rectangle" width x length (cm) and
  where its first corner is, or X, Y of any corner or spot (a post, a drinker's corner). Then, in order:
  a camera with 8 equations of the measured floor (2 per measured corner or point, 1 per edge point) gets
  its plane on its own; a camera that sees none follows one placed already through 4 spots it shares; two
  cameras with 5 spots in common are fitted together and put on the corners either of them sees (two
  cameras facing each other, each seeing the other's near corners); and "Fit / Refine all" adjusts every
  plane together - a camera that shares only a thin strip with its neighbour, its plane set roughly by
  hand first (tool Plane), is pulled onto the floor by the strip's spots and its edges. Every camera shows
  the floors' outlines (dashed where only the planes put them) and its grid only on the floors. A far
  corner lies on its edge already, so a camera that sees 2 corners and the side edges has 6 equations,
  not 8: the page says what to add (a measured post along each side, or spots shared with the camera
  opposite). A post on an edge: click its foot as an edge point, type how far it is from the edge's first
  corner along the fence (a tape) and it becomes a measured point - then put it on the other cameras
  that see it (tool Spots);
- **Spots** (`S`): one spot of the floor on every camera that sees it. Click it on one camera, then the
  same place on the others - once the planes are known, where it should be is ringed (dashed): click
  there. Each spot shows how far (px) it is from where the cameras together put it: green agrees, amber a
  little off, red off. Every two cameras with 5 spots in common are fitted at once (Auto-fit, as
  `autocalib --points`), all cameras they join on one plan;
- **Plane** (`F`): the floor grid on each calibrated camera (about 1 m when the plan is in cm; `G` shows
  it with any tool), and the plane moved as a whole. On the picture: drag and the floor follows the hand
  (the camera tilted and panned), Shift+drag rolls it. On the right: Tilt, Pan, Roll, Zoom and the Lens
  bend - how the floor sits in the picture - and Turn, Size and Move - where the camera's floor lies on
  the plan; Reset puts it back. On the **Plan** each camera's floor has a dot where it begins and an
  arrow where it looks: drag the dot to move it, the arrow to turn it. Lay another camera over this one
  (*over it*) and move the plane until the floor lines up. "Start a plane" gives a camera without one a
  plane to begin from; "Remove plane" drops a wrong one;
- **Select** (`V`): select and move anything, Del removes it. `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo,
  `Ctrl+wheel` zooms.

Nothing is written until **Save** (`Ctrl+S`): the zones of the cameras changed, the cameras' planes
(`floor_points` / `distortion` / `aspect`, set or removed) and, after a fit, `[mct]` (link on, `link_dist`, the calibrated pairs out
of `disjoint`). The old file is kept as `barn.toml.bak`, the spots go to `calib/points.json`. The floor
plan is in about cm when `calib/` holds the cows' boxes cached by an `autocalib --videos` run
(`boxes_<cam>.json`); without them `link_dist` is a guess. Without recordings:
`--image cam1=/data/cam1.jpg --image cam2=...`, or the pictures the spots were clicked on
(`calib/pick`) are used.

A still from a camera, and its zone drawn:

```bash
docker compose run --rm --entrypoint ffmpeg herd -rtsp_transport tcp -i "rtsp://user:pass@10.0.0.11:554/stream1" -frames:v 1 /data/cam1.jpg
docker compose run --rm herd show-zone --config /data/barn.toml --camera cam1 --image /data/cam1.jpg
```

## 5. The datasets

**Recordings.** Every camera at once, an hour:

```bash
docker compose run --rm herd record --config /data/barn.toml --out /data/rec --minutes 60 --fps 5 --gpu
```

The files are named as the tools read them (`rec/cam1_2026-10-20_06-00-00.mkv`, Matroska: a recording
cut short - Ctrl+C, the network gone - still reads up to where it stopped). While it records, each camera
goes into `<name>.mkv.part` as the camera sends it; at the end it becomes `<name>.mkv`:

- `--fps 5`: re-encoded to H.264 at 5 frames a second, steady (at each 1/5 s the frame the camera sent
  last). The tools take a frame a second and the tracker `track_fps` a few more, so 5 is plenty - and the
  files and the time to encode them are a fraction;
- without `--fps`: an HEVC (H.265) camera re-encoded to H.264 at its own rate (the tools' reader decodes
  some cameras' HEVC poorly); an H.264 camera kept as it came.
- `--gpu`: that re-encoding on the NVIDIA card (decoded with NVDEC, encoded with NVENC) - many times
  faster, the CPU left free. The log says at the start when the card cannot (`Cannot load libcuda.so.1`:
  no card given to the container; then a current NVIDIA driver and `docker compose build` again - the
  image asks the driver for its video part) and the CPU does it; a file the card refuses midway is
  re-encoded on the CPU.

The length goes by the computer's clock, whatever the cameras' own time stamps do (the barn's jump ahead,
which cut the first test recordings short); a stream that stops early is recorded on into a new file
(`cam1_<its start>.mkv`, read after the first). At the end the log says per camera how many of the minutes
asked were recorded, `!!` when short.

`--keep-raw` keeps the stream as it came too (`<name>.mkv.raw`, not read by the tools). After the
recording the log says per camera what share of its frames arrived (over the internet the barn's four
cameras lose ~40%). `record --check /data/rec` checks recordings already made, every frame. Recordings made
elsewhere work too: copy them into `docker/data/rec/`, named by camera id and local start time.

**Damaged recordings.** A recorder's file can play in a player and still be read wrong by the tools:
`h264 ... error while decoding` in the log, one camera making no bursts, the clock of a camera standing
still. Encode them again into clean .mp4 (same names, steady frame rate, broken frames dropped) and use
that folder:

```bash
docker compose run --rm herd fix-videos --videos /data/rec --out /data/rec_fixed
```

`--copy` only re-wraps the stream (fast; enough when just the container is broken). Then pass
`--videos /data/rec_fixed` to the commands below (and to `zones page`).

A stretch of a long recording, for a smaller set (`.ts` and the other formats alike):

```bash
docker compose run --rm herd fix-videos --videos /data/rec_new --out /data/rec_first10 --cameras cam1 cam2 --first 10
```

`--cameras`: only these cameras' files. `--first 10` / `--last 10`: the first / last 10 minutes, the
same stretch of time on every camera - it starts where the camera that started last started / ends where
the camera that stopped first stopped - each file named by its new start (the time in its name moved
on), so the tools still put the cameras side by side. Frame-exact, so not with `--copy`. `--first 10 --from
14:00`: ten minutes from 14:00 (the clock of the names) - another stretch of the same day, e.g. for
`crosscam --also`, best hours later, once the cows have moved and lie elsewhere.

**Naming the cows** (`herd/SETUP.md` section 11):

```bash
docker compose run --rm herd barn video --config /data/barn.toml --videos /data/rec --out /data/barn_video
docker compose run --rm herd barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
```

1. Open `docker/data/barn_video/tracks.html` in your browser. Name the tracks, then **Save
   merges.csv**.
2. Move the saved file from Downloads to `docker/data/merges.csv`.
3. Export:

   ```bash
   docker compose run --rm herd barn export --cache /data/barn_video/training_cache --merges /data/merges.csv
   ```

**The detector's boxes** (`herd/SETUP.md` section 12):

```bash
docker compose run --rm herd boxes frames --config /data/barn.toml --videos /data/rec --out /data/boxes
```

1. Open `docker/data/boxes/boxes.html` in your browser. Correct the boxes, then **Save
   labels.json**.
2. Move the saved file to `docker/data/labels.json`.
3. Export:

   ```bash
   docker compose run --rm herd boxes export --data /data/boxes --labels /data/labels.json
   ```

They write `docker/data/herd_barn_feat_<date>.tgz` and `docker/data/herd_det_barn_<date>.tgz`: these go to
the pod's `/workspace`. Training writes new archives
(`herd_reid_*.tgz`, a detector in `herd_models_*.tgz`). Put those in `docker/models/` (move the old
ones out) and build again.

**Cows named on the boxes** (instead of, or besides, naming tracks). On the box page select a box,
press `I`, type her ear tag, `Enter`: the next box without an id is selected. `P` takes the ids of this
camera's frame before for boxes in the same place (a lying cow keeps her place): check them and fix
the few that moved. The same cow seen by two cameras at once gets the same tag on both: that is what the
model needs most. Save labels.json, then:

```bash
docker compose run --rm herd boxes ids --data /data/boxes --labels /data/labels.json
```

Faster: type each cow's tag once, on one frame of a camera, and press `F`. The tag is carried through
all of that camera's frames, forwards and backwards, while her box clearly continues: the most similar
box on the next frame, and she the most similar to it, not across more than 5 minutes. Filled-in ids
are blue; check them. A typed id is never overwritten; fix a wrong blue one by typing it, and press `F`
again. "only frames I edited" limits it to frames you have opened.

More crops between the cut frames, from the video itself (`--fill 2`: a crop every 2 s between two
frames 10 s apart, where the cow's two boxes overlap, i.e. she stayed in place; her box moved
linearly):

```bash
docker compose run --rm herd boxes ids --data /data/boxes --labels /data/labels.json --fill 2 --config /data/barn.toml --videos /data/rec_fixed
```

It writes `docker/data/herd_barn_ids_<date>.tgz`: each named box cut as the barn cuts a cow, a sixth of
the cows held out. Every frame with named boxes counts, done or not (`--done-only`: only done ones). Put it in the pod's `/workspace`; `reid_pod.sh` adds the set `barn_ids` to training.
The same labels.json also makes the detector's set (`boxes export`).

**One id, one cow on every camera?** Cross-camera training and checking only work when a cow has the
same id on every camera. See it, and fix it:

```bash
docker compose run --rm herd boxes gallery --data /data/boxes --labels /data/labels.json
```

Open `docker/data/boxes/ids.html`: each id is a group, each row in it that id on one camera with her
crops. **match cameras** (e.g. name the cows of cam3 by those of cam1): the left column is the first
camera's cows, the right the second's; click the cow on the right that the selected left one is, or
"new cow". A camera with the same cow under two ids is flagged. Or type in a row the cow it really is (rows with the same name become one cow on every camera;
`x` removes the id), **Save labels.json**, move it to `docker/data/`, then `boxes ids` again (and on
the box page, Load labels.json).

**Check the models on your labels** (before training anything on them - none of these frames was
ever seen by the models):

```bash
docker compose run --rm herd check --config /data/barn.toml
```

It takes the newest `herd_det_barn_*.tgz` and `herd_barn_ids_*.tgz` in `docker/data/` (or `--det`,
`--ids`):

- **the detector** on every labelled frame: the share of the cows it finds and of its boxes that are
  right, at the config's threshold and at the best one for this barn (put that one in
  `[detector] threshold`);
- **the identity model** (and the herd model's own fingerprint, to compare) on the named cows: a cow
  from another camera, is the nearest of all the cows her - from a group of crops and from one crop.

`docker/data/check/check.json` keeps the numbers. After training on the pod, put the new models in
`docker/models/`, build again and run `check` again: the same frames, before and after.

**One camera's zone changed** (`zones page` → `zones apply` again). Redo only that camera:

```bash
docker compose run --rm herd barn video --config /data/barn.toml --videos /data/rec_fixed --out /data/barn_video --cameras cam2
docker compose run --rm herd barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
docker compose run --rm herd boxes frames --config /data/barn.toml --videos /data/rec_fixed --out /data/boxes --every 10 --cameras cam2 --again
```

- `barn video --cameras`: that camera's earlier bursts are removed and its recordings played again;
  the other cameras' bursts and names stay. Its tracks get new names on the page, so name them again.
- `boxes frames --cameras ... --again`: that camera's frames are cut again at the same moments with
  the new grey zone, and the detector looks again. Your boxes on them stay, minus those now in the
  grey. The page marks each of them "cut again - check it": look, press Enter, Save labels.json. Until
  then, `boxes export` leaves them out.

## 6. A dress rehearsal on the recordings

Before the barn goes live: the whole system with `barn.toml` - zones, masks, detector, tracker, posture,
the gallery - played on the recordings:

```bash
docker compose run --rm herd trial --config /data/barn.toml --videos /data/rec_fixed --out /data/trial
```

A gallery fills over a barn's first hours; a quarter of an hour of recordings would end with every
cow NaN. So the files are played twice. The first pass (a burst every 20 s) is followed by the night's
maintenance, which enrols every cow seen in 6 or more bursts. The second pass then runs with those
cows: who is who. It writes:

- `docker/data/trial/show_<cam>/index.html` and `ids.mp4`: every camera's first 10 minutes with each
  box's cow, track, posture and activity (one colour per cow, the same on every camera; grey = NaN).
  Check:
  - every cow has a box (missed ones: the detector, its threshold, `min_box_pct`);
  - no box sits on the neighbouring pen (`exclude`);
  - a cow in the cam1/cam2 overlap is counted by one camera only;
  - a cow keeps her colour while she lies.
- `docker/data/trial/trial.json`, and a line a camera in the log: the cow-seconds counted, the tracks,
  the share given a cow, posture and activity;
- "what cannot be": one cow at the same second twice on one camera, or on two cameras that share no
  floor - two cows given one id.

**How right it is, in numbers: IDF1 against your own ids.** The frames you labelled with cow ids
(section 5: `boxes`, the ids gallery) come from the same recordings: each of your boxes is looked up
in the rehearsal at the same camera and second, and its cow compared with yours.

```bash
docker compose run --rm herd trial --config /data/barn.toml --videos /data/rec_fixed --out /data/trial --boxes /data/boxes --labels /data/labels.json
docker compose run --rm herd trial --score-only --config /data/barn.toml --out /data/trial --boxes /data/boxes --labels /data/labels.json
```

The second scores a rehearsal already made. The log and `trial/score.json` give:

- IDF1: the share of your boxes that got the right cow, with the system's names matched one to one to
  yours. IDP: of the cows it gave, the right one. IDR: of your boxes, the right cow.
- The same per camera.
- Which of its cows took two of yours (merged), and which of yours it split.
- `trial/same_cow.html` when one of its cows took several of your names: they are shown side by side.
  Often it is one cow that two cameras see, named once on each - say "same cow", save labels.json,
  score again with it (and train with it). If they are two cows, "different": the system was wrong.

Only the boxes a camera counts (its mask) are scored. Use the labels.json with the ids as the gallery
page saved them (the same cow one id on every camera).

**Which cow on the neighbour camera is she, by where she stands** (`crosscam`). The cameras do not move,
so what surrounds a cow in one frame says where on the floor she is; your labels (one cow, one id on
every camera) show where that place is in the neighbour's frame. It learns that mapping from them -
nothing measured - and checks it honestly: each labelled moment is left out, the mapping fitted on the
rest, and that moment's cows put onto the other camera and matched by distance:

```bash
docker compose run --rm herd crosscam --config /data/barn.toml --boxes /data/boxes --labels /data/labels.json
```

Per pair of neighbours: the right cow how often (beside a guess), from the boxes' bottom centres and from
their centres, each lens's distortion as the pairs show it (the config's may be off), the cows it missed,
your cows put again and again where another of yours is (one cow under two names on the two cameras, or
two names swapped: check them in `ids.html`), `boxes/crosscam_<a>_<b>.jpg` - where each cow was put on
the second camera against where you named her, the far ones red and named - and `boxes/crosscam.json`.
Misses count half on each camera's picture (a cow far from one camera is small on it and large on the
other's). `--planes 3` (the default) also tries up to three floor planes, each pair on the one it fits
best - the bedding, the alley and a lens one plane cannot follow - and says how well that matches; `--apply`
still writes one plane.

No plane at all: the mapping learnt from the pairs themselves, from where each box is, how big and how
tall (a standing cow's centre is higher than a lying one's) - once as one smooth mapping, once from each
box's nearest pairs (a step between the bedding and the alley stays a step). Within one recording a lying
cow's other moments teach her own place, so check it on another recording too, where the cows lie elsewhere:

```bash
docker compose run --rm herd crosscam --config /data/barn.toml --boxes /data/boxes_first10 --labels /data/labels_first10.json --also /data/boxes /data/labels.json
```

`--also`: fitted on one recording, checked on the other, both ways. The cows' names need not match between
the two sets - only within each. `--apply`, when one of the two is
calibrated and the other not: the other gets floor_points on its floor (barn.toml.bak kept), so
`[mct] link` joins the two cameras' tracks by it. Both calibrated: `--apply --onto cam1` keeps cam1's
and replaces the other's. It maps by the better of the two points, and writes `floor_at = "centre"`
(or `"bottom"`) on both cameras: the point of a box the run then puts on the floor. `--gate` (0.15 of
the frame height): how far from every box of the other camera a cow may land and still match one.

**Box by box, every camera at once:** `trial/boxes.html` (written with the score; `--score-only` writes
it again for a trial already made). Each labelled moment shows its cameras side by side, your boxes
with your ids and the cow the system gave each: green the right cow, red another, orange not found,
violet found but given no name (a click says why: no burst on that short track, none sure enough, or split between cows), grey dotted its boxes you did not name, dashed outside the camera's mask (not
scored). **One cow on both cameras:** above the frames, every cow of yours both shown cameras have at that moment, judged:
`ONE TRACK #7` (green: the system follows her as one cow across the cameras - her boxes on both carry `⇄#7` and
a ring of one colour), `one id` (the same name, tracks not joined), `two cows` (red), `no name` or `not found` on
a camera; B keeps only such moments, a click on one shows her on every camera. ←/→ go through the moments, E keeps
only those with a mistake, the cameras switch on and off,
and "your cow" keeps the moments with her. Click a box: its whole frame large with the box marked (the
rest darker), and that cow of yours (or its cow) in every camera's frames at every moment, the box marked
and the other side's name under each; a frame takes you to its moment. Z (close-ups) shows the boxes cut out
instead. The
frames are read from `boxes/frames/` where they are, so open the page on the computer that has them.

**Does one cow get one id on two cameras?** The score says it for every cow of yours it scored on two
cameras: the same cow of its on all of them or not (and at one moment on two cameras, how often the same).
With masks that share no floor your cows are rarely scored on two cameras. `--overlap` is an experiment
for that: each camera counts its whole `tracking_mask`, and the two rules that make a cow counted by two
cameras at once two cows (`[identity] counted_apart_s`, `[mct] unlink_s`) are off. Never for a barn: on
shared floor its cows are counted twice.

```bash
docker compose run --rm herd trial --overlap --config /data/barn.toml --videos /data/rec_fixed --out /data/trial_overlap --boxes /data/boxes --labels /data/labels.json
```

**Is the joining on at all?** A run with `[mct] mode = "link"` ends with `[run] [mct] two cameras' cows each
other's nearest ...` and `tracks joined across cameras N`; with `link_rule = "llr"` the store's `track_evidence`
fills. None of it in the log: the image is older than the code (`docker compose build` after `git pull`) or the
container reads another barn.toml.

**Another calibration or [mct] setting, without playing the recordings again** (minutes, no GPU): each camera's
tracks as the trial made them, only the joining across cameras done anew with the config as it is now, then
scored on your labels:

```bash
docker compose run --rm herd trial --relink --overlap --config /data/barn.toml --out /data/trial_overlap --boxes /data/boxes --labels /data/labels.json
```

It writes `trial_overlap/relink/` (a copy of the store, the trial's own left as it was) and says how far apart
the two cameras put their nearest cows and how many tracks it joined. Recalibrate (studio, `autocalib --pick`),
relink, compare - then a full trial with the calibration that wins.

The score then says, for every moment one of your cows is in both cameras' frames and both found her: how
often she was on **one joined track** (the linker found her on both), had **one id** (the same cow on both -
joins, the night's merges and your `merges apply` answers followed), **two different cows**, or **no name** on
one camera (no burst sure enough there yet) - over all, per camera pair, and the cows of yours it got wrong.

**Several identity models, one rehearsal each, on a pod** (`herd/trial_pod.sh`): which identity model
is best on your own cows. Upload to the pod's `/workspace`:

| where | what |
|---|---|
| `/workspace/models/` | the archives in `docker/models/` (`herd_models_*.tgz`, the detector's, `herd_rumination_*.tgz`; those two may also lie in `/workspace` itself, and models the pod already has under `/workspace/herd/` are linked in) and a `herd_reid_*.tgz` of each identity model to compare (identity models trained on that pod are taken as they are) |
| `/workspace/data/` | `barn.toml` (the RTSP passwords out: the trial plays files), `rec_fixed/`, `boxes/` (`frames.json`; with its `frames/` the page of one cow under two names gets pictures), `labels.json` |

Then `bash herd/trial_pod.sh` (and `bash herd/trial_pod.sh log`). `/models` and `/data` point at those two
folders, so barn.toml's paths stay as they are; the script stops naming any `/models/...` file it lacks.
Each identity model gets `/workspace/data/trial_<model>/` and a line in the table at the end (IDF1 of all
cameras and of each); `/workspace/export/herd_trials_<date>.tgz` keeps every `score.json` and `trial.json`.

**Cameras that share floor, calibrated without measuring anything.** A cow seen by two cameras should
be one cow. `[mct] link` joins her two tracks by where she stands on one floor plan. Marks measured by
hand are not needed: a cow that gets up or walks does it at the same moment on both cameras. The
detector looks at every second of the recordings, on each camera's whole picture (the rehearsal's
store will not do: the masks cut the shared floor away from one of the two cameras). Then
`autocalib` does four things:

- finds those cows and the two cameras' clock difference;
- fits the floor mapping between the cameras and each wide lens's bend;
- checks it on every cow both see;
- writes it all into `barn.toml`.

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --pairs cam1:cam2 --apply
```

The log says, per camera pair:

- how many moving cows agree;
- what share of the other camera's cows lands on one of this camera's cows (above ~60% is good), and
  what share would by chance alone (in a full pen any point is near some cow): the first has to be
  well above the second;
- "no shared floor" for cameras that share none.

Look at `docker/data/calib/cam1_cam2.jpg`: the other camera's cows (red crosses) should sit on this
camera's cows (green circles). Then run `trial` again: a cow in the overlap should now have one id on
both cameras. The boxes are kept in `docker/data/calib/boxes_<cam>.json`: a second run on the same
recordings skips the detector. `docker/data/calib/cam1_cam2_moving.jpg` is drawn whatever came of
it: both cameras side by side, the paths of the cows that walked (orange), the pairs found (red,
numbered alike) - it shows whether the recording had enough cows walking. If no pair is found, too few cows moved through the overlap: record
an hour or more (the feeding time is best) and run `autocalib` on that. A few minutes of cows lying
is not enough: their boxes only twitch, and a twitch is no evidence of where a cow stands.

**Cows that walk keep their track: `track_fps`.** Once a second a walking cow moves more than her own
box and her track breaks (a new one, often a new name). With

```toml
[sampling]
track_fps = 4
```

the detector and the tracker also look at 3 frames between the seconds; what is written (posture,
identity, a row a second) stays once a second. The detector's work on the seconds is 4 times as much:
the rehearsal takes longer, and live the GPU must keep up - when it does not, seconds are missed
(`seconds_missed` in the camera's counts).

**One cow, one track across cameras.** A cow's tracks on two cameras are joined only when both see her
at once - so the masks (where a camera counts) share no floor, and at least one camera of each pair has
a `tracking_mask` reaching into the other's floor (where it follows cows it does not count; zones page,
the tracking layer). Calibrated pairs (cam1-cam2) are joined by where the cow stands; a pair that shares
only a strip along a fence (cam3-cam4: no floor mapping) by look - a cow one camera sees beyond its mask
is one of the other's counted cows, the one her frame ids are clearly nearest to over 10 s. `trial` says
how many tracks were joined per camera pair.

**When a trial joins nothing** (the log: "tracks joined across cameras 0", most seconds turned away by
`link_margin`): the floor mapping is right in only part of the seconds and the cows lie side by side. Join by
evidence instead of by seconds near (`herd/GUIDE.md` section 4.8):

```toml
[mct]
link_rule = "llr"           # every second both cameras see two tracks: evidence for or against one cow
link_learn_offset = true    # the floor's bend between the cameras learnt from the cows joined
```

and run `trial` again. With it the night also puts together one cow the gallery named twice - once by each
camera's side of her - by where she stood over the hours (`cow_merge_llr`): the trial says which names it
made one ("one cow under two names, made one by where she stood"), and the score counts them as one cow.

**Pairs the night was not sure of: you decide.** Pairs with some evidence but not enough wait for a person.
The trial says how many; look at them (each name's faces, camera by camera, side by side):

```bash
docker compose run --rm herd merges page --store /data/trial/pass2/herd.sqlite --out /data/trial/merges.html
```

Open `docker/data/trial/merges.html`: **S** the same cow, **D** two cows, then **Save decisions.json**, move it
to `docker/data/`, and:

```bash
docker compose run --rm herd merges apply --store /data/trial/pass2/herd.sqlite --decisions /data/decisions.json
docker compose run --rm herd trial --score-only --config /data/barn.toml --out /data/trial --boxes /data/boxes --labels /data/labels.json
```

Live, the same with `--config /data/barn.toml` instead of `--store`: "same" counts in the reports at once, the
gallery merges the two the next night; "different" is never asked again.

**Tracks that break in one camera** (a cow walking, a lying cow the detector loses for minutes) break the
joining too. Two settings from ByteTrack (`herd/GUIDE.md` section 4.8):

```toml
[tracker]
kalman = true        # a walking cow's box looked for where her speed puts it
low_score = 0.2      # boxes the detector is unsure of (under [detector] threshold) keep a cow's track, never start one
```

**One clock.** Joining compares two cameras at the same second. Turn on time sync in every camera's web page
(PTP if it has it, else NTP, against the same server as this computer). A camera's stream may still come late:
the end of a `trial` (and of a live run) says it, e.g. `cam2's frames come +1.4 s late against cam1's ...:
time_offset_s = -1.4 on cam2`. Put that line into cam2's `[[cameras]]` block and run `trial` again.

**No cows walking: click the same spots on both pictures.** Nothing is measured; you only show
which spot of one picture is which spot of the other:

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --pairs cam1:cam2 cam3:cam4 --pick
```

1. Open `docker/data/calib/pick.html` in a browser: each camera pair's pictures of one moment side by side.
2. Click a spot on one picture, then the same spot on the other (they get the same number). Spots **on
   the floor**: the foot of a post, a corner of the trough where it stands on the floor, a drain or a
   mark on the floor - not a cow's back or the top rail. 6-10 pairs, spread over the floor both
   cameras see (7 or more: the wide lens's bend is found too). `+` / `-` zoom; click a number to
   remove it.
3. **Save points.json** (it goes to your Downloads) and put it into `docker/data/calib/`.
4. Run:

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --points /data/calib/points.json --apply
```

The log says per pair how many spots one floor mapping agrees on and how close it puts them to your
clicks (a few px is good). A spot that disagrees with the others (clicked on two different places) is
left out and named; fewer than 5 spots that agree, or spots along one line (one fence), are refused
and nothing is written for that pair. Cameras whose shared floor is only a strip along a fence cannot
be calibrated this way: take that pair out of `[mct] disjoint`, put it in `neighbours`, and let the
gallery join their cows by look.
`calib/cam1_cam2_spots.jpg` shows the spots numbered as on the page: green agree, red are off (with
the px), grey were left out. Fix them on the page (it remembers your clicks), save, run again. Then the check on the cows (share, and what chance alone would give) and the
pictures `calib/cam1_cam2.jpg`, `calib/cam3_cam4.jpg`. With `--apply` the pairs also leave `[mct]
disjoint`; two groups of cameras that share no floor are kept far apart on the plan.

**Check the calibration by eye** - what barn.toml holds now (its `floor_points`, `distortion`), on the
pictures the spots were clicked on (`calib/pick`):

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --out /data/calib --overlay
```

For each calibrated pair, both ways, the other camera's picture is carried over the floor onto this one
and laid over it half-transparent: `calib/cam2_on_cam1.jpg`, `calib/cam1_on_cam2.jpg`, and
`calib/overlay.html` with a slider (0% this camera only, 100% the other's). Where the mapping is right
the floor's lines, stall edges and the feet of posts and drinkers stay in place as the slider moves;
cows' backs, walls and rails stand above the floor and never line up - that is expected. Circles are
the spots clicked on this picture, crosses where the other picture's clicks land (the line between: px
off); yellow / blue are the other camera's `mask` / `tracking_mask` drawn on this picture. Darkened: no
floor both cameras see. Options: `--pairs cam1:cam2`, `--videos /data/rec_fixed --at 300` (the
recordings' moment 300 s in instead of the clicked pictures), `--alpha 0.7` (the jpgs' opacity).

## 7. The barn, live

```bash
docker compose --profile live up -d barn            # starts, and starts again with Docker Desktop
docker compose logs -f barn                         # its log
docker compose --profile live down                  # stop it
```

Reports, while it runs or after:

```bash
docker compose run --rm herd report --config /data/barn.toml csv
docker compose run --rm herd report --config /data/barn.toml alerts --day 2026-10-20
docker compose run --rm herd gallery --config /data/barn.toml
```

`herd/SETUP.md` sections 8–9 say what they mean. In Docker Desktop → Settings → General, turn on
"Start Docker Desktop when you sign in": the barn then comes back by itself after a restart.

## 8. Other commands

`docker compose run --rm herd help` lists them. Anything `herd.py` takes runs as
`docker compose run --rm herd <command> ...`; `bash` opens a shell in the container.

## Updating

After `git pull`, run `docker compose build` again. Only the code is rebuilt: the models' layers stay
cached unless `docker/models/` changed. Then `docker compose --profile live up -d barn` restarts the
live barn on the new image.
