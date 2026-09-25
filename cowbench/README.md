# cowbench

Measures how well a vision-language model served by vLLM reads cow behaviour on
the [CBVD-5 dataset](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset)
([paper](https://www.nature.com/articles/s41598-024-65953-x)), and writes a
report with the example count, the error rate, the model name, the test date,
the quantization and the dataset link.

## What one "example" is

One annotated cow on one keyframe. The model sees the full 1920x1080 keyframe
with that cow outlined in bright green, and answers two fields.

## Stills or video

Both. `--frames 1` (the default) sends the annotated keyframe from
`labelframes/`. `--frames 5 --span 2.0` decodes that many frames from
`videos/videos/<id>.mp4`, evenly spread over the given number of seconds and
centred on the keyframe, and sends them as a sequence.

The clips are 1920x1080, 25 fps, 10 s. The keyframe
`labelframes/<id>_<t>.jpg` is exactly frame `t*25` of `<id>.mp4` — checked by
pixel comparison, the only difference is JPEG recompression noise — so the two
paths describe the same moment and their numbers are comparable.

Worth doing at least once, because of what `rumination` is: chewing cud, a
movement. On a single still frame a ruminating cow and an idle one are the same
picture, and an error there says more about the frame than about the model. The
cost is that the box is annotated on the middle frame only, so a cow that walks
drifts off its own outline — which is why the span is short by default and the
prompt tells the model to follow the animal rather than the rectangle.

## The annotation is not a 5-way label

CBVD-5 ships AVA-format annotations, which are multi-label: one box appears as
several CSV rows, one per action id.

```
393,4,0.330,0.230,0.417,0.365,2,1   <- lying down
393,4,0.330,0.230,0.417,0.365,5,1   <- rumination     (same box)
```

`labelmap.txt` numbers the classes 1-5 (the dataset card's "0-4" is wrong), and
they form two independent axes:

| axis | values | per box |
|---|---|---|
| posture | `1` stand, `2` lying down | exactly one, always |
| activity | `3` foraging, `4` drinking water, `5` rumination | zero or one |

So the model is asked for a posture *and* an activity, and three error rates are
reported: posture, activity, and exact match. Scoring this as a flat 5-way
problem would mark "lying" wrong on a box annotated `lying + rumination`, which
inflates the error rate for no reason.

## Which split

`annotations/ava_val_v2.1.csv` — 2533 boxes across 50 clips.

`ava_test_v2.1.csv` is `ava_val_v2.1.csv` with every row duplicated: 7522 rows,
the same 2533 boxes, nothing new. It is not used.

One box in val carries both `stand` and `lying down`. There is no correct answer
to grade against, so it is dropped before the run and listed in the report.

## Usage

```bash
pip install -r requirements.txt

python cowbench.py plan   --video 371        # -> out/manifest.jsonl
python cowbench.py run                       # -> out/results.jsonl
python cowbench.py score                     # -> out/metrics.json
python cowbench.py report                    # -> out/report.md
```

Clip **371** is the default target: 120 boxes (over the 100 minimum) from a
single clip, and the only val clip containing all five classes —
`stand` 96, `lying` 24, `foraging` 78, `rumination` 18, `drinking` 6.

Only `run` talks to the model. `score` and `report` are pure functions over
`results.jsonl`, so metrics and wording can be reworked without re-querying.
`run` is resumable: it skips ids already present in `results.jsonl`.

### Reaching a model on a pod

```bash
ssh -N -L 8000:127.0.0.1:8000 root@<pod>
python cowbench.py run --base-url http://127.0.0.1:8000 --model muse-glimmer
```

The dataset stays on the local machine; frames go over the tunnel as base64
data URLs. Clip 371 needs six JPEGs, about 440 KiB each after the downscale.

## Options that matter

| flag | default | why |
|---|---|---|
| `--mode` | `marked` | `marked` keeps the scene; `crop` makes the cow bigger but hides the trough, and `drinking` vs `feeding` is decided by what the head is over. `both` sends two images and lets the report separate "cannot see the cow" from "cannot read the scene". |
| `--frames` / `--span` | `1` / `2.0` | `>1` decodes from the mp4 instead of the keyframe; see above |
| `--max-width` | `1280` | downscale before encoding; trades image tokens against small-cow visibility. Boxes in clip 371 are 0.8–5.8% of the frame, none under 40px on a side after the downscale |
| `--temperature` | `0.0` | with `seed=0`, reruns are comparable |
| `--max-tokens` | `4096` | Muse Glimmer reasons before answering and reasons longer with more images. At 2048 a five-frame request is truncated mid-thought and returns nothing; on `finish_reason: length` the client retries once at double the budget |
| `--concurrency` | `4` | vLLM batches these; raise it if the GPU is idle |
| `--limit` / `--seed` | off / `0` | seeded subsample, recorded in the manifest |

## Output

`run` answers through vLLM's structured output (`response_format: json_schema`),
so a malformed answer is not a thing that can happen — a parser would otherwise
be one more source of error indistinguishable from the model being wrong:

```json
{"posture": "standing|lying", "activity": "feeding|drinking|ruminating|none", "confidence": 0.0}
```

`results.jsonl` keeps the raw response and, because Muse Glimmer is served with
`--reasoning-parser`, its `reasoning_content` too — useful for reading back why
a given box went wrong.

`report.md` carries the required header (model, served name, quantization, test
date, dataset link, split, clips, example count, exact-match error rate, vLLM
version, sampling params, image mode, prompt hash) plus:

- three error rates with 95% Wilson confidence intervals;
- the **majority-class baseline** for each axis. 80% of boxes are `standing`, so
  a constant answer already scores well, and an error rate quoted without that
  comparison says nothing;
- per-class precision / recall / F1 and both confusion matrices;
- failed requests and excluded boxes, counted separately and never hidden.

`drinking` has 6 boxes in clip 371 (53 in all of val). Any per-class number for
it is indicative only — that is what the confidence intervals are there to show.
