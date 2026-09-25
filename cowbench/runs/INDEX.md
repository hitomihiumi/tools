# Runs

Archived runs, kept in the repository. Each directory holds the whole record:
`manifest.jsonl` (which cows), `results.jsonl` (raw model answers, including the
model's reasoning), `run_meta.json` (model, quantization, date, sampling, prompt
hash), `metrics.json` and `report.md`.

`results.jsonl` is the thing worth keeping: metrics and reports are pure
functions over it, so a changed metric or a reworded report never needs the
model again. Working directories (`out/`, `out-*/`) are scratch and gitignored;
anything worth citing is copied here.

All against `RedHatAI/Muse-Glimmer-30B-FP8-block`, FP8 block-wise
(compressed-tensors), vLLM 0.30.0, temperature 0, seed 0, structured output,
image mode `marked`, 2026-09-25.

| Directory | Sample | Width | Frames | Exact error | Posture | Activity |
|---|---|---|---|---|---|---|
| `2026-09-25_clip371_w1280_f1` | clip 371, 120 cows | 1280 | 1 | 27.5% | 17.5% | 23.3% |
| `2026-09-25_clip371_w1280_f5` | same 120 cows | 1280 | 5 / 2.0s | 28.3% | 17.5% | 20.8% |
| `2026-09-25_val-full_w1280_f1` | **all of val, 2532 cows, 292 keyframes, 50 clips** | 1280 | 1 | **30.8%** | 9.8% | 25.6% |
| `2026-09-25_val300_w1280` | 300 cows sampled from val, seed 0 | 1280 | 1 | 28.0% | 9.3% | 22.7% |
| `2026-09-25_val300_w1920` | the same 300 cows | 1920 | 1 | 24.3% | 6.7% | 21.0% |
| `2026-09-25_val300_both` | the same 300 cows, marked+crop | 1920 | 1 | 24.7% | 5.3% | 20.7% |
| `2026-09-25_val300_jaw8` | the same 300 cows, crops only | 1920 | 8 / 1.5s | 32.0% | 5.7% | 29.7% |
| `2026-09-25_val300_jaw20` | the same 300 cows, crops only | 1920 | 20 / 1.52s = 12.5 fps | 33.7% | 5.0% | 31.7% |
| `2026-09-25_val-full_w1920_f1` | **all of val, 2532 cows** | 1920 | 1 | **29.3%** / **26.8%** voted | 7.4% / 6.6% | 25.4% / 23.1% |

## What the runs established

**One clip is not a measurement.** Per-clip exact-match error across the 50 val
clips runs from 0% to 100%, median 28%, IQR 20–45%. Clip 371's 27.5% landed near
the median by luck, and it badly misrepresented two things: posture (17.5% error
there vs 9.8% across val) and the box-size effect (87%→0% there vs 39%→12%).

**Half the errors are one class.** `ruminating` was predicted **zero times in
2532 requests**, against 383 annotated. Setting those examples aside drops
exact-match error from 30.8% to 18.4%. Nothing tried so far moves it: five
frames of video scored 1/18, native resolution 0/42.

**Resolution improves discrimination; temporal context only moved the
threshold.** Paired over 300 examples:

| Change | Exact | Fixed | Broken | p (McNemar) |
|---|---|---|---|---|
| 1280 -> 1920 | 28.0% -> 24.3% | 21 | 10 | 0.071 |
| 1 frame -> 5 frames (clip 371) | 27.5% -> 28.3% | 4 | 5 | 1.000 |

The two look superficially similar and are not. Going to 5 frames raised lying
recall 20.8%→41.7% while dropping standing recall 97.9%→92.7%, and the count of
`lying` answers went 7→17: the model became less biased, not more accurate.
Going to 1920 raised *both* recalls (lying 83.2%→86.6%, standing 95.6%→97.8%)
with the predicted counts unchanged (107→107, 193→193), and the gain falls off
monotonically with box size — 10 fixed in the smallest quartile, 2 in the
largest. That is a model that was short of pixels.

`--max-width` now defaults to 1920 for that reason. It costs ~2x the image
tokens (1448 -> 2943 per request). p=0.071 is suggestive rather than settled;
re-running all of val at 1920 would resolve it, but no decision depends on the
exact magnitude.

## Rumination is not visible to this model zero-shot

Tested at three frame rates on the same 42 ruminating cows, crops enlarged to
768px so the head is large:

| Setup | Said `ruminating` | Correct |
|---|---|---|
| 1 frame, 1920 marked | 0 | 0/42 |
| 8 crops, 4.67 fps | 2 | 1/42 |
| 20 crops, 12.5 fps | 5 | 2/42 |

12.5 fps is five times the Nyquist rate for a ~1 Hz chew. Paired against
4.67 fps, where only the frame rate differs: 11 fixed, 16 broken, p = 0.442.
The model is not ignoring the class - its reasoning on every ruminating cow
considers it and looks for jaw movement across the frames ("Can we see jaw
movement? ... the cow's head is relatively still ... hard to tell"), then
settles on `none`. It looks and does not see.

Note: the `reasoning` field on vLLM 0.30 with the `muse_glimmer` parser starts
with an echo of the prompt, which itself contains "jaw" and "chewing cud";
strip everything up to "Answer with JSON only." before searching it.

## Comparing

```bash
python cowbench.py compare --a runs/2026-09-25_val300_w1280 --b runs/2026-09-25_val300_w1920
```

Paired McNemar over the shared example ids, not two error rates side by side.
The runs answer the same cows, so the only informative examples are the ones
that changed, and a sample this size cannot resolve a few points any other way.
