# Runs

Archived runs, kept in the repository. Each directory holds the whole record:
`manifest.jsonl` (which cows), `results.jsonl` (raw model answers, including the
model's reasoning), `run_meta.json` (model, quantization, date, sampling, prompt
hash), `metrics.json` and `report.md`.

`results.jsonl` is the thing worth keeping: metrics and reports are pure
functions over it, so a changed metric or a reworded report never needs the
model again. Working directories (`out/`, `out-*/`) are scratch and gitignored;
anything worth citing is copied here.

| Directory | Sample | Image | Frames | Exact error | Notes |
|---|---|---|---|---|---|
| `2026-09-25_clip371_w1280_f1` | clip 371, 6 keyframes, 120 cows | marked, 1280px | 1 | 27.5% | first full run |
| `2026-09-25_clip371_w1280_f5` | same 120 cows | marked, 1280px | 5 over 2.0s | 28.3% | paired vs f1: p=1.000 on posture, 0.508 on activity — indistinguishable |

All against `RedHatAI/Muse-Glimmer-30B-FP8-block`, FP8 block-wise, vLLM 0.30.0,
temperature 0, seed 0, structured output.

## Comparing

```bash
python cowbench.py compare --a runs/<baseline> --b runs/<variant>
```

Paired McNemar over the shared example ids, not two error rates side by side.
The runs answer the same cows, so the only informative examples are the ones
that changed, and a sample this size cannot resolve a few points of difference
any other way.
