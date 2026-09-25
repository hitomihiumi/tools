# Muse Glimmer on CBVD-5: cow behaviour recognition

| | |
|---|---|
| Model | `RedHatAI/Muse-Glimmer-30B-FP8-block` |
| Served as | `muse-glimmer` |
| Quantization | FP8 block-wise (W8A8, compressed-tensors); KV cache: auto (bf16) |
| Test date | 2026-09-25T16:13:04+03:00 |
| Dataset | [CBVD-5 (Cow Behavior Video Dataset)](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset) — [paper](https://www.nature.com/articles/s41598-024-65953-x) |
| Split / source | `annotations\ava_val_v2.1.csv` |
| Clips | 50 clips |
| Keyframes | 183 |
| Examples (annotated boxes) | **300** |
| **Exact-match error rate** | **33.7%** |
| Serving | vLLM 0.30.0 |
| Sampling | temperature=0.0, seed=0, max_tokens=4096, structured output (json_schema) |
| Image mode | crop, max width 1920px |
| Temporal context | 20 frames over 1.52s from the clip |
| Prompt revision | `c97a452152e9` |

## What is being measured

One example = one annotated cow on one keyframe. The model sees the full keyframe with that cow outlined and answers two independent fields, matching how CBVD-5 annotates: a **posture** (standing / lying) and an **activity** (feeding / drinking / ruminating / none). Exact match means both are right.

The bounding box is annotated on the middle frame only; over 1.52s a moving cow drifts off its own outline. That is a cost of the temporal context, paid to make `ruminating` observable at all.

## Headline numbers

| Axis | Error rate | 95% CI | Majority baseline | Correct |
|---|---|---|---|---|
| Exact match (both fields) | **33.7%** | 28.6% – 39.2% | — | 199/300 |
| Posture only | **5.0%** | 3.1% – 8.1% | 39.7% (always `standing`) | 285/300 |
| Activity only | **31.7%** | 26.7% – 37.1% | 51.3% (always `none`) | 205/300 |

## Predictions given vs annotated

| Axis | Class | Annotated | Predicted |
|---|---|---|---|
| posture | `standing` | 181 | 184 |
| posture | `lying` | 119 | 116 |
| activity | `feeding` | 107 | 146 |
| activity | `drinking` | 5 | 4 |
| activity | `ruminating` | 42 | 5 |
| activity | `none` | 146 | 145 |

## Error rate by clip

50 clips, worst first. Spread 0.0% – 100.0%. A single-clip number sits somewhere in this range and says as much about that camera angle as about the model.

| Clip | Keyframes | Cows | Exact-match error |
|---|---|---|---|
| 345 | 2 | 3 | 100.0% |
| 353 | 1 | 1 | 100.0% |
| 354 | 2 | 2 | 100.0% |
| 364 | 1 | 1 | 100.0% |
| 374 | 3 | 3 | 100.0% |
| 386 | 3 | 4 | 100.0% |
| 394 | 1 | 1 | 100.0% |
| 358 | 5 | 11 | 72.7% |
| 355 | 2 | 3 | 66.7% |
| 372 | 6 | 12 | 66.7% |
| 377 | 3 | 3 | 66.7% |
| 378 | 3 | 3 | 66.7% |
| 351 | 3 | 5 | 60.0% |
| 381 | 5 | 5 | 60.0% |
| 350 | 3 | 4 | 50.0% |
| 375 | 3 | 6 | 50.0% |
| 392 | 6 | 8 | 50.0% |
| 384 | 4 | 7 | 42.9% |
| 367 | 6 | 12 | 41.7% |
| 379 | 4 | 5 | 40.0% |
| 380 | 3 | 5 | 40.0% |
| 347 | 5 | 13 | 38.5% |
| 348 | 6 | 13 | 38.5% |
| 357 | 5 | 8 | 37.5% |
| 346 | 4 | 9 | 33.3% |
| 352 | 3 | 3 | 33.3% |
| 370 | 4 | 6 | 33.3% |
| 385 | 4 | 6 | 33.3% |
| 376 | 5 | 7 | 28.6% |
| 356 | 3 | 4 | 25.0% |
| 389 | 3 | 4 | 25.0% |
| 373 | 5 | 13 | 23.1% |
| 362 | 4 | 5 | 20.0% |
| 393 | 2 | 5 | 20.0% |
| 360 | 3 | 6 | 16.7% |
| 366 | 4 | 6 | 16.7% |
| 391 | 3 | 6 | 16.7% |
| 371 | 6 | 13 | 15.4% |
| 365 | 6 | 16 | 12.5% |
| 344 | 1 | 1 | 0.0% |
| 341 | 2 | 3 | 0.0% |
| 349 | 6 | 9 | 0.0% |
| 359 | 4 | 5 | 0.0% |
| 361 | 3 | 3 | 0.0% |
| 363 | 5 | 9 | 0.0% |
| 382 | 5 | 7 | 0.0% |
| 383 | 5 | 7 | 0.0% |
| 387 | 3 | 3 | 0.0% |
| 388 | 4 | 5 | 0.0% |
| 390 | 1 | 1 | 0.0% |

## Error rate by how big the cow is in frame

Quartiles of bounding-box area. `y centre` is where those boxes sit vertically, which in this barn is also how far away they are: the near feed barrier is the lower half of the frame, the far cubicle row the upper.

| Quartile | Box area, % of frame | n | y centre | Annotated `lying` | Exact-match error |
|---|---|---|---|---|---|
| Q1 | 0.29 – 0.82 | 75 | 0.27 | 64.0% | **50.7%** |
| Q2 | 0.82 – 1.36 | 75 | 0.29 | 64.0% | **44.0%** |
| Q3 | 1.37 – 3.31 | 75 | 0.33 | 30.7% | **32.0%** |
| Q4 | 3.55 – 10.87 | 75 | 0.47 | 0.0% | **8.0%** |

## Per class

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `posture:standing` | 181 | 0.95 | 0.97 | 0.96 |
| `posture:lying` | 119 | 0.95 | 0.92 | 0.94 |
| `activity:feeding` | 107 | 0.68 | 0.93 | 0.78 |
| `activity:drinking` | 5 | 0.50 | 0.40 | 0.44 |
| `activity:ruminating` | 42 | 0.40 | 0.05 | 0.09 |
| `activity:none` | 146 | 0.70 | 0.70 | 0.70 |

## Confusion matrix — posture

| ground truth / predicted | `standing` | `lying` | `<failed>` |
|---|---|---|---|
| `standing` | 175 | 6 | 0 |
| `lying` | 9 | 110 | 0 |

## Confusion matrix — activity

| ground truth / predicted | `feeding` | `drinking` | `ruminating` | `none` | `<failed>` |
|---|---|---|---|---|---|
| `feeding` | 99 | 0 | 1 | 7 | 0 |
| `drinking` | 3 | 2 | 0 | 0 | 0 |
| `ruminating` | 4 | 0 | 2 | 36 | 0 |
| `none` | 40 | 2 | 2 | 102 | 0 |

## Every example

All 300 annotated cows, 183 keyframes. Ordered by keyframe, then left to right. `x` marks the horizontal centre of the box as a fraction of frame width, `y` the vertical centre.

| # | clip | t, s | x | y | ground truth | prediction | |
|---|---|---|---|---|---|---|---|
| 1 | 341 | 4 | 0.08 | 0.51 | standing / feeding | standing / feeding | ok |
| 2 | 341 | 6 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 3 | 341 | 6 | 0.63 | 0.48 | standing / feeding | standing / feeding | ok |
| 4 | 344 | 6 | 0.24 | 0.29 | standing / none | standing / none | ok |
| 5 | 345 | 4 | 0.24 | 0.20 | standing / none | standing / feeding | activity |
| 6 | 345 | 6 | 0.09 | 0.28 | standing / none | standing / feeding | activity |
| 7 | 345 | 6 | 0.31 | 0.19 | standing / none | standing / feeding | activity |
| 8 | 346 | 2 | 0.70 | 0.20 | standing / none | standing / none | ok |
| 9 | 346 | 2 | 0.94 | 0.26 | lying / none | standing / feeding | posture + activity |
| 10 | 346 | 3 | 0.28 | 0.24 | standing / none | standing / none | ok |
| 11 | 346 | 3 | 0.41 | 0.25 | standing / none | standing / feeding | activity |
| 12 | 346 | 5 | 0.11 | 0.26 | standing / none | standing / feeding | activity |
| 13 | 346 | 5 | 0.37 | 0.45 | standing / feeding | standing / feeding | ok |
| 14 | 346 | 5 | 0.48 | 0.21 | standing / none | standing / none | ok |
| 15 | 346 | 6 | 0.03 | 0.51 | standing / feeding | standing / feeding | ok |
| 16 | 346 | 6 | 0.64 | 0.51 | standing / feeding | standing / feeding | ok |
| 17 | 347 | 2 | 0.04 | 0.48 | standing / feeding | standing / feeding | ok |
| 18 | 347 | 3 | 0.17 | 0.39 | standing / feeding | standing / none | activity |
| 19 | 347 | 3 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 20 | 347 | 3 | 0.71 | 0.22 | standing / none | standing / feeding | activity |
| 21 | 347 | 4 | 0.01 | 0.47 | standing / feeding | standing / feeding | ok |
| 22 | 347 | 4 | 0.38 | 0.44 | standing / feeding | standing / feeding | ok |
| 23 | 347 | 4 | 0.72 | 0.25 | standing / drinking | standing / feeding | activity |
| 24 | 347 | 4 | 0.83 | 0.25 | standing / none | standing / feeding | activity |
| 25 | 347 | 4 | 0.88 | 0.49 | standing / feeding | standing / feeding | ok |
| 26 | 347 | 4 | 0.92 | 0.27 | standing / none | standing / feeding | activity |
| 27 | 347 | 5 | 0.17 | 0.39 | standing / feeding | standing / feeding | ok |
| 28 | 347 | 7 | 0.38 | 0.44 | standing / feeding | standing / feeding | ok |
| 29 | 347 | 7 | 0.78 | 0.51 | standing / feeding | standing / feeding | ok |
| 30 | 348 | 2 | 0.52 | 0.45 | standing / feeding | standing / feeding | ok |
| 31 | 348 | 2 | 0.54 | 0.20 | lying / none | lying / none | ok |
| 32 | 348 | 2 | 0.59 | 0.27 | standing / none | lying / none | posture |
| 33 | 348 | 2 | 0.71 | 0.50 | standing / feeding | standing / feeding | ok |
| 34 | 348 | 2 | 0.83 | 0.45 | standing / feeding | standing / feeding | ok |
| 35 | 348 | 3 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 36 | 348 | 4 | 0.26 | 0.30 | lying / ruminating | lying / none | activity |
| 37 | 348 | 4 | 0.71 | 0.21 | standing / none | standing / none | ok |
| 38 | 348 | 5 | 0.71 | 0.21 | standing / none | standing / feeding | activity |
| 39 | 348 | 6 | 0.81 | 0.23 | standing / none | standing / feeding | activity |
| 40 | 348 | 6 | 0.93 | 0.29 | standing / none | lying / none | posture |
| 41 | 348 | 7 | 0.30 | 0.41 | standing / feeding | standing / feeding | ok |
| 42 | 348 | 7 | 0.59 | 0.27 | standing / none | standing / none | ok |
| 43 | 349 | 2 | 0.35 | 0.50 | standing / feeding | standing / feeding | ok |
| 44 | 349 | 3 | 0.72 | 0.21 | standing / none | standing / none | ok |
| 45 | 349 | 4 | 0.94 | 0.47 | standing / feeding | standing / feeding | ok |
| 46 | 349 | 5 | 0.17 | 0.49 | standing / feeding | standing / feeding | ok |
| 47 | 349 | 6 | 0.17 | 0.49 | standing / feeding | standing / feeding | ok |
| 48 | 349 | 6 | 0.72 | 0.21 | standing / none | standing / none | ok |
| 49 | 349 | 7 | 0.10 | 0.51 | standing / feeding | standing / feeding | ok |
| 50 | 349 | 7 | 0.17 | 0.49 | standing / feeding | standing / feeding | ok |
| 51 | 349 | 7 | 0.72 | 0.21 | standing / none | standing / none | ok |
| 52 | 350 | 2 | 0.15 | 0.47 | standing / none | standing / feeding | activity |
| 53 | 350 | 5 | 0.77 | 0.43 | standing / feeding | standing / feeding | ok |
| 54 | 350 | 5 | 0.96 | 0.28 | lying / none | standing / feeding | posture + activity |
| 55 | 350 | 6 | 0.18 | 0.27 | lying / none | lying / none | ok |
| 56 | 351 | 3 | 0.83 | 0.29 | standing / drinking | standing / feeding | activity |
| 57 | 351 | 5 | 0.11 | 0.34 | lying / ruminating | lying / none | activity |
| 58 | 351 | 5 | 0.36 | 0.20 | standing / none | standing / none | ok |
| 59 | 351 | 6 | 0.21 | 0.47 | standing / feeding | standing / feeding | ok |
| 60 | 351 | 6 | 0.84 | 0.29 | standing / drinking | standing / feeding | activity |
| 61 | 352 | 4 | 0.09 | 0.51 | standing / none | standing / feeding | activity |
| 62 | 352 | 5 | 0.08 | 0.30 | lying / none | lying / none | ok |
| 63 | 352 | 6 | 0.07 | 0.31 | lying / none | lying / none | ok |
| 64 | 353 | 4 | 0.27 | 0.33 | lying / ruminating | lying / none | activity |
| 65 | 354 | 3 | 0.13 | 0.30 | lying / ruminating | lying / none | activity |
| 66 | 354 | 5 | 0.12 | 0.29 | lying / ruminating | lying / none | activity |
| 67 | 355 | 2 | 0.62 | 0.51 | standing / feeding | standing / feeding | ok |
| 68 | 355 | 2 | 0.92 | 0.44 | standing / none | standing / feeding | activity |
| 69 | 355 | 7 | 0.87 | 0.27 | lying / ruminating | lying / none | activity |
| 70 | 356 | 4 | 0.88 | 0.26 | lying / ruminating | lying / none | activity |
| 71 | 356 | 5 | 0.07 | 0.34 | lying / none | lying / none | ok |
| 72 | 356 | 7 | 0.30 | 0.47 | standing / feeding | standing / feeding | ok |
| 73 | 356 | 7 | 0.80 | 0.30 | standing / none | standing / none | ok |
| 74 | 357 | 2 | 0.30 | 0.44 | standing / feeding | standing / feeding | ok |
| 75 | 357 | 2 | 0.65 | 0.47 | standing / feeding | standing / feeding | ok |
| 76 | 357 | 3 | 0.95 | 0.49 | standing / feeding | standing / feeding | ok |
| 77 | 357 | 4 | 0.92 | 0.27 | standing / none | standing / feeding | activity |
| 78 | 357 | 5 | 0.65 | 0.47 | standing / feeding | standing / feeding | ok |
| 79 | 357 | 5 | 0.92 | 0.27 | standing / none | standing / feeding | activity |
| 80 | 357 | 5 | 0.98 | 0.32 | lying / none | standing / feeding | posture + activity |
| 81 | 357 | 7 | 0.95 | 0.49 | standing / feeding | standing / feeding | ok |
| 82 | 358 | 2 | 0.43 | 0.22 | lying / ruminating | lying / none | activity |
| 83 | 358 | 2 | 0.95 | 0.33 | lying / ruminating | lying / none | activity |
| 84 | 358 | 3 | 0.15 | 0.32 | lying / none | standing / feeding | posture + activity |
| 85 | 358 | 3 | 0.44 | 0.23 | lying / ruminating | lying / none | activity |
| 86 | 358 | 3 | 0.87 | 0.48 | standing / feeding | standing / feeding | ok |
| 87 | 358 | 4 | 0.44 | 0.23 | lying / ruminating | lying / none | activity |
| 88 | 358 | 4 | 0.87 | 0.48 | standing / feeding | standing / feeding | ok |
| 89 | 358 | 4 | 0.95 | 0.33 | lying / ruminating | lying / none | activity |
| 90 | 358 | 5 | 0.05 | 0.51 | standing / feeding | standing / none | activity |
| 91 | 358 | 5 | 0.95 | 0.33 | lying / ruminating | lying / ruminating | ok |
| 92 | 358 | 6 | 0.95 | 0.33 | lying / ruminating | lying / none | activity |
| 93 | 359 | 2 | 0.14 | 0.31 | lying / none | lying / none | ok |
| 94 | 359 | 3 | 0.25 | 0.27 | lying / none | lying / none | ok |
| 95 | 359 | 6 | 0.13 | 0.30 | lying / none | lying / none | ok |
| 96 | 359 | 6 | 0.25 | 0.27 | lying / none | lying / none | ok |
| 97 | 359 | 7 | 0.24 | 0.27 | lying / none | lying / none | ok |
| 98 | 360 | 2 | 0.05 | 0.34 | lying / none | lying / none | ok |
| 99 | 360 | 2 | 0.25 | 0.27 | lying / none | lying / none | ok |
| 100 | 360 | 2 | 0.90 | 0.34 | lying / none | lying / none | ok |
| 101 | 360 | 2 | 0.96 | 0.24 | standing / none | standing / feeding | activity |
| 102 | 360 | 4 | 0.25 | 0.26 | lying / none | lying / none | ok |
| 103 | 360 | 5 | 0.24 | 0.27 | lying / none | lying / none | ok |
| 104 | 361 | 3 | 0.89 | 0.31 | lying / none | lying / none | ok |
| 105 | 361 | 5 | 0.94 | 0.32 | lying / none | lying / none | ok |
| 106 | 361 | 6 | 0.89 | 0.32 | lying / none | lying / none | ok |
| 107 | 362 | 2 | 0.06 | 0.34 | lying / none | lying / feeding | activity |
| 108 | 362 | 2 | 0.89 | 0.32 | lying / none | lying / none | ok |
| 109 | 362 | 4 | 0.25 | 0.27 | lying / none | lying / none | ok |
| 110 | 362 | 5 | 0.15 | 0.29 | lying / none | lying / none | ok |
| 111 | 362 | 6 | 0.96 | 0.31 | standing / none | standing / none | ok |
| 112 | 363 | 2 | 0.53 | 0.30 | lying / none | lying / none | ok |
| 113 | 363 | 3 | 0.02 | 0.38 | lying / none | lying / none | ok |
| 114 | 363 | 3 | 0.90 | 0.34 | lying / none | lying / none | ok |
| 115 | 363 | 3 | 0.97 | 0.31 | lying / none | lying / none | ok |
| 116 | 363 | 4 | 0.25 | 0.26 | lying / none | lying / none | ok |
| 117 | 363 | 4 | 0.53 | 0.28 | lying / none | lying / none | ok |
| 118 | 363 | 6 | 0.25 | 0.26 | lying / none | lying / none | ok |
| 119 | 363 | 7 | 0.13 | 0.29 | lying / none | lying / none | ok |
| 120 | 363 | 7 | 0.26 | 0.27 | lying / none | lying / none | ok |
| 121 | 364 | 6 | 0.90 | 0.31 | standing / none | standing / feeding | activity |
| 122 | 365 | 2 | 0.04 | 0.47 | standing / feeding | standing / feeding | ok |
| 123 | 365 | 2 | 0.47 | 0.46 | standing / feeding | standing / feeding | ok |
| 124 | 365 | 2 | 0.67 | 0.50 | standing / feeding | standing / feeding | ok |
| 125 | 365 | 3 | 0.09 | 0.45 | standing / feeding | standing / feeding | ok |
| 126 | 365 | 3 | 0.47 | 0.46 | standing / feeding | standing / feeding | ok |
| 127 | 365 | 3 | 0.89 | 0.47 | standing / feeding | standing / feeding | ok |
| 128 | 365 | 4 | 0.37 | 0.52 | standing / feeding | standing / feeding | ok |
| 129 | 365 | 5 | 0.16 | 0.48 | standing / feeding | standing / feeding | ok |
| 130 | 365 | 5 | 0.34 | 0.24 | lying / ruminating | lying / none | activity |
| 131 | 365 | 5 | 0.47 | 0.46 | standing / feeding | standing / feeding | ok |
| 132 | 365 | 5 | 0.76 | 0.45 | standing / feeding | standing / feeding | ok |
| 133 | 365 | 5 | 0.95 | 0.45 | standing / feeding | standing / feeding | ok |
| 134 | 365 | 6 | 0.27 | 0.47 | standing / feeding | standing / feeding | ok |
| 135 | 365 | 6 | 0.43 | 0.23 | lying / ruminating | lying / none | activity |
| 136 | 365 | 7 | 0.09 | 0.45 | standing / feeding | standing / feeding | ok |
| 137 | 365 | 7 | 0.57 | 0.48 | standing / feeding | standing / feeding | ok |
| 138 | 366 | 2 | 0.10 | 0.35 | lying / none | lying / ruminating | activity |
| 139 | 366 | 3 | 0.29 | 0.46 | standing / feeding | standing / feeding | ok |
| 140 | 366 | 5 | 0.10 | 0.35 | lying / none | lying / none | ok |
| 141 | 366 | 5 | 0.80 | 0.46 | standing / feeding | standing / feeding | ok |
| 142 | 366 | 6 | 0.07 | 0.36 | lying / none | lying / none | ok |
| 143 | 366 | 6 | 0.72 | 0.26 | standing / drinking | standing / drinking | ok |
| 144 | 367 | 2 | 0.10 | 0.26 | standing / none | standing / feeding | activity |
| 145 | 367 | 2 | 0.15 | 0.24 | standing / none | lying / none | posture |
| 146 | 367 | 2 | 0.17 | 0.19 | standing / none | standing / none | ok |
| 147 | 367 | 2 | 0.32 | 0.21 | standing / none | standing / feeding | activity |
| 148 | 367 | 3 | 0.16 | 0.50 | standing / feeding | standing / feeding | ok |
| 149 | 367 | 3 | 0.17 | 0.19 | standing / none | standing / none | ok |
| 150 | 367 | 3 | 0.32 | 0.21 | standing / none | standing / feeding | activity |
| 151 | 367 | 4 | 0.90 | 0.32 | lying / none | lying / none | ok |
| 152 | 367 | 5 | 0.53 | 0.22 | lying / none | lying / feeding | activity |
| 153 | 367 | 5 | 0.90 | 0.32 | lying / none | lying / none | ok |
| 154 | 367 | 6 | 0.16 | 0.50 | standing / feeding | standing / feeding | ok |
| 155 | 367 | 7 | 0.53 | 0.22 | lying / none | lying / none | ok |
| 156 | 370 | 2 | 0.53 | 0.42 | standing / none | standing / none | ok |
| 157 | 370 | 2 | 0.54 | 0.21 | standing / none | standing / none | ok |
| 158 | 370 | 3 | 0.02 | 0.42 | standing / none | standing / none | ok |
| 159 | 370 | 3 | 0.52 | 0.23 | standing / none | standing / none | ok |
| 160 | 370 | 5 | 0.43 | 0.20 | standing / none | standing / feeding | activity |
| 161 | 370 | 6 | 0.03 | 0.28 | standing / none | standing / feeding | activity |
| 162 | 371 | 2 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 163 | 371 | 3 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 164 | 371 | 3 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 165 | 371 | 3 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 166 | 371 | 4 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 167 | 371 | 5 | 0.08 | 0.32 | lying / none | lying / none | ok |
| 168 | 371 | 5 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 169 | 371 | 5 | 0.26 | 0.48 | standing / feeding | lying / none | posture + activity |
| 170 | 371 | 6 | 0.08 | 0.32 | lying / none | lying / ruminating | activity |
| 171 | 371 | 6 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 172 | 371 | 6 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 173 | 371 | 7 | 0.45 | 0.21 | lying / ruminating | lying / ruminating | ok |
| 174 | 371 | 7 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 175 | 372 | 2 | 0.27 | 0.20 | standing / none | standing / feeding | activity |
| 176 | 372 | 2 | 0.40 | 0.20 | standing / none | standing / none | ok |
| 177 | 372 | 2 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 178 | 372 | 3 | 0.60 | 0.17 | standing / none | lying / none | posture |
| 179 | 372 | 4 | 0.04 | 0.48 | standing / feeding | standing / ruminating | activity |
| 180 | 372 | 4 | 0.95 | 0.30 | lying / ruminating | lying / none | activity |
| 181 | 372 | 5 | 0.40 | 0.20 | standing / none | standing / feeding | activity |
| 182 | 372 | 5 | 0.44 | 0.23 | lying / ruminating | lying / none | activity |
| 183 | 372 | 6 | 0.18 | 0.33 | lying / none | lying / none | ok |
| 184 | 372 | 6 | 0.55 | 0.33 | standing / none | standing / none | ok |
| 185 | 372 | 6 | 0.60 | 0.17 | standing / none | lying / none | posture |
| 186 | 372 | 7 | 0.27 | 0.20 | standing / none | standing / feeding | activity |
| 187 | 373 | 3 | 0.16 | 0.47 | standing / feeding | standing / feeding | ok |
| 188 | 373 | 3 | 0.93 | 0.42 | standing / feeding | standing / feeding | ok |
| 189 | 373 | 4 | 0.32 | 0.20 | standing / none | standing / feeding | activity |
| 190 | 373 | 4 | 0.90 | 0.45 | standing / feeding | standing / feeding | ok |
| 191 | 373 | 5 | 0.25 | 0.30 | lying / ruminating | standing / feeding | posture + activity |
| 192 | 373 | 5 | 0.37 | 0.41 | standing / feeding | standing / feeding | ok |
| 193 | 373 | 5 | 0.92 | 0.28 | lying / none | lying / none | ok |
| 194 | 373 | 6 | 0.75 | 0.19 | standing / drinking | standing / drinking | ok |
| 195 | 373 | 6 | 0.83 | 0.49 | standing / feeding | standing / none | activity |
| 196 | 373 | 7 | 0.37 | 0.41 | standing / feeding | standing / feeding | ok |
| 197 | 373 | 7 | 0.46 | 0.46 | standing / feeding | standing / feeding | ok |
| 198 | 373 | 7 | 0.57 | 0.48 | standing / feeding | standing / feeding | ok |
| 199 | 373 | 7 | 0.94 | 0.45 | standing / feeding | standing / feeding | ok |
| 200 | 374 | 2 | 0.94 | 0.33 | lying / ruminating | lying / feeding | activity |
| 201 | 374 | 6 | 0.94 | 0.31 | lying / ruminating | lying / none | activity |
| 202 | 374 | 7 | 0.83 | 0.23 | standing / none | standing / drinking | activity |
| 203 | 375 | 2 | 0.36 | 0.49 | standing / feeding | standing / feeding | ok |
| 204 | 375 | 5 | 0.34 | 0.20 | standing / feeding | standing / none | activity |
| 205 | 375 | 5 | 0.77 | 0.49 | standing / feeding | standing / feeding | ok |
| 206 | 375 | 7 | 0.08 | 0.35 | lying / none | lying / none | ok |
| 207 | 375 | 7 | 0.18 | 0.28 | lying / ruminating | lying / none | activity |
| 208 | 375 | 7 | 0.94 | 0.31 | lying / none | standing / feeding | posture + activity |
| 209 | 376 | 2 | 0.33 | 0.25 | lying / none | lying / none | ok |
| 210 | 376 | 3 | 0.13 | 0.29 | lying / none | standing / feeding | posture + activity |
| 211 | 376 | 4 | 0.19 | 0.28 | lying / none | lying / none | ok |
| 212 | 376 | 4 | 0.33 | 0.25 | lying / none | lying / none | ok |
| 213 | 376 | 5 | 0.20 | 0.28 | lying / none | lying / none | ok |
| 214 | 376 | 5 | 0.95 | 0.32 | lying / ruminating | lying / none | activity |
| 215 | 376 | 6 | 0.13 | 0.29 | lying / none | lying / none | ok |
| 216 | 377 | 3 | 0.43 | 0.24 | lying / ruminating | lying / none | activity |
| 217 | 377 | 4 | 0.71 | 0.46 | standing / feeding | standing / feeding | ok |
| 218 | 377 | 7 | 0.24 | 0.26 | lying / none | standing / none | posture |
| 219 | 378 | 3 | 0.07 | 0.36 | lying / none | lying / none | ok |
| 220 | 378 | 6 | 0.34 | 0.25 | lying / ruminating | lying / none | activity |
| 221 | 378 | 7 | 0.34 | 0.25 | lying / ruminating | lying / none | activity |
| 222 | 379 | 2 | 0.17 | 0.33 | lying / none | lying / none | ok |
| 223 | 379 | 3 | 0.38 | 0.29 | lying / ruminating | lying / none | activity |
| 224 | 379 | 3 | 0.88 | 0.29 | lying / none | lying / none | ok |
| 225 | 379 | 4 | 0.61 | 0.47 | standing / feeding | standing / feeding | ok |
| 226 | 379 | 6 | 0.26 | 0.31 | lying / ruminating | lying / none | activity |
| 227 | 380 | 2 | 0.07 | 0.38 | lying / none | lying / none | ok |
| 228 | 380 | 2 | 0.26 | 0.31 | lying / none | lying / none | ok |
| 229 | 380 | 3 | 0.19 | 0.28 | lying / ruminating | standing / feeding | posture + activity |
| 230 | 380 | 3 | 0.26 | 0.30 | lying / ruminating | lying / none | activity |
| 231 | 380 | 4 | 0.88 | 0.31 | lying / none | lying / none | ok |
| 232 | 381 | 2 | 0.68 | 0.49 | standing / feeding | standing / feeding | ok |
| 233 | 381 | 3 | 0.38 | 0.29 | lying / ruminating | lying / none | activity |
| 234 | 381 | 5 | 0.10 | 0.35 | lying / ruminating | lying / none | activity |
| 235 | 381 | 6 | 0.90 | 0.29 | lying / none | lying / none | ok |
| 236 | 381 | 7 | 0.38 | 0.30 | lying / ruminating | lying / none | activity |
| 237 | 382 | 2 | 0.17 | 0.33 | lying / none | lying / none | ok |
| 238 | 382 | 4 | 0.10 | 0.50 | standing / feeding | standing / feeding | ok |
| 239 | 382 | 5 | 0.59 | 0.47 | standing / feeding | standing / feeding | ok |
| 240 | 382 | 5 | 0.84 | 0.43 | standing / feeding | standing / feeding | ok |
| 241 | 382 | 6 | 0.84 | 0.44 | standing / feeding | standing / feeding | ok |
| 242 | 382 | 7 | 0.09 | 0.50 | standing / feeding | standing / feeding | ok |
| 243 | 382 | 7 | 0.95 | 0.34 | lying / none | lying / none | ok |
| 244 | 383 | 2 | 0.16 | 0.34 | lying / none | lying / none | ok |
| 245 | 383 | 3 | 0.24 | 0.30 | standing / none | standing / none | ok |
| 246 | 383 | 3 | 0.37 | 0.30 | lying / none | lying / none | ok |
| 247 | 383 | 3 | 0.84 | 0.42 | standing / feeding | standing / feeding | ok |
| 248 | 383 | 4 | 0.16 | 0.34 | lying / none | lying / none | ok |
| 249 | 383 | 5 | 0.53 | 0.29 | lying / none | lying / none | ok |
| 250 | 383 | 6 | 0.89 | 0.29 | lying / none | lying / none | ok |
| 251 | 384 | 2 | 0.11 | 0.36 | lying / none | lying / none | ok |
| 252 | 384 | 2 | 0.16 | 0.34 | lying / none | lying / none | ok |
| 253 | 384 | 3 | 0.10 | 0.36 | lying / none | lying / none | ok |
| 254 | 384 | 3 | 0.24 | 0.36 | standing / none | standing / feeding | activity |
| 255 | 384 | 4 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 256 | 384 | 4 | 0.90 | 0.30 | standing / none | standing / feeding | activity |
| 257 | 384 | 5 | 0.34 | 0.27 | standing / none | standing / feeding | activity |
| 258 | 385 | 2 | 0.37 | 0.28 | lying / ruminating | lying / none | activity |
| 259 | 385 | 2 | 0.86 | 0.45 | standing / feeding | standing / feeding | ok |
| 260 | 385 | 3 | 0.16 | 0.34 | lying / none | lying / none | ok |
| 261 | 385 | 4 | 0.04 | 0.37 | standing / none | standing / none | ok |
| 262 | 385 | 5 | 0.16 | 0.34 | lying / none | lying / none | ok |
| 263 | 385 | 5 | 0.31 | 0.25 | lying / ruminating | lying / feeding | activity |
| 264 | 386 | 3 | 0.26 | 0.31 | lying / ruminating | lying / none | activity |
| 265 | 386 | 4 | 0.30 | 0.24 | lying / ruminating | lying / none | activity |
| 266 | 386 | 4 | 0.43 | 0.27 | lying / ruminating | lying / none | activity |
| 267 | 386 | 5 | 0.42 | 0.28 | lying / ruminating | lying / none | activity |
| 268 | 387 | 2 | 0.42 | 0.30 | lying / none | lying / none | ok |
| 269 | 387 | 3 | 0.24 | 0.31 | standing / none | standing / none | ok |
| 270 | 387 | 7 | 0.42 | 0.29 | lying / none | lying / none | ok |
| 271 | 388 | 2 | 0.79 | 0.44 | standing / feeding | standing / feeding | ok |
| 272 | 388 | 4 | 0.49 | 0.49 | standing / feeding | standing / feeding | ok |
| 273 | 388 | 4 | 0.80 | 0.45 | standing / feeding | standing / feeding | ok |
| 274 | 388 | 6 | 0.04 | 0.50 | standing / feeding | standing / feeding | ok |
| 275 | 388 | 7 | 0.50 | 0.48 | standing / feeding | standing / feeding | ok |
| 276 | 389 | 2 | 0.79 | 0.46 | standing / feeding | standing / feeding | ok |
| 277 | 389 | 4 | 0.95 | 0.39 | standing / none | standing / feeding | activity |
| 278 | 389 | 7 | 0.31 | 0.51 | standing / feeding | standing / feeding | ok |
| 279 | 389 | 7 | 0.88 | 0.45 | standing / feeding | standing / feeding | ok |
| 280 | 390 | 6 | 0.57 | 0.50 | standing / feeding | standing / feeding | ok |
| 281 | 391 | 2 | 0.83 | 0.48 | standing / feeding | standing / feeding | ok |
| 282 | 391 | 6 | 0.37 | 0.18 | standing / none | standing / none | ok |
| 283 | 391 | 6 | 0.44 | 0.17 | standing / none | standing / none | ok |
| 284 | 391 | 7 | 0.37 | 0.17 | standing / none | standing / feeding | activity |
| 285 | 391 | 7 | 0.44 | 0.16 | standing / none | standing / none | ok |
| 286 | 391 | 7 | 0.84 | 0.43 | standing / feeding | standing / feeding | ok |
| 287 | 392 | 2 | 0.05 | 0.50 | standing / feeding | standing / feeding | ok |
| 288 | 392 | 3 | 0.39 | 0.18 | standing / none | standing / feeding | activity |
| 289 | 392 | 4 | 0.28 | 0.30 | standing / none | standing / none | ok |
| 290 | 392 | 4 | 0.70 | 0.22 | standing / none | standing / none | ok |
| 291 | 392 | 5 | 0.18 | 0.48 | standing / feeding | standing / none | activity |
| 292 | 392 | 5 | 0.33 | 0.48 | standing / feeding | standing / none | activity |
| 293 | 392 | 6 | 0.17 | 0.46 | standing / feeding | standing / feeding | ok |
| 294 | 392 | 7 | 0.71 | 0.21 | standing / none | standing / drinking | activity |
| 295 | 393 | 2 | 0.90 | 0.28 | standing / none | standing / none | ok |
| 296 | 393 | 3 | 0.25 | 0.27 | lying / ruminating | lying / none | activity |
| 297 | 393 | 3 | 0.34 | 0.25 | lying / none | lying / none | ok |
| 298 | 393 | 3 | 0.88 | 0.48 | standing / feeding | standing / feeding | ok |
| 299 | 393 | 3 | 0.90 | 0.29 | standing / none | standing / none | ok |
| 300 | 394 | 2 | 0.33 | 0.25 | lying / ruminating | lying / none | activity |

## Excluded from scoring

Boxes whose annotation has no single correct answer (contradictory or missing posture label). They are dropped before the model is queried, so they affect neither the count nor the error rate.

- `382_00005_ece9b588` — 2 posture labels, 0 activity labels: stand, lying down

