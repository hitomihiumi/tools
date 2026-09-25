# Muse Glimmer on CBVD-5: cow behaviour recognition

| | |
|---|---|
| Model | `RedHatAI/Muse-Glimmer-30B-FP8-block` |
| Served as | `muse-glimmer` |
| Quantization | FP8 block-wise (W8A8, compressed-tensors); KV cache: auto (bf16) |
| Test date | 2026-09-25T12:58:43+03:00 |
| Dataset | [CBVD-5 (Cow Behavior Video Dataset)](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset) — [paper](https://www.nature.com/articles/s41598-024-65953-x) |
| Split / source | `annotations\ava_val_v2.1.csv` |
| Clips | 371 |
| Keyframes | 6 |
| Examples (annotated boxes) | **120** |
| **Exact-match error rate** | **28.3%** |
| Serving | vLLM 0.30.0 |
| Sampling | temperature=0.0, seed=0, max_tokens=4096, structured output (json_schema) |
| Image mode | marked, max width 1280px |
| Temporal context | 5 frames over 2.0s from the clip |
| Prompt revision | `0f20701da5ab` |

## What is being measured

One example = one annotated cow on one keyframe. The model sees the full keyframe with that cow outlined and answers two independent fields, matching how CBVD-5 annotates: a **posture** (standing / lying) and an **activity** (feeding / drinking / ruminating / none). Exact match means both are right.

The bounding box is annotated on the middle frame only; over 2.0s a moving cow drifts off its own outline. That is a cost of the temporal context, paid to make `ruminating` observable at all.

## Headline numbers

| Axis | Error rate | 95% CI | Majority baseline | Correct |
|---|---|---|---|---|
| Exact match (both fields) | **28.3%** | 21.0% – 37.0% | — | 86/120 |
| Posture only | **17.5%** | 11.7% – 25.3% | 20.0% (always `standing`) | 99/120 |
| Activity only | **20.8%** | 14.5% – 28.9% | 35.0% (always `feeding`) | 95/120 |

## Reading the numbers

- On posture the model is within 2.5% of the majority baseline (17.5% vs 20.0%) — close enough that the headline number is carried by the class skew, not by the model.

## Predictions given vs annotated

| Axis | Class | Annotated | Predicted |
|---|---|---|---|
| posture | `standing` | 96 | 103 |
| posture | `lying` | 24 | 17 |
| activity | `feeding` | 78 | 79 |
| activity | `drinking` | 6 | 1 |
| activity | `ruminating` | 18 | 1 |
| activity | `none` | 18 | 39 |

## Error rate by how big the cow is in frame

Quartiles of bounding-box area. `y centre` is where those boxes sit vertically, which in this barn is also how far away they are: the near feed barrier is the lower half of the frame, the far cubicle row the upper.

| Quartile | Box area, % of frame | n | y centre | Annotated `lying` | Exact-match error |
|---|---|---|---|---|---|
| Q1 | 0.80 – 1.33 | 30 | 0.28 | 80.0% | **90.0%** |
| Q2 | 1.70 – 3.02 | 30 | 0.48 | 0.0% | **16.7%** |
| Q3 | 3.31 – 4.55 | 30 | 0.48 | 0.0% | **3.3%** |
| Q4 | 4.67 – 5.76 | 30 | 0.47 | 0.0% | **3.3%** |

## Per class

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `posture:standing` | 96 | 0.86 | 0.93 | 0.89 |
| `posture:lying` | 24 | 0.59 | 0.42 | 0.49 |
| `activity:feeding` | 78 | 0.96 | 0.97 | 0.97 |
| `activity:drinking` | 6 | 1.00 | 0.17 | 0.29 |
| `activity:ruminating` | 18 | 1.00 | 0.06 | 0.11 |
| `activity:none` | 18 | 0.44 | 0.94 | 0.60 |

## Confusion matrix — posture

| ground truth / predicted | `standing` | `lying` | `<failed>` |
|---|---|---|---|
| `standing` | 89 | 7 | 0 |
| `lying` | 14 | 10 | 0 |

## Confusion matrix — activity

| ground truth / predicted | `feeding` | `drinking` | `ruminating` | `none` | `<failed>` |
|---|---|---|---|---|---|
| `feeding` | 76 | 0 | 0 | 2 | 0 |
| `drinking` | 2 | 1 | 0 | 3 | 0 |
| `ruminating` | 0 | 0 | 1 | 17 | 0 |
| `none` | 1 | 0 | 0 | 17 | 0 |

## Every example

All 120 annotated cows, 6 keyframes. Ordered by keyframe, then left to right. `x` marks the horizontal centre of the box as a fraction of frame width, `y` the vertical centre.

| # | clip | t, s | x | y | ground truth | prediction | |
|---|---|---|---|---|---|---|---|
| 1 | 371 | 2 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 2 | 371 | 2 | 0.08 | 0.32 | lying / none | standing / none | posture |
| 3 | 371 | 2 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 4 | 371 | 2 | 0.17 | 0.33 | lying / ruminating | standing / none | posture + activity |
| 5 | 371 | 2 | 0.17 | 0.24 | standing / none | standing / none | ok |
| 6 | 371 | 2 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 7 | 371 | 2 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 8 | 371 | 2 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 9 | 371 | 2 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 10 | 371 | 2 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 11 | 371 | 2 | 0.45 | 0.21 | lying / ruminating | standing / none | posture + activity |
| 12 | 371 | 2 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 13 | 371 | 2 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 14 | 371 | 2 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 15 | 371 | 2 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 16 | 371 | 2 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 17 | 371 | 2 | 0.82 | 0.44 | standing / feeding | standing / none | activity |
| 18 | 371 | 2 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 19 | 371 | 2 | 0.86 | 0.24 | standing / drinking | lying / none | posture + activity |
| 20 | 371 | 2 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |
| 21 | 371 | 3 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 22 | 371 | 3 | 0.08 | 0.32 | lying / none | standing / none | posture |
| 23 | 371 | 3 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 24 | 371 | 3 | 0.17 | 0.33 | lying / ruminating | standing / none | posture + activity |
| 25 | 371 | 3 | 0.17 | 0.24 | standing / none | lying / none | posture |
| 26 | 371 | 3 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 27 | 371 | 3 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 28 | 371 | 3 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 29 | 371 | 3 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 30 | 371 | 3 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 31 | 371 | 3 | 0.45 | 0.21 | lying / ruminating | lying / none | activity |
| 32 | 371 | 3 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 33 | 371 | 3 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 34 | 371 | 3 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 35 | 371 | 3 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 36 | 371 | 3 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 37 | 371 | 3 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 38 | 371 | 3 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 39 | 371 | 3 | 0.86 | 0.24 | standing / drinking | lying / none | posture + activity |
| 40 | 371 | 3 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |
| 41 | 371 | 4 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 42 | 371 | 4 | 0.08 | 0.32 | lying / none | standing / none | posture |
| 43 | 371 | 4 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 44 | 371 | 4 | 0.17 | 0.33 | lying / ruminating | lying / none | activity |
| 45 | 371 | 4 | 0.17 | 0.24 | standing / none | lying / none | posture |
| 46 | 371 | 4 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 47 | 371 | 4 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 48 | 371 | 4 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 49 | 371 | 4 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 50 | 371 | 4 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 51 | 371 | 4 | 0.45 | 0.21 | lying / ruminating | lying / none | activity |
| 52 | 371 | 4 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 53 | 371 | 4 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 54 | 371 | 4 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 55 | 371 | 4 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 56 | 371 | 4 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 57 | 371 | 4 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 58 | 371 | 4 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 59 | 371 | 4 | 0.86 | 0.24 | standing / drinking | lying / none | posture + activity |
| 60 | 371 | 4 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |
| 61 | 371 | 5 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 62 | 371 | 5 | 0.08 | 0.32 | lying / none | standing / none | posture |
| 63 | 371 | 5 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 64 | 371 | 5 | 0.17 | 0.33 | lying / ruminating | standing / none | posture + activity |
| 65 | 371 | 5 | 0.17 | 0.24 | standing / none | lying / none | posture |
| 66 | 371 | 5 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 67 | 371 | 5 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 68 | 371 | 5 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 69 | 371 | 5 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 70 | 371 | 5 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 71 | 371 | 5 | 0.45 | 0.21 | lying / ruminating | standing / none | posture + activity |
| 72 | 371 | 5 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 73 | 371 | 5 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 74 | 371 | 5 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 75 | 371 | 5 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 76 | 371 | 5 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 77 | 371 | 5 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 78 | 371 | 5 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 79 | 371 | 5 | 0.86 | 0.24 | standing / drinking | standing / drinking | ok |
| 80 | 371 | 5 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |
| 81 | 371 | 6 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 82 | 371 | 6 | 0.08 | 0.32 | lying / none | standing / feeding | posture + activity |
| 83 | 371 | 6 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 84 | 371 | 6 | 0.17 | 0.33 | lying / ruminating | standing / none | posture + activity |
| 85 | 371 | 6 | 0.17 | 0.24 | standing / none | standing / none | ok |
| 86 | 371 | 6 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 87 | 371 | 6 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 88 | 371 | 6 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 89 | 371 | 6 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 90 | 371 | 6 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 91 | 371 | 6 | 0.45 | 0.21 | lying / ruminating | standing / none | posture + activity |
| 92 | 371 | 6 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 93 | 371 | 6 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 94 | 371 | 6 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 95 | 371 | 6 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 96 | 371 | 6 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 97 | 371 | 6 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 98 | 371 | 6 | 0.86 | 0.46 | standing / feeding | standing / none | activity |
| 99 | 371 | 6 | 0.86 | 0.24 | standing / drinking | standing / feeding | activity |
| 100 | 371 | 6 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |
| 101 | 371 | 7 | 0.04 | 0.49 | standing / feeding | standing / feeding | ok |
| 102 | 371 | 7 | 0.08 | 0.32 | lying / none | standing / none | posture |
| 103 | 371 | 7 | 0.08 | 0.49 | standing / feeding | standing / feeding | ok |
| 104 | 371 | 7 | 0.17 | 0.33 | lying / ruminating | standing / none | posture + activity |
| 105 | 371 | 7 | 0.17 | 0.24 | standing / none | lying / none | posture |
| 106 | 371 | 7 | 0.18 | 0.50 | standing / feeding | standing / feeding | ok |
| 107 | 371 | 7 | 0.24 | 0.51 | standing / feeding | standing / feeding | ok |
| 108 | 371 | 7 | 0.26 | 0.48 | standing / feeding | standing / feeding | ok |
| 109 | 371 | 7 | 0.37 | 0.48 | standing / feeding | standing / feeding | ok |
| 110 | 371 | 7 | 0.38 | 0.28 | lying / ruminating | lying / none | activity |
| 111 | 371 | 7 | 0.45 | 0.21 | lying / ruminating | lying / ruminating | ok |
| 112 | 371 | 7 | 0.48 | 0.47 | standing / feeding | standing / feeding | ok |
| 113 | 371 | 7 | 0.57 | 0.42 | standing / feeding | standing / feeding | ok |
| 114 | 371 | 7 | 0.66 | 0.45 | standing / feeding | standing / feeding | ok |
| 115 | 371 | 7 | 0.70 | 0.23 | standing / none | standing / none | ok |
| 116 | 371 | 7 | 0.73 | 0.49 | standing / feeding | standing / feeding | ok |
| 117 | 371 | 7 | 0.82 | 0.44 | standing / feeding | standing / feeding | ok |
| 118 | 371 | 7 | 0.86 | 0.46 | standing / feeding | standing / feeding | ok |
| 119 | 371 | 7 | 0.86 | 0.24 | standing / drinking | standing / feeding | activity |
| 120 | 371 | 7 | 0.93 | 0.48 | standing / feeding | standing / feeding | ok |

