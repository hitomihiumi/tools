# Muse Glimmer on CBVD-5: cow behaviour recognition

| | |
|---|---|
| Model | `RedHatAI/Muse-Glimmer-30B-FP8-block` |
| Served as | `muse-glimmer` |
| Quantization | FP8 block-wise (W8A8, compressed-tensors); KV cache: auto (bf16) |
| Test date | 2026-09-25T12:53:42+03:00 |
| Dataset | [CBVD-5 (Cow Behavior Video Dataset)](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset) — [paper](https://www.nature.com/articles/s41598-024-65953-x) |
| Split / source | `annotations\ava_val_v2.1.csv` |
| Clips | 371 |
| Examples (annotated boxes) | **120** |
| **Exact-match error rate** | **27.5%** |
| Serving | vLLM 0.30.0 |
| Sampling | temperature=0.0, seed=0, max_tokens=4096, structured output (json_schema) |
| Image mode | marked, max width 1280px |
| Temporal context | annotated keyframe only (single still) |
| Prompt revision | `187aec380cb8` |

## What is being measured

One example = one annotated cow on one keyframe. The model sees the full keyframe with that cow outlined and answers two independent fields, matching how CBVD-5 annotates: a **posture** (standing / lying) and an **activity** (feeding / drinking / ruminating / none). Exact match means both are right.

**Read `ruminating` with that in mind.** Rumination is chewing cud — a movement. On one still frame a ruminating cow and an idle one are the same picture, so errors on that class measure the frame, not the model. Re-run with `--frames 5` to give it the motion.

## Headline numbers

| Axis | Error rate | 95% CI | Majority baseline | Correct |
|---|---|---|---|---|
| Exact match (both fields) | **27.5%** | 20.3% – 36.1% | — | 87/120 |
| Posture only | **17.5%** | 11.7% – 25.3% | 20.0% (always `standing`) | 99/120 |
| Activity only | **23.3%** | 16.7% – 31.7% | 35.0% (always `feeding`) | 92/120 |

## Per class

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `posture:standing` | 96 | 0.83 | 0.98 | 0.90 |
| `posture:lying` | 24 | 0.71 | 0.21 | 0.32 |
| `activity:feeding` | 78 | 0.92 | 0.97 | 0.94 |
| `activity:drinking` | 6 | 1.00 | 0.17 | 0.29 |
| `activity:ruminating` | 18 | 0.00 | 0.00 | 0.00 |
| `activity:none` | 18 | 0.42 | 0.83 | 0.56 |

## Confusion matrix — posture

| ground truth / predicted | `standing` | `lying` | `<failed>` |
|---|---|---|---|
| `standing` | 94 | 2 | 0 |
| `lying` | 19 | 5 | 0 |

## Confusion matrix — activity

| ground truth / predicted | `feeding` | `drinking` | `ruminating` | `none` | `<failed>` |
|---|---|---|---|---|---|
| `feeding` | 76 | 0 | 0 | 2 | 0 |
| `drinking` | 1 | 1 | 0 | 4 | 0 |
| `ruminating` | 3 | 0 | 0 | 15 | 0 |
| `none` | 3 | 0 | 0 | 15 | 0 |

