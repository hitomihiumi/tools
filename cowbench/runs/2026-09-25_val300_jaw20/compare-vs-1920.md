# 1920 marked, 1 frame vs 20 crops, 12.5 fps

300 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **1920 marked, 1 frame** — image mode `marked`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:00:36+03:00
- **20 crops, 12.5 fps** — image mode `crop`, max width 1920px, 20 frames over 1.52s, prompt `c97a452152e9`, run 2026-09-25T16:13:04+03:00

## Paired result

| Axis | 1920 marked, 1 frame error | 20 crops, 12.5 fps error | Fixed by 20 crops, 12.5 fps | Broken by 20 crops, 12.5 fps | p (McNemar) |
|---|---|---|---|---|---|
| exact | 24.3% | 33.7% | 15 | 43 | 0.000 |
| posture | 6.7% | 5.0% | 14 | 9 | 0.405 |
| activity | 21.0% | 31.7% | 8 | 40 | 0.000 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 1920 marked, 1 frame | Recall 20 crops, 12.5 fps | Predicted 1920 marked, 1 frame | Predicted 20 crops, 12.5 fps |
|---|---|---|---|---|---|
| `posture:standing` | 181 | 97.8% | 96.7% | 193 | 184 |
| `posture:lying` | 119 | 86.6% | 92.4% | 107 | 116 |
| `activity:feeding` | 107 | 94.4% | 92.5% | 113 | 146 |
| `activity:drinking` | 5 | 60.0% | 40.0% | 4 | 4 |
| `activity:ruminating` | 42 | 0.0% | 4.8% | 0 | 5 |
| `activity:none` | 146 | 91.1% | 69.9% | 183 | 145 |

