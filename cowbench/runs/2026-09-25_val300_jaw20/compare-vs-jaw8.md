# 8 crops, 4.67 fps vs 20 crops, 12.5 fps

300 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **8 crops, 4.67 fps** — image mode `crop`, max width 1920px, 8 frames over 1.5s, prompt `5795abb8e505`, run 2026-09-25T14:23:57+03:00
- **20 crops, 12.5 fps** — image mode `crop`, max width 1920px, 20 frames over 1.52s, prompt `c97a452152e9`, run 2026-09-25T16:13:04+03:00

## Paired result

| Axis | 8 crops, 4.67 fps error | 20 crops, 12.5 fps error | Fixed by 20 crops, 12.5 fps | Broken by 20 crops, 12.5 fps | p (McNemar) |
|---|---|---|---|---|---|
| exact | 32.0% | 33.7% | 11 | 16 | 0.442 |
| posture | 5.7% | 5.0% | 7 | 5 | 0.774 |
| activity | 29.7% | 31.7% | 8 | 14 | 0.286 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 8 crops, 4.67 fps | Recall 20 crops, 12.5 fps | Predicted 8 crops, 4.67 fps | Predicted 20 crops, 12.5 fps |
|---|---|---|---|---|---|
| `posture:standing` | 181 | 96.1% | 96.7% | 184 | 184 |
| `posture:lying` | 119 | 91.6% | 92.4% | 116 | 116 |
| `activity:feeding` | 107 | 93.5% | 92.5% | 143 | 146 |
| `activity:drinking` | 5 | 40.0% | 40.0% | 3 | 4 |
| `activity:ruminating` | 42 | 2.4% | 4.8% | 2 | 5 |
| `activity:none` | 146 | 74.0% | 69.9% | 152 | 145 |

