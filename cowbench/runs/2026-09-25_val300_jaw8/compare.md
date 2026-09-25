# 1920 marked vs 8 crops / 1.5s

300 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **1920 marked** — image mode `marked`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:00:36+03:00
- **8 crops / 1.5s** — image mode `crop`, max width 1920px, 8 frames over 1.5s, prompt `5795abb8e505`, run 2026-09-25T14:23:57+03:00

## Paired result

| Axis | 1920 marked error | 8 crops / 1.5s error | Fixed by 8 crops / 1.5s | Broken by 8 crops / 1.5s | p (McNemar) |
|---|---|---|---|---|---|
| exact | 24.3% | 32.0% | 17 | 40 | 0.003 |
| posture | 6.7% | 5.7% | 15 | 12 | 0.701 |
| activity | 21.0% | 29.7% | 10 | 36 | 0.000 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 1920 marked | Recall 8 crops / 1.5s | Predicted 1920 marked | Predicted 8 crops / 1.5s |
|---|---|---|---|---|---|
| `posture:standing` | 181 | 97.8% | 96.1% | 193 | 184 |
| `posture:lying` | 119 | 86.6% | 91.6% | 107 | 116 |
| `activity:feeding` | 107 | 94.4% | 93.5% | 113 | 143 |
| `activity:drinking` | 5 | 60.0% | 40.0% | 4 | 3 |
| `activity:ruminating` | 42 | 0.0% | 2.4% | 0 | 2 |
| `activity:none` | 146 | 91.1% | 74.0% | 183 | 152 |

