# 1280px vs 1920px native

300 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **1280px** — image mode `marked`, max width 1280px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T13:55:28+03:00
- **1920px native** — image mode `marked`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:00:36+03:00

## Paired result

| Axis | 1280px error | 1920px native error | Fixed by 1920px native | Broken by 1920px native | p (McNemar) |
|---|---|---|---|---|---|
| exact | 28.0% | 24.3% | 21 | 10 | 0.071 |
| posture | 9.3% | 6.7% | 15 | 7 | 0.134 |
| activity | 22.7% | 21.0% | 15 | 10 | 0.424 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 1280px | Recall 1920px native | Predicted 1280px | Predicted 1920px native |
|---|---|---|---|---|---|
| `posture:standing` | 181 | 95.6% | 97.8% | 193 | 193 |
| `posture:lying` | 119 | 83.2% | 86.6% | 107 | 107 |
| `activity:feeding` | 107 | 93.5% | 94.4% | 118 | 113 |
| `activity:drinking` | 5 | 40.0% | 60.0% | 4 | 4 |
| `activity:ruminating` | 42 | 0.0% | 0.0% | 0 | 0 |
| `activity:none` | 146 | 89.0% | 91.1% | 178 | 183 |

