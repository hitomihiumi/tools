# 1280 vs 1920

2532 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **1280** — image mode `marked`, max width 1280px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T13:11:28+03:00
- **1920** — image mode `marked`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:49:43+03:00

## Paired result

| Axis | 1280 error | 1920 error | Fixed by 1920 | Broken by 1920 | p (McNemar) |
|---|---|---|---|---|---|
| exact | 30.8% | 29.3% | 171 | 135 | 0.045 |
| posture | 9.8% | 7.4% | 130 | 69 | 0.000 |
| activity | 25.6% | 25.4% | 120 | 114 | 0.744 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 1280 | Recall 1920 | Predicted 1280 | Predicted 1920 |
|---|---|---|---|---|---|
| `posture:standing` | 1512 | 95.7% | 97.6% | 1630 | 1625 |
| `posture:lying` | 1020 | 82.1% | 85.3% | 902 | 907 |
| `activity:feeding` | 791 | 91.5% | 91.5% | 905 | 909 |
| `activity:drinking` | 53 | 28.3% | 35.8% | 28 | 32 |
| `activity:ruminating` | 383 | 0.0% | 0.0% | 0 | 0 |
| `activity:none` | 1305 | 87.7% | 87.8% | 1599 | 1591 |

