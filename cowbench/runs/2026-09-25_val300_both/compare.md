# 1920 marked vs 1920 marked+crop

300 examples answered by both runs. Paired comparison: the same cow, the same annotation, two settings.

- **1920 marked** — image mode `marked`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:00:36+03:00
- **1920 marked+crop** — image mode `both`, max width 1920px, single keyframe, prompt `187aec380cb8`, run 2026-09-25T14:16:42+03:00

## Paired result

| Axis | 1920 marked error | 1920 marked+crop error | Fixed by 1920 marked+crop | Broken by 1920 marked+crop | p (McNemar) |
|---|---|---|---|---|---|
| exact | 24.3% | 24.7% | 15 | 16 | 1.000 |
| posture | 6.7% | 5.3% | 15 | 11 | 0.557 |
| activity | 21.0% | 20.7% | 9 | 8 | 1.000 |

`Fixed` and `broken` are the only examples that carry information about the difference; everything both runs agree on is excluded from the test. A p-value above 0.05 means the two settings are not distinguishable on this many examples — which is a finding, not a failure.

## Per class

| Class | Support | Recall 1920 marked | Recall 1920 marked+crop | Predicted 1920 marked | Predicted 1920 marked+crop |
|---|---|---|---|---|---|
| `posture:standing` | 181 | 97.8% | 91.7% | 193 | 167 |
| `posture:lying` | 119 | 86.6% | 99.2% | 107 | 133 |
| `activity:feeding` | 107 | 94.4% | 90.7% | 113 | 106 |
| `activity:drinking` | 5 | 60.0% | 100.0% | 4 | 6 |
| `activity:ruminating` | 42 | 0.0% | 0.0% | 0 | 0 |
| `activity:none` | 146 | 91.1% | 93.2% | 183 | 188 |

