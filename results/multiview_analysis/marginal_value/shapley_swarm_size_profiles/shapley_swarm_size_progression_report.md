# Size-Conditioned Shapley Progression

This analysis keeps the exact same fusion game as the controlled ring Shapley analysis.

- Coalition value: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Players: the 8 azimuths inside one fixed M4 ring.
- Difference from the standard report: Shapley is decomposed by coalition size.

For one viewpoint `u`, the exact Shapley value can be rewritten as:

`phi(u) = (1 / n) * sum_k average_{|S| = k}[v(S union {u}) - v(S)]`

So each row below answers a swarm-size question directly:

- `k = 1`: how much does a view add as the first drone?
- `k = 2`: how much does it add when one drone is already present?
- `k = 3`: how much does it add when a pair is already present?
- and so on.

## Overall progression across the 9 controlled rings

| Added drone number | Mean marginal gain | Mean exact Shapley component from this size | Mean cumulative share of final Shapley | Mean best azimuth gain | Mean top-vs-runner-up gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1783 | 0.0223 | 0.1981 | 0.2289 | 0.0189 |
| 2 | 0.1546 | 0.0193 | 0.3693 | 0.2032 | 0.0200 |
| 3 | 0.1339 | 0.0167 | 0.5170 | 0.1810 | 0.0214 |
| 4 | 0.1158 | 0.0145 | 0.6444 | 0.1615 | 0.0215 |
| 5 | 0.1002 | 0.0125 | 0.7541 | 0.1444 | 0.0216 |
| 6 | 0.0868 | 0.0108 | 0.8485 | 0.1298 | 0.0221 |
| 7 | 0.0752 | 0.0094 | 0.9299 | 0.1179 | 0.0241 |
| 8 | 0.0651 | 0.0081 | 1.0000 | 0.1080 | 0.0257 |

## Focus ring: elmid-radnear

This is the strongest ring by full 8-drone coalition value: `0.9434`.

| Added drone number | Mean marginal gain across azimuths | Best azimuth at this size | Top marginal gain | Runner-up | Gap |
| ---: | ---: | --- | ---: | --- | ---: |
| 1 | 0.1966 | `az135` | 0.2629 | `az270` | 0.0026 |
| 2 | 0.1679 | `az270` | 0.2367 | `az135` | 0.0133 |
| 3 | 0.1425 | `az270` | 0.2153 | `az135` | 0.0270 |
| 4 | 0.1203 | `az270` | 0.1959 | `az135` | 0.0386 |
| 5 | 0.1011 | `az270` | 0.1785 | `az135` | 0.0482 |
| 6 | 0.0846 | `az270` | 0.1629 | `az135` | 0.0558 |
| 7 | 0.0708 | `az270` | 0.1488 | `az135` | 0.0614 |
| 8 | 0.0595 | `az270` | 0.1363 | `az180` | 0.0650 |

## Thesis reading

- The average gain of the **2nd drone** is `0.1546`.
- The average gain of the **3rd drone** is `0.1339`.
- The average gain of the **4th drone** is `0.1158`.
- Gains remain positive but shrink with swarm size, so the main question becomes `how many angles are worth adding?` before `which exact angle is best?`.
- Within any fixed swarm size, the per-ring top azimuth still tells you which angle is the best teammate at that stage.

## Practical takeaway

- Use `aggregate_shapley_swarm_size_summary.csv` when the headline question is the number of views.
- Use `ring_shapley_swarm_size_summary.csv` when the question is which angle to add at swarm size `k`.
- Use the standard ring Shapley report when you want one overall teammate ranking aggregated across all coalition sizes.
