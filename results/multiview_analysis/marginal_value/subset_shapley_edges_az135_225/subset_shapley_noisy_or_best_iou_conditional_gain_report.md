# Exact Conditional Next-View Gains On The Subset Fusion Game

- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `elmid-radnear-az135` | 0.273823 | `elmid-radfar-az225` | 0.179355 | 0.453178 |
| 2 | `elmid-radnear-az135 + elmid-radfar-az225` | 0.453178 | `elhigh-radmid-az225` | 0.135523 | 0.588701 |
| 3 | `elmid-radnear-az135 + elmid-radfar-az225 + elhigh-radmid-az225` | 0.588701 | `ellow-radmid-az225` | 0.094402 | 0.683103 |
| 4 | `ellow-radmid-az225 + elmid-radnear-az135 + elmid-radfar-az225 + elhigh-radmid-az225` | 0.683103 | `elmid-radnear-az225` | 0.080772 | 0.763875 |
| 5 | `ellow-radmid-az225 + elmid-radnear-az135 + elmid-radnear-az225 + elmid-radfar-az225 + elhigh-radmid-az225` | 0.763875 | `ellow-radmid-az135` | 0.064452 | 0.828327 |
| 6 | `ellow-radmid-az135 + ellow-radmid-az225 + elmid-radnear-az135 + elmid-radnear-az225 + elmid-radfar-az225 + elhigh-radmid-az225` | 0.828327 | `elhigh-radmid-az135` | 0.049713 | 0.878040 |
| 7 | `ellow-radmid-az135 + ellow-radmid-az225 + elmid-radnear-az135 + elmid-radnear-az225 + elmid-radfar-az225 + elhigh-radmid-az135 + elhigh-radmid-az225` | 0.878040 | `elmid-radfar-az135` | 0.041171 | 0.919212 |