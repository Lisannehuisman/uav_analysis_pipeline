# Exact Conditional Next-View Gains On The Subset Fusion Game

- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `elmid-radnear-az270` | 0.258635 | `ellow-radnear-az225` | 0.180718 | 0.439353 |
| 2 | `ellow-radnear-az225 + elmid-radnear-az270` | 0.439353 | `elhigh-radmid-az000` | 0.149314 | 0.588667 |
| 3 | `ellow-radnear-az225 + elmid-radnear-az270 + elhigh-radmid-az000` | 0.588667 | `elhigh-radnear-az315` | 0.102687 | 0.691354 |
| 4 | `ellow-radnear-az225 + elmid-radnear-az270 + elhigh-radnear-az315 + elhigh-radmid-az000` | 0.691354 | `ellow-radfar-az135` | 0.086002 | 0.777357 |
| 5 | `ellow-radnear-az225 + ellow-radfar-az135 + elmid-radnear-az270 + elhigh-radnear-az315 + elhigh-radmid-az000` | 0.777357 | `elmid-radfar-az090` | 0.072685 | 0.850042 |
| 6 | `ellow-radnear-az225 + ellow-radfar-az135 + elmid-radnear-az270 + elmid-radfar-az090 + elhigh-radnear-az315 + elhigh-radmid-az000` | 0.850042 | `elhigh-radfar-az045` | 0.044482 | 0.894523 |
| 7 | `ellow-radnear-az225 + ellow-radfar-az135 + elmid-radnear-az270 + elmid-radfar-az090 + elhigh-radnear-az315 + elhigh-radmid-az000 + elhigh-radfar-az045` | 0.894523 | `ellow-radmid-az180` | 0.020683 | 0.915206 |