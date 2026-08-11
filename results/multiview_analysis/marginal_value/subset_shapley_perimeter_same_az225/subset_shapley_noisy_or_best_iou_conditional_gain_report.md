# Exact Conditional Next-View Gains On The Subset Fusion Game

- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `ellow-radnear-az225` | 0.230760 | `elhigh-radmid-az225` | 0.187403 | 0.418163 |
| 2 | `ellow-radnear-az225 + elhigh-radmid-az225` | 0.418163 | `elmid-radfar-az225` | 0.151227 | 0.569391 |
| 3 | `ellow-radnear-az225 + elmid-radfar-az225 + elhigh-radmid-az225` | 0.569391 | `elhigh-radnear-az225` | 0.103284 | 0.672675 |
| 4 | `ellow-radnear-az225 + elmid-radfar-az225 + elhigh-radnear-az225 + elhigh-radmid-az225` | 0.672675 | `ellow-radmid-az225` | 0.071124 | 0.743798 |
| 5 | `ellow-radnear-az225 + ellow-radmid-az225 + elmid-radfar-az225 + elhigh-radnear-az225 + elhigh-radmid-az225` | 0.743798 | `elhigh-radfar-az225` | 0.063136 | 0.806935 |
| 6 | `ellow-radnear-az225 + ellow-radmid-az225 + elmid-radfar-az225 + elhigh-radnear-az225 + elhigh-radmid-az225 + elhigh-radfar-az225` | 0.806935 | `ellow-radfar-az225` | 0.048736 | 0.855671 |
| 7 | `ellow-radnear-az225 + ellow-radmid-az225 + ellow-radfar-az225 + elmid-radfar-az225 + elhigh-radnear-az225 + elhigh-radmid-az225 + elhigh-radfar-az225` | 0.855671 | `elmid-radnear-az225` | 0.044002 | 0.899673 |