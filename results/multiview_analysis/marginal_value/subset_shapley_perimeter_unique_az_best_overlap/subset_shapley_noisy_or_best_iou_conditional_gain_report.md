# Exact Conditional Next-View Gains On The Subset Fusion Game

- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `ellow-radnear-az225` | 0.237215 | `ellow-radmid-az045` | 0.185549 | 0.422764 |
| 2 | `ellow-radnear-az225 + ellow-radmid-az045` | 0.422764 | `elmid-radfar-az180` | 0.126372 | 0.549136 |
| 3 | `ellow-radnear-az225 + ellow-radmid-az045 + elmid-radfar-az180` | 0.549136 | `elhigh-radnear-az270` | 0.104360 | 0.653496 |
| 4 | `ellow-radnear-az225 + ellow-radmid-az045 + elmid-radfar-az180 + elhigh-radnear-az270` | 0.653496 | `elmid-radnear-az090` | 0.097822 | 0.751319 |
| 5 | `ellow-radnear-az225 + ellow-radmid-az045 + elmid-radnear-az090 + elmid-radfar-az180 + elhigh-radnear-az270` | 0.751319 | `elhigh-radfar-az000` | 0.074645 | 0.825963 |
| 6 | `ellow-radnear-az225 + ellow-radmid-az045 + elmid-radnear-az090 + elmid-radfar-az180 + elhigh-radnear-az270 + elhigh-radfar-az000` | 0.825963 | `elhigh-radmid-az135` | 0.052165 | 0.878128 |
| 7 | `ellow-radnear-az225 + ellow-radmid-az045 + elmid-radnear-az090 + elmid-radfar-az180 + elhigh-radnear-az270 + elhigh-radmid-az135 + elhigh-radfar-az000` | 0.878128 | `ellow-radfar-az315` | 0.036149 | 0.914277 |