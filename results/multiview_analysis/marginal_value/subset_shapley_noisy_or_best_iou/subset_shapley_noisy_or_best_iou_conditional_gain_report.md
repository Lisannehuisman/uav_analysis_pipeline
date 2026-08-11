# Exact Conditional Next-View Gains On The Subset Fusion Game

- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `elhigh-radmid-az225` | 0.214604 | `elmid-radnear-az315` | 0.182022 | 0.396626 |
| 2 | `elhigh-radmid-az225 + elmid-radnear-az315` | 0.396626 | `ellow-radnear-az000` | 0.140199 | 0.536824 |
| 3 | `ellow-radnear-az000 + elhigh-radmid-az225 + elmid-radnear-az315` | 0.536824 | `ellow-radmid-az045` | 0.117809 | 0.654633 |
| 4 | `ellow-radnear-az000 + ellow-radmid-az045 + elhigh-radmid-az225 + elmid-radnear-az315` | 0.654633 | `elhigh-radnear-az270` | 0.101061 | 0.755693 |
| 5 | `ellow-radnear-az000 + ellow-radmid-az045 + elhigh-radmid-az225 + elhigh-radnear-az270 + elmid-radnear-az315` | 0.755693 | `elhigh-radfar-az180` | 0.062560 | 0.818254 |
| 6 | `ellow-radnear-az000 + ellow-radmid-az045 + elhigh-radfar-az180 + elhigh-radmid-az225 + elhigh-radnear-az270 + elmid-radnear-az315` | 0.818254 | `ellow-radfar-az090` | 0.053409 | 0.871663 |
| 7 | `ellow-radnear-az000 + ellow-radmid-az045 + ellow-radfar-az090 + elhigh-radfar-az180 + elhigh-radmid-az225 + elhigh-radnear-az270 + elmid-radnear-az315` | 0.871663 | `elmid-radfar-az135` | 0.040525 | 0.912188 |