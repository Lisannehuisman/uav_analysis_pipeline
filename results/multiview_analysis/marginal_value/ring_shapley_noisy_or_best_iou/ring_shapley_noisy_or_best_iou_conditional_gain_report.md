# Exact Conditional Next-View Gains On The Ring Fusion Game

- This report uses the same coalition value as ring Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.
- Shapley answers the average teammate question.
- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.

## Best next addition for the best singleton in each ring

| Ring | Current singleton C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |
| --- | --- | ---: | --- | ---: | ---: |
| `ellow-radnear` | `az180` | 0.221197 | `az225` | 0.178463 | 0.399660 |
| `ellow-radmid` | `az045` | 0.207697 | `az315` | 0.194813 | 0.402510 |
| `ellow-radfar` | `az000` | 0.203461 | `az135` | 0.156031 | 0.359492 |
| `elmid-radnear` | `az135` | 0.262943 | `az270` | 0.216005 | 0.478948 |
| `elmid-radmid` | `az000` | 0.251852 | `az270` | 0.205955 | 0.457806 |
| `elmid-radfar` | `az225` | 0.222927 | `az000` | 0.150758 | 0.373685 |
| `elhigh-radnear` | `az225` | 0.216642 | `az000` | 0.173906 | 0.390548 |
| `elhigh-radmid` | `az000` | 0.238772 | `az225` | 0.182465 | 0.421237 |
| `elhigh-radfar` | `az045` | 0.234660 | `az315` | 0.160971 | 0.395631 |

## Best next addition for the best pair in each ring

| Ring | Current pair C | v(C) | Best third azimuth u | Delta(u | C) | v(C union {u}) |
| --- | --- | ---: | --- | ---: | ---: |
| `ellow-radnear` | `az045 + az225` | 0.406686 | `az180` | 0.151658 | 0.558344 |
| `ellow-radmid` | `az045 + az315` | 0.402510 | `az090` | 0.126360 | 0.528870 |
| `ellow-radfar` | `az000 + az135` | 0.359492 | `az270` | 0.119093 | 0.478585 |
| `elmid-radnear` | `az135 + az270` | 0.478948 | `az180` | 0.137609 | 0.616557 |
| `elmid-radmid` | `az000 + az270` | 0.457806 | `az045` | 0.164790 | 0.622597 |
| `elmid-radfar` | `az000 + az225` | 0.373685 | `az090` | 0.140368 | 0.514053 |
| `elhigh-radnear` | `az000 + az225` | 0.390548 | `az135` | 0.146545 | 0.537092 |
| `elhigh-radmid` | `az000 + az225` | 0.421237 | `az090` | 0.130050 | 0.551288 |
| `elhigh-radfar` | `az045 + az315` | 0.395631 | `az090` | 0.150695 | 0.546326 |