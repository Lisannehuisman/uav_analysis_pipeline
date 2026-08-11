# Controlled 8-Azimuth Ring Shapley Analysis

## Setup

- Input scene records: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`
- Players are the 8 azimuth viewpoints within each fixed M4 ring (`elevation x radius`).
- Coalition value uses `Noisy-OR + best IoU` exactly, not the older independent `max` utility.
- Singleton coalition value means `v({u})`: the exact one-view coalition value of one azimuth inside one fixed ring.
- Shapley is computed exactly over all `2^8 = 256` coalitions.
- The `top Shapley azimuth` is therefore only the winner within that one fixed ring, not one global best angle across all rings.
- For a concrete existing coalition `C`, the next-view question uses the same game directly: `Delta(u | C) = v(C union {u}) - v(C)`.
- Because the current cache contains no scene with a full observed 8-view ring, missing ring viewpoints are treated as unavailable inside the coalition rather than pretending the ring is complete.
- Each ring is evaluated on the scenes that contain at least one observation from that ring.

## Ring Coverage

| Ring | Ring scenes | Complete 8-view scenes | Max views seen in one scene | Grand value | Best singleton | Top Shapley |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `ellow-radnear` | 154 | 0 | 4 | 0.920198 | `az180` (0.221197) | `az225` (0.146639) |
| `ellow-radmid` | 149 | 0 | 4 | 0.897452 | `az045` (0.207697) | `az315` (0.154264) |
| `ellow-radfar` | 154 | 0 | 4 | 0.860633 | `az000` (0.203461) | `az000` (0.136105) |
| `elmid-radnear` | 151 | 0 | 4 | 0.943444 | `az135` (0.262943) | `az270` (0.191846) |
| `elmid-radmid` | 148 | 0 | 4 | 0.919248 | `az000` (0.251852) | `az000` (0.171775) |
| `elmid-radfar` | 145 | 0 | 4 | 0.911379 | `az225` (0.222927) | `az225` (0.137633) |
| `elhigh-radnear` | 150 | 0 | 4 | 0.917643 | `az225` (0.216642) | `az225` (0.147523) |
| `elhigh-radmid` | 159 | 0 | 4 | 0.906895 | `az000` (0.238772) | `az000` (0.169891) |
| `elhigh-radfar` | 147 | 0 | 4 | 0.912211 | `az045` (0.234660) | `az045` (0.171733) |

## Top Azimuth Per Ring

- `ellow-radnear`: best singleton coalition is `az180` with `v({u}) = 0.221197`, while the top Shapley azimuth is `az225` (`ellow-radnear-az225`) with `phi = 0.146639`; runner-up `az180` with gap `0.001979`.
- `ellow-radmid`: best singleton coalition is `az045` with `v({u}) = 0.207697`, while the top Shapley azimuth is `az315` (`ellow-radmid-az315`) with `phi = 0.154264`; runner-up `az045` with gap `0.020994`.
- `ellow-radfar`: best singleton coalition is `az000` with `v({u}) = 0.203461`, while the top Shapley azimuth is `az000` (`ellow-radfar-az000`) with `phi = 0.136105`; runner-up `az135` with gap `0.012155`.
- `elmid-radnear`: best singleton coalition is `az135` with `v({u}) = 0.262943`, while the top Shapley azimuth is `az270` (`elmid-radnear-az270`) with `phi = 0.191846`; runner-up `az135` with gap `0.038354`.
- `elmid-radmid`: best singleton coalition is `az000` with `v({u}) = 0.251852`, while the top Shapley azimuth is `az000` (`elmid-radmid-az000`) with `phi = 0.171775`; runner-up `az270` with gap `0.020520`.
- `elmid-radfar`: best singleton coalition is `az225` with `v({u}) = 0.222927`, while the top Shapley azimuth is `az225` (`elmid-radfar-az225`) with `phi = 0.137633`; runner-up `az090` with gap `0.012418`.
- `elhigh-radnear`: best singleton coalition is `az225` with `v({u}) = 0.216642`, while the top Shapley azimuth is `az225` (`elhigh-radnear-az225`) with `phi = 0.147523`; runner-up `az135` with gap `0.009706`.
- `elhigh-radmid`: best singleton coalition is `az000` with `v({u}) = 0.238772`, while the top Shapley azimuth is `az000` (`elhigh-radmid-az000`) with `phi = 0.169891`; runner-up `az225` with gap `0.033839`.
- `elhigh-radfar`: best singleton coalition is `az045` with `v({u}) = 0.234660`, while the top Shapley azimuth is `az045` (`elhigh-radfar-az045`) with `phi = 0.171733`; runner-up `az315` with gap `0.042608`.

## Conditional Next-View Example Within The Same Ring Game

| Ring | Best singleton C | Best 2nd azimuth | Delta(u | C) | New value | Best pair C | Best 3rd azimuth | Delta(u | C) | New value |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| `ellow-radnear` | `ellow-radnear-az180` | `az225` | 0.178463 | 0.399660 | `az045 + az225` | `az180` | 0.151658 | 0.558344 |
| `ellow-radmid` | `ellow-radmid-az045` | `az315` | 0.194813 | 0.402510 | `az045 + az315` | `az090` | 0.126360 | 0.528870 |
| `ellow-radfar` | `ellow-radfar-az000` | `az135` | 0.156031 | 0.359492 | `az000 + az135` | `az270` | 0.119093 | 0.478585 |
| `elmid-radnear` | `elmid-radnear-az135` | `az270` | 0.216005 | 0.478948 | `az135 + az270` | `az180` | 0.137609 | 0.616557 |
| `elmid-radmid` | `elmid-radmid-az000` | `az270` | 0.205955 | 0.457806 | `az000 + az270` | `az045` | 0.164790 | 0.622597 |
| `elmid-radfar` | `elmid-radfar-az225` | `az000` | 0.150758 | 0.373685 | `az000 + az225` | `az090` | 0.140368 | 0.514053 |
| `elhigh-radnear` | `elhigh-radnear-az225` | `az000` | 0.173906 | 0.390548 | `az000 + az225` | `az135` | 0.146545 | 0.537092 |
| `elhigh-radmid` | `elhigh-radmid-az000` | `az225` | 0.182465 | 0.421237 | `az000 + az225` | `az090` | 0.130050 | 0.551288 |
| `elhigh-radfar` | `elhigh-radfar-az045` | `az315` | 0.160971 | 0.395631 | `az045 + az315` | `az090` | 0.150695 | 0.546326 |

## Per-Ring Rankings

### ellow-radnear

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `225` | 0.146639 | 0.220271 | 36 | 36 |
| 2 | `180` | 0.144660 | 0.221197 | 39 | 37 |
| 3 | `045` | 0.119516 | 0.186415 | 32 | 31 |
| 4 | `135` | 0.113340 | 0.192238 | 32 | 32 |
| 5 | `000` | 0.111512 | 0.188561 | 35 | 32 |
| 6 | `090` | 0.100729 | 0.188976 | 32 | 32 |
| 7 | `270` | 0.098554 | 0.173505 | 30 | 29 |
| 8 | `315` | 0.085247 | 0.161882 | 29 | 28 |

### ellow-radmid

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `315` | 0.154264 | 0.205188 | 33 | 33 |
| 2 | `045` | 0.133269 | 0.207697 | 37 | 35 |
| 3 | `000` | 0.117882 | 0.174120 | 31 | 29 |
| 4 | `090` | 0.115113 | 0.180679 | 32 | 30 |
| 5 | `225` | 0.103237 | 0.160674 | 29 | 27 |
| 6 | `135` | 0.097484 | 0.165632 | 30 | 28 |
| 7 | `270` | 0.094765 | 0.153637 | 26 | 25 |
| 8 | `180` | 0.081437 | 0.121435 | 21 | 20 |

### ellow-radfar

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `000` | 0.136105 | 0.203461 | 39 | 36 |
| 2 | `135` | 0.123950 | 0.181546 | 36 | 34 |
| 3 | `225` | 0.116745 | 0.151660 | 30 | 28 |
| 4 | `270` | 0.115581 | 0.151540 | 32 | 27 |
| 5 | `180` | 0.101107 | 0.139797 | 25 | 24 |
| 6 | `090` | 0.095029 | 0.148464 | 28 | 26 |
| 7 | `315` | 0.092839 | 0.135549 | 28 | 25 |
| 8 | `045` | 0.079278 | 0.115562 | 22 | 21 |

### elmid-radnear

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `270` | 0.191846 | 0.260348 | 43 | 43 |
| 2 | `135` | 0.153491 | 0.262943 | 43 | 43 |
| 3 | `180` | 0.118708 | 0.190977 | 32 | 32 |
| 4 | `315` | 0.112425 | 0.205056 | 33 | 33 |
| 5 | `090` | 0.112021 | 0.181936 | 31 | 31 |
| 6 | `045` | 0.104091 | 0.197108 | 32 | 32 |
| 7 | `225` | 0.079358 | 0.145214 | 25 | 24 |
| 8 | `000` | 0.071505 | 0.129485 | 21 | 21 |

### elmid-radmid

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `000` | 0.171775 | 0.251852 | 42 | 42 |
| 2 | `270` | 0.151254 | 0.244716 | 40 | 40 |
| 3 | `045` | 0.131345 | 0.212595 | 35 | 35 |
| 4 | `090` | 0.115278 | 0.177221 | 29 | 29 |
| 5 | `180` | 0.112661 | 0.183881 | 31 | 31 |
| 6 | `315` | 0.093214 | 0.175938 | 29 | 29 |
| 7 | `135` | 0.073057 | 0.120866 | 20 | 20 |
| 8 | `225` | 0.070664 | 0.125401 | 22 | 21 |

### elmid-radfar

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `225` | 0.137633 | 0.222927 | 36 | 36 |
| 2 | `090` | 0.125215 | 0.172478 | 29 | 29 |
| 3 | `135` | 0.119060 | 0.180368 | 31 | 31 |
| 4 | `045` | 0.114782 | 0.177552 | 30 | 30 |
| 5 | `000` | 0.111921 | 0.169181 | 27 | 27 |
| 6 | `180` | 0.101382 | 0.157620 | 26 | 26 |
| 7 | `270` | 0.101352 | 0.158688 | 26 | 26 |
| 8 | `315` | 0.100035 | 0.157270 | 26 | 26 |

### elhigh-radnear

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `225` | 0.147523 | 0.216642 | 36 | 36 |
| 2 | `135` | 0.137818 | 0.187174 | 32 | 32 |
| 3 | `000` | 0.132071 | 0.210594 | 34 | 34 |
| 4 | `180` | 0.126527 | 0.185542 | 31 | 31 |
| 5 | `315` | 0.110765 | 0.169378 | 29 | 29 |
| 6 | `270` | 0.090033 | 0.169668 | 30 | 30 |
| 7 | `090` | 0.087897 | 0.163249 | 28 | 28 |
| 8 | `045` | 0.085008 | 0.135652 | 23 | 23 |

### elhigh-radmid

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `000` | 0.169891 | 0.238772 | 43 | 43 |
| 2 | `225` | 0.136052 | 0.202456 | 36 | 36 |
| 3 | `090` | 0.116183 | 0.182344 | 34 | 34 |
| 4 | `180` | 0.107146 | 0.165714 | 30 | 30 |
| 5 | `135` | 0.105382 | 0.162175 | 30 | 30 |
| 6 | `045` | 0.098768 | 0.164536 | 30 | 30 |
| 7 | `270` | 0.088414 | 0.142727 | 27 | 27 |
| 8 | `315` | 0.085058 | 0.134748 | 24 | 24 |

### elhigh-radfar

| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `045` | 0.171733 | 0.234660 | 39 | 39 |
| 2 | `315` | 0.129125 | 0.185000 | 30 | 30 |
| 3 | `270` | 0.112678 | 0.182465 | 30 | 30 |
| 4 | `090` | 0.112121 | 0.173670 | 29 | 29 |
| 5 | `180` | 0.097229 | 0.151818 | 25 | 25 |
| 6 | `000` | 0.096668 | 0.172021 | 29 | 29 |
| 7 | `135` | 0.096462 | 0.153069 | 25 | 25 |
| 8 | `225` | 0.096195 | 0.162109 | 27 | 27 |

