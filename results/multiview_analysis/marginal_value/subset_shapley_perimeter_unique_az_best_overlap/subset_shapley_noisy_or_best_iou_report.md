# Exact Viewpoint-Subset Shapley Analysis

## Setup

- Input scene records: `C:\Users\lisan\OneDrive\Documents\New project\m4_two_drone_operational_analysis\outputs\scene_view_records.csv`
- Subset name: `perimeter_unique_az_best_overlap`
- Selection mode: `manual`
- Selection rule: manual viewpoint subset
- Player count: `8`
- Coalition value uses `Noisy-OR + best IoU` exactly, matching the ring Shapley analysis.
- Shapley is computed exactly over all `2^8 = 256` coalitions.
- Missing selected viewpoints inside a scene are treated as unavailable rather than imputed.
- The subset is evaluated on scenes that contain at least one of the selected viewpoints.

## Selected Players

| Player | Viewpoint | Ring | Azimuth |
| ---: | --- | --- | ---: |
| 1 | `ellow-radnear-az225` | `ellow-radnear` | `az225` |
| 2 | `ellow-radmid-az045` | `ellow-radmid` | `az045` |
| 3 | `ellow-radfar-az315` | `ellow-radfar` | `az315` |
| 4 | `elmid-radnear-az090` | `elmid-radnear` | `az090` |
| 5 | `elmid-radfar-az180` | `elmid-radfar` | `az180` |
| 6 | `elhigh-radnear-az270` | `elhigh-radnear` | `az270` |
| 7 | `elhigh-radmid-az135` | `elhigh-radmid` | `az135` |
| 8 | `elhigh-radfar-az000` | `elhigh-radfar` | `az000` |

## Coverage

- Scenes with any selected viewpoint: `143`
- Complete scenes with all selected viewpoints observed: `1`
- Maximum selected views observed in one scene: `8`
- Grand coalition value: `0.914277`

## Top Players

- Best singleton: `ellow-radnear-az225` with `v({u}) = 0.237215`.
- Top Shapley player: `ellow-radnear-az225` with `phi = 0.154156`.
- Runner-up: `ellow-radmid-az045` with gap `0.027475`.

## Conditional Next-View Example Inside The Same Subset Game

| Current coalition C | Best addition u | Delta(u | C) | New value |
| --- | --- | ---: | ---: |
| `ellow-radnear-az225` | `ellow-radmid-az045` | 0.185549 | 0.422764 |
| `ellow-radnear-az225 + ellow-radmid-az045` | `elmid-radfar-az180` | 0.126372 | 0.549136 |

## Player Ranking

| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | `ellow-radnear-az225` | 0.154156 | 0.237215 | 36 | 36 |
| 2 | 2 | `ellow-radmid-az045` | 0.126681 | 0.216412 | 37 | 35 |
| 3 | 8 | `elhigh-radfar-az000` | 0.117324 | 0.176833 | 29 | 29 |
| 4 | 4 | `elmid-radnear-az090` | 0.116705 | 0.192115 | 31 | 31 |
| 5 | 6 | `elhigh-radnear-az270` | 0.106509 | 0.177973 | 30 | 30 |
| 6 | 7 | `elhigh-radmid-az135` | 0.105994 | 0.180321 | 30 | 30 |
| 7 | 5 | `elmid-radfar-az180` | 0.103651 | 0.159825 | 26 | 26 |
| 8 | 3 | `ellow-radfar-az315` | 0.083257 | 0.145976 | 28 | 25 |