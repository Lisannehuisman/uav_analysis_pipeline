# Exact Viewpoint-Subset Shapley Analysis

## Setup

- Input scene records: `C:\Users\lisan\OneDrive\Documents\New project\m4_two_drone_operational_analysis\outputs\scene_view_records.csv`
- Subset name: `full_grid_perimeter_8`
- Selection mode: `full_grid_perimeter`
- Selection rule: 8 viewpoints on the perimeter of the 3x3 elevation/radius grid, one distinct azimuth each
- Player count: `8`
- Coalition value uses `Noisy-OR + best IoU` exactly, matching the ring Shapley analysis.
- Shapley is computed exactly over all `2^8 = 256` coalitions.
- Missing selected viewpoints inside a scene are treated as unavailable rather than imputed.
- The subset is evaluated on scenes that contain at least one of the selected viewpoints.

## Selected Players

| Player | Viewpoint | Ring | Azimuth |
| ---: | --- | --- | ---: |
| 1 | `ellow-radnear-az000` | `ellow-radnear` | `az000` |
| 2 | `ellow-radmid-az045` | `ellow-radmid` | `az045` |
| 3 | `ellow-radfar-az090` | `ellow-radfar` | `az090` |
| 4 | `elmid-radfar-az135` | `elmid-radfar` | `az135` |
| 5 | `elhigh-radfar-az180` | `elhigh-radfar` | `az180` |
| 6 | `elhigh-radmid-az225` | `elhigh-radmid` | `az225` |
| 7 | `elhigh-radnear-az270` | `elhigh-radnear` | `az270` |
| 8 | `elmid-radnear-az315` | `elmid-radnear` | `az315` |

## Coverage

- Scenes with any selected viewpoint: `150`
- Complete scenes with all selected viewpoints observed: `0`
- Maximum selected views observed in one scene: `4`
- Grand coalition value: `0.912188`

## Top Players

- Best singleton: `elhigh-radmid-az225` with `v({u}) = 0.214604`.
- Top Shapley player: `elhigh-radmid-az225` with `phi = 0.142802`.
- Runner-up: `elmid-radnear-az315` with gap `0.014031`.

## Conditional Next-View Example Inside The Same Subset Game

| Current coalition C | Best addition u | Delta(u | C) | New value |
| --- | --- | ---: | ---: |
| `elhigh-radmid-az225` | `elmid-radnear-az315` | 0.182022 | 0.396626 |
| `elhigh-radmid-az225 + elmid-radnear-az315` | `ellow-radnear-az000` | 0.140199 | 0.536824 |

## Player Ranking

| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 6 | `elhigh-radmid-az225` | 0.142802 | 0.214604 | 36 | 36 |
| 2 | 8 | `elmid-radnear-az315` | 0.128771 | 0.206423 | 33 | 33 |
| 3 | 1 | `ellow-radnear-az000` | 0.123612 | 0.193589 | 35 | 32 |
| 4 | 2 | `ellow-radmid-az045` | 0.121150 | 0.206312 | 37 | 35 |
| 5 | 7 | `elhigh-radnear-az270` | 0.115402 | 0.169668 | 30 | 30 |
| 6 | 5 | `elhigh-radfar-az180` | 0.098968 | 0.148782 | 25 | 25 |
| 7 | 4 | `elmid-radfar-az135` | 0.093555 | 0.174356 | 31 | 31 |
| 8 | 3 | `ellow-radfar-az090` | 0.087928 | 0.152423 | 28 | 26 |