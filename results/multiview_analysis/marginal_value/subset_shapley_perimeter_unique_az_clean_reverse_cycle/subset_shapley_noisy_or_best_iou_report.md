# Exact Viewpoint-Subset Shapley Analysis

## Setup

- Input scene records: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`
- Subset name: `perimeter_unique_az_clean_reverse_cycle`
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
| 2 | `ellow-radmid-az180` | `ellow-radmid` | `az180` |
| 3 | `ellow-radfar-az135` | `ellow-radfar` | `az135` |
| 4 | `elmid-radnear-az270` | `elmid-radnear` | `az270` |
| 5 | `elmid-radfar-az090` | `elmid-radfar` | `az090` |
| 6 | `elhigh-radnear-az315` | `elhigh-radnear` | `az315` |
| 7 | `elhigh-radmid-az000` | `elhigh-radmid` | `az000` |
| 8 | `elhigh-radfar-az045` | `elhigh-radfar` | `az045` |

## Coverage

- Scenes with any selected viewpoint: `152`
- Complete scenes with all selected viewpoints observed: `0`
- Maximum selected views observed in one scene: `5`
- Grand coalition value: `0.915206`

## Top Players

- Best singleton: `elmid-radnear-az270` with `v({u}) = 0.258635`.
- Top Shapley player: `elmid-radnear-az270` with `phi = 0.144919`.
- Runner-up: `ellow-radnear-az225` with gap `0.002381`.

## Conditional Next-View Example Inside The Same Subset Game

| Current coalition C | Best addition u | Delta(u | C) | New value |
| --- | --- | ---: | ---: |
| `elmid-radnear-az270` | `ellow-radnear-az225` | 0.180718 | 0.439353 |
| `ellow-radnear-az225 + elmid-radnear-az270` | `elhigh-radmid-az000` | 0.149314 | 0.588667 |

## Player Ranking

| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 4 | `elmid-radnear-az270` | 0.144919 | 0.258635 | 43 | 43 |
| 2 | 1 | `ellow-radnear-az225` | 0.142538 | 0.223169 | 36 | 36 |
| 3 | 7 | `elhigh-radmid-az000` | 0.141700 | 0.249769 | 43 | 43 |
| 4 | 8 | `elhigh-radfar-az045` | 0.112403 | 0.226941 | 39 | 39 |
| 5 | 3 | `ellow-radfar-az135` | 0.108176 | 0.183935 | 36 | 34 |
| 6 | 6 | `elhigh-radnear-az315` | 0.104343 | 0.167150 | 29 | 29 |
| 7 | 5 | `elmid-radfar-az090` | 0.102134 | 0.164535 | 29 | 29 |
| 8 | 2 | `ellow-radmid-az180` | 0.058992 | 0.119038 | 21 | 20 |
