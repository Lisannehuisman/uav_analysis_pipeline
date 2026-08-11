# Exact Viewpoint-Subset Shapley Analysis

## Setup

- Input scene records: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`
- Subset name: `perimeter_same_az225`
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
| 2 | `ellow-radmid-az225` | `ellow-radmid` | `az225` |
| 3 | `ellow-radfar-az225` | `ellow-radfar` | `az225` |
| 4 | `elmid-radnear-az225` | `elmid-radnear` | `az225` |
| 5 | `elmid-radfar-az225` | `elmid-radfar` | `az225` |
| 6 | `elhigh-radnear-az225` | `elhigh-radnear` | `az225` |
| 7 | `elhigh-radmid-az225` | `elhigh-radmid` | `az225` |
| 8 | `elhigh-radfar-az225` | `elhigh-radfar` | `az225` |

## Coverage

- Scenes with any selected viewpoint: `147`
- Complete scenes with all selected viewpoints observed: `0`
- Maximum selected views observed in one scene: `6`
- Grand coalition value: `0.899673`

## Top Players

- Best singleton: `ellow-radnear-az225` with `v({u}) = 0.230760`.
- Top Shapley player: `elhigh-radmid-az225` with `phi = 0.145883`.
- Runner-up: `ellow-radnear-az225` with gap `0.015611`.

## Conditional Next-View Example Inside The Same Subset Game

| Current coalition C | Best addition u | Delta(u | C) | New value |
| --- | --- | ---: | ---: |
| `ellow-radnear-az225` | `elhigh-radmid-az225` | 0.187403 | 0.418163 |
| `ellow-radnear-az225 + elhigh-radmid-az225` | `elmid-radfar-az225` | 0.151227 | 0.569391 |

## Player Ranking

| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 7 | `elhigh-radmid-az225` | 0.145883 | 0.218983 | 36 | 36 |
| 2 | 1 | `ellow-radnear-az225` | 0.130271 | 0.230760 | 36 | 36 |
| 3 | 6 | `elhigh-radnear-az225` | 0.126635 | 0.221063 | 36 | 36 |
| 4 | 5 | `elmid-radfar-az225` | 0.120507 | 0.219894 | 36 | 36 |
| 5 | 2 | `ellow-radmid-az225` | 0.102804 | 0.162860 | 29 | 27 |
| 6 | 8 | `elhigh-radfar-az225` | 0.093913 | 0.162109 | 27 | 27 |
| 7 | 3 | `ellow-radfar-az225` | 0.092912 | 0.158882 | 30 | 28 |
| 8 | 4 | `elmid-radnear-az225` | 0.086749 | 0.149165 | 25 | 24 |
