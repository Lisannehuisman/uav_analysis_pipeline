# Exact Viewpoint-Subset Shapley Analysis

## Setup

- Input scene records: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`
- Subset name: `edges_az135_225`
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
| 1 | `ellow-radmid-az135` | `ellow-radmid` | `az135` |
| 2 | `ellow-radmid-az225` | `ellow-radmid` | `az225` |
| 3 | `elmid-radnear-az135` | `elmid-radnear` | `az135` |
| 4 | `elmid-radnear-az225` | `elmid-radnear` | `az225` |
| 5 | `elmid-radfar-az135` | `elmid-radfar` | `az135` |
| 6 | `elmid-radfar-az225` | `elmid-radfar` | `az225` |
| 7 | `elhigh-radmid-az135` | `elhigh-radmid` | `az135` |
| 8 | `elhigh-radmid-az225` | `elhigh-radmid` | `az225` |

## Coverage

- Scenes with any selected viewpoint: `145`
- Complete scenes with all selected viewpoints observed: `0`
- Maximum selected views observed in one scene: `6`
- Grand coalition value: `0.919212`

## Top Players

- Best singleton: `elmid-radnear-az135` with `v({u}) = 0.273823`.
- Top Shapley player: `elmid-radnear-az135` with `phi = 0.165821`.
- Runner-up: `elhigh-radmid-az225` with gap `0.025071`.

## Conditional Next-View Example Inside The Same Subset Game

| Current coalition C | Best addition u | Delta(u | C) | New value |
| --- | --- | ---: | ---: |
| `elmid-radnear-az135` | `elmid-radfar-az225` | 0.179355 | 0.453178 |
| `elmid-radnear-az135 + elmid-radfar-az225` | `elhigh-radmid-az225` | 0.135523 | 0.588701 |

## Player Ranking

| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 3 | `elmid-radnear-az135` | 0.165821 | 0.273823 | 43 | 43 |
| 2 | 8 | `elhigh-radmid-az225` | 0.140750 | 0.222004 | 36 | 36 |
| 3 | 6 | `elmid-radfar-az225` | 0.124899 | 0.222927 | 36 | 36 |
| 4 | 1 | `ellow-radmid-az135` | 0.102783 | 0.170201 | 30 | 28 |
| 5 | 7 | `elhigh-radmid-az135` | 0.100169 | 0.177833 | 30 | 30 |
| 6 | 2 | `ellow-radmid-az225` | 0.096597 | 0.165106 | 29 | 27 |
| 7 | 4 | `elmid-radnear-az225` | 0.094434 | 0.151223 | 25 | 24 |
| 8 | 5 | `elmid-radfar-az135` | 0.093758 | 0.180368 | 31 | 31 |
