# 1/2/3-View Per-View Mean Marginal Contributions

For each view, this table averages its marginal contribution over every same-object coalition of size 1, 2, or 3 that contains that view:

`value(coalition) - value(coalition without this view)`

This is a diagnostic mean marginal table for small coalitions only. It is not the exact Shapley result; use `exact_shapley_all_views.csv` for standard Shapley attribution.

## real UAV fine-tuned / `la_souris_truck`

Target class: `whitevan`. Usable views: 10.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-07-12 21.05.57.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | `2026-07-12 21.06.04.jpg` | 46 | 0.000 | 0.001 | 0.007 | 0.018 |
| 3 | `2026-07-12 21.01.06.jpg` | 46 | 0.087 | 0.087 | 0.140 | 0.607 |
| 4 | `2026-07-12 21.01.23.jpg` | 46 | 0.087 | 0.209 | 0.249 | 0.860 |
| 5 | `2026-07-12 21.01.30.jpg` | 46 | 0.087 | 0.199 | 0.238 | 0.851 |
| 6 | `2026-07-12 21.01.33.jpg` | 46 | 0.087 | 0.049 | 0.105 | 0.447 |
| 7 | `2026-07-12 21.01.49.jpg` | 46 | 0.087 | 0.156 | 0.200 | 0.797 |
| 8 | `2026-07-12 21.01.50.jpg` | 46 | 0.087 | 0.149 | 0.197 | 0.786 |
| 9 | `2026-07-12 21.01.51.jpg` | 46 | 0.087 | 0.149 | 0.194 | 0.786 |
| 10 | `2026-07-12 21.05.27.jpg` | 46 | 0.087 | 0.024 | 0.059 | 0.285 |

## real UAV fine-tuned / `m_truck`

Target class: `whitevan`. Usable views: 5.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-07-12 21.22.14.jpg` | 11 | 0.091 | 0.120 | 0.167 | 0.845 |
| 2 | `2026-07-12 21.21.43.jpg` | 11 | 0.091 | 0.064 | 0.109 | 0.709 |
| 3 | `2026-07-12 21.21.46.jpg` | 11 | 0.091 | 0.114 | 0.141 | 0.838 |
| 4 | `2026-07-12 21.21.47.jpg` | 11 | 0.091 | 0.107 | 0.137 | 0.828 |
| 5 | `2026-07-12 21.22.11.jpg` | 11 | 0.091 | 0.065 | 0.107 | 0.712 |

## real UAV fine-tuned / `ooij_tower`

Target class: `tower`. Usable views: 7.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-29 17.37.15.jpg` | 22 | 0.045 | 0.065 | 0.134 | 0.505 |
| 3 | `2026-06-29 17.36.06.jpg` | 22 | 0.045 | 0.011 | 0.122 | 0.246 |
| 4 | `2026-06-29 17.36.07.jpg` | 22 | 0.045 | 0.024 | 0.091 | 0.376 |
| 5 | `2026-06-29 17.36.40.jpg` | 22 | 0.045 | 0.140 | 0.179 | 0.653 |
| 6 | `2026-06-29 17.36.50.jpg` | 22 | 0.045 | 0.213 | 0.233 | 0.754 |
| 7 | `2026-06-29 17.36.51.jpg` | 22 | 0.045 | 0.260 | 0.239 | 0.801 |
| 8 | `2026-06-29 17.36.53.jpg` | 22 | 0.045 | 0.022 | 0.082 | 0.360 |

## real UAV fine-tuned / `parkinglot_car`

Target class: `suv`. Usable views: 6.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-28 20.56.45.jpg` | 16 | 0.062 | 0.082 | 0.102 | 0.789 |
| 2 | `2026-06-28 20.56.46.jpg` | 16 | 0.062 | 0.052 | 0.096 | 0.756 |
| 3 | `2026-06-28 20.56.47.jpg` | 16 | 0.062 | 0.049 | 0.093 | 0.745 |
| 4 | `2026-06-28 20.56.54.jpg` | 16 | 0.062 | 0.054 | 0.119 | 0.760 |
| 5 | `2026-06-28 20.58.04.jpg` | 16 | 0.062 | 0.044 | 0.118 | 0.707 |
| 6 | `2026-06-28 20.58.01.jpg` | 16 | 0.062 | 0.054 | 0.106 | 0.761 |

## real UAV fine-tuned / `white_truck_bottendaal`

Target class: `whitevan`. Usable views: 7.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-30 15.33.46.jpg` | 22 | 0.045 | 0.089 | 0.109 | 0.929 |
| 2 | `2026-06-30 15.33.52.jpg` | 22 | 0.045 | 0.080 | 0.085 | 0.920 |
| 3 | `2026-06-30 15.34.24.jpg` | 22 | 0.045 | 0.036 | 0.048 | 0.786 |
| 4 | `2026-06-30 15.34.33.jpg` | 22 | 0.045 | 0.049 | 0.064 | 0.867 |
| 5 | `2026-06-30 15.35.01.jpg` | 22 | 0.045 | 0.040 | 0.055 | 0.837 |
| 6 | `2026-06-30 15.33.29.jpg` | 22 | 0.045 | 0.042 | 0.055 | 0.843 |
| 7 | `2026-06-30 15.33.34.jpg` | 22 | 0.045 | 0.065 | 0.075 | 0.899 |

## synthetic M4 only / `la_souris_truck`

Target class: `whitevan`. Usable views: 10.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-07-12 21.05.57.jpg` | 46 | 0.478 | 0.182 | 0.230 | 0.380 |
| 2 | `2026-07-12 21.06.04.jpg` | 46 | 0.478 | 0.377 | 0.404 | 0.636 |
| 3 | `2026-07-12 21.01.06.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | `2026-07-12 21.01.23.jpg` | 46 | 0.478 | 0.218 | 0.265 | 0.438 |
| 5 | `2026-07-12 21.01.30.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | `2026-07-12 21.01.33.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | `2026-07-12 21.01.49.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | `2026-07-12 21.01.50.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9 | `2026-07-12 21.01.51.jpg` | 46 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | `2026-07-12 21.05.27.jpg` | 46 | 0.478 | 0.587 | 0.588 | 0.846 |

## synthetic M4 only / `m_truck`

Target class: `whitevan`. Usable views: 5.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-07-12 21.22.14.jpg` | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | `2026-07-12 21.21.43.jpg` | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | `2026-07-12 21.21.46.jpg` | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | `2026-07-12 21.21.47.jpg` | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | `2026-07-12 21.22.11.jpg` | 11 | 0.000 | 0.000 | 0.000 | 0.000 |

## synthetic M4 only / `ooij_tower`

Target class: `tower`. Usable views: 7.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-29 17.37.15.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | `2026-06-29 17.36.06.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | `2026-06-29 17.36.07.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | `2026-06-29 17.36.40.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | `2026-06-29 17.36.50.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | `2026-06-29 17.36.51.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | `2026-06-29 17.36.53.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |

## synthetic M4 only / `parkinglot_car`

Target class: `suv`. Usable views: 6.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-28 20.56.45.jpg` | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | `2026-06-28 20.56.46.jpg` | 16 | 0.000 | 0.001 | 0.001 | 0.001 |
| 3 | `2026-06-28 20.56.47.jpg` | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | `2026-06-28 20.56.54.jpg` | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | `2026-06-28 20.58.04.jpg` | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | `2026-06-28 20.58.01.jpg` | 16 | 0.000 | 0.000 | 0.000 | 0.000 |

## synthetic M4 only / `white_truck_bottendaal`

Target class: `whitevan`. Usable views: 7.

| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-06-30 15.33.46.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | `2026-06-30 15.33.52.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | `2026-06-30 15.34.24.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | `2026-06-30 15.34.33.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | `2026-06-30 15.35.01.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | `2026-06-30 15.33.29.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | `2026-06-30 15.33.34.jpg` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
