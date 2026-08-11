# Real Box Method Illustration

![Real-data method illustration](real_box_method_illustration.png)

## What This Figure Is Showing

- `View 1`: `S0-SM_tank2-elmid-radmid-az000.png`
- `View 2`: `S0-SM_tank2-elmid-radnear-az315.png`
- Green boxes are the target ground truth.
- Orange boxes are the matched target predictions that define the target metrics used in the analysis.
- Yellow boxes are other final detector outputs from the prediction JSON.
- The important methodological point is that the current multiview rules operate on per-view target evidence, not on a cross-view fused 2D box.

## Method Scores For This Exact Pair

| Rank | Method | Family | Score |
| --- | --- | --- | ---: |
| 1 | `Geometry-weighted OR + best IoU` | Geometry-aware accumulation | 0.9800 |
| 2 | `Hybrid geometry+cell OR` | Geometry-aware accumulation | 0.9800 |
| 3 | `Viewpoint-cell OR + best IoU` | Geometry-aware accumulation | 0.9799 |
| 4 | `Noisy-OR + best IoU` | Evidence accumulation | 0.9780 |
| 5 | `Support-weighted OR` | Evidence accumulation | 0.9704 |
| 6 | `Viewpoint-cell selector` | Geometry-aware selection | 0.9281 |
| 7 | `Best box (max)` | Selection | 0.9281 |
| 8 | `Single View 1` | Reference | 0.9281 |
| 9 | `Geometry prior selector` | Geometry-aware selection | 0.9281 |
| 10 | `Geometry-calibrated selector` | Geometry-aware selection | 0.9281 |
| 11 | `Mean quality` | Naive averaging | 0.9273 |
| 12 | `Single View 2` | Reference | 0.9266 |

## How To Read It

- If a method is a `selector`, it ends by keeping one view and throwing the other away.
- If a method is `OR-style accumulation`, it lets both views contribute evidence to the final pair score.
- If a method is `geometry-aware`, the geometry part mostly changes how much we trust each view before combining them.
- That is why the key step in your results is usually not `geometry` versus `no geometry`, but `selection` versus `evidence accumulation`.
