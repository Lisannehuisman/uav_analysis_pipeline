# Method Combination Schematic

![Method schematic](C:/Users/lisan/OneDrive/Documents/New project/geometry_aware_fusion_analysis/outputs/method_combination_schematic.png)

## Plain-Language Summary

Every method starts with two views. The real difference is whether the method:

- keeps only one view,
- averages views naively, or
- accumulates evidence from both views.

That is the central interpretation key for your results.

## What Each Family Really Does

- `Single-view reference`: uses one view only. No multiview benefit is possible.
- `Best box (max)`: keeps the better of the two single-view target qualities. This is mostly a rescue rule. Current score: `0.9088`.
- `Mean quality`: averages the two view qualities. This is too pessimistic when one view is weak.
- `Noisy-OR + best IoU`: accumulates evidence from both views, then keeps the best localization term. This is the main strong non-geometry baseline.
- `Support-weighted OR`: similar to noisy-OR, but more conservative because it rewards agreement/support.
- `Geometry prior selector`: chooses the view that geometry predicts will be best. It still throws away one view.
- `Geometry calibrated selector`: chooses the view with the strongest geometry-adjusted confidence. Still a selector.
- `Viewpoint-cell prior selector`: chooses the view from the best exact lattice cell for that class. Still a selector.
- `Geometry-weighted OR`: keeps both views, but geometry reweights their confidence before OR fusion.
- `Viewpoint-cell OR`: keeps both views and uses class-specific lattice-cell reliability as the reweighting term.
- `Hybrid geometry+cell OR`: combines the smooth geometry prior and the discrete cell prior before OR fusion.

## How To Interpret The Current Results

- `Noisy-OR + best IoU` = `0.9515`
- `Viewpoint-cell OR + best IoU` = `0.9515`
- Best overall method right now = `Viewpoint-cell OR + best IoU` at `0.9515`

So the main performance jump comes from moving away from selection/averaging and toward evidence accumulation.

## The Important Nuance

`Viewpoint-cell OR + best IoU` wins against plain `Noisy-OR + best IoU` on `0.7207` of pair rows, but its mean pair-score gap is only `-0.0001`.

That means you should present it as a near-tie with slightly better local behavior, not as a dramatic breakthrough.

## Best Way To Show This In Results

- Use `overall_method_summary.csv` for the end-to-end ranking.
- Use `method_value_summary.csv` to separate rescue from true corroboration.
- Use `pairwise_method_comparison.csv` to show direct method-versus-method comparisons.
- Use `added_viewpoint_headlines.csv` to show which specific second viewpoints help most.

## Practical Reading Rule

- If `mean_lift_vs_best_constituent` is near `0`, the method mainly acts like a smart selector/rescue rule.
- If `mean_lift_vs_best_constituent` is clearly positive, the method gets real cross-view corroboration value.
- If `rescue_rate_given_primary_miss` is high, the second drone is especially useful when the first drone fails.
- If `win_rate` is high but `mean_score_gap` is near zero, the comparison is practically a tie.
