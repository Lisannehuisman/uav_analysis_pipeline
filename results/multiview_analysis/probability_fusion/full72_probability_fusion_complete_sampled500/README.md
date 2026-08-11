# Synthetic Multi-Perspective Probability Fusion

## What was tested

- Synthetic-only target-level fusion using cached YOLOv8l M4 detections.
- Evaluation split: `full72`.
- Coalition sizes: `1, 2, 3`.
- Fusion rules: `max`, `mean`, `noisy_or`, `product`, and `log_product`.

## Why this is synthetic-only

- The experiment groups repeated views of the same synthetic object instance (`scene_key`) under known azimuth, elevation, and radius geometry.
- No real-image files are used anywhere in this analysis.

## How scores were extracted

- Per-view target score `p_{i,v}` is the best correct-class detection confidence with IoU >= 0.5 against a target-class ground-truth box.
- If no valid target match exists in a view, that view score is set to `0` and the view-level found label is `0`.
- Coalition labels are operational target-recovery labels: a coalition is labeled `found=1` if at least one constituent view contains a valid target match.

## Coverage note

- Expected full grid size: `72` views per instance.
- Observed cached coverage: min `72`, max `72`, mean `72.00` views per instance.
- Instances with the full expected grid present in the current cache: `205` / `205`.
- The script therefore reports the actual cached view coverage and proceeds without pretending that a full 72-view reconstruction exists when it does not.

## Calibration

- Requested calibration mode: `none`.
- Effective calibration mode: `none`.
- Note: Calibration disabled by configuration.
- Calibration fit samples: `0`.

## Main result

- Best multi-view summary row: `Log-product` at `k=2`.
- Mean fused score: `-29.2822`.
- Average precision: `1.0000`.
- Best-threshold F1: `1.0000` at threshold `-33.9946`.
- Mean uplift vs best constituent single view: `-29.8977`.
- Rate of beating the best constituent single view: `0.0000`.
- Best rescue row: `Noisy-OR` at `k=2` with uplift `+0.0113`.
- Best product row: `Product` at `k=3` with uplift `-0.5954`.
- The found-label AP and F1 metrics are near-ceiling for `k=1` because the per-view score is itself defined from valid target matches; the multi-view interpretation should therefore focus mainly on coalition-size trends and uplift over the best constituent single view.

## Rescue vs agreement interpretation

- This experiment tests multi-perspective probability fusion only in the synthetic setting, where repeated observations of the same object are available under known viewpoint geometry.
- Product-based fusion is interpreted as an agreement model, while max and noisy-OR are interpreted as rescue models.
- In this cache, rescue-style fusion improved over the best constituent single view more than product fusion did, which suggests that multi-view benefit mainly comes from increasing the chance that at least one viewpoint sees the target clearly.

## Geometry signal

- Strongest pairwise geometry summary row: `different_azimuth_and_different_elevation` with `Mean`.
- Pair-group mean uplift vs best constituent single view: `-0.1961`.

## Important limitations

- The current cached prediction files do not provide persistent 3D object IDs inside the annotation JSON, so IoU validation is target-class based rather than explicit world-instance-ID based.
- If the requested calibration mode is enabled, it is fit at prediction level and then applied to matched per-view scores; zero-score views remain zero.
- When `evaluation_split=combined`, view density improves, but the same object instances appear across val and test, so split purity is weaker than in `test`-only evaluation.

## Output files

- `manifest_proxy.csv`
- `view_coverage_summary.csv`
- `per_view_scores.csv`
- `per_coalition_scores.csv`
- `fusion_summary.csv`
- `geometry_group_summary.csv`
- `pair_geometry_focus_summary.csv`
- `fusion_method_performance_by_coalition_size.png`
- `marginal_gain_from_adding_views.png`
- `max_vs_noisy_or_vs_product.png`
- `geometry_group_comparison.png`

## Thesis-ready interpretation

This experiment tests multi-perspective probability fusion only in the synthetic setting, where repeated observations of the same object are available under known viewpoint geometry. Product-based fusion is interpreted as an agreement model, while max and noisy-OR fusion are interpreted as rescue models. If max or noisy-OR outperform product fusion, this suggests that multi-view benefit in object detection arises mainly from increasing the probability that at least one viewpoint observes the target clearly, rather than from multiplying mutually reinforcing probabilities across all views.

Generated in `C:\Users\lisan\OneDrive\Documents\New project\probability_fusion\outputs\full72_probability_fusion_complete_sampled500`.
