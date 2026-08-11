# Per-class Shapley plots

This folder contains three actual-data plots derived from the exact `Noisy-OR + best IoU` ring-based Shapley analysis.

## Files

- `per_class_top_shapley_heatmap.png`
  - Rows are object classes, columns are controlled rings.
  - Each cell shows the top azimuth and top Shapley score for that class-ring combination.
- `per_class_best_ring_bar.png`
  - One row per object class.
  - Shows the strongest class-specific Shapley score anywhere in the ring grid, with its ring and azimuth.
- `per_class_elmid_radnear_azimuth_facets.png`
  - Small multiples for the focused ring `elmid-radnear`.
  - Shows the full azimuth-wise Shapley profile for each object class, not just the winner.

## Best use

- Use the heatmap for a compact supervisor or thesis overview.
- Use the bar chart if you want the cleanest summary answer to `what is the best collaborative angle per object?`.
- Use the focused ring facets when you want to show that pooled results hide strong object-specific differences.

## Input sources

- `ring_shapley_noisy_or_best_iou_by_class.csv`
- `ring_shapley_noisy_or_best_iou_by_class_summary.csv`