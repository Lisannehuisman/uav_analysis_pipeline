# Viewpoint Data Separation

This folder is the clean separation between the two sources of viewpoint results in this project.

## Use These Two Folders

- `s4_individual_viewpoints/`
  - Pure results derived from the S4 single-viewpoint / individual-viewpoint analyses.
  - Contains copied outputs from:
    - `factor_level_viewpoint_analysis/results_s0_m4`
    - `comparison_output/*_s0_m4`
    - `outputs/object_viewpoint_metric_grid_s0_m4`
    - `outputs/thesis_tools/best_viewpoints_72`

- `ponyland_pairs/`
  - Pure results derived from the Ponyland pair-training sweep.
  - Contains:
    - raw synced Ponyland snapshot (`snapshot/`)
    - pair-only derived summaries and trend files (`derived_outputs/`)
    - local mirror of pair reports/manifests/plots (`cluster_mirror/`)

## Mixed Files Removed

The following mixed artifacts were removed from `m4_pair_partial_analysis` because they combined S4-derived visuals with Ponyland pair outputs:

- `create_better_story_visuals.py`
- `outputs/story_visuals_summary.md`
- `outputs/today_snapshot_update.md`
- `outputs/plots/per_object_angle_gradients.png`
- `outputs/plots/drone_protocol_quality_by_object.png`
- `outputs/plots/drone_protocol_lift_vs_single.png`
- `outputs/plots/overall_protocol_tradeoff.png`

## Pair-Only Workspace Still Kept

`m4_pair_partial_analysis/` still exists, but it now contains only pair-sweep snapshot files and pair-only derived outputs.
