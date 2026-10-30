# Clutter Grouping Analysis

This folder analyzes whether clutter groupings correlate with target detection quality and whether those groupings differ significantly.

## Inputs

- `geometry_ground_truth_analysis/outputs/view_geometry_table.csv`
- `m4_two_drone_operational_analysis/outputs/scene_view_records.csv`

## Operationalization

- `clutter = num_label_boxes - num_target_class_boxes`
- view-level clutter bins: `0`, `1-4`, `5-9`, `10+`
- scene-level clutter groups: quartiles of the mean distractor count per scene

## Outputs

Running `run_clutter_grouping_analysis.py` writes:

- joined CSVs
- clutter-group summaries
- scene-level correlations
- omnibus and pairwise significance tests
- plots for clutter distributions and quality-group relationships
- `analysis_report.md`
