# M4 Viewpoint Selection Analysis

This folder adds the missing thesis-facing selection layer on top of the cached
fixed-detector M4 analyses.

It does not retrain detectors and it does not invent camera geometry. It reuses
the evidence that is already present in the repository:

- `m4_two_drone_operational_analysis/outputs/scene_view_records.csv`
- `viewpoint_data_separated/72_trained_models/manifests/viewpoint_inventory.csv`
- `m4_cross_view_box_fusion_analysis/outputs/pair_combo_rows.csv`
- `m4_cross_view_box_fusion_analysis/outputs/triple_combo_rows.csv`
- `outputs/thesis_tools/dataset_structure_audit/*`

## Recommended One-Command Pipeline

Run the integrated analysis first:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\run_swarm_viewpoint_analysis.py
```

This writes one self-contained output tree to
`m4_viewpoint_selection_analysis/outputs/integrated/`.

It covers:

- clean viewpoint inventory and metadata
- single-view global and per-class rankings
- elevation/radius/azimuth grouped analysis
- factor-explanation scores for viewpoint quality
- pair/triple relationship taxonomy and performance summaries
- pairwise synergy matrix and complementarity groups
- weighted practical subset recommendations
- 1/2/3-view budget comparison with marginal gains

Main outputs:

- `viewpoint_table.csv`
- `per_viewpoint_metrics.csv`
- `per_class_viewpoint_metrics.csv`
- `viewpoint_grouped_analysis.csv`
- `viewpoint_factor_explanation.csv`
- `pair_triple_relationship_metadata.csv`
- `pair_triple_performance.csv`
- `relationship_type_performance_summary.csv`
- `pairwise_synergy_matrix.csv`
- `top_k_best_pairs.csv`
- `top_k_most_complementary_pairs.csv`
- `top_k_redundant_pairs.csv`
- `recommended_viewpoint_subsets.csv`
- `swarm_budget_comparison.csv`
- `VIEWPOINT_SWARM_ANALYSIS_SUMMARY.md`
- `PRACTICAL_VIEWPOINT_SELECTION_CHECKLIST.md`
- plots under `outputs/integrated/plots/`

## Script: `build_viewpoint_subset_matrix.py`

### What it does

Builds a reproducible all-viewpoint selection table from the cached M4
scene/view records.

It computes:

- all observed 1-view, 2-view, and 3-view subsets
- best fixed subsets by drone count
- 72x72 pair matrices for strict target quality, support, complementarity, and
  support-weighted late-fusion quality
- factor-pattern summaries for near/far, elevation, and azimuth diversity
- target-absent false-alarm rates for selected subsets

The key metric is `mean_best_strict_quality`, which is the mean best
`confidence * IoU` target match available among the selected views.

### How to run

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\build_viewpoint_subset_matrix.py
```

Optional example:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\build_viewpoint_subset_matrix.py `
  --min-support 3 `
  --top-n 25 `
  --output-dir .\m4_viewpoint_selection_analysis\outputs
```

### Main outputs

Written to `m4_viewpoint_selection_analysis/outputs/`:

- `viewpoint_inventory.csv`
- `subset_scores.csv`
- `best_subsets_by_k.csv`
- `pair_matrix_strict_quality.csv`
- `pair_matrix_complementarity.csv`
- `pair_matrix_support.csv`
- `pair_matrix_support_weighted_fusion.csv`
- `factor_pattern_summary.csv`
- `viewpoint_subset_matrix_report.md`
- `plots/pair_matrix_strict_quality.png`
- `plots/pair_matrix_complementarity.png`
- `plots/pair_matrix_support.png`
- `plots/pair_matrix_support_weighted_fusion.png`
- `plots/subset_size_gain_curve.png`
- `plots/top_subsets_by_k.png`
- `plots/factor_pattern_summary.png`

## Script: `validate_viewpoint_subset_generalization.py`

### What it does

Performs a conservative validation of viewpoint selection stability.

It repeatedly splits the cached M4 test scenes into:

- selection scenes
- held-out scenes

For each split, it selects the best fixed 1-view, 2-view, and 3-view subsets on
the selection scenes, then evaluates those same subsets on held-out scenes.

This is scene-split held-out validation. Because the dataset audit indicates
object instances overlap across train, val, and test, this script does not call
the held-out fold true novel-object generalization.

It also audits target-absent views, where the filename target has no ground
truth box, and measures false alarms in those views.

### How to run

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\validate_viewpoint_subset_generalization.py
```

Optional example:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\validate_viewpoint_subset_generalization.py `
  --folds 5 `
  --repeats 10 `
  --top-n 5 `
  --min-support 3
```

### Main outputs

Written to `m4_viewpoint_selection_analysis/outputs/generalization/`:

- `scene_split_generalization_trials.csv`
- `scene_split_generalization_summary.csv`
- `top_subset_selection_frequency.csv`
- `recommended_subsets_scene_split.csv`
- `generalization_readiness.csv`
- `target_absent_viewpoint_audit.csv`
- `target_absent_false_alarm_examples.csv`
- `generalization_validation_report.md`
- `plots/scene_split_generalization_gap.png`
- `plots/selection_stability_top_subsets.png`
- `plots/target_absent_false_alarm_rates.png`

## Script: `robust_viewpoint_diversity_analysis.py`

### What it does

Adds the robustness layer for the pair/triple diversity conclusions.

It implements two follow-up checks:

- factor-level pair/triple analysis with scene-bootstrap confidence intervals
- matched-scene relationship comparisons, where each relationship difference is
  computed only on scenes where both relationship types are available

This is intended for thesis claims about whether distance, azimuth, elevation,
or mixed diversity explains stronger multi-view performance. It should be used
to support relationship-pattern conclusions, while exact top viewpoint IDs
should still be interpreted with their support counts.

### How to run

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\robust_viewpoint_diversity_analysis.py
```

Optional example:

```powershell
.\.venv\Scripts\python.exe .\m4_viewpoint_selection_analysis\robust_viewpoint_diversity_analysis.py `
  --bootstrap-iters 5000 `
  --min-scenes 10
```

### Main outputs

Written to `m4_viewpoint_selection_analysis/outputs/robustness/`:

- `scene_relationship_metric_table.csv`
- `relationship_bootstrap_ci.csv`
- `matched_scene_relationship_differences.csv`
- `common_scene_axis_summary.csv`
- `robust_relationship_recommendations.csv`
- `ROBUST_VIEWPOINT_DIVERSITY_SUMMARY.md`
- `plots/bootstrap_relationship_ci.png`
- `plots/matched_scene_differences.png`

## Recommended Use In Thesis

Use these outputs as the final viewpoint-selection layer:

1. `best_subsets_by_k.csv`
   Answers which fixed 1-view, 2-view, and 3-view configurations work best.
2. `pair_matrix_strict_quality.csv` and `pair_matrix_complementarity.csv`
   Answer whether a pair matrix can identify useful viewpoint subsets.
3. `subset_size_gain_curve.png`
   Shows the effect of using 1, 2, or 3 viewpoints.
4. `generalization_validation_report.md`
   Shows how stable recommendations are under scene-split held-out validation
   and flags exact subsets with poor held-out support.
5. `target_absent_viewpoint_audit.csv`
   Documents performance when the filename target is absent.
6. `outputs/robustness/ROBUST_VIEWPOINT_DIVERSITY_SUMMARY.md`
   Supports robust claims about which kinds of viewpoint diversity help most.

## Interpretation Boundary

The scripts support viewpoint selection and late-fusion analysis from cached
detections. They do not establish calibrated 3D fusion, cross-view identity
tracking, or true novel-object-instance generalization without additional data
or split definitions.
