# Repository Audit And Viewpoint Selection Roadmap

This note records the project state observed before adding the scripts in this
folder, and the concrete analysis plan those scripts implement.

## Audit Summary

### Existing fixed-detector multiview analysis

The repository already contains a strong fixed-detector M4 analysis pipeline:

- `m4_two_drone_operational_analysis`
  - reuses cached full-M4 YOLOv8l predictions and COCO ground truth
  - writes per-view target-quality records
  - compares 1-view, 2-view, and 3-view operational protocols
  - includes OR, confirmation, and unanimous-style protocols
- `m4_cross_view_box_fusion_analysis`
  - reuses the same M4 prediction cache
  - compares conservative late-fusion policies
  - writes 2-view and 3-view fusion rows
- `m4_marginal_viewpoint_value_analysis`
  - turns fixed-detector outputs into marginal-value, complementarity, and
    diminishing-return summaries

The main fixed-detector cache currently contains:

- `2214` image/view records
- `205` scene keys
- `72` absolute viewpoints
- `10` object classes
- `40` target-absent views, about `1.8%`

The scene availability summary shows:

- minimum views per scene: `4`
- mean views per scene: `10.8`
- maximum views per scene: `18`
- all `205` scenes support at least 2 and 3 available views

### Existing training-side viewpoint analysis

The repository also contains viewpoint-restricted training results:

- `72/72` single-view models completed
- `2535/2556` pair-view models completed
- current pair sweep completion: about `99.2%`
- full-M4, single-view, pair-view, and equal-image-count M4 controls are already
  summarized in `full_m4_vs_single_pair_operational_analysis`

These are training-side detector generalization results. They should be kept
separate from fixed-detector operational multiview results.

### Existing detector-family comparison

The repository includes standardized detector-family comparisons for:

- `YOLOv8n`
- `YOLOv8l`
- `Faster R-CNN`

across regimes:

- `M1`
- `M2a`
- `M2b`
- `M3`
- `M4`

The strongest cached M4 fixed-detector source for this new analysis is:

- `outputs/detector_family_comparison/standardized_test_eval/ground_truth/M4_test_gt.json`
- `outputs/detector_family_comparison/standardized_test_eval/predictions/YOLOv8l_M4_test_predictions.json`

### Existing split and absence audit

The dataset audit in `outputs/thesis_tools/dataset_structure_audit` reports:

- train images: `10332`
- validation images: `2214`
- test images: `2214`

It also reports object-instance overlap across splits. This means the current
test split should be described carefully as held-out images/views, not as a
guaranteed novel-object-instance split.

Target-absent examples are present but sparse. The absence analysis should be
treated as an audit of false alarms, not as a large negative benchmark.

## Missing Pieces Identified

The project already had most of the raw ingredients, but four thesis-facing
artifacts were not yet cleanly exposed:

1. A 72x72 viewpoint-pair matrix that can be used directly to identify strong
   viewpoint pairs.
2. A single best-subset table for 1, 2, and 3 viewpoints generated from one
   reproducible command.
3. A scene-split held-out validation loop that measures whether selected
   viewpoint subsets remain strong outside the scenes used for selection.
4. A compact target-absent false-alarm audit attached to the viewpoint-selection
   story.

## Analysis Roadmap

### Step 1: Build the viewpoint subset matrix

Use `build_viewpoint_subset_matrix.py`.

This step answers:

- What is the best fixed 1-view, 2-view, or 3-view configuration?
- Can a matrix over all viewpoints identify useful pairs?
- Which elevation/radius/azimuth patterns recur in strong subsets?
- How do late-fusion quality scores compare with target-quality subset scores?

Primary outputs:

- `subset_scores.csv`
- `best_subsets_by_k.csv`
- `pair_matrix_strict_quality.csv`
- `pair_matrix_complementarity.csv`
- `pair_matrix_support_weighted_fusion.csv`
- `plots/subset_size_gain_curve.png`

### Step 2: Validate recommendations on held-out scenes

Use `validate_viewpoint_subset_generalization.py`.

This step answers:

- Are top viewpoint subsets stable when selected on one scene subset and tested
  on held-out scenes?
- How large is the selection-to-held-out quality gap?
- Which fixed subsets are repeatedly selected across splits?

Primary outputs:

- `scene_split_generalization_trials.csv`
- `scene_split_generalization_summary.csv`
- `top_subset_selection_frequency.csv`
- `recommended_subsets_scene_split.csv`
- `plots/scene_split_generalization_gap.png`
- `plots/selection_stability_top_subsets.png`

### Step 3: Audit target absence

Also handled by `validate_viewpoint_subset_generalization.py`.

This step answers:

- Which classes or viewpoints have target-absent examples?
- When the filename target is absent, how often does the detector still produce
  target-class false positives?
- Are selected subsets vulnerable to absent-view false alarms?

Primary outputs:

- `target_absent_viewpoint_audit.csv`
- `target_absent_false_alarm_examples.csv`
- `plots/target_absent_false_alarm_rates.png`

## Recommended Thesis Wording

Use careful language:

- "fixed-detector multiview viewpoint selection"
- "scene-split held-out validation"
- "held-out images/views"
- "target-absent false-alarm audit"

Avoid claiming:

- true calibrated geometric fusion
- cross-view object identity tracking
- novel-object-instance generalization
- a large negative-only benchmark

Those claims would require additional metadata, split definitions, or data.
