# Multiview method comparison report

## Purpose

This report compares viewpoint-combination rules that can be evaluated honestly with the current cached project data.
It is designed to answer two questions:

- if one view is already available, which two-view combination rules improve target-centric detection quality most;
- and which second viewpoints add the most value when they are appended to a primary view.

## How this comparison is computed

- Input data: per-scene, per-view target metrics from `scene_view_records.csv`.
- Single-view reference: the scene-balanced mean of `target_strict_quality_iou50` over all available single viewpoints.
- Two-view comparison: for every scene and every unordered pair of viewpoints, the script evaluates each fusion rule on the same underlying per-view target records.
- Added-viewpoint analysis: for every ordered pair `(primary, secondary)`, the script measures how much the pair score improves over the primary-only score and over the best constituent single score.
- Headline added viewpoints: the report ranks secondary viewpoints by their average lift across all primaries they were paired with, subject to the support threshold.

## Important boundary

The current cache supports score-level and conservative late-fusion comparisons, but it does not yet support full geometry-aware box fusion across views.
That means methods such as Weighted Boxes Fusion or calibration-based reprojection fusion are discussed in the literature map, but not evaluated as if they were already valid on the present files.

## Single-view reference

- Scene-balanced single-view reference quality: `0.8732`

## Best two-view method in the current cache

- Best method: `Noisy-OR + best IoU`
- Method family: `Probabilistic accumulation`
- Scene-balanced two-view quality: `0.9515`
- Gain versus single-view reference: `+0.0784`

## Weakest two-view method in the current cache

- Weakest method: `Mean quality`
- Scene-balanced two-view quality: `0.8732`
- Gain versus single-view reference: `+0.0000`

## Interpreting the method families

- `Best box (max)` is the rescue-view baseline: the second view helps if either view is good.
- `Mean quality` tests the naive intuition that all views should simply be averaged.
- `2-of-2 unanimous best box` is a strict confirmation rule and will often trade recall for agreement.
- `Noisy-OR` methods treat multiple views as accumulating evidence rather than competing single boxes.
- `Support-weighted OR` is the most conservative evaluable corroboration rule because it rewards both evidence strength and multi-view agreement.

## Feasibility map

The CSV `method_feasibility_matrix.csv` records which families are evaluable now and which require extra calibration, correspondence, or multiview training.

## Strong added viewpoints by method

### Best box (max)

- `elmid-radnear-az090`: mean lift vs primary `+0.0740`, lift vs best constituent `+0.0000`, support `346`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0630`, lift vs best constituent `+0.0000`, support `338`, rescue|primary miss `1.0000`
- `ellow-radnear-az090`: mean lift vs primary `+0.0579`, lift vs best constituent `+0.0000`, support `362`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0574`, lift vs best constituent `+0.0000`, support `427`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0533`, lift vs best constituent `+0.0000`, support `234`, rescue|primary miss `1.0000`

### Mean quality

- `elmid-radnear-az090`: mean lift vs primary `+0.0298`, lift vs best constituent `-0.0442`, support `346`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0276`, lift vs best constituent `-0.0354`, support `338`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0239`, lift vs best constituent `-0.0335`, support `427`, rescue|primary miss `1.0000`
- `ellow-radnear-az225`: mean lift vs primary `+0.0228`, lift vs best constituent `-0.0255`, support `394`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0215`, lift vs best constituent `-0.0318`, support `234`, rescue|primary miss `1.0000`

### Noisy-OR + best IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1248`, lift vs best constituent `+0.0508`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.1039`, lift vs best constituent `+0.0465`, support `427`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0996`, lift vs best constituent `+0.0463`, support `234`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.0976`, lift vs best constituent `+0.0465`, support `374`, rescue|primary miss `1.0000`
- `elmid-radfar-az090`: mean lift vs primary `+0.0958`, lift vs best constituent `+0.0491`, support `288`, rescue|primary miss `1.0000`

### Noisy-OR + mean IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1011`, lift vs best constituent `+0.0271`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0893`, lift vs best constituent `+0.0318`, support `427`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0870`, lift vs best constituent `+0.0337`, support `234`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.0840`, lift vs best constituent `+0.0330`, support `374`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0788`, lift vs best constituent `+0.0157`, support `338`, rescue|primary miss `1.0000`

### Support-weighted OR

- `elmid-radnear-az090`: mean lift vs primary `+0.0897`, lift vs best constituent `+0.0157`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0753`, lift vs best constituent `+0.0178`, support `427`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.0725`, lift vs best constituent `+0.0214`, support `374`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0705`, lift vs best constituent `+0.0172`, support `234`, rescue|primary miss `1.0000`
- `elmid-radmid-az180`: mean lift vs primary `+0.0676`, lift vs best constituent `+0.0241`, support `324`, rescue|primary miss `1.0000`

### 2-of-2 unanimous best box

- `elmid-radnear-az090`: mean lift vs primary `+0.0512`, lift vs best constituent `-0.0228`, support `346`, rescue|primary miss `0.0000`
- `ellow-radnear-az090`: mean lift vs primary `+0.0428`, lift vs best constituent `-0.0151`, support `362`, rescue|primary miss `0.0000`
- `ellow-radnear-az225`: mean lift vs primary `+0.0411`, lift vs best constituent `-0.0071`, support `394`, rescue|primary miss `0.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0355`, lift vs best constituent `-0.0276`, support `338`, rescue|primary miss `0.0000`
- `elmid-radnear-az180`: mean lift vs primary `+0.0348`, lift vs best constituent `-0.0172`, support `352`, rescue|primary miss `0.0000`

## Files

- `overall_method_summary.csv`
- `pair_method_rows.csv`
- `ordered_pair_method_rows.csv`
- `ordered_pair_gain_summary.csv`
- `added_viewpoint_summary.csv`
- `added_viewpoint_headlines.csv`
- `overall_method_comparison.png`
- `top_added_viewpoints_by_method.png`