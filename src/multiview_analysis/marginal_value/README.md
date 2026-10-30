# M4 Marginal Viewpoint Value Analysis

This analysis examines the contribution of additional UAV viewpoints to object detection performance using predictions from a fixed detector.

The main objective is to quantify how detection performance changes as additional views become available, identify diminishing returns, and investigate which viewpoints provide complementary information.

## What This Folder Adds

The analysis consists of four complementary components:
- `single-view landscape`
- `multi-view gain`
- `complementarity`
- `swarm-size-conditioned Shapley`

## Data Sources and Scope

The analysis uses predictions and evaluation results generated with the fixed YOLOv8l-M4 detector.

The available scene-level observation records are stored in:

- `results/multiview_analysis/two_drone_operational/scene_view_records.csv`
- `results/multiview_analysis/two_drone_operational_test/scene_view_records.csv`

These files contain the observations used to examine viewpoint performance and complementarity.

Some scripts also depend on the original protocol-level summary files (`protocol_overall_summary.csv` and `protocol_class_summary.csv`). These intermediate files are not currently included in the repository.

The previously generated analysis results are available in `results/multiview_analysis/marginal_value/`.

The detector remains fixed throughout the multiview evaluation. Differences in performance therefore reflect the available observations and their combination rather than changes in model training.

## Main Definitions

### Pair Complementarity

For a viewpoint pair `(i, j)`, this folder computes:

`E[max(i, j)] - max(E[i], E[j])`

on the matched scene subset where both viewpoints are available.

### Third-View Gain

For a triple `(i, j, k)`, this folder computes:

`E[max(i, j, k)] - max(E[max(i, j)], E[max(i, k)], E[max(j, k)])`

on the matched scene subset where all three viewpoints are available.

### Observed Marginal Contribution Score

Each viewpoint gets a marginal score based on how much it helps when added to:

- one existing view
- an existing pair

This is intentionally not called `Shapley`, because it only averages observed pair-stage and third-stage marginal gains from the current cache. The only Shapley analysis retained in this project is the exact fusion-aware ring analysis based on `Noisy-OR + best IoU`.

The following scripts implement the fusion-aware Shapley analyses:

- `compute_ring_shapley_noisy_or_best_iou.py` for ring-level exact Shapley
- `compute_subset_shapley_noisy_or_best_iou.py` for exact Shapley on a custom or full-grid-distributed 8-view subset
- `build_shapley_swarm_size_profiles.py` for exact Shapley decomposed by coalition size / swarm size
- `build_noisy_or_coalition_size_breakdown.py` for coalition-value growth by number of drones
- `build_image_count_shapley_proxy.py` for an angle-agnostic count-first Shapley-style proxy
- `ring_shapley_noisy_or_best_iou_conditional_gain_report.md` for exact `Delta(u | C)` next-view gains inside the same ring game

## Angle-Agnostic Count-First Shapley Proxy

This angle-agnostic analysis is useful when camera-angle estimates are unavailable or unreliable.

It answers:

`what is the marginal contribution of the 2nd, 3rd, 4th, ... image, even if we ignore angle labels?`
Run from the repository root:

```cmd
python src\multiview_analysis\marginal_value\build_image_count_shapley_proxy.py --scene-records results\multiview_analysis\two_drone_operational\scene_view_records.csv --output-dir results\recomputed\marginal_value\image_count_shapley_proxy
```

This script:
- groups images by target instance / scene;
- ignores viewpoint labels after grouping;
- samples random image orderings;
- measures the expected marginal value of the `k`th added image.

Outputs:

- `outputs/image_count_shapley_proxy/image_count_shapley_proxy_report.md`
- `outputs/image_count_shapley_proxy/image_count_shapley_proxy_summary.csv`
- `outputs/image_count_shapley_proxy/image_count_shapley_proxy_by_class.csv`

Interpretation:

- use this when the thesis question is `how many images are worth adding?`;
- use angle-aware Shapley only after that, when the question becomes `which angle is the most valuable teammate?`.

## Exact Shapley Progression By Swarm Size

This analysis examines how the contribution of individual viewpoints changes with coalition size.

Instead of only asking:

`which angle is best?`

the analysis also considers:

`what does the 2nd, 3rd, 4th, ... angle add, and which angle is best at each swarm size?`

Run:

```cmd
python src\multiview_analysis\marginal_value\build_shapley_swarm_size_profiles.py --scene-records results\multiview_analysis\two_drone_operational\scene_view_records.csv --output-dir results\recomputed\marginal_value\shapley_swarm_size_profiles
```

This writes exact size-conditioned Shapley outputs for the controlled 8-azimuth rings:

- `outputs/shapley_swarm_size_profiles/ring_shapley_swarm_size_detail.csv`
- `outputs/shapley_swarm_size_profiles/ring_shapley_swarm_size_summary.csv`
- `outputs/shapley_swarm_size_profiles/aggregate_shapley_swarm_size_summary.csv`
- `outputs/shapley_swarm_size_profiles/shapley_swarm_size_progression_report.md`

Interpretation:

- `aggregate_shapley_swarm_size_summary.csv` answers the number-of-angles question;
- `ring_shapley_swarm_size_summary.csv` answers which azimuth is best when adding the `k`th drone;
- standard ring Shapley still answers the overall teammate question aggregated across all coalition sizes.
## Exact Full-Grid Distributed 8-View Shapley

This analysis computes exact Shapley values for a subset of eight viewpoints using Noisy-OR confidence aggregation and the best available IoU as the coalition value.

Unlike the ring-based analysis, the selected viewpoints can differ in elevation, distance, and azimuth.

### Default viewpoint selection

By default, eight viewpoints are selected across the perimeter of the 3 x 3 elevation-radius grid, covering all eight azimuths.

Run from the repository root:

```cmd
python src\multiview_analysis\marginal_value\compute_subset_shapley_noisy_or_best_iou.py --scene-records results\multiview_analysis\two_drone_operational\scene_view_records.csv --output-dir results\recomputed\marginal_value\subset_shapley
```

### Manual viewpoint selection

A specific set of viewpoints can also be provided using `--selection-mode manual`.

For example:

```cmd
python src\multiview_analysis\marginal_value\compute_subset_shapley_noisy_or_best_iou.py --scene-records results\multiview_analysis\two_drone_operational\scene_view_records.csv --output-dir results\recomputed\marginal_value\manual_subset --selection-mode manual --subset-name my_subset --viewpoints ellow-radnear-az000 ellow-radmid-az045 ellow-radfar-az090 elmid-radfar-az135 elhigh-radfar-az180 elhigh-radmid-az225 elhigh-radnear-az270 elmid-radnear-az315
```

Both configurations use the same coalition-value definition. The difference is which viewpoints are included in the analysis.

Results are written to the specified output directory.


## Outputs

Running the script writes the following to `outputs/`:

- `single_view_landscape.csv`
- `single_view_landscape_by_class.csv`
- `multi_view_gain_summary.csv`
- `gain_capture_targets.csv`
- `pair_complementarity_summary.csv`
- `pair_complementarity_supported_summary.csv`
- `triple_third_view_gain_summary.csv`
- `triple_third_view_gain_supported_summary.csv`
- `viewpoint_marginal_contribution_summary.csv`
- `viewpoint_marginal_contribution_by_class.csv`
- `class_multi_view_gain_summary.csv`
- `best_combinations_by_drone_count.csv`
- `per_class_plot_index.csv`
- `marginal_value_report.md`
- `coalition_size_breakdown/*`
- `image_count_shapley_proxy/*`
- `shapley_swarm_size_profiles/*`

Plots:

- `plots/single_view_landscape_top.png`
- `plots/multi_view_diminishing_returns.png`
- `plots/class_gain_comparison.png`
- `plots/pair_complementarity_top.png`
- `plots/third_view_gain_top.png`
- `plots/viewpoint_marginal_scores_top.png`
- `plots/per_class/*`

## How To Run
## Running the Marginal-Value Analysis

The main analysis script is located at:

`src/multiview_analysis/marginal_value/run_marginal_value_analysis.py`

It combines scene-level viewpoint observations with protocol-level summaries to evaluate single-view performance, multiview gains, diminishing returns, and viewpoint complementarity.

Three input files are required:

* `scene_view_records.csv`
* `protocol_overall_summary.csv`
* `protocol_class_summary.csv`

The scene-level observations are included in this repository. The original protocol-level summary files are not currently included and must be restored to reproduce the complete analysis.

Once all three input files are available, the analysis can be run from the repository root:

```cmd
python src\multiview_analysis\marginal_value\run_marginal_value_analysis.py --scene-records results\multiview_analysis\two_drone_operational\scene_view_records.csv --protocol-overall "PATH_TO_PROTOCOL_OVERALL_CSV" --protocol-class "PATH_TO_PROTOCOL_CLASS_CSV" --output-dir results\recomputed\marginal_value\main_analysis
```

Replace the two `PATH_TO_...` placeholders with the actual locations of the protocol summary files.

Additional parameters include `--top-n`, `--min-pair-support`, and `--min-triple-support`, which control the reported combinations and minimum observation support.

Previously generated results are available in:

`results/multiview_analysis/marginal_value/`

These results can be inspected without rerunning the original analysis.

## Relevant Outputs

The main output files are:

1. `multi_view_gain_summary.csv`
2. `gain_capture_targets.csv`
3. `single_view_landscape.csv`
4. `pair_complementarity_summary.csv`
5. `triple_third_view_gain_summary.csv`
6. `viewpoint_marginal_contribution_summary.csv`

Class-specific results are available in:

1. `class_multi_view_gain_summary.csv`
2. `plots/class_gain_comparison.png`
3. `single_view_landscape_by_class.csv`
4. `viewpoint_marginal_contribution_by_class.csv`
5. `plots/per_class/*`

## Main Interpretation

`A practical viewpoint budget can be chosen by considering the smallest number of observations that captures most of the measured multiview gain. Useful additional views need not be those with the strongest standalone scores.`

