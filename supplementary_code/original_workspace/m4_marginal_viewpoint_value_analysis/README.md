# M4 Marginal Viewpoint Value Analysis

This folder applies your newer thesis framing directly to the outputs that already exist in this project.

The question is no longer only:

`does one angle work better than another?`

It becomes:

`if multiple UAV views are available, how much genuinely new information does each extra view still add, and when do the returns start to flatten?`

## What This Folder Adds

This folder turns the existing fixed-detector M4 swarm outputs into three thesis-ready layers:

- `single-view landscape`
- `multi-view gain`
- `complementarity`
- `swarm-size-conditioned Shapley`

## Why This Fits The Current Project

This analysis reuses the outputs you already generated:

- `m4_two_drone_operational_analysis/outputs/scene_view_records.csv`
- `m4_two_drone_operational_analysis/thesis_swarm_outputs/protocol_overall_summary.csv`
- `m4_two_drone_operational_analysis/thesis_swarm_outputs/protocol_class_summary.csv`

That keeps the main story aligned with your stronger thesis route:

- fixed detector
- inference-time multiview analysis
- no need to retrain a new detector for every possible viewpoint subset

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

If you need the exact late-fusion teammate story, use:

- `compute_ring_shapley_noisy_or_best_iou.py` for ring-level exact Shapley
- `compute_subset_shapley_noisy_or_best_iou.py` for exact Shapley on a custom or full-grid-distributed 8-view subset
- `build_shapley_swarm_size_profiles.py` for exact Shapley decomposed by coalition size / swarm size
- `build_noisy_or_coalition_size_breakdown.py` for coalition-value growth by number of drones
- `build_image_count_shapley_proxy.py` for an angle-agnostic count-first Shapley-style proxy
- `ring_shapley_noisy_or_best_iou_conditional_gain_report.md` for exact `Delta(u | C)` next-view gains inside the same ring game

## Angle-Agnostic Count-First Shapley Proxy

This is the best bridge to raw drone imagery when exact angles are not yet trusted.

It answers:

`what is the marginal contribution of the 2nd, 3rd, 4th, ... image, even if we ignore angle labels?`

Run:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\build_image_count_shapley_proxy.py
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

This is the bridge to your supervisor's new framing.

Instead of only asking:

`which angle is best?`

you can now ask:

`what does the 2nd, 3rd, 4th, ... angle add, and which angle is best at each swarm size?`

Run:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\build_shapley_swarm_size_profiles.py
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

If you want the same exact `Noisy-OR + best IoU` Shapley analysis, but not constrained
to one ring, run:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\compute_subset_shapley_noisy_or_best_iou.py
```

By default this selects 8 viewpoints spread across the outer perimeter of the
full `3 x 3 x 8` M4 grid:

- one viewpoint from each perimeter `(elevation, radius)` cell
- all 8 azimuths used exactly once

You can also pass a manual subset:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\compute_subset_shapley_noisy_or_best_iou.py `
  --selection-mode manual `
  --subset-name my_subset `
  --viewpoints `
    ellow-radnear-az000 `
    ellow-radmid-az045 `
    ellow-radfar-az090 `
    elmid-radfar-az135 `
    elhigh-radfar-az180 `
    elhigh-radmid-az225 `
    elhigh-radnear-az270 `
    elmid-radnear-az315
```

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

From the project root:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\run_marginal_value_analysis.py
```

Optional:

```powershell
.\.venv\Scripts\python.exe .\m4_marginal_viewpoint_value_analysis\run_marginal_value_analysis.py `
  --top-n 20 `
  --min-pair-support 8 `
  --min-triple-support 3 `
  --output-dir .\m4_marginal_viewpoint_value_analysis\outputs
```

## How To Use It In The Thesis

Use the outputs in this order:

1. `multi_view_gain_summary.csv`
2. `gain_capture_targets.csv`
3. `single_view_landscape.csv`
4. `pair_complementarity_summary.csv`
5. `triple_third_view_gain_summary.csv`
6. `viewpoint_marginal_contribution_summary.csv`

For class-specific interpretation, use:

1. `class_multi_view_gain_summary.csv`
2. `plots/class_gain_comparison.png`
3. `single_view_landscape_by_class.csv`
4. `viewpoint_marginal_contribution_by_class.csv`
5. `plots/per_class/*`

## Intended Thesis Claim

`the best swarm size is the smallest number of viewpoints that captures most of the available multi-view gain, and the best viewpoints are the ones with the largest marginal contribution rather than merely the strongest standalone score.`
