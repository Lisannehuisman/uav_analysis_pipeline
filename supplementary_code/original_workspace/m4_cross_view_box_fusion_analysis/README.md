# M4 Cross-View Box Fusion Analysis

This folder adds the box-level extension that was still missing from the existing `m4_two_drone_operational_analysis` workflow.

The earlier operational analysis already answered:

- `1-of-1`, `1-of-2`, `2-of-2`, `1-of-3`, `2-of-3`, `3-of-3`
- best 2-view and 3-view combinations
- rescue views and viewpoint patterns

What this folder adds is narrower and more box-aware:

- it keeps the fixed `YOLOv8l_M4` detector
- it reuses the saved M4 test predictions and ground truth
- it fuses matched target-box evidence across selected views

## What This Experiment Is

This is a conservative late-fusion benchmark built from the data you actually have now.

Each selected view can contribute a matched target box with:

- confidence
- matched IoU
- strict quality = `confidence * IoU`

Those matched detections are then combined into a single coalition-level quality score for:

- `2-view` combinations
- `3-view` combinations

## What This Experiment Is Not

This folder does not claim:

- camera-calibrated geometric fusion
- weighted box fusion in a shared image plane
- learned multiview fusion
- cross-view 3D object identity reasoning

That boundary matters because the current files only provide per-image detections:

- `outputs/detector_family_comparison/standardized_test_eval/ground_truth/M4_test_gt.json`
- `outputs/detector_family_comparison/standardized_test_eval/predictions/YOLOv8l_M4_test_predictions.json`

They do not include camera extrinsics, calibration, or explicit cross-view correspondences.

## Fusion Policies

The script compares three policies:

- `best_box`
  Baseline: keep the strongest matched target box from the selected views.
- `noisy_or_max_iou`
  Combine support confidences with noisy-OR, then multiply by the best matched IoU.
- `support_weighted_or`
  Combine support confidences with noisy-OR, use mean matched IoU, and weight by the fraction of views that actually support the target.

`support_weighted_or` is the most conservative multiview policy in this folder because it rewards both evidence strength and agreement across views.

## Project Entry Points

Recommended on this Windows project:

```powershell
powershell -ExecutionPolicy Bypass -File .\m4_cross_view_box_fusion_analysis\run_box_fusion_analysis.ps1
```

Direct Python run:

```powershell
.\.venv\Scripts\python.exe .\m4_cross_view_box_fusion_analysis\run_box_fusion_analysis.py
```

Challenge-slice companion export:

```powershell
.\.venv\Scripts\python.exe .\m4_cross_view_box_fusion_analysis\build_challenge_slice_summary.py
```

Optional arguments:

```powershell
powershell -ExecutionPolicy Bypass -File .\m4_cross_view_box_fusion_analysis\run_box_fusion_analysis.ps1 `
  --score-threshold 0.001 `
  --min-pair-support 8 `
  --min-triple-support 6
```

You can also override the inputs explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\m4_cross_view_box_fusion_analysis\run_box_fusion_analysis.ps1 `
  --gt-json outputs\detector_family_comparison\standardized_test_eval\ground_truth\M4_test_gt.json `
  --pred-json outputs\detector_family_comparison\standardized_test_eval\predictions\YOLOv8l_M4_test_predictions.json
```

## Main Outputs

The script writes to `m4_cross_view_box_fusion_analysis/outputs/`.

Main tables:

- `pair_combo_rows.csv`
- `triple_combo_rows.csv`
- `policy_overall_summary.csv`
- `pair_policy_summary.csv`
- `triple_policy_summary.csv`
- `pair_support_weighted_pattern_summary.csv`
- `box_fusion_report.md`
- `challenge_slice_method_summary.csv`
- `challenge_slice_headline_summary.csv`
- `challenge_slice_report.md`

Plots:

- `plots/policy_overall_quality.png`
- `plots/pair_fusion_gain_distribution.png`
- `plots/top_pairs_support_weighted.png`
- `plots/top_triples_support_weighted.png`
- `plots/pair_support_weighted_pattern_heatmap.png`
- `plots/challenge_slice_top5_deltas.png`

## How To Read It

Use the outputs in this order:

1. `policy_overall_summary.csv`
   This tells you whether cross-view evidence accumulation beats the single best matched box on average.
2. `box_fusion_report.md`
   This is the fastest human-readable summary.
3. `pair_policy_summary.csv` and `triple_policy_summary.csv`
   These rank the best viewpoint combinations under each policy.
4. `pair_support_weighted_pattern_summary.csv`
   This shows where the conservative fusion policy gains or loses against the best-box baseline.

## Relationship To The Operational Analysis

These two folders answer different questions:

- `m4_two_drone_operational_analysis`
  Operational availability question: if I have one, two, or three views, how often do I find the target or improve strict target quality?
- `m4_cross_view_box_fusion_analysis`
  Box-level evidence question: when several views support the target, does combining their matched-box evidence improve over simply taking the best single box?

Together, they give you:

- the operational mission story
- the conservative box-level late-fusion story

without claiming geometric multiview fusion that the current files cannot support.
