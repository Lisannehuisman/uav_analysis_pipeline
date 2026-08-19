# M4 Two-Drone Operational Analysis

This folder applies `method 2`: keep the detector fixed and compare what happens when a scene is observed with `1 drone` versus `2 drones`.

## What This Folder Answers

- Does a second simultaneous viewpoint improve the chance that the intended target object is found?
- Which single viewpoints are strongest on their own?
- Which viewpoint pairs are strongest together?
- Which second-drone viewpoints most often rescue a miss from the primary drone?
- Are some factor levels or azimuth gaps more useful than others?

## What This Folder Does Not Claim

- It is **not** a retraining experiment.
- It does **not** test multi-view fusion or a special swarm model.
- It does **not** deduplicate the same real-world object across two images for the general COCO-style image metrics.

The folder now exposes two target-centric layers:

- `binary target found rate`
  This remains as a reference and answers whether the target class was found at least once.
- `strict target-quality metrics`
  These are the new headline metrics and answer how well the target was detected, not just whether it was found.

The stricter target metrics are:

- `target_match_confidence_iou50`
  The detector confidence of the strongest correct target detection at IoU `>= 0.50`.
- `target_ap50_95`
  Target-only AP50-95 for the filename target class in that view.
- `target_strict_quality_iou50`
  A stricter composite score: matched confidence multiplied by matched IoU.

Operationally:

- a `1-drone` mission uses one available viewpoint from that scene
- a `2-drone` mission uses two available viewpoints from that same scene
- the stricter two-drone summaries use the best available target quality across the selected views

This gives a direct operational answer to:

`does having a second viewpoint help me detect the intended object better, and from which angles?`

## Data Sources

By default the script uses:

- `outputs/detector_family_comparison/standardized_test_eval/ground_truth/M4_test_gt.json`
- `outputs/detector_family_comparison/standardized_test_eval/predictions/YOLOv8l_M4_test_predictions.json`

## Main Outputs

After running the script, the `outputs/` folder contains:

- `scene_view_records.csv`
  One row per test image/viewpoint with overall image metrics, binary target flags, and strict target-quality fields.
- `single_viewpoint_summary.csv`
  Absolute single-view viewpoint ranking.
- `pair_viewpoint_summary.csv`
  Absolute two-view pair ranking.
- `second_drone_viewpoint_rescue_summary.csv`
  Which second viewpoints most often rescue a miss.
- `scene_expectation_summary.csv`
  Scene-level expected 1-drone and 2-drone performance.
- `overall_one_vs_two_summary.csv`
  Headline expected comparison.
- `class_level_one_vs_two_summary.csv`
  Which object classes gain the most from a second viewpoint.
- `two_drone_operational_report.md`
  Human-readable summary.

## How To Run

```powershell
python .\m4_two_drone_operational_analysis\analyze_two_drone_operational.py
```

Optional flags:

```powershell
python .\m4_two_drone_operational_analysis\analyze_two_drone_operational.py `
  --score-threshold 0.001 `
  --min-single-support 10 `
  --min-pair-support 8 `
  --min-rescue-support 10
```

## Interpretation Guide

Use the outputs in this order:

1. `overall_one_vs_two_summary.csv`
   This answers whether 2 viewpoints beat 1 on average under the stricter target-quality metrics.
2. `class_level_one_vs_two_summary.csv`
   This shows which object classes benefit most in strict target quality.
3. `single_viewpoint_summary.csv`
   This shows which single viewpoints are strongest for the target object.
4. `pair_viewpoint_summary.csv`
   This shows which viewpoint pairs are strongest together.
5. `second_drone_viewpoint_rescue_summary.csv`
   This shows which angles are best as the added second view.

## Key Limitation

The stricter target metrics are still `target-class` metrics, not unique 3D object identity metrics across views. They are much more informative than plain found-or-not, but they still do not deduplicate identical real-world objects across multiple views.

## Thesis-Style Swarm Extension

For a fuller `1 vs 2 vs 3 drone` swarm study, including `OR`, `confirmation`, and `unanimous` protocols plus data-readiness checks, run:

```powershell
python .\m4_two_drone_operational_analysis\run_swarm_thesis_analysis.py
```

That expanded pipeline writes to:

- `m4_two_drone_operational_analysis/thesis_swarm_outputs/`

Key extra outputs include:

- `data_readiness_summary.csv`
- `protocol_overall_summary.csv`
- `protocol_class_summary.csv`
- `pair_protocol_summary.csv`
- `triple_protocol_summary.csv`
- `second_drone_rescue_summary.csv`
- `third_drone_or_rescue_summary.csv`
- `third_drone_confirmation_upgrade_summary.csv`
- `swarm_thesis_report.md`
