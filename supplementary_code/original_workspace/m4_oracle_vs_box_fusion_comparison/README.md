# M4 Current Method vs Late Fusion Comparison

This folder compares:

- the current `best available / oracle max` multiview evaluation
- the existing late-fusion policies from `m4_cross_view_box_fusion_analysis`

The comparison is done on the same scene-balanced basis.

## Why This Exists

The current operational multiview analysis is useful as an information-availability benchmark, but it is optimistic:

- it measures what useful target evidence is present somewhere in the chosen views
- it does not require a deployable fusion rule to recover that evidence

The box-fusion folder is more deployment-oriented:

- `noisy_or_max_iou`
- `support_weighted_or`

`best_box` is intentionally omitted from this comparison project's outputs because it is exactly identical to the current/oracle score on this dataset, so keeping both would only duplicate the same line in the tables and plots.

This comparison lets you see the gap between:

- `what was available in principle`
- `what a concrete late-fusion policy extracts`

## Inputs

- `m4_two_drone_operational_analysis/thesis_swarm_outputs/protocol_scene_expectation_summary.csv`
- `m4_cross_view_box_fusion_analysis/outputs/pair_combo_rows.csv`
- `m4_cross_view_box_fusion_analysis/outputs/triple_combo_rows.csv`

## Outputs

- `matched_scene_policy_comparison.csv`
- `overall_policy_comparison.csv`
- `class_policy_comparison.csv`
- `oracle_vs_box_fusion_report.md`

Plots:

- `plots/overall_policy_comparison.png`
- `plots/per_class_policy_comparison_k2.png`
- `plots/per_class_policy_comparison_k3.png`

## How To Run

```powershell
.\.venv\Scripts\python.exe .\m4_oracle_vs_box_fusion_comparison\compare_oracle_to_box_fusion.py
```
