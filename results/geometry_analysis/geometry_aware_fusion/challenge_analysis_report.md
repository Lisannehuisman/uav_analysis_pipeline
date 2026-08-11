# Challenge Analysis Report

## Purpose

This analysis tests whether the centered-target benchmark becomes more discriminative when evaluation is restricted to the harder tail of the current cache.
Instead of claiming general scene-wide detection difficulty, it isolates target-centric challenge slices that are still honest to the current dataset design.
All numbers in this report are based on ordered primary->secondary pair rows, so they are challenge-slice robustness probes rather than replacements for the scene-balanced summaries used elsewhere.

## Slice Design

- `All primary views`: the full ordered-pair analysis baseline.
- `Weak primary quality (Q1)`: primary views already in the weakest quartile by target strict quality.
- `Small target box (Q1)`: primary views where the projected target box is in the smallest quartile.
- `Low primary IoU (Q1)`: primary views where matched localization is in the weakest quartile.
- `Far primary views`: primary radius is `far`.
- `Hard scenes by mean quality (Q1)`: scenes whose mean single-view quality is in the weakest quartile.
- `Primary miss only`: rescue-only rows where the primary view failed to detect the target.

## Main Reading Rule

The main question is not only which method wins overall, but whether the gap between accumulation-based methods and selection-based methods widens when the primary view is harder.

## Headline Findings

- On `All primary views`, the best accumulation method is `noisy_or_best_iou` at `0.9518`. Its gap over `best_box` is `+0.0419`.
- On `Weak primary quality (Q1)`, the best accumulation method is `noisy_or_best_iou` at `0.8660`. Its gap over `best_box` is `+0.0824`.
- On `Small target box (Q1)`, the best accumulation method is `noisy_or_best_iou` at `0.8846`. Its gap over `best_box` is `+0.0837`.
- On `Far primary views`, the best accumulation method is `noisy_or_best_iou` at `0.9451`. Its gap over `best_box` is `+0.0447`.
- On `Primary miss only`, the best accumulation method is `noisy_or_best_iou` at `0.7591`. In this slice every positive pair score is a rescue event, so several methods collapse to the same secondary-view success ceiling.

## Interpretation

If the accumulation-versus-selection gap grows on the harder slices, then the challenge analysis is doing useful work: it shows where a centered-target benchmark still differentiates multiview policies.
If the top accumulation methods remain almost tied with each other, that should still be interpreted as a small within-family difference rather than evidence that the challenge slices fully remove the ceiling effect.

## Files

- `challenge_slice_method_summary.csv`
- `challenge_slice_headlines.csv`
- `challenge_slice_score_table.csv`
- `challenge_slice_method_heatmap.png`

## Recommended thesis use

Use this analysis as a robustness probe: the benchmark remains target-centric and center-biased, but these slices show whether the main conclusions still hold when the primary observation is less favorable than average.