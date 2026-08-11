# Full M4 vs Single-View vs Pair-View vs 1-of-2/1-of-3

## What This Comparison Does

- This analysis places the main training-side baselines and the operational multiview protocols in one summary.
- Left side of the story: detector generalization after training on different viewpoint subsets.
- Right side of the story: target-centric multiview performance when the detector is held fixed and extra views are made available at inference time.

## Important Interpretation Boundary

- `Full M4`, `single-view`, and `pair-view` are per-image detector evaluations on the full fixed M4 test split.
- `1-of-1`, `1-of-2`, and `1-of-3` are operational target-centric protocols built on fixed YOLOv8l_M4 predictions.
- These two panels should therefore be compared for trend and takeaway, not as identical metrics.

## Training-Side Comparison

- Full M4 baseline: `mAP50-95 = 0.6396`, `mAP50 = 0.8367`, `F1 = 0.8457`
- Mean single-view model (72 runs): `mAP50-95 = 0.3384`, gap to full M4 `-0.3012`
- Best single-view model (`elmid-radmid-az045`): `mAP50-95 = 0.4164`, gap to full M4 `-0.2232`
- Equal-budget M4 for single-view mean (1 run(s), `train=144`, `val=31`): `mAP50-95 = 0.4843`, gap to single-view mean `-0.1460`
- Equal-budget M4 for best single (1 run(s), `train=135`, `val=35`): `mAP50-95 = 0.4803`, gap to best single `-0.0638`
- Mean pair-view model (2535 / 2556 completed): `mAP50-95 = 0.4207`, gap to full M4 `-0.2190`
- Best pair-view model (`ellow-radmid-az000` + `elmid-radmid-az225`): `mAP50-95 = 0.4958`, gap to full M4 `-0.1438`, lift over best single `+0.0794`
- Equal-budget M4 for pair-view mean (1 run(s), `train=287`, `val=61`): `mAP50-95 = 0.5234`, gap to pair-view mean `-0.1028`
- Equal-budget M4 for best pair (1 run(s), `train=298`, `val=59`): `mAP50-95 = 0.5196`, gap to best pair `-0.0238`

## Operational Comparison

- 1-of-1: target found `0.9811`, target AP50-95 `0.8516`, strict quality `0.8732`
- 1-of-2 OR: target found `0.9986`, target AP50-95 `0.9277` (`+0.0761` vs 1-of-1), strict quality `0.9088` (`+0.0357`)
- 1-of-3 OR: target found `0.9999`, target AP50-95 `0.9487` (`+0.0971` vs 1-of-1), strict quality `0.9189` (`+0.0457`)

## Main Takeaway

- The equal-count M4 controls are the fairer headline comparison when the question is about viewpoint diversity at fixed image budget.
- Use the equal-budget M4 gap to judge whether restricted-view training is still weaker than an equally sized diverse-view subset.
- Full M4 remains the strongest training-side detector baseline.
- Pair-view training clearly improves over single-view training, but it still does not close the gap to the strongest diverse-view baseline.
- Extra views at inference time help a lot under 1-of-2 and 1-of-3 OR protocols, but that is a different kind of gain than retraining on richer viewpoint subsets.
- The clean story is therefore: viewpoint diversity helps both in training and at inference, but those gains appear through different mechanisms.

## Generated Files

- `training_side_summary.csv`
- `operational_side_summary.csv`
- `comparison_summary.md`
- `plots/training_side_comparison.png`
- `fair_training_side_table.csv`
- `fair_training_side_table.md`
- `plots/fair_training_side_comparison.png`
- `plots/operational_side_comparison.png`
- `plots/headline_comparison_dashboard.png`
