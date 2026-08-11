# Robust Viewpoint Diversity Analysis

## What This Adds

This analysis strengthens the factor-level pair/triple conclusions using two checks:

1. Bootstrap confidence intervals over scenes, not over raw combinations.
2. Matched-scene pairwise comparisons between relationship types, so each difference is computed only on scenes where both relationship types are available.

This directly addresses the sparse exact-pair/exact-triple support problem.

## Strongest Scene-Normalized Relationship Types

- k=`2`, `azimuth` best type: `diagonal_135` with AP50-95 `0.9306` (95% CI `0.9182` to `0.9420`, `204` scenes)
- k=`2`, `distance` best type: `near_far` with AP50-95 `0.9285` (95% CI `0.9165` to `0.9415`, `194` scenes)
- k=`2`, `elevation` best type: `adjacent_elevation` with AP50-95 `0.9336` (95% CI `0.9216` to `0.9444`, `201` scenes)
- k=`2`, `mixed_diversity` best type: `elevation_only` with AP50-95 `0.9381` (95% CI `0.9234` to `0.9518`, `167` scenes)
- k=`3`, `azimuth` best type: `medium_spread_azimuths` with AP50-95 `0.9495` (95% CI `0.9395` to `0.9591`, `204` scenes)
- k=`3`, `distance` best type: `near_far` with AP50-95 `0.9493` (95% CI `0.9386` to `0.9592`, `194` scenes)
- k=`3`, `elevation` best type: `adjacent_elevation` with AP50-95 `0.9514` (95% CI `0.9415` to `0.9610`, `201` scenes)
- k=`3`, `mixed_diversity` best type: `distance+elevation` with AP50-95 `0.9541` (95% CI `0.9412` to `0.9659`, `137` scenes)

## Matched-Scene Differences

- k=`3`, `mixed_diversity`, `distance+elevation minus distance_only`: mean difference `+0.0464`, 95% CI `+0.0050` to `+0.0935`, common scenes `15`
- k=`3`, `mixed_diversity`, `distance+elevation+azimuth minus distance_only`: mean difference `+0.0373`, 95% CI `+0.0054` to `+0.0737`, common scenes `17`
- k=`2`, `mixed_diversity`, `distance_only minus elevation_only`: mean difference `-0.0360`, 95% CI `-0.0626` to `-0.0114`, common scenes `135`
- k=`3`, `mixed_diversity`, `distance+azimuth minus distance_only`: mean difference `+0.0346`, 95% CI `+0.0036` to `+0.0730`, common scenes `17`
- k=`2`, `mixed_diversity`, `distance+elevation minus distance_only`: mean difference `+0.0335`, 95% CI `+0.0147` to `+0.0561`, common scenes `148`
- k=`2`, `mixed_diversity`, `distance_only minus elevation+azimuth`: mean difference `-0.0310`, 95% CI `-0.0490` to `-0.0143`, common scenes `155`
- k=`2`, `mixed_diversity`, `distance+elevation+azimuth minus distance_only`: mean difference `+0.0306`, 95% CI `+0.0143` to `+0.0500`, common scenes `155`
- k=`2`, `mixed_diversity`, `azimuth_only minus distance_only`: mean difference `+0.0240`, 95% CI `+0.0079` to `+0.0425`, common scenes `152`
- k=`2`, `mixed_diversity`, `distance+azimuth minus distance_only`: mean difference `+0.0217`, 95% CI `+0.0060` to `+0.0395`, common scenes `155`
- k=`2`, `elevation`, `adjacent_elevation minus same_elevation`: mean difference `+0.0122`, 95% CI `+0.0070` to `+0.0179`, common scenes `201`

## Robust Recommendations

- k=`2`, `azimuth=diagonal_135`: mean AP50-95 `0.9306`, CI `0.9182` to `0.9420`, wins `1`, losses `0`
- k=`2`, `azimuth=adjacent_45`: mean AP50-95 `0.9290`, CI `0.9169` to `0.9400`, wins `0`, losses `0`
- k=`2`, `azimuth=quarter_turn_90`: mean AP50-95 `0.9269`, CI `0.9146` to `0.9387`, wins `0`, losses `0`
- k=`2`, `azimuth=opposite_180`: mean AP50-95 `0.9259`, CI `0.9128` to `0.9381`, wins `0`, losses `0`
- k=`2`, `azimuth=same_azimuth`: mean AP50-95 `0.9217`, CI `0.9088` to `0.9339`, wins `0`, losses `1`
- k=`2`, `distance=near_far`: mean AP50-95 `0.9285`, CI `0.9165` to `0.9415`, wins `0`, losses `0`
- k=`2`, `distance=same_radius`: mean AP50-95 `0.9285`, CI `0.9169` to `0.9392`, wins `0`, losses `0`
- k=`2`, `distance=adjacent_radius`: mean AP50-95 `0.9250`, CI `0.9120` to `0.9367`, wins `0`, losses `0`
- k=`2`, `elevation=adjacent_elevation`: mean AP50-95 `0.9336`, CI `0.9216` to `0.9444`, wins `2`, losses `0`
- k=`2`, `elevation=low_high`: mean AP50-95 `0.9276`, CI `0.9140` to `0.9404`, wins `0`, losses `1`
- k=`2`, `elevation=same_elevation`: mean AP50-95 `0.9208`, CI `0.9084` to `0.9327`, wins `0`, losses `1`
- k=`2`, `mixed_diversity=elevation+azimuth`: mean AP50-95 `0.9325`, CI `0.9204` to `0.9440`, wins `3`, losses `0`
- k=`2`, `mixed_diversity=distance+elevation+azimuth`: mean AP50-95 `0.9314`, CI `0.9197` to `0.9432`, wins `3`, losses `0`
- k=`2`, `mixed_diversity=distance+elevation`: mean AP50-95 `0.9288`, CI `0.9155` to `0.9414`, wins `3`, losses `0`
- k=`2`, `mixed_diversity=elevation_only`: mean AP50-95 `0.9381`, CI `0.9234` to `0.9518`, wins `2`, losses `0`
- k=`2`, `mixed_diversity=distance+azimuth`: mean AP50-95 `0.9252`, CI `0.9129` to `0.9367`, wins `1`, losses `3`
- k=`2`, `mixed_diversity=azimuth_only`: mean AP50-95 `0.9202`, CI `0.9063` to `0.9323`, wins `1`, losses `4`
- k=`2`, `mixed_diversity=distance_only`: mean AP50-95 `0.8987`, CI `0.8763` to `0.9198`, wins `0`, losses `6`

## Thesis Interpretation

Use this robustness layer to support claims about relationship patterns, not exact viewpoint identities. If a relationship type has a high scene-normalized mean but its matched-scene CI overlaps competing types, phrase the conclusion as a tendency rather than a statistically decisive result.

A strong thesis-safe claim is: evaluate exact top pairs/triples as candidates, but base general swarm-design guidance on relationship types that remain strong under scene-normalized bootstrap and matched-scene checks.

## Generated Files

- `scene_relationship_metric_table.csv`
- `relationship_bootstrap_ci.csv`
- `matched_scene_relationship_differences.csv`
- `common_scene_axis_summary.csv`
- `robust_relationship_recommendations.csv`
- `plots/bootstrap_relationship_ci.png`
- `plots/matched_scene_differences.png`
