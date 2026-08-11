# Clutter Grouping Analysis

## Purpose

This analysis tests whether clutter groupings, operationalized as non-target objects in view, are associated with target detection quality and whether the clutter groups differ significantly.

## Inputs

- Geometry table: `results/geometry_analysis/ground_truth/view_geometry_table.csv`
- Scene detection table: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`

## Operationalization

- View-level clutter is defined as `num_label_boxes - num_target_class_boxes`.
- View-level bins: `0`, `1-4`, `5-9`, `10+` distractors.
- Scene-level clutter uses the mean distractor count across evaluated views of a scene.
- Significance testing is done at the scene level to avoid treating multiple correlated views from the same scene as fully independent.

## Scene clutter quartile cutoffs

- Q1 cutoff: `4.0000` mean distractors
- Median cutoff: `5.3333` mean distractors
- Q3 cutoff: `7.2727` mean distractors

## Dataset coverage

- Joined evaluation views: `2214`
- Unique evaluated scenes: `205`

## Headline findings

- Strongest scene-level monotonic association: `Target AP50-95` with Spearman rho `-0.325` and p `<0.0001`.
- Mean target AP50-95 falls from `0.8927` in `Q1 low clutter` to `0.8038` in `Q4 high clutter`.
- `2/4` scene-level omnibus tests are significant at `p < 0.05`.

## View-level clutter summary

- Group `0`: `n=190`, mean AP50-95 `0.9126`, mean strict quality `0.9052`, mean detected rate `1.0000`.
- Group `1-4`: `n=1038`, mean AP50-95 `0.8774`, mean strict quality `0.8734`, mean detected rate `0.9769`.
- Group `5-9`: `n=585`, mean AP50-95 `0.8594`, mean strict quality `0.8809`, mean detected rate `0.9863`.
- Group `10+`: `n=401`, mean AP50-95 `0.7505`, mean strict quality `0.8511`, mean detected rate `0.9776`.

## Scene-level group summary

- `Q1 low clutter`: `n=53` scenes, mean clutter `3.2828`, mean AP50-95 `0.8927`, mean strict quality `0.9052`, mean detected rate `0.9917`.
- `Q2 mid-low clutter`: `n=52` scenes, mean clutter `4.7796`, mean AP50-95 `0.8675`, mean strict quality `0.8792`, mean detected rate `0.9772`.
- `Q3 mid-high clutter`: `n=49` scenes, mean clutter `6.0879`, mean AP50-95 `0.8402`, mean strict quality `0.8691`, mean detected rate `0.9762`.
- `Q4 high clutter`: `n=51` scenes, mean clutter `9.4917`, mean AP50-95 `0.8038`, mean strict quality `0.8375`, mean detected rate `0.9787`.

## Correlations

- `Target AP50-95`: Spearman rho `-0.325`, p `<0.0001`.
- `Target Strict Quality`: Spearman rho `-0.168`, p `0.0157`.
- `Target Match Confidence`: Spearman rho `-0.141`, p `0.0439`.
- `Target Detected Rate`: Spearman rho `-0.076`, p `0.2775`.

## Omnibus tests across scene clutter quartiles

- `Target AP50-95`: Kruskal-Wallis H `22.792`, p `<0.0001`.
- `Target Strict Quality`: Kruskal-Wallis H `7.889`, p `0.0484`.
- `Target Match Confidence`: Kruskal-Wallis H `5.922`, p `0.1155`.
- `Target Detected Rate`: Kruskal-Wallis H `4.170`, p `0.2436`.

## Pairwise scene-group differences after Holm correction

- `Target AP50-95`: `Q1 low clutter` vs `Q4 high clutter`, Holm-adjusted p `<0.0001`, mean delta `0.0889`, Cliff's delta `0.507`.
- `Target AP50-95`: `Q2 mid-low clutter` vs `Q4 high clutter`, Holm-adjusted p `0.0033`, mean delta `0.0637`, Cliff's delta `0.390`.
- `Target AP50-95`: `Q1 low clutter` vs `Q3 mid-high clutter`, Holm-adjusted p `0.0301`, mean delta `0.0525`, Cliff's delta `0.308`.

