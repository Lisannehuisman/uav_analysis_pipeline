# Revised Real-World Multi-View Shapley Analysis

## Scope

- Uses exactly the five same-object real UAV sequences specified by the user.
- Duplicate filenames inside the supplied object lists are removed before analysis and are not counted as separate players.
- Coalitions are formed only within the same physical object instance.
- Target-class mapping: `parkinglot_car` -> `suv`, `ooij_tower` -> `tower`, and truck objects -> `whitevan`.
- Target detection rule: correct target class, IoU >= 0.50, and confidence >= 0.25.
- The Ooij tower image `2026-06-29 17.35.32.jpg` remains excluded because its label file has no tower ground-truth target.
- No model retraining is performed; the analysis reuses the existing per-view prediction table when `--skip-predict` is used.

## Object Groups

| Object | Target class | Unique listed views | Usable target views |
| --- | --- | ---: | ---: |
| `parkinglot_car` | `suv` | 6 | 6 |
| `ooij_tower` | `tower` | 8 | 7 |
| `la_souris_truck` | `whitevan` | 10 | 10 |
| `m_truck` | `whitevan` | 5 | 5 |
| `white_truck_bottendaal` | `whitevan` | 7 | 7 |

## Target-GT Exclusions

- `ooij_tower` view 2 (`2026-06-29 17.35.32.jpg`) has no `tower` target box and is excluded from Shapley games.

## Analysis A: Coalition-Size Progression

Question: how much does performance improve, on average, when moving from one to two to three UAV views?

Values are macro-averaged over physical object instances, so the 10-view object does not dominate the 5-view object.

| Model | Views | Object support | Detection | Gain detection | Best strict quality | Gain strict quality | Noisy-OR x best IoU | Gain Noisy-OR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real UAV fine-tuned | 1 | 5 | 0.960 | 0.960 | 0.696 | 0.696 | 0.696 | 0.696 |
| Real UAV fine-tuned | 2 | 5 | 0.996 | 0.036 | 0.778 | 0.082 | 0.859 | 0.163 |
| Real UAV fine-tuned | 3 | 5 | 1.000 | 0.004 | 0.812 | 0.034 | 0.912 | 0.053 |
| Synthetic M4 only | 1 | 5 | 0.080 | 0.080 | 0.046 | 0.046 | 0.046 | 0.046 |
| Synthetic M4 only | 2 | 5 | 0.133 | 0.053 | 0.080 | 0.034 | 0.082 | 0.036 |
| Synthetic M4 only | 3 | 5 | 0.167 | 0.033 | 0.105 | 0.025 | 0.110 | 0.028 |

## Analysis B: Exact Shapley Viewpoint Attribution

Question: across all possible same-object view combinations, which individual viewpoints account for coalition value?

For an object with `n` usable views, every subset of those views is enumerated exactly. The empty coalition has value 0, and each view receives the standard factorial-weighted Shapley value over predecessor coalitions of size 0 through `n-1`.

Value functions:

- Primary: `best strict target quality`, defined as the maximum strict target-quality score in the coalition.
- Detection/rescue: 1 if any view correctly detects the target, otherwise 0.
- Secondary sensitivity metric: `Noisy-OR x best IoU`. This is a heuristic fusion score. YOLO confidence is not treated as a calibrated probability, and the Noisy-OR component naturally tends to increase when additional positive-confidence observations are added.

## Shapley Efficiency Validation

Each row checks that the sum of view Shapley values equals the grand-coalition value. The script asserts an absolute error below `1e-8`; it also asserts no duplicate filenames remain, null views receive zero contribution, symmetric views receive equal values, and the coalition count equals `2^n`.

| Model | Object | Value function | Views | Expected coalitions | Enumerated | Grand value | Sum Shapley | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real UAV fine-tuned | `la_souris_truck` | `detection` | 10 | 1024 | 1024 | 1.000000 | 1.000000 | 0.0000000000 |
| Real UAV fine-tuned | `la_souris_truck` | `noisy_or_best_iou` | 10 | 1024 | 1024 | 0.964499 | 0.964499 | 0.0000000000 |
| Real UAV fine-tuned | `la_souris_truck` | `strict_quality` | 10 | 1024 | 1024 | 0.859941 | 0.859941 | 0.0000000000 |
| Real UAV fine-tuned | `m_truck` | `detection` | 5 | 32 | 32 | 1.000000 | 1.000000 | 0.0000000000 |
| Real UAV fine-tuned | `m_truck` | `noisy_or_best_iou` | 5 | 32 | 32 | 0.956009 | 0.956009 | 0.0000000000 |
| Real UAV fine-tuned | `m_truck` | `strict_quality` | 5 | 32 | 32 | 0.844995 | 0.844995 | 0.0000000000 |
| Real UAV fine-tuned | `ooij_tower` | `detection` | 7 | 128 | 128 | 1.000000 | 1.000000 | 0.0000000000 |
| Real UAV fine-tuned | `ooij_tower` | `noisy_or_best_iou` | 7 | 128 | 128 | 0.945559 | 0.945559 | 0.0000000000 |
| Real UAV fine-tuned | `ooij_tower` | `strict_quality` | 7 | 128 | 128 | 0.801366 | 0.801366 | 0.0000000000 |
| Real UAV fine-tuned | `parkinglot_car` | `detection` | 6 | 64 | 64 | 1.000000 | 1.000000 | 0.0000000000 |
| Real UAV fine-tuned | `parkinglot_car` | `noisy_or_best_iou` | 6 | 64 | 64 | 0.927744 | 0.927744 | 0.0000000000 |
| Real UAV fine-tuned | `parkinglot_car` | `strict_quality` | 6 | 64 | 64 | 0.788678 | 0.788678 | 0.0000000000 |
| Real UAV fine-tuned | `white_truck_bottendaal` | `detection` | 7 | 128 | 128 | 1.000000 | 1.000000 | 0.0000000000 |
| Real UAV fine-tuned | `white_truck_bottendaal` | `noisy_or_best_iou` | 7 | 128 | 128 | 0.978601 | 0.978601 | 0.0000000000 |
| Real UAV fine-tuned | `white_truck_bottendaal` | `strict_quality` | 7 | 128 | 128 | 0.929118 | 0.929118 | 0.0000000000 |
| Synthetic M4 only | `la_souris_truck` | `detection` | 10 | 1024 | 1024 | 1.000000 | 1.000000 | 0.0000000000 |
| Synthetic M4 only | `la_souris_truck` | `noisy_or_best_iou` | 10 | 1024 | 1024 | 0.869880 | 0.869880 | 0.0000000000 |
| Synthetic M4 only | `la_souris_truck` | `strict_quality` | 10 | 1024 | 1024 | 0.845653 | 0.845653 | 0.0000000000 |
| Synthetic M4 only | `m_truck` | `detection` | 5 | 32 | 32 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `m_truck` | `noisy_or_best_iou` | 5 | 32 | 32 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `m_truck` | `strict_quality` | 5 | 32 | 32 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `ooij_tower` | `detection` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `ooij_tower` | `noisy_or_best_iou` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `ooij_tower` | `strict_quality` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `parkinglot_car` | `detection` | 6 | 64 | 64 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `parkinglot_car` | `noisy_or_best_iou` | 6 | 64 | 64 | 0.000744 | 0.000744 | 0.0000000000 |
| Synthetic M4 only | `parkinglot_car` | `strict_quality` | 6 | 64 | 64 | 0.000744 | 0.000744 | 0.0000000000 |
| Synthetic M4 only | `white_truck_bottendaal` | `detection` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `white_truck_bottendaal` | `noisy_or_best_iou` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |
| Synthetic M4 only | `white_truck_bottendaal` | `strict_quality` | 7 | 128 | 128 | 0.000000 | 0.000000 | 0.0000000000 |

## Thesis-Level Shapley Summary

Main metric: exact Shapley value for strict target quality.

| Model | Object | View | Single-view quality | Exact Shapley value | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| Real UAV fine-tuned | `la_souris_truck` | 1 | 0.000 | 0.000 | 10 |
| Real UAV fine-tuned | `la_souris_truck` | 2 | 0.018 | 0.002 | 9 |
| Real UAV fine-tuned | `la_souris_truck` | 3 | 0.607 | 0.085 | 6 |
| Real UAV fine-tuned | `la_souris_truck` | 4 | 0.860 | 0.161 | 1 |
| Real UAV fine-tuned | `la_souris_truck` | 5 | 0.851 | 0.152 | 2 |
| Real UAV fine-tuned | `la_souris_truck` | 6 | 0.447 | 0.059 | 7 |
| Real UAV fine-tuned | `la_souris_truck` | 7 | 0.797 | 0.124 | 3 |
| Real UAV fine-tuned | `la_souris_truck` | 8 | 0.786 | 0.121 | 4 |
| Real UAV fine-tuned | `la_souris_truck` | 9 | 0.786 | 0.121 | 5 |
| Real UAV fine-tuned | `la_souris_truck` | 10 | 0.285 | 0.035 | 8 |
| Real UAV fine-tuned | `m_truck` | 1 | 0.845 | 0.193 | 1 |
| Real UAV fine-tuned | `m_truck` | 2 | 0.709 | 0.142 | 5 |
| Real UAV fine-tuned | `m_truck` | 3 | 0.838 | 0.186 | 2 |
| Real UAV fine-tuned | `m_truck` | 4 | 0.828 | 0.181 | 3 |
| Real UAV fine-tuned | `m_truck` | 5 | 0.712 | 0.143 | 4 |
| Real UAV fine-tuned | `ooij_tower` | 1 | 0.505 | 0.090 | 4 |
| Real UAV fine-tuned | `ooij_tower` | 3 | 0.246 | 0.035 | 7 |
| Real UAV fine-tuned | `ooij_tower` | 4 | 0.376 | 0.057 | 5 |
| Real UAV fine-tuned | `ooij_tower` | 5 | 0.653 | 0.139 | 3 |
| Real UAV fine-tuned | `ooij_tower` | 6 | 0.754 | 0.189 | 2 |
| Real UAV fine-tuned | `ooij_tower` | 7 | 0.801 | 0.237 | 1 |
| Real UAV fine-tuned | `ooij_tower` | 8 | 0.360 | 0.054 | 6 |
| Real UAV fine-tuned | `parkinglot_car` | 1 | 0.789 | 0.158 | 1 |
| Real UAV fine-tuned | `parkinglot_car` | 2 | 0.756 | 0.128 | 4 |
| Real UAV fine-tuned | `parkinglot_car` | 3 | 0.745 | 0.126 | 5 |
| Real UAV fine-tuned | `parkinglot_car` | 4 | 0.760 | 0.130 | 3 |
| Real UAV fine-tuned | `parkinglot_car` | 5 | 0.707 | 0.118 | 6 |
| Real UAV fine-tuned | `parkinglot_car` | 6 | 0.761 | 0.130 | 2 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 1 | 0.929 | 0.158 | 1 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 2 | 0.920 | 0.149 | 2 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 3 | 0.786 | 0.112 | 7 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 4 | 0.867 | 0.128 | 4 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 5 | 0.837 | 0.121 | 6 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 6 | 0.843 | 0.122 | 5 |
| Real UAV fine-tuned | `white_truck_bottendaal` | 7 | 0.899 | 0.138 | 3 |
| Synthetic M4 only | `la_souris_truck` | 1 | 0.380 | 0.095 | 4 |
| Synthetic M4 only | `la_souris_truck` | 2 | 0.636 | 0.213 | 2 |
| Synthetic M4 only | `la_souris_truck` | 3 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 4 | 0.438 | 0.114 | 3 |
| Synthetic M4 only | `la_souris_truck` | 5 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 6 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 7 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 8 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 9 | 0.000 | 0.000 | 5 |
| Synthetic M4 only | `la_souris_truck` | 10 | 0.846 | 0.423 | 1 |
| Synthetic M4 only | `m_truck` | 1 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `m_truck` | 2 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `m_truck` | 3 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `m_truck` | 4 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `m_truck` | 5 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 1 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 3 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 4 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 5 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 6 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 7 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `ooij_tower` | 8 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `parkinglot_car` | 1 | 0.000 | 0.000 | 2 |
| Synthetic M4 only | `parkinglot_car` | 2 | 0.001 | 0.001 | 1 |
| Synthetic M4 only | `parkinglot_car` | 3 | 0.000 | 0.000 | 2 |
| Synthetic M4 only | `parkinglot_car` | 4 | 0.000 | 0.000 | 2 |
| Synthetic M4 only | `parkinglot_car` | 5 | 0.000 | 0.000 | 2 |
| Synthetic M4 only | `parkinglot_car` | 6 | 0.000 | 0.000 | 2 |
| Synthetic M4 only | `white_truck_bottendaal` | 1 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 2 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 3 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 4 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 5 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 6 | 0.000 | 0.000 | 1 |
| Synthetic M4 only | `white_truck_bottendaal` | 7 | 0.000 | 0.000 | 1 |

## Highest Strict-Quality Shapley Views

| Model | Object | Highest-ranked view(s) | Exact Shapley value |
| --- | --- | --- | ---: |
| Real UAV fine-tuned | `la_souris_truck` | v4 `2026-07-12 21.01.23.jpg` | 0.161 |
| Real UAV fine-tuned | `m_truck` | v1 `2026-07-12 21.22.14.jpg` | 0.193 |
| Real UAV fine-tuned | `ooij_tower` | v7 `2026-06-29 17.36.51.jpg` | 0.237 |
| Real UAV fine-tuned | `parkinglot_car` | v1 `2026-06-28 20.56.45.jpg` | 0.158 |
| Real UAV fine-tuned | `white_truck_bottendaal` | v1 `2026-06-30 15.33.46.jpg` | 0.158 |
| Synthetic M4 only | `la_souris_truck` | v10 `2026-07-12 21.05.27.jpg` | 0.423 |
| Synthetic M4 only | `m_truck` | v1 `2026-07-12 21.22.14.jpg`, v2 `2026-07-12 21.21.43.jpg`, v3 `2026-07-12 21.21.46.jpg`, v4 `2026-07-12 21.21.47.jpg`, v5 `2026-07-12 21.22.11.jpg` | 0.000 |
| Synthetic M4 only | `ooij_tower` | v1 `2026-06-29 17.37.15.jpg`, v3 `2026-06-29 17.36.06.jpg`, v4 `2026-06-29 17.36.07.jpg`, v5 `2026-06-29 17.36.40.jpg`, v6 `2026-06-29 17.36.50.jpg`, v7 `2026-06-29 17.36.51.jpg`, v8 `2026-06-29 17.36.53.jpg` | 0.000 |
| Synthetic M4 only | `parkinglot_car` | v2 `2026-06-28 20.56.46.jpg` | 0.001 |
| Synthetic M4 only | `white_truck_bottendaal` | v1 `2026-06-30 15.33.46.jpg`, v2 `2026-06-30 15.33.52.jpg`, v3 `2026-06-30 15.34.24.jpg`, v4 `2026-06-30 15.34.33.jpg`, v5 `2026-06-30 15.35.01.jpg`, v6 `2026-06-30 15.33.29.jpg`, v7 `2026-06-30 15.33.34.jpg` | 0.000 |

## Fine-Tuned Versus Synthetic-Only Comparison

| Model | Macro grand detection | Macro grand strict quality | Macro grand Noisy-OR x best IoU | Near-zero strict-Shapley views |
| --- | ---: | ---: | ---: | ---: |
| Real UAV fine-tuned | 1.000 | 0.845 | 0.954 | 1 / 35 |
| Synthetic M4 only | 0.200 | 0.169 | 0.174 | 30 / 35 |

## Interpretation

- Adding a second real UAV view improves macro detection by `0.036`, strict quality by `0.082`, and Noisy-OR x best IoU by `0.163`.
- Adding a third real UAV view gives smaller gains: detection `0.004`, strict quality `0.034`, and Noisy-OR x best IoU `0.053`.
- Viewpoints do not contribute equally: exact strict-quality Shapley values vary by object, with some views carrying most of the grand-coalition quality and null/weak views contributing approximately zero.
- The descriptive Pearson correlation between single-view strict quality and exact strict-quality Shapley contribution for the real fine-tuned model is `0.869` across 35 views. This is descriptive only; the independent experimental unit is the physical object instance, not each view or coalition.
- Fine-tuning changes both the absolute detection quality and the usefulness of multi-view attribution. Synthetic-only Shapley attribution is limited when the detector fails under the synthetic-to-real domain gap.
- These results should not be framed as statistically significant population estimates, because there are only five independent physical object instances.

## Suggested Results Paragraph

Across five real UAV object instances with multiple same-object viewpoints, the fine-tuned detector showed a clear multi-view benefit in the coalition-size analysis. Macro-averaged detection was already high with one view, but adding a second view increased both detection and target quality, while the third view produced smaller additional gains, indicating diminishing returns. Exact all-view Shapley attribution showed that the contribution of individual viewpoints was uneven: for several objects, a small number of views accounted for most of the strict target-quality value, while weak or failed views contributed little or nothing. The synthetic-only model produced much lower grand-coalition quality and many near-zero viewpoint contributions, so its Shapley values are mainly evidence of limited synthetic-to-real transfer rather than reliable viewpoint usefulness. Overall, the experiment supports the thesis distinction between synthetic controlled viewpoint analysis, real-world fine-tuning for transfer, and a limited but measurable additional benefit from combining real UAV viewpoints. Because the independent sample consists of only five physical objects, these findings should be interpreted as focused case-study evidence rather than broad statistical proof.

## Output Files

- `exact_shapley_all_views.csv`: exact standard Shapley values for every usable view of every object and model.
- `exact_shapley_validation.csv`: Shapley efficiency and coalition-count validation table.
- `exact_coalition_values_all_sizes.csv`: raw value of every coalition, including the empty and grand coalitions.
- `shapley_marginal_by_predecessor_size.csv`: mean marginal contributions by predecessor coalition size; this explains Shapley values but is not a substitute for them.
- `coalition_scores_k1_to_k3.csv`: retained 1-, 2-, and 3-view coalitions for Analysis A.
- `overall_coalition_size_summary.csv`: macro-averaged 1 -> 2 -> 3 progression.
- `object_coalition_size_summary.csv`: per-object 1 -> 2 -> 3 progression.
- `class_coalition_size_summary.csv`: class-level 1 -> 2 -> 3 progression.
- `exact_shapley_strict_by_view_real_finetuned.png`: requested Figure 1.
- `single_view_vs_exact_shapley_strict_real_finetuned.png`: requested Figure 2.
- `real_multiview_progression.png`: retained progression figure.
