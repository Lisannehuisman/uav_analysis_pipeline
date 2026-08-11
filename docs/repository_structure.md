# Thesis-to-Project Mapping

I used the thesis PDF as the authoritative source for deciding what belongs in this standalone project. When the larger working folder contained extra experiments, drafts, caches, or public datasets that were not needed for the PDF narrative, I left them out.

## Main Mapping

| Thesis component | Project location |
| --- | --- |
| AirSim synthetic data generation and 72-view sampling | `data_collection/scripts/airsim_generation/`, `data_collection/raw_data/synthetic_subset/` |
| Real dataset preparation and annotation conversion | `data_collection/scripts/roboflow_annotation/`, `data_collection/raw_data/same_taxonomy_54_images/`, `data_collection/raw_data/self_collected_uav_validation/` |
| Detector-family benchmark, YOLOv8n/YOLOv8l/Faster R-CNN, and M1-M4 regimes | `src/training/`, `results/tables/detector_family_*`, `results/figures/figure_4_1_detector_regime_map50_95.png` |
| Single-view viewpoint sweep | `src/viewpoint_training/single_view_sweep/`, `results/tables/single_view_sweep_*`, `results/figures/figure_4_2_single_view_heatmap.png` |
| Pair-view sweep and equal-budget comparison | `src/viewpoint_training/pair_view_sweep/`, `src/viewpoint_training/compare_restricted_vs_equal_budget.py`, `results/tables/pair_view_sweep_master_results.csv`, `results/tables/equal_budget_control_table.csv` |
| Fixed-detector viewpoint sensitivity and factor-level analysis | `src/viewpoint_analysis/`, `results/tables/factor_*`, `results/figures/figure_4_4_factor_level_viewpoint_dependence.png` |
| Clutter analysis | `src/viewpoint_analysis/run_clutter_grouping_analysis.py`, `results/tables/clutter_quartile_summary.csv`, `results/reports/clutter_analysis_report.md` |
| Inference-time one/two/three-view protocols, fusion, and marginal value | `src/multiview_analysis/`, `results/tables/one_vs_two_*`, `results/tables/box_fusion_*`, `results/tables/harmonized_coalition_method_summary.csv`, `results/figures/figure_4_5_top_coalition_methods.png` |
| Synthetic image-count Shapley proxy | `src/multiview_analysis/build_image_count_shapley_proxy.py`, `results/tables/image_count_shapley_proxy_summary.csv`, `results/figures/figure_4_6_image_count_shapley_proxy.png` |
| Real-world transfer and fine-tuning | `src/real_world_transfer/`, `results/tables/zero_shot_transfer_same_taxonomy_and_public.csv`, `results/tables/real_uav_finetune_*`, `results/reports/real_uav_finetune_report.md` |
| Real same-object multiview Shapley analysis | `src/multiview_analysis/run_real_multiview_shapley.py`, `data_collection/raw_data/real_same_object_multiview/` (36 source views, 35 usable target views), `results/tables/real_multiview_*`, `results/figures/figure_4_7_real_multiview_progression.png`, `results/figures/figure_4_8_real_shapley_by_view.png` |
| Thesis figures and appendix plots | `src/figures/`, `results/figures/` |

## Included Result Artifacts

`results/tables/` contains the CSV summaries that support the main thesis claims. `results/figures/` contains the rendered figures used in the thesis and appendix. `results/reports/` contains the accompanying markdown summaries used while interpreting the experiments.

## Exclusions

- The complete AirSim synthetic image dataset is not copied because it is too large for a clean standalone archive. A representative 200-image subset and the full source manifest are included instead.
- Model checkpoints are not copied. Scripts use `models/` as the expected location for restored or rerun checkpoints.
- Public datasets such as VisDrone and AU-AIR are not copied. Their aggregate thesis metrics remain in the result tables.
- Temporary training caches, notebook-style scratch folders, old supervisor-update artifacts, LaTeX build products, and exploratory experiments outside the final thesis narrative are excluded.

## Uncertainties

- The parent workspace also contains a LaTeX thesis folder, but the downloaded PDF was treated as the final source for this archive.
- Some scripts need full intermediate files, prediction JSONs, or checkpoints to rerun exactly. Where those large files are omitted, the already computed thesis summaries are kept in `results/`.
