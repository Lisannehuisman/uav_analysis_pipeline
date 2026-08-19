# Thesis Content Mapping

This file maps the thesis PDF to the repository files. The PDF used for this mapping is `thesis/Msc_thesis_Lisanne_Huisman_firstfulldraft.pdf`.

## Methods And Experiments

| Thesis section | Content | Repository files |
| --- | --- | --- |
| 3.2 | Synthetic Unreal Engine/AirSim benchmark with ten classes: tent, tank, tower, container, whitevan, suv, male, rock, barrel, tree | `data_collection/scripts/airsim_generation/`, `data_collection/raw_data/synthetic_subset/`, `data_collection/raw_data/synthetic_subset/source_manifest_full.csv` |
| 3.2.2 / Table 3.1 | 72-view grid: 8 azimuths, 3 elevation bands, 3 radii | `data_collection/scripts/airsim_generation/airsim_capture_utils.py`, `data_collection/scripts/airsim_generation/capture_multi_env.py`, `results/figures/figure_3_1_viewpoint_grid.png` |
| 3.3 | RGB and segmentation capture, automatic box extraction, YOLO export | `data_collection/scripts/airsim_generation/thesis_capt_imgs_lbls1.py`, `data_collection/scripts/airsim_generation/yolo_to_coco.py`, `data_collection/scripts/airsim_generation/make_data_yaml.py` |
| 3.3.1 / Appendix A.2.1 | Dataset accounting: 14,760 images, 10,332 train, 2,214 validation, 2,214 test, 205 inferred instances | `data_collection/raw_data/synthetic_subset/source_manifest_full.csv`, `results/tables/target_visibility_summary.csv` |
| 3.4 | Detector families: YOLOv8n, YOLOv8l, Faster R-CNN | `src/training/`, `results/tables/detector_family_standardized_test_summary.csv` |
| 3.5 / Table 3.2 | Evaluation metrics: precision, recall, F1, AP50, mAP50:95, per-image AP proxy, strict target quality | `src/training/standardized_test_eval.py`, `src/multiview_analysis/`, `results/tables/` |
| 3.6 / Table 3.3 | Experiment 1: M1, M2a, M2b, M3, M4 training-viewpoint regimes | `src/training/`, `src/viewpoint_training/`, `results/tables/detector_family_regime_metric_table.csv` |
| 3.6.1 | Equal-budget controls for restricted-view training | `src/viewpoint_training/compare_restricted_vs_equal_budget.py`, `results/tables/equal_budget_control_table.csv`, `results/reports/equal_budget_control_summary.md` |
| 3.7 | Experiment 2: fixed YOLOv8l-M4 viewpoint sensitivity by azimuth, elevation, radius | `src/viewpoint_analysis/run_factor_level_analysis.py`, `results/tables/factor_*.csv`, `results/figures/figure_4_4_factor_level_viewpoint_dependence.png` |
| 3.7.1 / Appendix A.2.3 | Clutter-stratified difficulty check | `src/viewpoint_analysis/run_clutter_grouping_analysis.py`, `results/tables/clutter_quartile_summary.csv`, `results/reports/clutter_analysis_report.md` |
| 3.8 | Experiment 3: inference-time multiview analysis with a fixed detector | `src/multiview_analysis/analyze_two_drone_operational.py`, `src/multiview_analysis/compare_view_combination_methods.py`, `results/tables/one_vs_two_*.csv` |
| 3.8.2 / Appendix A.2.4 | Matched-box fusion policies, Noisy-OR, odds-product, support-weighted variants | `src/multiview_analysis/run_box_fusion_analysis.py`, `src/multiview_analysis/build_harmonized_method_comparison.py`, `results/tables/harmonized_coalition_method_summary.csv` |
| Appendix A.2.4 | Coalition-size marginal / image-count Shapley proxy | `src/multiview_analysis/build_image_count_shapley_proxy.py`, `results/tables/image_count_shapley_proxy_summary.csv`, `results/reports/image_count_shapley_proxy_report.md` |
| 3.9.1 / 4.4.1 | Zero-shot transfer to VisDrone, AU-AIR, same-taxonomy real images, and self-collected UAV images | `src/real_world_transfer/analyze_real_uav_results.py`, `results/tables/zero_shot_transfer_same_taxonomy_and_public.csv`, `results/reports/real_world_validation_comparison_summary.md` |
| 3.9.2 | Self-collected 156-image UAV dataset | `data_collection/raw_data/self_collected_uav_validation/`, `data_collection/scripts/roboflow_annotation/` |
| 3.9.3 / 4.4.3 | Real-image fine-tuning from YOLOv8l-M4, evaluated on the 39-image held-out real-UAV test subset | `src/real_world_transfer/analyze_real_uav_results.py`, `results/tables/real_uav_finetune_*.csv`, `results/reports/real_uav_finetune_report.md` |
| 3.9.5 / 4.4.4 / 4.4.5 | Real same-object multiview coalitions and exact Shapley values | `data_collection/raw_data/real_same_object_multiview/`, `src/multiview_analysis/run_real_multiview_shapley.py`, `results/tables/real_multiview_*.csv` |

## Main Figures

| Figure | Thesis content | Repository files |
| --- | --- | --- |
| Figure 3.1 | Deterministic 72-view capture grid | `results/figures/figure_3_1_viewpoint_grid.png`, `src/figures/generate_viewpoint_sampling_figure.py`, `src/figures/generate_viewpoint_capture_grid_3d.py` |
| Figure 3.2 | Representative synthetic UAV images | `results/figures/figure_3_2_synthetic_examples.png`, `data_collection/raw_data/synthetic_subset/` |
| Figure 3.3 | Dataset pipeline diagram | Described in `data_collection/scripts/airsim_generation/`; the final layout is embedded in the thesis PDF rather than stored as a separate source image. |
| Figure 3.4 | YOLOv8l versus Faster R-CNN qualitative comparison | `results/figures/figure_3_4_yolo_frcnn_comparison.png`, `src/figures/generate_detector_comparison_figure.py` |
| Figure 3.5 | Synthetic, AU-AIR, same-taxonomy, and self-collected domain examples | `data_collection/raw_data/synthetic_subset/`, `data_collection/raw_data/same_taxonomy_54_images/`, `data_collection/raw_data/self_collected_uav_validation/`; AU-AIR images are external and not stored. |
| Figure 3.6 | Real-UAV acquisition setup | Final image is in the thesis PDF; underlying self-collected dataset is in `data_collection/raw_data/self_collected_uav_validation/`. |
| Figure 3.7 | La Souris same-object real UAV views | `data_collection/raw_data/real_same_object_multiview/images/la_souris_truck_*.jpg` |
| Figure 4.1 | Detector-family mAP50:95 by viewpoint regime | `results/figures/figure_4_1_detector_regime_map50_95.png`, `results/tables/detector_family_regime_metric_table.csv`, `src/training/plot_regime_map50_95.py` |
| Figure 4.2 | Single-view training heatmap | `results/figures/figure_4_2_single_view_heatmap.png`, `results/tables/single_view_sweep_master_results.csv` |
| Figure 4.3 | Equal-budget training controls | `results/figures/figure_4_3_equal_budget_controls.png`, `results/tables/equal_budget_control_table.csv` |
| Figure 4.4 | Factor-level viewpoint dependence | `results/figures/figure_4_4_factor_level_viewpoint_dependence.png`, `src/viewpoint_analysis/build_main_section_visuals.py` |
| Figure 4.5 | Top two-view and three-view coalition methods | `results/figures/figure_4_5_top_coalition_methods.png`, `results/tables/top5_two_vs_three_method_comparison.csv` |
| Figure 4.6 | Coalition-size marginal analysis | `results/figures/figure_4_6_image_count_shapley_proxy.png`, `results/tables/image_count_shapley_proxy_summary.csv` |
| Figure 4.7 | Real same-object multiview progression | `results/figures/figure_4_7_real_multiview_progression.png`, `results/tables/real_multiview_overall_coalition_size_summary.csv` |
| Figure 4.8 | Exact strict-quality Shapley contributions for real views | `results/figures/figure_4_8_real_shapley_by_view.png`, `results/tables/real_multiview_exact_shapley_all_views.csv` |

## Main Tables

| Table | Thesis content | Repository files |
| --- | --- | --- |
| Table 2.1 | Representative UAV/synthetic/multiview datasets | Literature table in thesis PDF; no experiment code. |
| Table 3.1 | Deterministic viewpoint grid | `data_collection/scripts/airsim_generation/airsim_capture_utils.py`, `results/figures/figure_3_1_viewpoint_grid.png` |
| Table 3.2 | Metric roles | `src/training/standardized_test_eval.py`, `src/multiview_analysis/` |
| Table 3.3 | M1-M4 regime notation | `src/training/comparison_config.py`, `src/viewpoint_training/` |
| Table 4.1 | Restricted-view training versus equal-budget controls | `results/tables/equal_budget_control_table.csv` |
| Table 4.2 | Permutation-based coalition-size marginal contribution | `results/tables/image_count_shapley_proxy_summary.csv` |
| Table 4.3 | Aggregate zero-shot real-world transfer | `results/tables/zero_shot_transfer_same_taxonomy_and_public.csv` |
| Table 4.4 | Synthetic-only versus real-image fine-tuned detector | `results/tables/real_uav_finetune_stable_class_summary.csv`, `results/tables/real_uav_finetune_supported_class_summary.csv` |
| Table 4.5 | Exact Shapley summary for the fine-tuned detector | `results/tables/real_multiview_exact_shapley_validation.csv`, `results/tables/real_multiview_exact_shapley_all_views.csv` |
| Appendix A.2-A.10 | Dataset accounting, hyperparameters, clutter quartiles, detector benchmark, coalition methods, and transfer breakdowns | `results/tables/*.csv`, `results/reports/*.md`, `src/training/`, `src/viewpoint_analysis/`, `src/real_world_transfer/` |

## Results Summary

| Thesis result | Files |
| --- | --- |
| YOLOv8l-M4 is the strongest synthetic detector-regime combination, reaching mAP50:95 = 0.6396. | `results/tables/detector_family_standardized_test_summary.csv`, `results/figures/figure_4_1_detector_regime_map50_95.png` |
| Single-view training is much weaker than full M4 training; best single view reaches mAP50:95 = 0.4164. | `results/tables/single_view_sweep_master_results.csv`, `results/figures/figure_4_2_single_view_heatmap.png` |
| Equal-budget M4 subsets beat restricted single-view and pair-view training. | `results/tables/equal_budget_control_table.csv`, `results/figures/figure_4_3_equal_budget_controls.png` |
| Fixed-detector observation quality is most consistently affected by elevation, with class-specific azimuth effects. | `results/tables/factor_best_summary.csv`, `results/figures/figure_4_4_factor_level_viewpoint_dependence.png` |
| Multiview inference gains are strongest from the first added view and show diminishing returns after two to three views. | `results/tables/image_count_shapley_proxy_summary.csv`, `results/figures/figure_4_6_image_count_shapley_proxy.png` |
| Zero-shot real-world transfer is weak across VisDrone, AU-AIR, same-taxonomy real images, and self-collected UAV images. | `results/tables/zero_shot_transfer_same_taxonomy_and_public.csv` |
| Fine-tuning on real UAV images substantially improves the 39-image held-out test subset. | `results/tables/real_uav_finetune_*.csv`, `results/reports/real_uav_finetune_report.md` |
| Real same-object exact Shapley values show that view contributions vary by physical target. | `results/tables/real_multiview_exact_shapley_all_views.csv`, `results/figures/figure_4_8_real_shapley_by_view.png` |
