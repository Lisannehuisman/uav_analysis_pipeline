# Project Mapping

This repository follows the four experiment blocks in the thesis.

| Thesis block | Main folders |
| --- | --- |
| Synthetic AirSim benchmark and 72-view sampling | `data_collection/scripts/airsim_generation/`, `data_collection/raw_data/synthetic_subset/` |
| Real-image annotation and dataset preparation | `data_collection/scripts/roboflow_annotation/`, `data_collection/raw_data/same_taxonomy_54_images/`, `data_collection/raw_data/self_collected_uav_validation/` |
| Experiment 1: detector-family benchmark and training-side viewpoint diversity | `src/training/`, `src/viewpoint_training/`, `results/tables/detector_family_*`, `results/tables/single_view_sweep_*`, `results/tables/pair_view_sweep_*` |
| Experiment 2: fixed-detector viewpoint sensitivity | `src/viewpoint_analysis/`, `results/tables/factor_*`, `results/tables/clutter_quartile_summary.csv` |
| Experiment 3: inference-time multiview and coalition analysis | `src/multiview_analysis/`, `results/tables/one_vs_two_*`, `results/tables/harmonized_coalition_method_summary.csv`, `results/tables/image_count_shapley_proxy_summary.csv` |
| Experiment 4: real-world transfer, fine-tuning, and real same-object Shapley analysis | `src/real_world_transfer/`, `src/multiview_analysis/run_real_multiview_shapley.py`, `data_collection/raw_data/real_same_object_multiview/`, `results/tables/real_*`, `results/tables/zero_shot_transfer_same_taxonomy_and_public.csv` |
| Thesis figures | `src/figures/`, `results/figures/` |
| Thesis table outputs and short reports | `results/tables/`, `results/reports/` |

The full AirSim image set, model checkpoints, public VisDrone/AU-AIR datasets, and large intermediate prediction caches are not stored here. The repository keeps a representative synthetic subset, the thesis-owned real datasets, the scripts, and the result summaries needed to inspect the thesis claims.

For a figure-by-figure and table-by-table map, see `THESIS_CONTENT_MAPPING.md`.
