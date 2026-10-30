# Repository structure

The repository is organized around the main experiments in the thesis.

- `src/data_preparation/` contains data-generation and annotation utilities.
- `src/training/` contains detector training and standardized evaluation scripts.
- `src/detector_analysis/` contains detector comparisons and related analyses.
- `src/viewpoint_analysis/` examines the effect of camera-object geometry on a fixed detector.
- `src/viewpoint_training/` contains single-view, pair-view, and equal-budget training experiments.
- `src/multiview_analysis/` contains multiview fusion, viewpoint selection, and Shapley analyses.
- `src/geometry_analysis/` contains geometry-aware fusion experiments.
- `src/real_world_transfer/` contains real-UAV evaluation and fine-tuning analyses.
- `src/figures/` contains figure-generation scripts.
- `src/experimental/` contains exploratory work outside the main thesis analyses.

`data/sample/` holds a small representative dataset, and `data/analysis_inputs/` contains selected cached evaluation inputs. Additional research datasets are stored in `data_collection/raw_data/`.

`results/` contains the main tables, figures, and reports. `supplementary_code/original_workspace/` preserves older scripts for historical reference; those scripts can retain original workspace paths and dependencies.

See `docs/REPRODUCIBILITY.md` for the limits of what can be rerun from the repository as published.
