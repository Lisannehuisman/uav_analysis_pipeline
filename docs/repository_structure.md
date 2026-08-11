\# Repository structure



The repository is organized by the main experimental components of the thesis.



\- `src/data\_preparation/` contains the synthetic-data generation, conversion and annotation utilities.

\- `src/detector\_analysis/` contains detector evaluation and architecture-comparison scripts.

\- `src/viewpoint\_analysis/` contains the fixed-detector viewpoint-dependence analyses.

\- `src/viewpoint\_training/` contains the single-view, pair-view and matched-control experiments.

\- `src/multiview\_analysis/` contains multiview fusion, viewpoint-selection and Shapley-based analyses.

\- `src/geometry\_analysis/` contains the geometry-aware fusion experiments.

\- `src/real\_world\_transfer/` contains the real-UAV evaluation and transfer analyses.

\- `src/figures/` contains scripts used for thesis figures.

\- `src/experimental/` contains additional exploratory experiments.



Small example data are stored under `data/sample/`. Compact cached inputs needed for selected analyses are stored under `data/analysis\_inputs/`.



The `results/` directory contains the main tables, reports and figures generated during the experiments. Large prediction caches, raw datasets and model checkpoints are excluded from Git because of their size.

