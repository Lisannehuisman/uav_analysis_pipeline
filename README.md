# Viewpoint Diversity for UAV Object Detection

Code, datasets, analyses, and result artifacts for my MSc thesis on how camera viewpoint affects UAV object detection, multi-view inference, and synthetic-to-real transfer.

The research combines a controlled synthetic UAV benchmark generated in Unreal Engine/AirSim with YOLOv8 and Faster R-CNN detectors, viewpoint-specific training experiments, multi-view analysis, and evaluation on real UAV imagery.

## Key findings

The main findings of the thesis are:

- **Viewpoint diversity during training matters.** The fully viewpoint-diverse YOLOv8l-M4 model achieved the strongest synthetic performance with **mAP50:95 = 0.640**, compared with **0.416** for the best single-view training configuration.
- **More training images alone do not explain the improvement.** Equal-budget subsets sampled from the diverse M4 regime outperformed restricted single-view and pair-view training.
- **Viewpoint effects depend on geometry and object class.** Elevation showed the most consistent influence on fixed-detector performance, while azimuth effects were more class-specific.
- **A second UAV viewpoint can provide substantial additional information.** Multi-view gains were largest when moving from one to two views, with diminishing returns as more views were added.
- **Strong synthetic performance does not automatically transfer to real UAV imagery.** Zero-shot transfer was limited across the evaluated real-world datasets, while fine-tuning on real UAV images substantially improved performance.
- **There is no universally optimal viewpoint.** Real same-object Shapley analysis showed that the contribution of individual viewpoints varies between physical targets.

## Experimental pipeline

The thesis investigates viewpoint effects at several stages of the object-detection pipeline:

```text
Controlled synthetic UAV dataset
        │
        ├── 10 object classes
        └── 72 viewpoints per object
            8 azimuths × 3 elevations × 3 radii
        │
        ▼
Viewpoint-diversity training
        │
        ├── M1–M4 training regimes
        ├── Single-view training
        ├── Pair-view training
        └── Equal-budget controls
        │
        ▼
Detector evaluation
        │
        ├── YOLOv8n
        ├── YOLOv8l
        └── Faster R-CNN
        │
        ▼
Fixed-detector viewpoint analysis
        │
        ├── Azimuth
        ├── Elevation
        ├── Radius
        └── Scene clutter
        │
        ▼
Multi-view analysis
        │
        ├── Two- and three-view combinations
        ├── Fusion strategies
        └── Shapley-based viewpoint contribution
        │
        ▼
Synthetic-to-real evaluation
        │
        ├── Public UAV datasets
        ├── Same-taxonomy real imagery
        └── Self-collected UAV imagery
        │
        ▼
Real-data fine-tuning and
real same-object multi-view analysis
```

The controlled synthetic benchmark makes it possible to change camera viewpoint while keeping other experimental factors relatively controlled. The subsequent experiments examine whether the resulting viewpoint effects remain consistent across detector architectures, whether multiple viewpoints provide complementary information, and how well conclusions obtained from synthetic imagery transfer to real UAV data.

## Repository structure

```text
uav_analysis_pipeline/
├── data_collection/
│   ├── raw_data/
│   │   ├── synthetic_subset/
│   │   ├── same_taxonomy_54_images/
│   │   ├── self_collected_uav_validation/
│   │   └── real_same_object_multiview/
│   └── scripts/
│
├── src/
│   ├── training/
│   ├── viewpoint_training/
│   ├── viewpoint_analysis/
│   ├── multiview_analysis/
│   ├── real_world_transfer/
│   └── figures/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   ├── recomputed/
│   └── intermediate/
│
├── models/
├── thesis/
├── supplementary_code/
├── PROJECT_MAPPING.md
├── THESIS_CONTENT_MAPPING.md
└── requirements.txt
```

The main components are:

- [`data_collection/`](./data_collection/) — synthetic-data utilities, annotation tools, and the datasets that can be distributed with the repository.
- [`src/training/`](./src/training/) — detector-family training, standardized evaluation, and M1–M4 regime comparison.
- [`src/viewpoint_training/`](./src/viewpoint_training/) — single-view, pair-view, and equal-budget viewpoint-training experiments.
- [`src/viewpoint_analysis/`](./src/viewpoint_analysis/) — fixed-detector analyses of azimuth, elevation, distance, and scene difficulty.
- [`src/multiview_analysis/`](./src/multiview_analysis/) — multi-view combination, fusion, coalition, and Shapley analyses.
- [`src/real_world_transfer/`](./src/real_world_transfer/) — synthetic-to-real evaluation and analysis of real-UAV fine-tuning.
- [`src/figures/`](./src/figures/) — scripts used to generate thesis figures.
- [`results/`](./results/) — result tables, figures, and analysis reports used throughout the thesis.
- [`supplementary_code/`](./supplementary_code/) — additional scripts retained from the broader experimental workspace.

For a compact mapping between thesis experiments and repository folders, see [`PROJECT_MAPPING.md`](./PROJECT_MAPPING.md).

For a detailed mapping between thesis sections, figures, tables, results, and their corresponding repository files, see [`THESIS_CONTENT_MAPPING.md`](./THESIS_CONTENT_MAPPING.md).

## Synthetic dataset

The controlled synthetic benchmark was generated using Unreal Engine and AirSim.

It contains **14,760 images** spanning ten object classes:

`tent`, `tank`, `tower`, `container`, `whitevan`, `suv`, `male`, `rock`, `barrel`, and `tree`.

The camera-object geometry follows a structured 72-view grid:

| Viewpoint factor | Values |
|---|---:|
| Azimuth | 8 |
| Elevation | 3 |
| Radius | 3 |
| Total viewpoints | 72 |

The complete AirSim capture is too large to distribute through GitHub. A representative subset is therefore provided in [`data_collection/raw_data/synthetic_subset/`](./data_collection/raw_data/synthetic_subset/).

The full source manifest is retained as:

[`source_manifest_full.csv`](./data_collection/raw_data/synthetic_subset/source_manifest_full.csv)

## Real UAV data

The repository also includes the thesis-owned real datasets:

- [`same_taxonomy_54_images/`](./data_collection/raw_data/same_taxonomy_54_images/) — 54 real images using the same object taxonomy as the synthetic benchmark.
- [`self_collected_uav_validation/`](./data_collection/raw_data/self_collected_uav_validation/) — 156 self-collected UAV images used for real-world validation and fine-tuning experiments.
- [`real_same_object_multiview/`](./data_collection/raw_data/real_same_object_multiview/) — repeated UAV observations of the same physical targets used for real-world multi-view and Shapley analysis.

Public external datasets such as **VisDrone** and **AU-AIR** are not redistributed in this repository. Result summaries derived from these datasets are retained under [`results/`](./results/).

## Main analysis stages

### 1. Detector training and comparison

The detector experiments compare YOLOv8n, YOLOv8l, and Faster R-CNN and evaluate the effect of different training-viewpoint distributions.

Relevant code:

[`src/training/`](./src/training/)

Example evaluation commands:

```powershell
python src/training/standardized_test_eval.py --help
python src/training/create_regime_metric_table.py --help
```

### 2. Viewpoint-training experiments

These experiments test how restricting the viewpoints available during training affects generalization across the complete viewpoint space.

They include:

- 72 single-viewpoint training conditions;
- pair-viewpoint experiments;
- M1–M4 viewpoint-diversity regimes;
- equal-budget controls separating viewpoint diversity from training-set size.

Relevant code:

[`src/viewpoint_training/`](./src/viewpoint_training/)

```powershell
python src/viewpoint_training/single_view_sweep/enumerate_single_viewpoints.py --help
python src/viewpoint_training/pair_view_sweep/enumerate_viewpoint_pairs.py --help
python src/viewpoint_training/compare_restricted_vs_equal_budget.py
```

### 3. Fixed-detector viewpoint analysis

A fixed YOLOv8l-M4 detector is used to study how observation geometry affects detection without retraining the detector for each viewpoint.

The analysis considers:

- azimuth;
- elevation;
- radius;
- object class;
- scene clutter.

Relevant code:

[`src/viewpoint_analysis/`](./src/viewpoint_analysis/)

```powershell
python src/viewpoint_analysis/run_factor_level_analysis.py --help
python src/viewpoint_analysis/run_clutter_grouping_analysis.py --help
```

### 4. Multi-view analysis

The multi-view experiments investigate whether observations from multiple UAV viewpoints contain complementary information.

The repository includes analyses of:

- one-view versus multi-view performance;
- two- and three-view combinations;
- prediction-combination strategies;
- coalition-size effects;
- Shapley-based viewpoint contribution.

Relevant code:

[`src/multiview_analysis/`](./src/multiview_analysis/)

```powershell
python src/multiview_analysis/build_harmonized_method_comparison.py --help
python src/multiview_analysis/build_image_count_shapley_proxy.py --help
python src/multiview_analysis/run_real_multiview_shapley.py --help
```

### 5. Synthetic-to-real transfer

The final experiments test how well the conclusions obtained from controlled synthetic data transfer to real UAV imagery.

This includes:

- zero-shot evaluation on real imagery;
- comparison across several real UAV datasets;
- fine-tuning the synthetic YOLOv8l-M4 detector on self-collected real UAV images;
- real same-object multi-view evaluation.

Relevant code:

[`src/real_world_transfer/`](./src/real_world_transfer/)

```powershell
python src/real_world_transfer/analyze_real_uav_results.py
```

## Setup

Clone the repository and create a Python environment:

```powershell
git clone https://github.com/Lisannehuisman/uav_analysis_pipeline.git
cd uav_analysis_pipeline

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some experiments additionally require:

- AirSim and an Unreal Engine environment for synthetic data generation;
- Detectron2 for Faster R-CNN experiments;
- a CUDA-capable GPU for practical detector training and fine-tuning.

## Reproducibility

Most analysis scripts use project-relative paths.

The repository contains the compact data, tables, figures, reports, and scripts needed to inspect the thesis experiments. Some large artifacts are intentionally not stored in normal Git, including:

- the complete 14,760-image AirSim dataset;
- large detector checkpoints;
- full prediction caches and COCO JSON files;
- large intermediate experimental outputs.

Full detector retraining therefore requires either restoring the corresponding model checkpoints under `models/` or rerunning the training experiments.

Recomputed outputs are written to [`results/recomputed/`](./results/recomputed/) where supported.

## Thesis

**Viewpoint Diversity for UAV Object Detection: From Controlled Synthetic Training to Multiview Diagnostics and Real-World Transfer**

The thesis PDF is available in the [`thesis/`](./thesis/) directory.

The repository's detailed thesis-to-code mapping is available in:

[`THESIS_CONTENT_MAPPING.md`](./THESIS_CONTENT_MAPPING.md)

## Related repository

The original scripts and configurations preserved from the Radboud University Ponyland GPU cluster are available in the companion repository:

[`uav_pipeline_ponyland`](https://github.com/Lisannehuisman/uav_pipeline_ponyland)

This repository focuses on the organized analysis and reproducibility pipeline, while the companion repository preserves the original cluster-side experimental code and configurations.