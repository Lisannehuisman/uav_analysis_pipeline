# Lisanne's Master's Thesis Project

This repository contains the code, selected datasets, and result artifacts for my MSc thesis on viewpoint diversity for UAV object detection. I organized it as a standalone Python project so the thesis experiments, figures, tables, and datasets can be inspected without the large exploratory workspace around it.

The thesis PDF used as the reference document is included in `thesis/Msc_thesis_Lisanne_Huisman_firstfulldraft.pdf`.

## Project Layout

```text
lisannesmasterthesis/
  data_collection/
    raw_data/
      synthetic_subset/                 representative AirSim subset
      same_taxonomy_54_images/          full 54-image real same-taxonomy set
      self_collected_uav_validation/    full 156-image UAV validation set
      real_same_object_multiview/       35-image same-object multiview set
    scripts/                            AirSim and annotation utilities
  src/
    training/                           detector-family training/evaluation code
    viewpoint_training/                 single-view and pair-view sweep code
    viewpoint_analysis/                 fixed-detector factor and clutter analysis
    multiview_analysis/                 fusion, swarm, and Shapley analyses
    real_world_transfer/                real-world transfer/fine-tuning analysis
    figures/                            figure-generation scripts
  results/
    figures/                            thesis figures and appendix figures
    tables/                             thesis CSV summaries
    reports/                            markdown analysis reports
  thesis/                               reference PDF
```

## Setup

Create a Python environment and install the main dependencies:
=======
This repository contains the code and data used for my master's thesis on viewpoint diversity for UAV object detection. The thesis studies how camera-object geometry affects detector training, fixed-detector evaluation, multiview inference, and transfer from synthetic to real drone imagery. The main synthetic benchmark was generated in Unreal Engine/AirSim with a controlled 72-view grid around ten object classes, then evaluated with YOLOv8 and Faster R-CNN style detectors.

The thesis PDF is included at `thesis/Msc_thesis_Lisanne_Huisman_firstfulldraft.pdf`.

## Thesis

Viewpoint Diversity for UAV Object Detection: From Controlled Synthetic Training to Multiview Diagnostics and Real-World Transfer

## Repository Layout

```text
data_collection/
  raw_data/
    synthetic_subset/                 small AirSim subset with YOLO labels
    same_taxonomy_54_images/          real same-taxonomy transfer set
    self_collected_uav_validation/    156-image self-collected UAV dataset
    real_same_object_multiview/       repeated real views of five physical targets
  scripts/
    airsim_generation/                AirSim capture, split, and format utilities
    roboflow_annotation/              CVAT/Roboflow review and conversion utilities
models/                               placeholder for restored or rerun checkpoints
src/
  training/                           detector-family benchmark and evaluation
  viewpoint_training/                 single-view, pair-view, and equal-budget sweeps
  viewpoint_analysis/                 fixed-detector factor and clutter analysis
  multiview_analysis/                 fusion, coalition, and Shapley analyses
  real_world_transfer/                real-image transfer and fine-tuning analysis
  figures/                            scripts for thesis figures
results/
  figures/                            rendered thesis figures
  tables/                             CSV summaries behind thesis tables
  reports/                            short markdown analysis reports
  recomputed/                         default place for rerun outputs
  intermediate/                       optional restored large prediction files
thesis/                               thesis PDF
supplementary_code/                   additional associated code from the original workspace
```

`PROJECT_MAPPING.md` gives the compact thesis-to-folder overview. `THESIS_CONTENT_MAPPING.md` gives the more detailed mapping from thesis methods, figures, tables, and results to files in this repository.

supplementary_code/ keeps extra associated scripts from the broader working project. These files are useful as reference material, but the main reproducible thesis workflow is documented through the folders above.

## Setup

The project uses Python. 

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Data Included

The full synthetic AirSim capture used in the thesis is very large, so I included a representative subset with the same 10-class taxonomy and viewpoint naming convention. The full source manifest is retained as `data_collection/raw_data/synthetic_subset/source_manifest_full.csv` so the subset can be traced back to the complete capture.

The thesis-owned real datasets are included directly:

- `same_taxonomy_54_images`: the full 54-image real same-taxonomy dataset.
- `self_collected_uav_validation`: the 156-image UAV validation set.
- `real_same_object_multiview`: 36 same-object UAV source views; one Ooij tower view has no target box and is excluded by the analysis, leaving 35 usable Shapley views.

Public external datasets such as VisDrone and AU-AIR are not copied into this archive; only the thesis result summaries that depend on them are kept.

## Reproduction Notes

Most scripts default to project-relative paths. Full retraining still requires the original model checkpoints or rerunning training into `models/`, which is intentionally left as a placeholder. Large intermediate COCO prediction JSONs and full AirSim captures are also not included when their results are already represented by thesis figures, tables, and reports.

For a detailed thesis-to-project mapping, see `PROJECT_MAPPING.md`.
=======
The Visual Studio solution is `lisannesmasterthesis.sln`. The VS Code workspace is `lisannesmasterthesis.code-workspace`.

Some scripts also need optional tools that are not installed automatically here:

- AirSim and an Unreal Engine scene for synthetic data capture.
- Detectron2 for Faster R-CNN training/evaluation.
- A GPU for realistic YOLOv8 training or fine-tuning.

## Data

The full AirSim dataset used in the thesis has 14,760 images. I keep a representative subset in `data_collection/raw_data/synthetic_subset/` so the folder structure, labels, and scripts can be inspected without putting the whole generated dataset in Git. The full manifest is kept as `data_collection/raw_data/synthetic_subset/source_manifest_full.csv`.

The real datasets that I collected or assembled for the thesis are included:

- `same_taxonomy_54_images`: 54 real images using the synthetic taxonomy.
- `self_collected_uav_validation`: 156 UAV images used for zero-shot validation and the real fine-tuning split.
- `real_same_object_multiview`: 36 labelled source views across five physical targets. One Ooij tower view has no target box, so the Shapley analysis uses 35 views.

The public external datasets are not stored in this repository:

- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- AU-AIR: https://bozcani.github.io/auairdataset

Those datasets are large and have their own distribution pages. The repository keeps the aggregate thesis result tables that depend on them.

## Running The Main Stages

Synthetic data utilities:

```powershell
python data_collection/scripts/airsim_generation/make_data_yaml.py --help
python data_collection/scripts/airsim_generation/yolo_to_coco.py --help
```

Detector benchmark and evaluation:

```powershell
python src/training/standardized_test_eval.py --help
python src/training/create_regime_metric_table.py --help
```

Single-view and pair-view training sweeps:

```powershell
python src/viewpoint_training/single_view_sweep/enumerate_single_viewpoints.py --help
python src/viewpoint_training/pair_view_sweep/enumerate_viewpoint_pairs.py --help
python src/viewpoint_training/compare_restricted_vs_equal_budget.py
```

Fixed-detector viewpoint analysis:

```powershell
python src/viewpoint_analysis/run_factor_level_analysis.py --help
python src/viewpoint_analysis/run_clutter_grouping_analysis.py --help
```

Multiview and Shapley analysis:

```powershell
python src/multiview_analysis/build_harmonized_method_comparison.py --help
python src/multiview_analysis/build_image_count_shapley_proxy.py --help
python src/multiview_analysis/run_real_multiview_shapley.py --help
```

Real-world transfer:

```powershell
python src/real_world_transfer/analyze_real_uav_results.py
```

Most reruns write into `results/recomputed/` by default. Full retraining or exact reproduction of every result needs the omitted full AirSim dataset, restored checkpoints under `models/`, and sometimes large prediction JSONs under `results/intermediate/`.

