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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project also includes `lisannesmasterthesis.pyproj` and `lisannesmasterthesis.sln` for Visual Studio, plus `lisannesmasterthesis.code-workspace` for VS Code.

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
