# Reproducibility

This repository contains the analysis code, selected inputs, and result files for the MSc thesis on viewpoint diversity in UAV object detection. Not every experiment can be rerun from this repository alone.

## Data

A small synthetic sample is available in `data/sample/`, and a larger synthetic subset is included in `data_collection/raw_data/synthetic_subset/`. The complete original AirSim dataset is not included. Some self-collected real UAV datasets are included under `data_collection/raw_data/`, while third-party datasets such as VisDrone and AU-AIR are not redistributed here.

A list of the tracked files is available through `git ls-files`. The presence of a `.gitignore` rule does not remove datasets already tracked by Git.

Machine-specific locations can be configured using `configs/paths.local.yaml`, based on `configs/paths.example.yaml`.

## Models and environments

Model checkpoints are not distributed with the repository. See `models/README.md` for the model files used in the experiments. Python dependencies are listed in `requirements.txt`; the separate Ponyland repository documents the original cluster training environment.

## Cached inputs and outputs

Selected cached viewpoint inputs are in `data/analysis_inputs/`. Final tables, figures, and reports are in `results/`.

Some scripts still refer to paths from the original research workspace. For example, `src/multiview_analysis/marginal_value/run_marginal_value_analysis.py` requires the original protocol-level summary CSV files, which are not currently tracked in this repository. Its saved results can be inspected under `results/multiview_analysis/marginal_value/`.

## Example

From the repository root, the factor-level analysis script can be inspected with:

```cmd
python src\viewpoint_analysis\factor_level\run_factor_level_analysis.py --help
```

Check the input arguments and available data before executing an analysis. When regenerating results, use a separate output directory to avoid overwriting the original thesis results.
