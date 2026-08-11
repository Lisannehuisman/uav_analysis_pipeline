\# Reproducibility



This repository contains the local analysis code used for the master's thesis.



\## Data

The complete synthetic and real UAV datasets are not stored in Git because of

their size. A small representative sample is included under data/sample/.

Machine-specific paths can be configured using configs/paths.local.yaml, based

on configs/paths.example.yaml.



\## Models

Large model checkpoints are not included in Git. See models/README.md for the

checkpoints used in the experiments.



\## Cached analysis inputs

Small intermediate files required for selected analyses are included under

data/analysis\_inputs/. This makes it possible to reproduce several analyses

without rerunning detector inference.



\## Training and cluster experiments

Large-scale model training and evaluation were carried out on the Ponyland

cluster. The cluster-side training pipeline is stored separately from this

local analysis repository.



\## Example

From the repository root:



python -m src.viewpoint\_analysis.factor\_level.run\_factor\_level\_analysis



This regenerates the factor-level viewpoint outputs under:

results/viewpoint\_analysis/factor\_level/

