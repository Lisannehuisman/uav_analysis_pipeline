# M4 Single-Viewpoint Training Experiment

This folder implements a controlled training-subset experiment for the M4 dataset.

It trains one normal single-image `YOLOv8l` detector per individual training viewpoint:

- each model sees only the training images from 1 viewpoint
- labels are reused exactly from the original M4 dataset
- train/val/test split boundaries are preserved
- evaluation is run consistently with the same COCO-style metric pipeline used for the duo-viewpoint sweep

## Scripts

- `enumerate_single_viewpoints.py`
  - discovers the 72 viewpoints from the base M4 dataset
  - writes the full 72-view manifest
  - flags 5 representative pilot viewpoints

- `build_single_subsets.py`
  - builds viewpoint-specific YOLO list files and dataset YAMLs
  - validates that only the selected viewpoint is present
  - checks that split leakage is not introduced

- `run_single_experiment.py`
  - resumable per-viewpoint worker
  - stages: `subset`, `train`, `eval`, `all`, `sanity`
  - writes one status JSON per training viewpoint

- `launch_single_sweep.py`
  - emits generic bash launchers or Slurm-array launchers
  - supports `pilot` and `full` modes

- `aggregate_single_results.py`
  - scans status JSONs
  - writes master CSVs
  - produces plots and a short experiment report

## Scientific Question

This experiment asks how much detector generalization can be learned from training on only one viewpoint.

It complements the duo-viewpoint sweep by providing a matched single-view baseline:

- single-view training: one viewpoint in `train` and `val`
- duo-view training: two viewpoints in `train` and `val`
- same detector family
- same optimization protocol
- same evaluation on the full fixed M4 test split

## Typical Workflow

1. Generate the single-viewpoint manifest.
2. Build subsets for the 5 pilot viewpoints.
3. Run `sanity` on those pilot viewpoints.
4. Launch real pilot training on the cluster.
5. Aggregate pilot outputs.
6. Launch the full sweep only after the pilot checks out.

## Notes

- The subset builder uses YOLO list files instead of copying images, which keeps the pipeline light and reproducible.
- The worker is resumable because each viewpoint owns its own directory and status file.
- The full M4 YOLOv8l baseline is reused from the existing standardized evaluation summary when available.
- For the real GPU pilot and full sweep on Ponyland, use `PONYLAND_RUNBOOK.md`.

