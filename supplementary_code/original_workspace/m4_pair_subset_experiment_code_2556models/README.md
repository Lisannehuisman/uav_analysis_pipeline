# M4 Pair-Subset Experiment

This folder implements a controlled training-subset experiment for the M4 dataset.

It does **not** do multiview fusion.

It trains one normal single-image `YOLOv8l` detector per viewpoint pair:

- each model sees only the union of training images from 2 viewpoints
- labels are reused exactly from the original M4 dataset
- train/val/test split boundaries are preserved
- evaluation is run consistently with a COCO-style metric pipeline

## Scripts

- `enumerate_viewpoint_pairs.py`
  - discovers the 72 viewpoints from the base M4 dataset
  - writes the full `72 choose 2 = 2556` pair manifest
  - flags 5 representative pilot pairs

- `build_pair_subsets.py`
  - builds pair-specific YOLO list files and pair dataset YAMLs
  - validates that only the selected viewpoints are present
  - checks that split leakage is not introduced

- `run_pair_experiment.py`
  - resumable per-pair worker
  - stages: `subset`, `train`, `eval`, `all`, `sanity`
  - writes one status JSON per pair

- `launch_pair_sweep.py`
  - emits generic bash launchers or Slurm-array launchers
  - supports `pilot` and `full` modes

- `aggregate_pair_results.py`
  - scans status JSONs
  - writes master CSVs
  - produces plots and a short experiment report

## Protocol

Two evaluation options are implemented:

- `Option A`
  - evaluate each pair-trained model on the full fixed M4 test split
  - this is the recommended default because it measures generalization from 2 training viewpoints to the full viewpoint space

- `Option B`
  - evaluate each pair-trained model only on the same 2 viewpoints in the fixed M4 test split
  - this is useful as a diagnostic for in-subset fit, but should not be the headline result

## Typical Workflow

1. Generate the pair manifest.
2. Build subsets for the 5 pilot pairs.
3. Run `sanity` on those pilot pairs.
4. Launch real pilot training on the cluster.
5. Aggregate pilot outputs.
6. Launch the full sweep only after the pilot checks out.

## Notes

- The subset builder uses YOLO list files instead of copying images, which keeps the pipeline light and reproducible.
- The worker is resumable because each pair owns its own directory and status file.
- The full M4 YOLOv8l baseline is reused from the existing standardized evaluation summary when available.
- For the real GPU pilot and full sweep on Ponyland, use `PONYLAND_RUNBOOK.md`.
