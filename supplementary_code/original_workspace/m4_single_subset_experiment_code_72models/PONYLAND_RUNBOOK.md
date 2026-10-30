# M4 Single-Viewpoint Ponyland Runbook

This runbook is for the controlled M4 single-viewpoint training experiment:

- one normal single-image `YOLOv8l` model per viewpoint
- viewpoint-specific train and val subsets
- fixed evaluation on the full M4 test set
- pilot-first execution on the Ponyland GPU cluster

## 1. Recommended Scientific Protocol

- train each model on only one selected viewpoint
- evaluate every viewpoint-trained model on the full fixed M4 `test` split across all 72 viewpoints
- this measures generalization from a single training viewpoint to the full viewpoint space

## 2. What Has Already Been Validated Locally

- all `72` training viewpoints were enumerated
- five representative pilot viewpoints were selected
- viewpoint-filtered subset manifests can be built and validated
- split leakage checks can be run before cluster execution
- resumable status tracking, aggregation, plots, and reporting are implemented

The real training and evaluation still need to run on Ponyland because the local machine here does not have the required CUDA setup for the full experiment.

## 3. Prepare the Cluster Workspace

Copy or sync the project to Ponyland, then move to the project root.

Example placeholder:

```bash
cd /vol/tensusers6/lisannehuisman/New-project
```

## 4. Prepare a Python Environment

Use the same environment style that already worked for your earlier Ponyland runs. At minimum, the environment needs:

```bash
pip install numpy matplotlib pyyaml pillow pycocotools ultralytics
```

It also needs GPU-enabled:

- `torch`
- `torchvision`

## 5. Stage Model Weights

If Ponyland cannot download `yolov8l.pt` from the internet, place a local copy on the cluster and pass it explicitly with `--model`.

Example placeholder:

```bash
/vol/tensusers6/lisannehuisman/models/yolov8l.pt
```

## 6. Regenerate the Pilot Launcher on Ponyland

Regenerate the launcher on Ponyland so the script uses the cluster-side paths and Python executable rather than the local Windows defaults.

Example:

```bash
python m4_single_subset_experiment/launch_single_sweep.py \
  --experiment-root outputs/m4_single_subset_experiment \
  --mode pilot \
  --launcher slurm \
  --python-executable python \
  --workspace-root /vol/tensusers6/lisannehuisman/New-project \
  --base-data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml \
  --device 0 \
  --eval-batch 16 \
  --resume-training \
  --slurm-gres gpu:1 \
  --slurm-cpus-per-task 8 \
  --slurm-mem 24G \
  --slurm-time 04:00:00 \
  --model /vol/tensusers6/lisannehuisman/models/yolov8l.pt
```

What this does:

- creates a Slurm array launcher for the 5 pilot viewpoints
- writes logs into `outputs/m4_single_subset_experiment/logs`
- preserves resumability with one status file per viewpoint
- evaluates each trained model with the same fixed pipeline used for pair training

## 7. Launch the Pilot

Submit the generated Slurm array:

```bash
sbatch outputs/m4_single_subset_experiment/launchers/launch_pilot_slurm.sh
```

After the pilot jobs finish, aggregate the results:

```bash
python m4_single_subset_experiment/aggregate_single_results.py \
  --experiment-root outputs/m4_single_subset_experiment
```

## 8. Pilot Checks Before Full Sweep

Before launching all `72` viewpoints, verify:

- each pilot viewpoint has `subset.status = completed`
- each pilot viewpoint has `training.status = completed`
- each pilot viewpoint has `evaluation.status = completed`
- the `master_results.csv` rows contain sensible image counts and metrics
- the viewpoint-specific datasets contain only the requested viewpoint
- model outputs are stored under the expected viewpoint directory

Files to inspect:

- `outputs/m4_single_subset_experiment/reports/master_results.csv`
- `outputs/m4_single_subset_experiment/reports/single_viewpoint_experiment_report.md`
- `outputs/m4_single_subset_experiment/plots/top_performing_single_viewpoints.png`

## 9. Launch the Full Sweep

Only after the pilot passes, regenerate the full launcher:

```bash
python m4_single_subset_experiment/launch_single_sweep.py \
  --experiment-root outputs/m4_single_subset_experiment \
  --mode full \
  --launcher slurm \
  --python-executable python \
  --workspace-root /vol/tensusers6/lisannehuisman/New-project \
  --base-data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml \
  --device 0 \
  --eval-batch 16 \
  --resume-training \
  --slurm-gres gpu:1 \
  --slurm-cpus-per-task 8 \
  --slurm-mem 24G \
  --slurm-time 04:00:00 \
  --max-parallel 16 \
  --model /vol/tensusers6/lisannehuisman/models/yolov8l.pt
```

Submit it:

```bash
sbatch outputs/m4_single_subset_experiment/launchers/launch_full_slurm.sh
```

Then re-aggregate after batches complete:

```bash
python m4_single_subset_experiment/aggregate_single_results.py \
  --experiment-root outputs/m4_single_subset_experiment
```

## 10. Rough Compute Budget

Using the existing full M4 YOLOv8l run as a reference:

- full M4 training took about `6.99` hours for `100` epochs on `10332` train images
- each single-viewpoint subset contains only the images from one of the 72 viewpoints
- that is much smaller than the full M4 training set and about half of a duo-viewpoint subset

Naive scaling suggests that single-viewpoint jobs should be materially cheaper than the duo-view sweep, but still nontrivial because each run includes validation, full-test evaluation, checkpointing, and scheduler overhead.

The pilot should be used to replace rough estimates with measured Ponyland timings before launching all `72` viewpoints.

## 11. Main Risks

- if `yolov8l.pt` is not locally available and internet access is blocked, training will fail at model initialization
- if Slurm resource requests do not match Ponyland policy, adjust the generated launcher arguments and regenerate
- if the cluster Python path differs from `python`, pass the correct executable with `--python-executable`
- do not trust the full sweep until the pilot confirms dataset construction, training behavior, and evaluation outputs end to end
