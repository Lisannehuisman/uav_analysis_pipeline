# M4 Pair-Subset Ponyland Runbook

This runbook is for the controlled M4 viewpoint-pair training experiment:

- one normal single-image `YOLOv8l` model per viewpoint pair
- pair-specific train and val subsets
- fixed evaluation protocol with `Option A` and `Option B`
- pilot-first execution on the Ponyland GPU cluster

## 1. Recommended Scientific Protocol

- `Option A` is the recommended default:
  - train on the two selected viewpoints
  - evaluate on the full fixed M4 `test` split across all 72 viewpoints
  - this measures generalization from a restricted viewpoint subset to the full viewpoint space

- `Option B` is still implemented:
  - train on the two selected viewpoints
  - evaluate only on those same two viewpoints in `test`
  - treat this as a diagnostic, not the headline result

## 2. What Has Already Been Validated Locally

- all `2556` viewpoint pairs were enumerated
- five representative pilot pairs were selected
- pair-filtered subset manifests were built and validated
- split leakage checks passed for the pilot subsets
- resumable status tracking, aggregation, plots, and reporting were generated successfully

The real pilot training and evaluation still need to run on Ponyland because the local machine here does not have the required CUDA setup for the full experiment.

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
python m4_pair_subset_experiment/launch_pair_sweep.py \
  --experiment-root outputs/m4_pair_subset_experiment \
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

- creates a Slurm array launcher for the 5 pilot pairs
- writes logs into `outputs/m4_pair_subset_experiment/logs`
- preserves resumability with one status file per pair
- evaluates each trained model with the same fixed pipeline

## 7. Launch the Pilot

Submit the generated Slurm array:

```bash
sbatch outputs/m4_pair_subset_experiment/launchers/launch_pilot_slurm.sh
```

After the pilot jobs finish, aggregate the results:

```bash
python m4_pair_subset_experiment/aggregate_pair_results.py \
  --experiment-root outputs/m4_pair_subset_experiment
```

## 8. Pilot Checks Before Full Sweep

Before launching all `2556` pairs, verify:

- each pilot pair has `subset.status = completed`
- each pilot pair has `training.status = completed`
- each pilot pair has `evaluations.option_a_full_test.status = completed`
- the `master_results.csv` rows contain sensible image counts and metrics
- the pair-specific datasets contain only the two requested viewpoints
- model outputs are stored under the expected pair directory

Files to inspect:

- `outputs/m4_pair_subset_experiment/reports/master_results.csv`
- `outputs/m4_pair_subset_experiment/reports/pair_subset_experiment_report.md`
- `outputs/m4_pair_subset_experiment/plots/top_performing_pairs.png`

## 9. Launch the Full Sweep

Only after the pilot passes, regenerate the full launcher:

```bash
python m4_pair_subset_experiment/launch_pair_sweep.py \
  --experiment-root outputs/m4_pair_subset_experiment \
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
sbatch outputs/m4_pair_subset_experiment/launchers/launch_full_slurm.sh
```

Then re-aggregate after batches complete:

```bash
python m4_pair_subset_experiment/aggregate_pair_results.py \
  --experiment-root outputs/m4_pair_subset_experiment
```

## 10. Rough Compute Budget

Using the existing full M4 YOLOv8l run as a reference:

- full M4 training took about `6.99` hours for `100` epochs on `10332` train images
- pilot pair subsets currently contain about `282` to `288` train images each
- that is roughly `36x` fewer training images than full M4

Naive training-only scaling suggests about:

- `~12` minutes of pure training time per pair

A more realistic operational budget, including validation, evaluation on the full fixed test set, checkpointing, and cluster overhead, is:

- `~20` to `35` minutes per pair

For the full sweep:

- serialized budget: about `850` to `1490` GPU-hours
- at `16` concurrent array tasks: about `53` to `93` wall-clock hours, plus queue delays

Treat this as a planning estimate, not a guarantee. The pilot should be used to replace these rough numbers with measured Ponyland timings.

## 11. Main Risks

- if `yolov8l.pt` is not locally available and internet access is blocked, training will fail at model initialization
- if Slurm resource requests do not match Ponyland policy, adjust the generated launcher arguments and regenerate
- if the cluster Python path differs from `python`, pass the correct executable with `--python-executable`
- do not trust the full sweep until the pilot confirms dataset construction, training behavior, and evaluation outputs end to end
