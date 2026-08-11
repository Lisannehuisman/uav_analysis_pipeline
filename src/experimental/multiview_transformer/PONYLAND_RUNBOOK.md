# Ponyland Runbook

This runbook is for training the multiview transformer baseline on Ponyland with GPU access.

## 1. What To Run On Ponyland

Use Ponyland for:

- real training
- full single-view ranking
- shortlisted pair and triple ranking

Use the local CPU only for:

- smoke tests
- syntax checks
- small manifest checks

## 2. Prepare The Workspace

Copy or sync the project to Ponyland, then move to the repo root.

Example placeholder:

```bash
cd /vol/tensusers6/lisannehuisman/New-project
```

## 3. Prepare The Environment

Install the multiview requirements in your Ponyland environment:

```bash
pip install -r multiview_transformer/requirements.txt
```

You also need a GPU-enabled PyTorch and torchvision install.

## 4. Build The Scene Manifest

This works directly from the YOLO dataset YAML and labels:

```bash
python multiview_transformer/build_scene_manifest.py \
  --data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml \
  --output-csv outputs/multiview_transformer/manifests/m4_scene_manifest.csv
```

If you later create a detector-quality CSV for train and val as well, merge it here:

```bash
python multiview_transformer/build_scene_manifest.py \
  --data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml \
  --output-csv outputs/multiview_transformer/manifests/m4_scene_manifest.csv \
  --quality-csv /vol/tensusers6/lisannehuisman/quality/m4_scene_quality.csv \
  --quality-key file_name
```

## 5. Train The Model

Start with the provided config:

```bash
python multiview_transformer/train.py \
  --config multiview_transformer/configs/m4_mv3.yaml
```

Helpful overrides:

```bash
python multiview_transformer/train.py \
  --config multiview_transformer/configs/m4_mv3.yaml \
  --epochs 60 \
  --device cuda:0
```

## 6. Rank Single, Pair, And Triple Viewpoint Sets

First rank all single viewpoints:

```bash
python multiview_transformer/evaluate_sets.py \
  --checkpoint outputs/multiview_transformer/m4_mv3_resnet18/checkpoints/best.pt \
  --combo-size 1 \
  --split test \
  --require-complete
```

Then build pairs and triples only from the top singles:

```bash
python multiview_transformer/evaluate_sets.py \
  --checkpoint outputs/multiview_transformer/m4_mv3_resnet18/checkpoints/best.pt \
  --combo-size 2 \
  --split test \
  --require-complete \
  --shortlist-from outputs/multiview_transformer/m4_mv3_resnet18/eval/test/combo_1_summary.csv \
  --top-k 16

python multiview_transformer/evaluate_sets.py \
  --checkpoint outputs/multiview_transformer/m4_mv3_resnet18/checkpoints/best.pt \
  --combo-size 3 \
  --split test \
  --require-complete \
  --shortlist-from outputs/multiview_transformer/m4_mv3_resnet18/eval/test/combo_1_summary.csv \
  --top-k 16
```

Finally aggregate:

```bash
python multiview_transformer/rank_viewpoints.py \
  --eval-dir outputs/multiview_transformer/m4_mv3_resnet18/eval/test
```

## 7. Generate A Slurm Launcher

You can generate a ready-to-submit launcher:

```bash
python multiview_transformer/launch_ponyland.py \
  --config multiview_transformer/configs/m4_mv3.yaml \
  --data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml \
  --launcher slurm \
  --stage pipeline \
  --workspace-root /vol/tensusers6/lisannehuisman/New-project \
  --output-script outputs/multiview_transformer/launchers/launch_pipeline_slurm.sh
```

Submit it:

```bash
sbatch outputs/multiview_transformer/launchers/launch_pipeline_slurm.sh
```

## 8. Recommended First Ponyland Pilot

Start with:

- `resnet18`
- image size `224`
- `40` epochs
- batch size `8`
- one GPU
- shortlist pairs and triples from the top `16` singles

That is a good pilot before spending more cluster time.

## 9. Recommended Thesis Upgrade Path

The default target is label-derived visibility/area, because that runs immediately.

For the stronger version:

1. generate detector-quality supervision for train and val as well
2. rebuild the manifest with that quality CSV
3. switch `score_column` in `configs/m4_mv3.yaml` to `target_strict_quality_iou50`
4. retrain and rerun the ranking pipeline

That gives you angle rankings tied much more directly to detection performance.

