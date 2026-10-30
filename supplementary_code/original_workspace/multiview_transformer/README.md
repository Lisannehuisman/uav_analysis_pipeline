# Multiview Transformer

This folder contains a clean baseline for learning **which combinations of views are good** from the images themselves.

It is intentionally **not** a full multi-view detector yet. Instead, it is a practical first step that is much easier to train and rank on Ponyland:

- one shared image encoder processes each selected view
- viewpoint metadata is added to the image features
- a small transformer fuses the selected views
- the model predicts a **set score** for the chosen multi-view combination
- after training, fixed single, pair, and triple viewpoint sets can be ranked on a held-out split

## What It Is Good For

- learning image-aware viewpoint utility rather than hand-designed angle rules
- comparing `1-view`, `2-view`, and `3-view` combinations with one model
- ranking the best fixed angles overall and per object class
- running cleanly on Ponyland with one GPU

## What It Is Not Yet

- not a DETR-style multi-view detector
- not cross-view box matching
- not object-identity tracking across views

That is deliberate. This baseline gets the **scene grouping, image loading, transformer fusion, training loop, and angle ranking** in place first.

## Files

- `build_scene_manifest.py`
  Builds one CSV row per image/view with scene id, viewpoint metadata, and label-derived targets.
- `dataset.py`
  Groups manifest rows by scene and samples `1..K` views per training example.
- `model.py`
  Shared CNN encoder plus cross-view transformer.
- `train.py`
  Trains the model and saves `best.pt` and `last.pt`.
- `evaluate_sets.py`
  Scores fixed viewpoint sets on a chosen split.
- `rank_viewpoints.py`
  Aggregates evaluation CSVs into top-combo and per-viewpoint rankings.
- `launch_ponyland.py`
  Writes a simple Ponyland launcher script.
- `configs/m4_mv3.yaml`
  Default experiment config.

## Targets

The default config uses:

- `score_column = target_max_area_norm`
- `visible_column = target_visible`

Those come directly from the YOLO labels, so the pipeline can run immediately.

For a stronger thesis version, you can merge a detector-quality CSV into the manifest and switch the score column to something like:

- `target_ap50_95`
- `target_strict_quality_iou50`

That makes the ranking target much closer to detection utility instead of pure geometric visibility.

## Local CPU Smoke Test

This is safe to do locally before going to Ponyland:

```powershell
python multiview_transformer/build_scene_manifest.py `
  --data-yaml .\my_data.yaml `
  --output-csv .\outputs\multiview_transformer\manifests\m4_scene_manifest.csv

python multiview_transformer/train.py `
  --config .\multiview_transformer\configs\m4_mv3.yaml `
  --device cpu `
  --epochs 1
```

That is only a smoke test. Real training should happen on Ponyland.

## Ranking Workflow

After training:

```powershell
python multiview_transformer/evaluate_sets.py `
  --checkpoint .\outputs\multiview_transformer\m4_mv3_resnet18\checkpoints\best.pt `
  --combo-size 1 `
  --split test `
  --require-complete

python multiview_transformer/evaluate_sets.py `
  --checkpoint .\outputs\multiview_transformer\m4_mv3_resnet18\checkpoints\best.pt `
  --combo-size 2 `
  --split test `
  --require-complete `
  --shortlist-from .\outputs\multiview_transformer\m4_mv3_resnet18\eval\test\combo_1_summary.csv `
  --top-k 16

python multiview_transformer/evaluate_sets.py `
  --checkpoint .\outputs\multiview_transformer\m4_mv3_resnet18\checkpoints\best.pt `
  --combo-size 3 `
  --split test `
  --require-complete `
  --shortlist-from .\outputs\multiview_transformer\m4_mv3_resnet18\eval\test\combo_1_summary.csv `
  --top-k 16

python multiview_transformer/rank_viewpoints.py `
  --eval-dir .\outputs\multiview_transformer\m4_mv3_resnet18\eval\test
```

## Why The Shortlist Matters

Exhaustive singles are easy. Exhaustive pairs and triples can get large quickly.

The intended workflow is:

1. rank all single viewpoints
2. keep the best `N`
3. enumerate pairs and triples only inside that shortlist

That keeps Ponyland runtime reasonable while still answering the angle-selection question cleanly.

