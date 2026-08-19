# Detector Family Comparison

This folder organizes a fair comparison between three detector families across the same five regimes:

- `YOLOv8n`
- `YOLOv8l`
- `Faster R-CNN`

and these regimes:

- `M1`
- `M2a`
- `M2b`
- `M3`
- `M4`

## Files

- `comparison_config.py`
- `compare_saved_results.py`
- `standardized_test_eval.py`
- `generate_frcnn_predictions.py`
- `generate_test_reports.py`
- `create_ponyland_config.sh`
- `ponyland_config.template.json`
- `run_test_pipeline_gpu.sh`

## Recommended Metrics

For a clean conclusion, compare at least:

- `precision`
- `recall`
- `F1`
- `mAP50`
- `mAP50-95`
- `matched_mean_iou`

Two extra metrics that are often useful:

- `AP75`
- per-class `AP50-95`

## Workflow

### 1. Quick comparison from existing result folders

This gives you a fast overview from the files you already have:

```powershell
python .\detector_family_comparison\compare_saved_results.py
```

This is useful for initial organization, but it is not the final fair comparison because:

- YOLO values come from saved training/validation outputs
- Faster R-CNN provides COCO AP metrics but not directly comparable saved `precision`/`recall`/`F1`

### 2. Standardized comparison on a shared split

This is the stronger script to support your conclusion.

The configuration in [comparison_config.py](original_workspace/detector_family_comparison/comparison_config.py) is already filled with your five regime dataset YAML paths.

For the fair three-family comparison that is ready right now, run the shared `val` evaluation:

```powershell
python .\detector_family_comparison\standardized_test_eval.py --split val
```

If you want to run it in smaller resumable chunks, you can limit regimes explicitly:

```powershell
python .\detector_family_comparison\standardized_test_eval.py --split val --regimes M1 M2a
```

That run will:

- reuse the official COCO validation annotations for each regime
- run `YOLOv8n` and `YOLOv8l` on the shared validation images
- reuse Faster R-CNN COCO prediction JSONs from the saved result folders
- evaluate all three families with one shared COCO-style pipeline
- save summary CSVs and comparison plots

For a two-model comparison on the `test` split, you can still run:

```powershell
python .\detector_family_comparison\standardized_test_eval.py --split test --detectors YOLOv8n YOLOv8l
```

## Running On Ponyland / Linux

`comparison_config.py` now supports a JSON override via the `DETECTOR_COMPARISON_CONFIG_JSON` environment variable.

Use [ponyland_config.template.json](original_workspace/detector_family_comparison/ponyland_config.template.json) as the template for your Linux-side paths, then run:

```bash
bash detector_family_comparison/run_test_pipeline_gpu.sh /absolute/path/to/ponyland_config.json 0 cuda
```

That GPU pipeline will:

- rerun `YOLOv8n` and `YOLOv8l` on the full `test` split
- generate fresh Faster R-CNN test predictions from `model_final.pth`
- rebuild the shared 3-way `test` summary CSV
- regenerate the regime comparison plot, metric table, and per-class plots

## Important Note About Faster R-CNN

Your saved Faster R-CNN `inference/coco_instances_results.json` files align with the regime `val` COCO annotations, not with the `test` split. That means:

- a fair `YOLOv8n` vs `YOLOv8l` vs `Faster R-CNN` comparison is available now on `val`
- a fair three-family comparison on `test` needs fresh Faster R-CNN test inference from `model_final.pth`

Until that test inference exists, do not use the saved Faster R-CNN outputs as if they were test-set results.

