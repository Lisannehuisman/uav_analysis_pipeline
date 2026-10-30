# YOLO Model Comparison

This workspace contains a small script to compare two trained YOLO detection models on the same dataset split.

It also now includes project-specific experiment folders for the fixed-M4 viewpoint study, including:

- `m4_two_drone_operational_analysis`
  Operational `1/2/3-view` inference-time analysis with the full `YOLOv8l_M4` detector held fixed.
- `m4_cross_view_box_fusion_analysis`
  Conservative matched-box late fusion across views using the saved M4 test predictions and ground truth.

It produces:

- dataset-level `precision`, `recall`, `F1`, `mAP50`, and `mAP50-95`
- per-class `AP50-95` comparison across the two models
- per-image CSV files for both models
- boxplots over the test images for per-image `precision`, `recall`, `F1`, `AP50`, and `AP50-95`

## Files

- `compare_yolo_models.py`
- `requirements.txt`

## What you need

You need a YOLO dataset YAML file with a valid `test:` entry, for example:

```yaml
path: C:/DATA/airsim/thesis/datasets/my_dataset
train: images/train
val: images/val
test: images/test
names:
  0: object
```

The script expects standard YOLO folder layout, where labels mirror the image tree:

- `images/test/...`
- `labels/test/...`

## Install

Open this folder in Visual Studio Code and run:

```powershell
pip install -r requirements.txt
```

## Run

Example command:

```powershell
python compare_yolo_models.py `
  --model-a "C:\DATA\airsim\thesis\results\yolov8l_m4\M4_clean_yolov8l_run1\weights\best.pt" `
  --model-b "C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M4_yolov8l\weights\best.pt" `
  --label-a "M4 clean" `
  --label-b "S0_M4" `
  --data "C:\path\to\data.yaml" `
  --split test `
  --imgsz 640 `
  --batch 16 `
  --output-dir "comparison_output"
```

## Output

The script writes these files into `--output-dir`:

- `aggregate_metrics.csv`
- `per_class_ap50_95.csv`
- `per_image_metrics_model_a.csv`
- `per_image_metrics_model_b.csv`
- `aggregate_metrics.png`
- `per_class_ap50_95.png`
- `per_image_boxplots.png`

## Reuse Cached Results

If you already ran the main comparison once, you can create boxplots grouped by object type without rerunning inference:

```powershell
python plot_per_object_boxplots.py `
  --per-image-a ".\comparison_output\per_image_metrics_model_a.csv" `
  --per-image-b ".\comparison_output\per_image_metrics_model_b.csv" `
  --per-class-csv ".\comparison_output\per_class_ap50_95.csv" `
  --label-a "M4_clean" `
  --label-b "S0_M4" `
  --output-dir ".\comparison_output\per_object_boxplots"
```

This creates one plot per metric, grouped by object type, using the cached CSV files you already produced.

## Related M4 Analyses

Run the operational multiview analysis:

```powershell
.\.venv\Scripts\python.exe .\m4_two_drone_operational_analysis\analyze_two_drone_operational.py
```

Run the conservative box-level fusion extension:

```powershell
powershell -ExecutionPolicy Bypass -File .\m4_cross_view_box_fusion_analysis\run_box_fusion_analysis.ps1
```

## Notes

- Dataset-level metrics come from Ultralytics validation.
- Per-class `AP50-95` also comes directly from Ultralytics validation output.
- Per-image `precision`, `recall`, and `F1` use IoU `0.50`.
- Per-image `AP50` and `AP50-95` are computed image-by-image from matched detections so you can compare the spread across the test set.
- The boxplots include light jittered points so you can see how densely the image-level scores are distributed, inspired by your notebook's boxplot-plus-points style.
