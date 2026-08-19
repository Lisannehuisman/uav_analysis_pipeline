# Ponyland Test-Set Runbook

This runbook is for the final thesis comparison on the `test` split across:

- `YOLOv8n`
- `YOLOv8l`
- `Faster R-CNN`

and the five regimes:

- `M1`
- `M2a`
- `M2b`
- `M3`
- `M4`

## 1. Copy the comparison folder to Ponyland

Copy this folder to the cluster:

- `detector_family_comparison`

You can use `scp`, `rsync`, or your normal SSH workflow.

## 2. Prepare a Python environment on Ponyland

Install the packages needed by the scripts:

```bash
pip install numpy matplotlib pyyaml pillow pycocotools ultralytics
```

Your Ponyland environment also needs:

- `torch`
- `torchvision`

with GPU support if available.

## 3. Create a Ponyland path config

If your Ponyland layout matches the standard structure, generate the config automatically:

```bash
bash detector_family_comparison/create_ponyland_config.sh /vol/tensusers6/lisannehuisman detector_family_comparison/ponyland_config.json
```

If needed, you can still edit manually from:

- `detector_family_comparison/ponyland_config.template.json`

## 4. Run the full GPU test pipeline

From the project root on Ponyland:

```bash
bash detector_family_comparison/run_test_pipeline_gpu.sh detector_family_comparison/ponyland_config.json 0 cuda
```

That pipeline will:

- regenerate `YOLOv8n` and `YOLOv8l` test predictions
- generate Faster R-CNN test predictions from `model_final.pth`
- rebuild the shared 3-way test summary CSV
- regenerate the regime comparison figure
- regenerate the regime metric table
- regenerate the per-class AP figures

## 5. Collect the final outputs

The final outputs will be under the `default_output_dir` you set in the Ponyland config JSON.

Important files include:

- `standardized_test_summary.csv`
- `standardized_test_summary.png`
- `regime_metric_comparison.png`
- `regime_metric_table.csv`
- `regime_metric_table.png`
- `per_class_plots/per_class_ap50_95_test.png`
- `per_class_plots/per_class_ap50_test.png`
- `per_class_plots/per_class_ap75_test.png`
