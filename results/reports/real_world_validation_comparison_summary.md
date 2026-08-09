# Synthetic M4 Model on Real-World Validation Data

## Headline

The YOLOv8l model trained on synthetic M4 data performs very well on synthetic validation data, but collapses on real-world imagery. The new Lisanne UAV validation run is not an outlier; it is almost identical to the earlier curated 54-image real-world set and much closer to AU-AIR than to synthetic validation.

## Overall Comparison

| Evaluation set | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic M4 validation | n/a | n/a | 0.951 | 0.812 | 0.868 | 0.688 | `outputs/imported_runs/yolov8l_m4/M4_clean_yolov8l_run1/results.csv` |
| New Lisanne UAV real validation | 156 | 592 | 0.144 | 0.050 | 0.00976 | 0.00441 | `results/reports/real_world_validation_comparison_summary.md` |
| Earlier real_world_10_better set | 54 | n/a | n/a | n/a | ~0.011 | n/a | `external_public_real_world_baseline` |
| VisDrone MOT overlap validation | n/a | n/a | n/a | n/a | ~0.057 | n/a | `external_visdrone_baseline` |
| AU-AIR overlap validation | 32,823 | 127,265 | 0.028 | 0.0136 | 0.00380 | 0.00113 | `external_auair_baseline` |

## New Lisanne UAV Run

The new real-world validation set has 156 images and 592 labeled object instances after one duplicate label was removed. Overall performance is extremely low:

- Precision: 0.144
- Recall: 0.0501
- mAP50: 0.00976
- mAP50-95: 0.00441
- Best all-class F1 from the F1-confidence curve: about 0.02 at confidence 0.229

Per-class mAP50 from the PR curve:

| Class | mAP50 |
|---|---:|
| tent | 0.000 |
| tank | 0.018 |
| tower | 0.000 |
| container | 0.010 |
| whitevan | 0.021 |
| suv | 0.045 |
| male | 0.003 |
| rock | 0.000 |
| barrel | 0.000 |
| tree | 0.000 |

The normalized confusion matrix shows the main failure mode is missed detections: most true objects are assigned to background. Several classes are effectively never recovered, including male, rock, and barrel in this run. Threshold tuning is therefore not enough; the model is not localizing or classifying the real objects reliably.

## Interpretation

The model learned the synthetic M4 domain well, but it does not transfer to real UAV imagery. The drop from synthetic M4 `mAP50 = 0.868` to new real UAV `mAP50 = 0.00976` is about a 99 percent relative loss in mAP50. The earlier curated real-world set was also around `mAP50 = 0.011`, so the new result confirms the same domain gap on a larger, self-collected dataset.

The VisDrone overlap set is better at `mAP50 ~0.057`, driven mainly by SUV performance (`0.151`), but this is still very low compared with synthetic validation. AU-AIR is even lower than the new Lisanne set (`mAP50 = 0.0038`), so the broader pattern is consistent: synthetic-only training is not enough for robust real-world detection.

## Caveat

The local `data_collection/raw_data/self_collected_uav_validation` copy has the same number of images as the new validation run, but the class-id order in the local YAML does not cleanly match the displayed validation-table order. For thesis reporting, use the validation terminal output and saved PR/confusion plots as the authoritative metric sources unless the dataset YAML/class mapping is re-exported and verified.

