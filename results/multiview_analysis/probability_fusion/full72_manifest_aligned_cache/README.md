# Full 72-View Manifest-Aligned Cache

## What this cache is

- An exact synthetic object-instance manifest over `train_M4 + val + test`.
- Each row is aligned to one `(instance_id, viewpoint)` cell in the intended 72-view grid.
- Cached target-match scores are reused for `val` and `test` from the existing M4 scene-view caches.
- `train` rows are included in the manifest now, but their detector scores remain pending until a train prediction JSON is generated.

## Coverage

- Total rows: `14760`.
- Instance count: `205`.
- Instances with a complete 72-view union: `205` / `205`.
- Rows with cached scores available now: `14760`.
- Rows still missing scores now: `0`.

## Split structure

- The full 72-view grid is distributed across splits, not stored inside one split.
- In this dataset, `train` carries the majority of viewpoints per instance, while `val` and `test` provide the remaining viewpoint cells.

## Viewpoint inventory check

- Viewpoint rows matching expected train+val+test counts: `72` / `72`.

## Current score sources

- Train predictions expected at: `C:\Users\lisan\OneDrive\Documents\New project\outputs\detector_family_comparison\standardized_train_eval\predictions\YOLOv8l_M4_train_predictions.json`.
- Val predictions reused from: `C:\Users\lisan\OneDrive\Documents\New project\outputs\detector_family_comparison\standardized_val_eval\predictions\YOLOv8l_M4_val_predictions.json`.
- Test predictions reused from: `C:\Users\lisan\OneDrive\Documents\New project\outputs\detector_family_comparison\standardized_test_eval\predictions\YOLOv8l_M4_test_predictions.json`.
- Imported M4 model weights stored at: `C:\Users\lisan\OneDrive\Documents\New project\outputs\imported_runs\yolov8l_m4\M4_clean_yolov8l_run1\weights\best.pt`.

## Files

- `full72_manifest.csv`
- `full72_manifest_aligned_cache.csv`
- `instance_coverage_summary.csv`
- `viewpoint_split_check.csv`

Generated in `C:\Users\lisan\OneDrive\Documents\New project\probability_fusion\outputs\full72_manifest_aligned_cache`.
