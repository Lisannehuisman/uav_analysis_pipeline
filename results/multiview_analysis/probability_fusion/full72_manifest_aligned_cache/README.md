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

The train, validation and test prediction caches and the YOLOv8l-M4 checkpoint are not included in this repository because of their size. They can be regenerated using the detector-evaluation code under `src/detector_analysis/detector_family_comparison/`.

## Files

- `full72_manifest.csv`
- `full72_manifest_aligned_cache.csv`
- `instance_coverage_summary.csv`
- `viewpoint_split_check.csv`

The generated outputs for this experiment are stored in this directory.