# Probability Fusion

This folder contains the synthetic-only `Synthetic Multi-Perspective Probability Fusion` experiment.

Run it from the workspace root with:

```powershell
python .\probability_fusion\run_probability_fusion_experiment.py
```

Useful options:

```powershell
python .\probability_fusion\run_probability_fusion_experiment.py `
  --evaluation-split test `
  --calibration none `
  --coalition-sizes 1 2 3 `
  --max-combinations-per-k 5000 `
  --seed 42
```

To use the full 72-view manifest-aligned cache built from `train_M4 + val + test`:

```powershell
python .\probability_fusion\run_probability_fusion_experiment.py `
  --aligned-cache-csv ".\probability_fusion\outputs\full72_manifest_aligned_cache\full72_manifest_aligned_cache.csv" `
  --evaluation-split test `
  --calibration none `
  --coalition-sizes 1 2 3
```

Notes:

- The script reuses `m4_two_drone_operational_analysis/outputs_{val,test}/scene_view_records.csv` when present.
- The aligned cache under `probability_fusion/outputs/full72_manifest_aligned_cache/` reconstructs the exact 72-view union per instance across `train_M4 + val + test`.
- If those caches are missing, it tries to rebuild them from the saved M4 GT and prediction JSON files without overwriting older experiment folders.
- Output is written to a new folder under `probability_fusion/outputs/`.
- The script reports actual cached view coverage per object instance, so it stays honest when the available cache is denser or sparser than the ideal 72-view grid.

