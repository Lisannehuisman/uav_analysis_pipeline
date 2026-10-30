# M4 Matched-Control Experiment

This folder adds a fairer control for the headline training comparison.

Instead of comparing the best single-view or pair-view model directly against the full `YOLOv8l_M4` model, it builds a control that:

- still draws images from the full M4 viewpoint space
- uses the exact same `train` and `val` image counts as the source model
- keeps the same YOLOv8l training and full-test evaluation protocol

## Typical Workflow

1. Generate a manifest for the matched controls:

```powershell
.\.venv\Scripts\python.exe .\m4_matched_control_experiment\prepare_control_manifest.py
```

2. Run one or more controls from that manifest:

```powershell
.\.venv\Scripts\python.exe .\m4_matched_control_experiment\run_control_experiment.py --control-id mc_single_best_s00
.\.venv\Scripts\python.exe .\m4_matched_control_experiment\run_control_experiment.py --control-id mc_pair_best_s00
```

3. Aggregate the completed controls:

```powershell
.\.venv\Scripts\python.exe .\m4_matched_control_experiment\aggregate_control_results.py
```

4. Regenerate the headline comparison:

```powershell
.\.venv\Scripts\python.exe .\full_m4_vs_single_pair_operational_analysis\run_comparison.py
```

When the matched-control results exist, the comparison summary will automatically include the equal-image-count M4 controls.
