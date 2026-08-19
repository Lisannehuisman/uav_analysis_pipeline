# Factor-Level Viewpoint Analysis

This folder contains a higher-support alternative to the exact viewpoint analysis.

Instead of ranking full `azimuth + elevation + radius` combinations, it analyzes:

- `azimuth` only
- `elevation` only
- `radius` only

This pools over the other two viewpoint dimensions and gives each estimate many more
supporting images, which makes the conclusions more defensible for thesis writing.

Default run:

```powershell
.\.venv\Scripts\python.exe .\factor_level_viewpoint_analysis\run_factor_level_analysis.py
```

Default outputs are written to:

`factor_level_viewpoint_analysis/results_s0_m4`
