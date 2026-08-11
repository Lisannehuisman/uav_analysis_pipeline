## M4 Pair Partial Analysis

This folder is the local Visual Studio / VS Code workspace for analyzing the
current partial Ponyland pair-sweep results.

It is intentionally separate from `outputs/m4_pair_subset_experiment/`, which
still contains older local artifacts and should not be treated as the source of
truth for the running cluster sweep.

### Layout

- `sync_from_ponyland.ps1`
  - Pulls the latest aggregated reports, plots, and pair manifest from Ponyland.
- `analyze_partial_pair_results.py`
  - Reads the synced partial results and generates current rankings, viewpoint
    contribution summaries, and a markdown report.
- `data/current_snapshot/`
  - Latest synced Ponyland snapshot.
- `outputs/`
  - Analysis products generated from the current snapshot.

### Typical workflow

1. Sync the latest partial results from Ponyland.
2. Run the partial analysis script locally.
3. Read `outputs/current_partial_results_summary.md`.

### Example

```powershell
powershell -ExecutionPolicy Bypass -File .\m4_pair_partial_analysis\sync_from_ponyland.ps1
python .\m4_pair_partial_analysis\analyze_partial_pair_results.py
```
