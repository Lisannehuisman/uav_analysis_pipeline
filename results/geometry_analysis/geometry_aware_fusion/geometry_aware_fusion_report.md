# Geometry-Aware Fusion Report

## Purpose

This analysis tests geometry-aware target fusion baselines on the same scene-view evaluation base used by the earlier multiview comparisons.
It uses manifest-derived camera-target geometry to predict a per-view reliability prior, then injects that prior into view selection and late fusion.

## Important boundary

This is a geometry-aware reliability baseline, not full 3D outline reprojection.
It uses camera-target pose metadata but does not yet use camera intrinsics, a 3D object mesh, or explicit shared-plane box reprojection.

## Geometry prior quality

- Geometry-regression prior correlation: `0.6079`
- Viewpoint-cell prior correlation: `0.6149`
- Hybrid prior correlation: `0.6313`
- Mean geometry weight: `0.9983`
- Mean viewpoint-cell weight: `0.9995`
- Mean hybrid weight: `0.9986`

## Single-view reference

- Scene-balanced single-view reference quality: `0.8732`

## Best overall two-view method

- Best method: `Viewpoint-cell OR + best IoU`
- Method family: `Geometry-aware accumulation`
- Scene-balanced two-view quality: `0.9515`
- Gain versus single-view reference: `+0.0784`

## Best geometry-aware two-view method

- Best geometry-aware method: `Viewpoint-cell OR + best IoU`
- Scene-balanced two-view quality: `0.9515`
- Gain versus single-view reference: `+0.0784`
- Gap versus `Noisy-OR + best IoU`: `+0.0000`
- Pair-row win rate versus `Noisy-OR + best IoU`: `0.7207`
- Mean pair-score gap versus `Noisy-OR + best IoU`: `-0.0001`

## Interpreting the geometry-aware methods

- `Geometry prior selector` chooses the view with the strongest predicted geometry reliability prior and keeps that view's actual target quality.
- `Geometry-calibrated selector` chooses the view with the strongest confidence after geometry reweighting.
- `Viewpoint-cell prior selector` uses leave-one-scene-out class-specific evidence for the exact lattice cell rather than a smooth regression surface.
- `Geometry-weighted OR` keeps the noisy-OR late-fusion logic, but calibrates each view's confidence with a leave-one-scene-out geometry prior.
- `Viewpoint-cell OR` and `Hybrid geometry+cell OR` test whether a discrete lattice prior helps more than a smooth geometric prior.

## How to show the value of one technique over another

- Use `overall_method_summary.csv` and `overall_geometry_method_comparison.png` to show end-to-end quality. Right now the best deployable geometry-aware method is `Viewpoint-cell OR + best IoU` at `0.9515`, but its gap to `Noisy-OR + best IoU` is essentially zero.
- Use `method_value_summary.csv` and `method_tradeoff_scatter.png` to separate rescue from true corroboration. For the best geometry-aware method, mean lift versus best constituent is `+0.0418` and rescue rate is `0.9284`.
- Use `pairwise_method_comparison.csv` and `pairwise_method_gap_heatmap.png` to answer the direct question 'which method beats which'. Here the key nuance is that `Viewpoint-cell OR + best IoU` wins on `0.7207` of pair rows against `Noisy-OR + best IoU`, yet its mean pair-score gap stays `-0.0001`, so the result should be presented as a near-tie rather than a decisive win.

## Strong added viewpoints by method

### Geometry prior selector

- `elmid-radnear-az090`: mean lift vs primary `+0.0572`, lift vs best constituent `-0.0167`, support `346`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0539`, lift vs best constituent `-0.0091`, support `338`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0462`, lift vs best constituent `-0.0112`, support `427`, rescue|primary miss `0.8571`
- `elmid-radmid-az045`: mean lift vs primary `+0.0415`, lift vs best constituent `-0.0095`, support `374`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0411`, lift vs best constituent `-0.0122`, support `234`, rescue|primary miss `0.8889`

### Geometry-calibrated selector

- `elmid-radnear-az090`: mean lift vs primary `+0.0640`, lift vs best constituent `-0.0099`, support `346`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0569`, lift vs best constituent `-0.0061`, support `338`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0521`, lift vs best constituent `-0.0054`, support `427`, rescue|primary miss `1.0000`
- `ellow-radnear-az090`: mean lift vs primary `+0.0496`, lift vs best constituent `-0.0083`, support `362`, rescue|primary miss `1.0000`
- `elmid-radnear-az180`: mean lift vs primary `+0.0472`, lift vs best constituent `-0.0049`, support `352`, rescue|primary miss `1.0000`

### Viewpoint-cell prior selector

- `elmid-radnear-az090`: mean lift vs primary `+0.0499`, lift vs best constituent `-0.0241`, support `346`, rescue|primary miss `0.7778`
- `ellow-radnear-az225`: mean lift vs primary `+0.0394`, lift vs best constituent `-0.0088`, support `394`, rescue|primary miss `1.0000`
- `ellow-radnear-az090`: mean lift vs primary `+0.0391`, lift vs best constituent `-0.0189`, support `362`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0383`, lift vs best constituent `-0.0192`, support `427`, rescue|primary miss `0.6429`
- `elmid-radnear-az045`: mean lift vs primary `+0.0380`, lift vs best constituent `-0.0250`, support `338`, rescue|primary miss `0.4545`

### Geometry-weighted OR + best IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1342`, lift vs best constituent `+0.0602`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.1073`, lift vs best constituent `+0.0499`, support `427`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.1018`, lift vs best constituent `+0.0507`, support `374`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0999`, lift vs best constituent `+0.0466`, support `234`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0996`, lift vs best constituent `+0.0366`, support `338`, rescue|primary miss `1.0000`

### Geometry-weighted OR + mean IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1101`, lift vs best constituent `+0.0361`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.0926`, lift vs best constituent `+0.0351`, support `427`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.0881`, lift vs best constituent `+0.0370`, support `374`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.0873`, lift vs best constituent `+0.0340`, support `234`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0838`, lift vs best constituent `+0.0208`, support `338`, rescue|primary miss `1.0000`

### Viewpoint-cell OR + best IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1288`, lift vs best constituent `+0.0548`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.1066`, lift vs best constituent `+0.0491`, support `427`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.1003`, lift vs best constituent `+0.0470`, support `234`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.0998`, lift vs best constituent `+0.0487`, support `374`, rescue|primary miss `1.0000`
- `elmid-radfar-az090`: mean lift vs primary `+0.0965`, lift vs best constituent `+0.0498`, support `288`, rescue|primary miss `1.0000`

### Hybrid geometry+cell OR + best IoU

- `elmid-radnear-az090`: mean lift vs primary `+0.1322`, lift vs best constituent `+0.0582`, support `346`, rescue|primary miss `1.0000`
- `elmid-radmid-az000`: mean lift vs primary `+0.1070`, lift vs best constituent `+0.0496`, support `427`, rescue|primary miss `1.0000`
- `elmid-radmid-az045`: mean lift vs primary `+0.1009`, lift vs best constituent `+0.0498`, support `374`, rescue|primary miss `1.0000`
- `elhigh-radnear-az045`: mean lift vs primary `+0.1006`, lift vs best constituent `+0.0473`, support `234`, rescue|primary miss `1.0000`
- `elmid-radnear-az045`: mean lift vs primary `+0.0984`, lift vs best constituent `+0.0353`, support `338`, rescue|primary miss `1.0000`

## Files

- `geometry_priors.csv`
- `geometry_prior_diagnostics.csv`
- `overall_method_summary.csv`
- `method_value_summary.csv`
- `pairwise_method_comparison.csv`
- `ordered_pair_gain_summary.csv`
- `added_viewpoint_summary.csv`
- `added_viewpoint_headlines.csv`
- `overall_geometry_method_comparison.png`
- `predicted_prior_vs_actual.png`
- `method_tradeoff_scatter.png`
- `pairwise_method_gap_heatmap.png`