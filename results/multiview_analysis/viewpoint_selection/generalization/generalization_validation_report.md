# Viewpoint Subset Generalization Report

## Interpretation Boundary

This script performs repeated scene-split validation inside the cached M4 test records. The held-out fold is unseen by the subset-selection step, but it should not be described as a guaranteed novel-object-instance split.

Dataset audit fraction of instances present in train, val, and test: `1.000000`.

## Scene-Split Result

- k=`1`: selected strict quality `0.9355`, held-out strict quality `0.9196`, mean gap `0.0159`
- k=`2`: selected strict quality `0.9731`, held-out strict quality `0.5460`, mean gap `0.4272`
- k=`3`: selected strict quality `0.9753`, held-out strict quality `0.0859`, mean gap `0.8894`

## Selection Frequency And Held-Out Support

Rows with near-zero held-out support are evidence that exact fixed 2-view or 3-view combinations are sparse in the current cache. Treat those as a data-coverage limitation, not as deployable recommendations.

### k=1
- `ellow-radnear-az225`: selected in `1.000` of scene splits, held-out strict quality `0.9417`, held-out support `7.20` scenes
- `elmid-radnear-az315`: selected in `1.000` of scene splits, held-out strict quality `0.9389`, held-out support `6.60` scenes
- `elmid-radnear-az000`: selected in `0.760` of scene splits, held-out strict quality `0.9213`, held-out support `4.08` scenes
- `elmid-radnear-az045`: selected in `0.700` of scene splits, held-out strict quality `0.9211`, held-out support `6.00` scenes
- `elhigh-radnear-az000`: selected in `0.500` of scene splits, held-out strict quality `0.9171`, held-out support `7.20` scenes

### k=2
- `ellow-radnear-az225 + elmid-radfar-az315`: selected in `0.480` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `ellow-radnear-az270 + elmid-radmid-az225`: selected in `0.480` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `elmid-radnear-az270 + elmid-radnear-az315`: selected in `0.420` of scene splits, held-out strict quality `0.1845`, held-out support `0.19` scenes
- `ellow-radnear-az270 + ellow-radmid-az315`: selected in `0.400` of scene splits, held-out strict quality `0.3868`, held-out support `0.40` scenes
- `elmid-radnear-az045 + elhigh-radfar-az180`: selected in `0.340` of scene splits, held-out strict quality `0.2856`, held-out support `0.29` scenes

### k=3
- `ellow-radnear-az225 + ellow-radfar-az180 + elhigh-radnear-az225`: selected in `0.560` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `ellow-radnear-az225 + ellow-radfar-az180 + elhigh-radfar-az315`: selected in `0.520` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `ellow-radnear-az225 + ellow-radmid-az225 + ellow-radfar-az180`: selected in `0.520` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `ellow-radnear-az225 + elmid-radnear-az315 + elhigh-radfar-az270`: selected in `0.500` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes
- `ellow-radnear-az270 + elmid-radmid-az180 + elhigh-radnear-az135`: selected in `0.480` of scene splits, held-out strict quality `0.0000`, held-out support `0.00` scenes


## Target-Absent Audit

- Target-absent views: `40`; false-alarm views: `3`; false-alarm rate `0.0750`.

## Generated Files

- `scene_split_generalization_trials.csv`
- `scene_split_generalization_summary.csv`
- `top_subset_selection_frequency.csv`
- `recommended_subsets_scene_split.csv`
- `generalization_readiness.csv`
- `target_absent_viewpoint_audit.csv`
- `target_absent_false_alarm_examples.csv`
- `plots/scene_split_generalization_gap.png`
- `plots/selection_stability_top_subsets.png`
- `plots/target_absent_false_alarm_rates.png`
