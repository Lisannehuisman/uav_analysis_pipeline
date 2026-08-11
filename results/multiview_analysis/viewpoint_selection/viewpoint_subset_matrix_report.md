# Viewpoint Subset Matrix Report

## Inputs
- Scene/view records: `C:\Users\lisan\OneDrive\Documents\New project\m4_two_drone_operational_analysis\outputs\scene_view_records.csv`
- Records: `2214`
- Scene keys: `205`
- Absolute viewpoints: `72`
- Target-absent views: `40` (`0.0181`)

## Method Boundary

This analysis reuses cached fixed-detector M4 records. It ranks viewpoint subsets by target-centric evidence available in the selected views. It does not retrain detectors and does not assume calibrated 3D geometry.

## Support Distribution

Exact fixed subsets are only evaluated on scenes where all selected viewpoints are present. This matters most for triples.

- k=`1`: `72` observed subsets, support min/median/max = `20`/`30`/`43` scenes
- k=`2`: `2532` observed subsets, support min/median/max = `1`/`4`/`14` scenes
- k=`3`: `29882` observed subsets, support min/median/max = `1`/`1`/`6` scenes

Use the scene-split validation script before treating high-scoring sparse 2-view or 3-view exact subsets as robust recommendations.

## Best Supported Subsets

Minimum support for the headline list: `3` scenes.

### 1 Viewpoint(s)
- `ellow-radnear-az225`: strict quality `0.9423`, target found OR `1.0000`, support `36` scenes
- `elmid-radnear-az315`: strict quality `0.9383`, target found OR `1.0000`, support `33` scenes
- `elmid-radnear-az000`: strict quality `0.9311`, target found OR `1.0000`, support `21` scenes
- `elmid-radnear-az045`: strict quality `0.9301`, target found OR `1.0000`, support `32` scenes
- `elhigh-radnear-az000`: strict quality `0.9291`, target found OR `1.0000`, support `34` scenes

### 2 Viewpoint(s)
- `ellow-radnear-az270 + elmid-radmid-az225`: strict quality `0.9741`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az225 + elmid-radfar-az315`: strict quality `0.9736`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az270 + ellow-radmid-az315`: strict quality `0.9721`, target found OR `1.0000`, support `4` scenes
- `elmid-radnear-az270 + elmid-radnear-az315`: strict quality `0.9717`, target found OR `1.0000`, support `4` scenes
- `elmid-radnear-az045 + elhigh-radfar-az180`: strict quality `0.9717`, target found OR `1.0000`, support `4` scenes

### 3 Viewpoint(s)
- `ellow-radnear-az225 + elmid-radnear-az315 + elhigh-radfar-az270`: strict quality `0.9759`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az270 + elmid-radmid-az180 + elhigh-radnear-az135`: strict quality `0.9754`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az225 + ellow-radfar-az180 + elhigh-radnear-az225`: strict quality `0.9754`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az225 + ellow-radfar-az180 + elhigh-radfar-az315`: strict quality `0.9754`, target found OR `1.0000`, support `3` scenes
- `ellow-radnear-az225 + ellow-radmid-az225 + ellow-radfar-az180`: strict quality `0.9754`, target found OR `1.0000`, support `3` scenes

## Generated Files

- `viewpoint_inventory.csv`
- `subset_scores.csv`
- `best_subsets_by_k.csv`
- `pair_matrix_strict_quality.csv`
- `pair_matrix_complementarity.csv`
- `pair_matrix_support.csv`
- `pair_matrix_support_weighted_fusion.csv`
- `factor_pattern_summary.csv`
- `plots/pair_matrix_strict_quality.png`
- `plots/pair_matrix_complementarity.png`
- `plots/subset_size_gain_curve.png`
- `plots/top_subsets_by_k.png`
- `plots/factor_pattern_summary.png`
