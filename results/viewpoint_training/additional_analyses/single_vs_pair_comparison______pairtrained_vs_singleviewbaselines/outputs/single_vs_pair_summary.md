# Single vs Pair Training Comparison

## Coverage

- Completed single-view models: 72 / 72
- Completed pair models: 2535 / 2556

## Best Models

- Best single-view training viewpoint: `sv0034` = `elmid-radmid-az045` with `mAP50-95 = 0.4164`
- Best pair-trained model: `p0569` = `ellow-radmid-az000` + `elmid-radmid-az225` with `mAP50-95 = 0.4958`
- Best pair improvement over best single overall: `+0.0794` mAP50-95

## Pair vs Single Baselines

- Pairs beating their better constituent single: `2380 / 2535` (93.9%)
- Mean pair lift over best constituent single: `+0.0598` mAP50-95
- Median pair lift over best constituent single: `+0.0360` mAP50-95

## Data Volume Caveat

- Mean single-view training images: `143.5`
- Mean pair-view training images: `287.0`
- Pair models therefore usually see about twice as many training images as single-view models.
- This comparison is still scientifically useful, but it reflects both viewpoint complementarity and added image count.

## Recommended Fair Comparison

- The cleanest fairness fix is **not** to duplicate single-view images. That would only repeat the same viewpoint evidence and would not create a genuinely stronger baseline.
- Instead, compare each restricted model against an **equal-image-count M4 control** trained on the full viewpoint space with the same train/val image counts.
- This isolates `image count` from `viewpoint restriction`: if the matched M4 control still wins, the gap is due to viewpoint diversity rather than just having fewer images.

## Top Synergy Pairs

### Current matched-control status

- No completed matched-control runs were found yet under `outputs/m4_matched_control_experiment/reports/master_results.csv`.
- The repo now includes tooling to generate those controls by budget before rerunning this comparison.

### Practical recommendation

- For the thesis headline, compare `best single` against its matched M4 control and `best pair` against its matched M4 control.
- For the sweep-level story, keep the existing pair-vs-single plots, but frame them explicitly as `restricted-view vs restricted-view` comparisons rather than fully fair count-controlled comparisons.

- `p0069`: `ellow-radnear-az000` + `elhigh-radfar-az225` -> `+0.1561` over the better constituent single
- `p0724`: `ellow-radmid-az090` + `elhigh-radfar-az225` -> `+0.1560` over the better constituent single
- `p0843`: `ellow-radmid-az180` + `elhigh-radfar-az225` -> `+0.1548` over the better constituent single
- `p0601`: `ellow-radmid-az000` + `elhigh-radfar-az225` -> `+0.1529` over the better constituent single
- `p0901`: `ellow-radmid-az225` + `elhigh-radfar-az225` -> `+0.1525` over the better constituent single
- `p1123`: `ellow-radfar-az045` + `elhigh-radfar-az225` -> `+0.1507` over the better constituent single
- `p0538`: `ellow-radnear-az315` + `elhigh-radfar-az225` -> `+0.1504` over the better constituent single
- `p1115`: `ellow-radfar-az045` + `elhigh-radmid-az225` -> `+0.1502` over the better constituent single
- `p1014`: `ellow-radmid-az315` + `elhigh-radfar-az225` -> `+0.1498` over the better constituent single
- `p0593`: `ellow-radmid-az000` + `elhigh-radmid-az225` -> `+0.1493` over the better constituent single

## Viewpoints That Gain Most From Pairing

- `ellow-radnear-az270`: single `mAP50-95 = 0.2678`, mean pair lift `+0.1443`
- `ellow-radmid-az315`: single `mAP50-95 = 0.2748`, mean pair lift `+0.1421`
- `ellow-radnear-az315`: single `mAP50-95 = 0.2783`, mean pair lift `+0.1394`
- `ellow-radfar-az315`: single `mAP50-95 = 0.2759`, mean pair lift `+0.1390`
- `ellow-radfar-az225`: single `mAP50-95 = 0.2848`, mean pair lift `+0.1381`
- `ellow-radmid-az135`: single `mAP50-95 = 0.2778`, mean pair lift `+0.1368`
- `ellow-radfar-az000`: single `mAP50-95 = 0.2817`, mean pair lift `+0.1366`
- `ellow-radfar-az135`: single `mAP50-95 = 0.2821`, mean pair lift `+0.1338`
- `ellow-radnear-az135`: single `mAP50-95 = 0.2840`, mean pair lift `+0.1297`
- `ellow-radnear-az045`: single `mAP50-95 = 0.2851`, mean pair lift `+0.1288`
