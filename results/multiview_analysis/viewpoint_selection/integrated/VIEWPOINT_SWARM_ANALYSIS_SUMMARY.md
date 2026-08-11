# Integrated Viewpoint Swarm Analysis

## Viewpoint Representation

Viewpoints are represented as filename/cache tokens of the form `el{elevation}-rad{radius}-az{azimuth}`.

- elevation: `low`, `mid`, `high`
- radius/distance: `near`, `mid`, `far`
- azimuth: `000`, `045`, `090`, `135`, `180`, `225`, `270`, `315` degrees
- full space: `3 x 3 x 8 = 72` possible viewpoints

## Cached Evidence

- scene/view rows: `2214`
- scenes: `205`
- observed viewpoints: `72`
- target-absent rows: `40`

## Strongest Single Viewpoints

- `elmid-radnear-az315`: target AP50-95 `0.9613`, strict quality `0.9383`, support `33` scenes
- `elmid-radnear-az180`: target AP50-95 `0.9379`, strict quality `0.9012`, support `32` scenes
- `elhigh-radnear-az000`: target AP50-95 `0.9378`, strict quality `0.9291`, support `34` scenes
- `elhigh-radfar-az270`: target AP50-95 `0.9324`, strict quality `0.8941`, support `30` scenes
- `elhigh-radnear-az225`: target AP50-95 `0.9315`, strict quality `0.9027`, support `36` scenes

## Factor Explanation

- `target_ap50_95` explained by `full_viewpoint`: eta-squared `0.1348`
- `target_ap50_95` explained by `elevation_radius`: eta-squared `0.1111`
- `target_ap50_95` explained by `elevation_azimuth`: eta-squared `0.0966`
- `target_ap50_95` explained by `elevation`: eta-squared `0.0890`
- `target_ap50_95` explained by `radius_azimuth`: eta-squared `0.0180`
- `target_ap50_95` explained by `radius`: eta-squared `0.0126`
- `target_ap50_95` explained by `azimuth`: eta-squared `0.0019`
- `target_strict_quality_iou50` explained by `full_viewpoint`: eta-squared `0.0688`

## Relationship Types

- k=`2`, `azimuth=quarter_turn_90`: mean AP50-95 `0.9302`, mean strict quality `0.9100`, configs `642`
- k=`2`, `azimuth=diagonal_135`: mean AP50-95 `0.9301`, mean strict quality `0.9104`, configs `642`
- k=`2`, `azimuth=opposite_180`: mean AP50-95 `0.9292`, mean strict quality `0.9083`, configs `319`
- k=`2`, `azimuth=adjacent_45`: mean AP50-95 `0.9278`, mean strict quality `0.9090`, configs `643`
- k=`2`, `azimuth=same_azimuth`: mean AP50-95 `0.9223`, mean strict quality `0.9076`, configs `286`
- k=`2`, `distance=near_far`: mean AP50-95 `0.9353`, mean strict quality `0.9155`, configs `571`
- k=`2`, `distance=same_radius`: mean AP50-95 `0.9287`, mean strict quality `0.9086`, configs `818`
- k=`2`, `distance=adjacent_radius`: mean AP50-95 `0.9251`, mean strict quality `0.9068`, configs `1143`

## Explicit Synergy Definition

For a pair `(i, j)`, synergy is computed on matched scenes where both viewpoints exist: `mean(max(strict_i, strict_j)) - max(mean(strict_i), mean(strict_j))`. Positive values mean the pair adds complementary evidence beyond its better single viewpoint.
Most complementary observed pair: `ellow-radmid-az270 + elmid-radmid-az225` with synergy `0.3014` over `3` matched scenes.

## Practical Recommendations

The scripted weighted score uses: 0.35 raw detection, 0.20 complementarity, 0.15 class robustness, 0.15 scene support, 0.10 absence safety, and 0.05 deployability.
### Budget k=1
- `elmid-radnear-az270`: weighted score `0.8132`, AP50-95 `0.9137`, support `43` scenes
- `elmid-radnear-az135`: weighted score `0.8104`, AP50-95 `0.9076`, support `43` scenes
- `elhigh-radmid-az000`: weighted score `0.8063`, AP50-95 `0.9127`, support `43` scenes
- `elmid-radnear-az315`: weighted score `0.7971`, AP50-95 `0.9613`, support `33` scenes
- `elmid-radmid-az270`: weighted score `0.7864`, AP50-95 `0.9074`, support `40` scenes

### Budget k=2
- `ellow-radnear-az180 + elhigh-radmid-az180`: weighted score `0.7818`, AP50-95 `0.9671`, support `10` scenes
- `ellow-radfar-az000 + elmid-radmid-az000`: weighted score `0.7567`, AP50-95 `0.9106`, support `13` scenes
- `elmid-radnear-az270 + elmid-radmid-az270`: weighted score `0.7434`, AP50-95 `0.9808`, support `14` scenes
- `ellow-radfar-az135 + elmid-radnear-az180`: weighted score `0.7390`, AP50-95 `0.9636`, support `11` scenes
- `ellow-radmid-az045 + elmid-radnear-az090`: weighted score `0.7343`, AP50-95 `0.9364`, support `11` scenes

### Budget k=3
- `ellow-radmid-az045 + ellow-radmid-az270 + elmid-radnear-az090`: weighted score `0.7762`, AP50-95 `0.9800`, support `5` scenes
- `ellow-radmid-az090 + ellow-radfar-az000 + elmid-radmid-az000`: weighted score `0.7238`, AP50-95 `0.9437`, support `4` scenes
- `ellow-radmid-az135 + ellow-radfar-az090 + elmid-radnear-az135`: weighted score `0.7180`, AP50-95 `1.0000`, support `4` scenes
- `ellow-radnear-az180 + elmid-radnear-az180 + elhigh-radmid-az180`: weighted score `0.7158`, AP50-95 `1.0000`, support `5` scenes
- `ellow-radnear-az000 + elmid-radnear-az090 + elhigh-radfar-az090`: weighted score `0.7121`, AP50-95 `1.0000`, support `4` scenes

## Budget Comparison

- k=`1` best supported subset `elmid-radnear-az270`: AP50-95 `0.9137`, gain vs best single `+0.0000`, marginal gain `+0.0000`
- k=`2` best supported subset `ellow-radnear-az180 + elhigh-radmid-az180`: AP50-95 `0.9671`, gain vs best single `+0.0535`, marginal gain `+0.0535`
- k=`3` best supported subset `ellow-radmid-az045 + ellow-radmid-az270 + elmid-radnear-az090`: AP50-95 `0.9800`, gain vs best single `+0.0663`, marginal gain `+0.0129`

## Interpretation Boundary

This is a fixed-detector, cached-prediction analysis. Pair/triple performance uses best-available target evidence across selected views, not calibrated 3D fusion or cross-view object identity tracking. Exact pair/triple subsets can be sparse because not every scene contains every viewpoint.
