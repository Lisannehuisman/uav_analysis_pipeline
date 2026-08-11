# Practical Viewpoint Selection Checklist

Use this checklist when choosing a drone-swarm viewpoint set from the cached M4 evidence.

1. Start with a high raw detection viewpoint: high `AP50-95` and high strict target quality.
2. Require enough scene support; exact pairs/triples with only a few matched scenes are exploratory.
3. Add viewpoints only when pair synergy is positive under the explicit matched-scene definition.
4. Prefer subsets that improve class robustness rather than only improving one dominant class.
5. Check target-absent false alarms; sparse absence rows mean this is a safety audit, not a full negative benchmark.
6. Keep deployment simple when scores are close: fewer changes in elevation/radius are easier to fly.

## Scripted Ranking Rule

`weighted_selection_score = 0.35 raw_detection + 0.20 complementarity + 0.15 class_robustness + 0.15 scene_support + 0.10 absence_safety + 0.05 deployability`

Top recommendations by budget:

## 1 Viewpoint(s)
- `elmid-radnear-az270`: score `0.8132`, AP50-95 `0.9137`, synergy `0.0000`, class robustness `0.9103`, support `43` scenes
- `elmid-radnear-az135`: score `0.8104`, AP50-95 `0.9076`, synergy `0.0000`, class robustness `0.9189`, support `43` scenes
- `elhigh-radmid-az000`: score `0.8063`, AP50-95 `0.9127`, synergy `0.0000`, class robustness `0.8944`, support `43` scenes
- `elmid-radnear-az315`: score `0.7971`, AP50-95 `0.9613`, synergy `0.0000`, class robustness `0.9143`, support `33` scenes
- `elmid-radmid-az270`: score `0.7864`, AP50-95 `0.9074`, synergy `0.0000`, class robustness `0.9076`, support `40` scenes
- `elhigh-radnear-az225`: score `0.7715`, AP50-95 `0.9315`, synergy `0.0000`, class robustness `0.8734`, support `36` scenes
- `elmid-radmid-az000`: score `0.7664`, AP50-95 `0.8732`, synergy `0.0000`, class robustness `0.9106`, support `42` scenes
- `elhigh-radfar-az045`: score `0.7563`, AP50-95 `0.9059`, synergy `0.0000`, class robustness `0.8490`, support `39` scenes
- `elmid-radnear-az180`: score `0.7519`, AP50-95 `0.9379`, synergy `0.0000`, class robustness `0.8734`, support `32` scenes
- `elhigh-radmid-az225`: score `0.7509`, AP50-95 `0.9146`, synergy `0.0000`, class robustness `0.8633`, support `36` scenes

## 2 Viewpoint(s)
- `ellow-radnear-az180 + elhigh-radmid-az180`: score `0.7818`, AP50-95 `0.9671`, synergy `0.1109`, class robustness `0.8537`, support `10` scenes
- `ellow-radfar-az000 + elmid-radmid-az000`: score `0.7567`, AP50-95 `0.9106`, synergy `0.1148`, class robustness `0.7234`, support `13` scenes
- `elmid-radnear-az270 + elmid-radmid-az270`: score `0.7434`, AP50-95 `0.9808`, synergy `0.0165`, class robustness `0.7286`, support `14` scenes
- `ellow-radfar-az135 + elmid-radnear-az180`: score `0.7390`, AP50-95 `0.9636`, synergy `0.0672`, class robustness `0.7094`, support `11` scenes
- `ellow-radmid-az045 + elmid-radnear-az090`: score `0.7343`, AP50-95 `0.9364`, synergy `0.0828`, class robustness `0.7502`, support `11` scenes
- `ellow-radnear-az180 + elhigh-radmid-az000`: score `0.7318`, AP50-95 `0.9526`, synergy `0.0360`, class robustness `0.7600`, support `14` scenes
- `ellow-radnear-az180 + elhigh-radmid-az225`: score `0.7204`, AP50-95 `0.9732`, synergy `0.0608`, class robustness `0.5833`, support `11` scenes
- `ellow-radmid-az270 + elmid-radmid-az225`: score `0.7201`, AP50-95 `0.9875`, synergy `0.3110`, class robustness `0.1943`, support `3` scenes
- `elmid-radnear-az270 + elhigh-radmid-az000`: score `0.7127`, AP50-95 `0.9465`, synergy `0.0149`, class robustness `0.7403`, support `14` scenes
- `ellow-radmid-az135 + elmid-radnear-az135`: score `0.7031`, AP50-95 `0.9869`, synergy `0.0767`, class robustness `0.5960`, support `8` scenes

## 3 Viewpoint(s)
- `ellow-radmid-az045 + ellow-radmid-az270 + elmid-radnear-az090`: score `0.7762`, AP50-95 `0.9800`, synergy `0.1027`, class robustness `0.4790`, support `5` scenes
- `ellow-radmid-az090 + ellow-radfar-az000 + elmid-radmid-az000`: score `0.7238`, AP50-95 `0.9437`, synergy `0.2259`, class robustness `0.3555`, support `4` scenes
- `ellow-radmid-az135 + ellow-radfar-az090 + elmid-radnear-az135`: score `0.7180`, AP50-95 `1.0000`, synergy `0.0987`, class robustness `0.3980`, support `4` scenes
- `ellow-radnear-az180 + elmid-radnear-az180 + elhigh-radmid-az180`: score `0.7158`, AP50-95 `1.0000`, synergy `0.0233`, class robustness `0.4697`, support `5` scenes
- `ellow-radnear-az000 + elmid-radnear-az090 + elhigh-radfar-az090`: score `0.7121`, AP50-95 `1.0000`, synergy `0.1448`, class robustness `0.2944`, support `4` scenes
- `ellow-radmid-az000 + ellow-radfar-az135 + elmid-radnear-az180`: score `0.7028`, AP50-95 `1.0000`, synergy `0.0932`, class robustness `0.3762`, support `4` scenes
- `ellow-radnear-az180 + elhigh-radmid-az180 + elhigh-radmid-az225`: score `0.7006`, AP50-95 `0.9824`, synergy `0.1068`, class robustness `0.3871`, support `4` scenes
- `ellow-radnear-az000 + elmid-radnear-az045 + elmid-radnear-az090`: score `0.6985`, AP50-95 `0.9750`, synergy `0.1167`, class robustness `0.3777`, support `4` scenes
- `ellow-radmid-az315 + ellow-radfar-az270 + elhigh-radfar-az090`: score `0.6929`, AP50-95 `0.9800`, synergy `0.0186`, class robustness `0.4850`, support `5` scenes
- `ellow-radnear-az090 + ellow-radmid-az045 + elmid-radnear-az090`: score `0.6911`, AP50-95 `0.9783`, synergy `0.1003`, class robustness `0.3884`, support `4` scenes

