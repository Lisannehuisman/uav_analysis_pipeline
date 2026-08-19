# M4 Two-Drone Operational Analysis

## What This Method Measures

- The detector is held fixed: full M4 `YOLOv8l` predictions on the fixed M4 test split are reused.
- One-drone mode selects one available viewpoint per scene.
- Two-drone mode selects two available viewpoints from the same scene.
- Binary target-found rate is still reported as a reference.
- The stricter headline metrics score the target with matched confidence, target-only AP50-95, and a strict quality score equal to confidence multiplied by matched IoU at IoU>=0.50.
- Standard detector image scores are also summarized over the selected image sets.

## Important Interpretation Boundary

- This is an operational viewpoint-availability simulation, not a retraining experiment.
- The target-centric mission metric is the cleanest answer to the question 'does a second view help find the intended object?'
- The accompanying precision/recall/F1/AP values describe the selected image sets, but they do not deduplicate identical real-world objects across views.

## Overall Expected Comparison

- Expected target confidence, 1 drone: 0.9162
- Expected target confidence, 2 drones: 0.9437
- Expected target strict quality, 1 drone: 0.8704
- Expected target strict quality, 2 drones: 0.9065
- Gain in target strict quality: 0.0361
- Expected target AP50-95, 1 drone: 0.8495
- Expected best target AP50-95, 2 drones: 0.9274
- Binary target found rate, 1 drone: 0.9778
- Binary target found rate, 2 drones: 0.9974
- Expected mean AP50-95, 1 drone: 0.7900
- Expected mean AP50-95, 2 drones: 0.7900
- Expected best available AP50-95, 1 drone: 0.7900
- Expected best available AP50-95, 2 drones: 0.8881
- Expected mean F1, 1 drone: 0.6401
- Expected mean F1, 2 drones: 0.5775

## Strongest Single Viewpoints

- `elmid-radnear-az135`: target strict quality 0.9489, target AP50-95 0.9568, support 20
- `elmid-radnear-az180`: target strict quality 0.9266, target AP50-95 0.9178, support 29
- `elmid-radnear-az315`: target strict quality 0.9264, target AP50-95 0.9579, support 30
- `elmid-radnear-az045`: target strict quality 0.9185, target AP50-95 0.9342, support 35
- `elmid-radmid-az135`: target strict quality 0.9126, target AP50-95 0.8898, support 38

## Strongest Two-View Combinations

- `ellow-radfar-az045 + elmid-radmid-az225`: best target strict quality 0.9662, best target AP50-95 0.9820, support 8
- `ellow-radmid-az000 + elmid-radnear-az045`: best target strict quality 0.9610, best target AP50-95 0.9322, support 8
- `ellow-radmid-az180 + elmid-radmid-az135`: best target strict quality 0.9605, best target AP50-95 0.9381, support 8
- `elmid-radnear-az045 + elhigh-radnear-az090`: best target strict quality 0.9604, best target AP50-95 0.9891, support 9
- `ellow-radmid-az135 + elmid-radmid-az135`: best target strict quality 0.9604, best target AP50-95 0.9439, support 8

## Best Second-Drone Rescue Viewpoints

- `elhigh-radnear-az135`: rescue rate given primary miss 1.0000, primary-miss support 15
- `elhigh-radnear-az045`: rescue rate given primary miss 1.0000, primary-miss support 13
- `elmid-radmid-az090`: rescue rate given primary miss 1.0000, primary-miss support 11
- `elmid-radmid-az270`: rescue rate given primary miss 1.0000, primary-miss support 11
- `elhigh-radmid-az045`: rescue rate given primary miss 1.0000, primary-miss support 11

## Object Classes With The Largest 2-Drone Gains

- `male`: delta target strict quality 0.0871, delta target AP50-95 0.1190
- `barrel`: delta target strict quality 0.0866, delta target AP50-95 0.1114
- `tank`: delta target strict quality 0.0486, delta target AP50-95 0.0924
- `suv`: delta target strict quality 0.0382, delta target AP50-95 0.0823
- `whitevan`: delta target strict quality 0.0282, delta target AP50-95 0.0814

## Pair-Relation Snapshot

- `same_elevation_0__same_radius_0`: target strict quality 0.8764, target AP50-95 0.8524
- `same_elevation_0__same_radius_1`: target strict quality 0.8717, target AP50-95 0.8459
- `same_elevation_1__same_radius_0`: target strict quality 0.8721, target AP50-95 0.8507
- `same_elevation_1__same_radius_1`: target strict quality 0.8739, target AP50-95 0.8544
