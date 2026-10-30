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

- Expected target confidence, 1 drone: 0.9198
- Expected target confidence, 2 drones: 0.9460
- Expected target strict quality, 1 drone: 0.8732
- Expected target strict quality, 2 drones: 0.9088
- Gain in target strict quality: 0.0357
- Expected target AP50-95, 1 drone: 0.8516
- Expected best target AP50-95, 2 drones: 0.9277
- Binary target found rate, 1 drone: 0.9811
- Binary target found rate, 2 drones: 0.9986
- Expected mean AP50-95, 1 drone: 0.7857
- Expected mean AP50-95, 2 drones: 0.7857
- Expected best available AP50-95, 1 drone: 0.7857
- Expected best available AP50-95, 2 drones: 0.8850
- Expected mean F1, 1 drone: 0.6345
- Expected mean F1, 2 drones: 0.5721

## Strongest Single Viewpoints

- `ellow-radnear-az225`: target strict quality 0.9423, target AP50-95 0.8585, support 36
- `elmid-radnear-az315`: target strict quality 0.9383, target AP50-95 0.9613, support 33
- `elmid-radnear-az000`: target strict quality 0.9311, target AP50-95 0.9263, support 21
- `elmid-radnear-az045`: target strict quality 0.9301, target AP50-95 0.9156, support 32
- `elhigh-radnear-az000`: target strict quality 0.9291, target AP50-95 0.9378, support 34

## Strongest Two-View Combinations

- `elmid-radnear-az135 + elmid-radnear-az315`: best target strict quality 0.9659, best target AP50-95 1.0000, support 8
- `ellow-radnear-az225 + elmid-radfar-az000`: best target strict quality 0.9656, best target AP50-95 0.9688, support 8
- `ellow-radmid-az135 + elmid-radnear-az315`: best target strict quality 0.9633, best target AP50-95 0.9914, support 8
- `ellow-radmid-az045 + ellow-radfar-az315`: best target strict quality 0.9632, best target AP50-95 0.9087, support 8
- `ellow-radfar-az000 + elmid-radnear-az045`: best target strict quality 0.9631, best target AP50-95 0.9362, support 9

## Best Second-Drone Rescue Viewpoints

- `elmid-radmid-az000`: rescue rate given primary miss 1.0000, primary-miss support 14
- `elmid-radnear-az045`: rescue rate given primary miss 1.0000, primary-miss support 11
- `elmid-radfar-az090`: rescue rate given primary miss 1.0000, primary-miss support 11
- `elmid-radmid-az045`: rescue rate given primary miss 1.0000, primary-miss support 10
- `elhigh-radmid-az180`: rescue rate given primary miss 1.0000, primary-miss support 10

## Object Classes With The Largest 2-Drone Gains

- `barrel`: delta target strict quality 0.0806, delta target AP50-95 0.1094
- `male`: delta target strict quality 0.0670, delta target AP50-95 0.0942
- `suv`: delta target strict quality 0.0528, delta target AP50-95 0.0895
- `tank`: delta target strict quality 0.0521, delta target AP50-95 0.0999
- `whitevan`: delta target strict quality 0.0363, delta target AP50-95 0.0859

## Pair-Relation Snapshot

- `same_elevation_0__same_radius_0`: target strict quality 0.8730, target AP50-95 0.8527
- `same_elevation_0__same_radius_1`: target strict quality 0.8783, target AP50-95 0.8563
- `same_elevation_1__same_radius_0`: target strict quality 0.8740, target AP50-95 0.8515
- `same_elevation_1__same_radius_1`: target strict quality 0.8762, target AP50-95 0.8576
