# Thesis-Style Swarm Detection Analysis

## Data Readiness

- Images available: 2214
- Scenes available: 205
- Absolute viewpoints available: 72
- Views per scene: min 4, mean 10.80, max 18
- Scenes supporting 3-drone simulation: 205
- Filename target visible in GT across all views: 0.9819
- Filename target detected across all views: 0.9815
- Target mean AP50-95 across all views: 0.8527
- Target mean matched confidence at IoU>=0.50: 0.9202
- Target mean strict quality at IoU>=0.50: 0.8741

## Protocol Overview

- `1-of-1`: threshold target confidence 0.9198, threshold strict quality 0.8732, threshold target AP50-95 0.8516, binary found reference 0.9811
- `1-of-2`: threshold target confidence 0.9460, threshold strict quality 0.9088, threshold target AP50-95 0.9277, binary found reference 0.9986
- `2-of-2`: threshold target confidence 0.8936, threshold strict quality 0.8375, threshold target AP50-95 0.7755, binary found reference 0.9635
- `1-of-3`: threshold target confidence 0.9516, threshold strict quality 0.9189, threshold target AP50-95 0.9487, binary found reference 0.9999
- `2-of-3`: threshold target confidence 0.9347, threshold strict quality 0.8887, threshold target AP50-95 0.8858, binary found reference 0.9961
- `3-of-3`: threshold target confidence 0.8730, threshold strict quality 0.8119, threshold target AP50-95 0.7204, binary found reference 0.9473

## Best 2-Drone OR Pairs

- `elmid-radnear-az135 + elmid-radnear-az315`: threshold strict quality 0.9659, threshold target AP50-95 1.0000, support 8
- `ellow-radnear-az225 + elmid-radfar-az000`: threshold strict quality 0.9656, threshold target AP50-95 0.9688, support 8
- `ellow-radmid-az135 + elmid-radnear-az315`: threshold strict quality 0.9633, threshold target AP50-95 0.9914, support 8
- `ellow-radmid-az045 + ellow-radfar-az315`: threshold strict quality 0.9632, threshold target AP50-95 0.9087, support 8
- `ellow-radfar-az000 + elmid-radnear-az045`: threshold strict quality 0.9631, threshold target AP50-95 0.9362, support 9

## Best 2-Drone Confirmation Pairs

- `elmid-radnear-az135 + elmid-radnear-az315`: threshold strict quality 0.9521, threshold target AP50-95 0.9132, support 8
- `elmid-radnear-az315 + elhigh-radnear-az000`: threshold strict quality 0.9453, threshold target AP50-95 0.9379, support 9
- `ellow-radnear-az090 + elmid-radnear-az135`: threshold strict quality 0.9444, threshold target AP50-95 0.8605, support 8
- `ellow-radnear-az225 + elmid-radfar-az000`: threshold strict quality 0.9425, threshold target AP50-95 0.7866, support 8
- `elhigh-radnear-az000 + elhigh-radfar-az315`: threshold strict quality 0.9387, threshold target AP50-95 0.9220, support 9

## Best 3-Drone OR Triples

- `elmid-radnear-az270 + elhigh-radmid-az000 + elhigh-radfar-az045`: threshold strict quality 0.8950, threshold target AP50-95 0.9337, support 6

## Best 3-Drone 2-of-3 Confirmation Triples

- `elmid-radnear-az270 + elhigh-radmid-az000 + elhigh-radfar-az045`: threshold strict quality 0.8636, threshold target AP50-95 0.8653, support 6

## Strongest Incremental Rescue Views

- Second drone `elmid-radnear-az045`: rescue | primary miss 1.0000, mean strict-quality lift 0.0630
- Second drone `elmid-radmid-az000`: rescue | primary miss 1.0000, mean strict-quality lift 0.0574
- Second drone `elmid-radmid-az045`: rescue | primary miss 1.0000, mean strict-quality lift 0.0511
- Second drone `elmid-radfar-az090`: rescue | primary miss 1.0000, mean strict-quality lift 0.0467
- Second drone `elhigh-radmid-az180`: rescue | primary miss 1.0000, mean strict-quality lift 0.0388
- Third drone `elmid-radnear-az045`: rescue | first two miss 1.0000, mean strict-quality lift 0.0212
- Third drone `elmid-radnear-az090`: rescue | first two miss 1.0000, mean strict-quality lift 0.0245
- Third drone `elmid-radnear-az225`: rescue | first two miss 1.0000, mean strict-quality lift 0.0152
- Third drone `elmid-radfar-az090`: rescue | first two miss 1.0000, mean strict-quality lift 0.0087
- Third drone `elhigh-radmid-az135`: rescue | first two miss 1.0000, mean strict-quality lift 0.0070

## Largest Class Gains For 2-Drone OR

- `barrel`: delta threshold strict quality 0.0806, delta threshold AP50-95 0.1094
- `male`: delta threshold strict quality 0.0670, delta threshold AP50-95 0.0942
- `suv`: delta threshold strict quality 0.0528, delta threshold AP50-95 0.0895
- `tank`: delta threshold strict quality 0.0521, delta threshold AP50-95 0.0999
- `whitevan`: delta threshold strict quality 0.0363, delta threshold AP50-95 0.0859

## Largest Class Gains For 3-Drone OR

- `barrel`: delta threshold strict quality 0.1080, delta threshold AP50-95 0.1495
- `male`: delta threshold strict quality 0.0899, delta threshold AP50-95 0.1306
- `tank`: delta threshold strict quality 0.0667, delta threshold AP50-95 0.1316
- `suv`: delta threshold strict quality 0.0640, delta threshold AP50-95 0.1103
- `whitevan`: delta threshold strict quality 0.0428, delta threshold AP50-95 0.1002
