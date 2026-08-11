# Factor-level viewpoint summary for S0_M4

Metric analyzed: `ap50_95`. The cached per-image file does not contain a raw IoU column, so `ap50_95` is used as the available localization-quality proxy.
Total test images analyzed: 2214.

## Why this factor-level analysis is stronger
- Mean support per azimuth bin: 27.7 images.
- Mean support per elevation bin: 73.8 images.
- Mean support per radius bin: 73.8 images.
- Aggregating over the remaining viewpoint dimensions increases the number of observations per factor level compared with exact viewpoint cells.

## Main results
- Most common best elevation: high (10 of 10 objects).
- Most common best radius: mid (5 of 10 objects).
- Most common best azimuth: 315 (4 of 10 objects).
- Strongest factor most often: elevation (10).
- Objects with the clearest factor-level viewpoint dependence: tank (elevation), whitevan (elevation), suv (elevation).

## Interpretation
- It is more reliable to state which elevation, radius, or azimuth band tends to work best than to claim a single exact viewpoint cell.
- Elevation and radius can now be discussed with substantially more statistical support because each factor pools over the other two dimensions.
- These summaries complement the exact-cell analysis by emphasizing broader viewpoint trends.


