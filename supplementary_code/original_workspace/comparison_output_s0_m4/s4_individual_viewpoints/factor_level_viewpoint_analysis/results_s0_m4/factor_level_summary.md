# Factor-level viewpoint summary for S0_M4

Metric analyzed: `ap50_95`. The cached per-image file does not contain a raw IoU column, so `ap50_95` is used as the available localization-quality proxy.
Total test images analyzed: 2214.

## Why this factor-level analysis is stronger
- Mean support per azimuth bin: 27.7 images.
- Mean support per elevation bin: 73.8 images.
- Mean support per radius bin: 73.8 images.
- These support levels are much stronger than exact azimuth+elevation+radius cells, so the conclusions below are more defensible for the thesis.

## Main results
- Most common best elevation: high (10 of 10 objects).
- Most common best radius: mid (5 of 10 objects).
- Most common best azimuth: 315 (4 of 10 objects).
- Strongest factor most often: elevation (10).
- Objects with the clearest factor-level viewpoint dependence: tank (elevation), whitevan (elevation), suv (elevation).

## Thesis-ready interpretation
- It is more reliable to state which elevation, radius, or azimuth band tends to work best than to claim a single exact viewpoint cell.
- Elevation and radius can now be discussed with substantially more statistical support because each factor pools over the other two dimensions.
- These outputs are intended to replace the sparse exact-combination claims in the thesis discussion.