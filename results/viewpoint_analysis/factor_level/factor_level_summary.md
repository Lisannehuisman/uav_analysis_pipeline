# Factor-level viewpoint summary for YOLOv8l-M4

Metric analyzed: `ap50_95`. This is the cached per-image detection metric used for the viewpoint-dependence analysis.
Total test images analyzed: 2214.

## Analysis support
- Mean support per azimuth bin: 27.7 images.
- Mean support per elevation bin: 73.8 images.
- Mean support per radius bin: 73.8 images.
- Aggregating over the remaining viewpoint dimensions increases the number of observations per factor level compared with exact 		viewpoint cells.

## Main results
- Most common best elevation: high (10 of 10 objects).
- Most common best radius: mid (5 of 10 objects).
- Most common best azimuth: 315 (4 of 10 objects).
- Strongest factor most often: elevation (10).
- Objects with the clearest factor-level viewpoint dependence: tank (elevation), whitevan (elevation), suv (elevation).

- Factor-level trends are more robust than conclusions based on individual azimuth-elevation-radius cells.
- Elevation and radius estimates pool observations across the other two dimensions, increasing the number of observations per 	group.
- The factor-level summaries complement the exact-cell analysis by emphasizing broader viewpoint trends.