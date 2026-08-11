# Thesis viewpoint analysis summary for S0_M4

Metric analyzed: `ap50_95`. The cached per-image file does not include a raw IoU column, so `ap50_95` is used as the available localization-quality proxy throughout this analysis.

## 1. Ideal viewpoint per object
- Highest mean score overall: container at az090, elevation low, radius near with mean 1.000.
- Most common optimal elevation band: mid (6 of 10 objects).
- Most common optimal radius band: near (7 of 10 objects).
- Most stable best viewpoints (lowest std): container (0.000), male (0.000), suv (0.000).
- Caution: 9 of 10 exact best viewpoints are supported by only 1-2 images, so the heatmaps, top-3 tables, and regression trends are more reliable than any single exact cell on its own.

## 2. Viewpoint heatmaps
- Heatmaps were averaged over radius so the azimuth x elevation pattern is easier to compare across objects.
- White circled cells indicate the highest mean cell per object in the saved heatmaps.

## 3. Object grouping
- Hierarchical clustering produced 3 clusters with sizes: cluster 1: 4, cluster 2: 5, cluster 3: 1.
- Cluster membership is based on each class-specific optimal azimuth, elevation, radius, and best mean score.

## 4. Regression analysis
- Strongest parameter by regression effect range: elevation (10).
- The regression surface uses azimuth sine/cosine terms, elevation, radius, and pairwise interactions to approximate local viewpoint effects.

## 5. Robustness analysis
- Most viewpoint-sensitive objects by mean local drop: tree (0.054), barrel (0.019), rock (0.015).
- Robustness was estimated from the fitted regression surface using azimuth perturbations (+/-5 and +/-10 degrees) and small elevation/radius perturbations in normalized band units.

## 6. Thesis-ready conclusions
- Optimal viewpoints are object-specific, but the best solutions are concentrated in the mid elevation band and the near radius band.
- Objects do share patterns: the clustering and heatmaps show repeated families of preferred viewing geometry rather than completely unique optima.
- Elevation appears most influential most often, although several classes also show clear interaction effects.
- The most sensitive objects should be approached with tighter viewpoint control, while the most stable classes can be searched more flexibly.