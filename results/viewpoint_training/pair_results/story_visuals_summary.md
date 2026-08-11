# Better Visual Story Summary

## Angle Gradients
- Best elevation counts by object: {'high': 10}
- Best radius counts by object: {'mid': 5, 'near': 5}
- Azimuth remains object-specific, so the gradient heatmap is more informative than a single winner table.

## Drone Effect
- Best protocol counts by object: {'1-of-3': 10}
- `1-of-3` gives the strongest object-level AP50-95 almost everywhere, while stricter confirmation rules trade quality for certainty.
- The lift heatmap shows where 2-drone and 3-drone setups actually help relative to a single-drone baseline.