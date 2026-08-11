# Current Method vs Late Fusion Comparison

This report compares the current method to the late-fusion policies on the same scene-balanced evaluation base:

- `oracle / current method`: the current evaluation, which takes the best target strict-quality available inside the selected view set
- `late fusion`: the existing deployable policies from `m4_cross_view_box_fusion_analysis`

The `best_box` policy is intentionally omitted from the outputs here because it is exactly identical to the oracle/current-method score for every overall and per-class row in this dataset.

## Overall

### 2 views
- Oracle / current method: `0.9088`
- Noisy-OR + best IoU: `0.9515` (gap vs oracle `0.0427`)
- Support-weighted OR: `0.9212` (gap vs oracle `0.0124`)

### 3 views
- Oracle / current method: `0.9189`
- Noisy-OR + best IoU: `0.9661` (gap vs oracle `0.0472`)
- Support-weighted OR: `0.9269` (gap vs oracle `0.0081`)

## Per Class Highlights

### 2 views
- Highest support-weighted fused quality: `tent` at `0.9823`
- Closest support-weighted result to oracle: `tank` at gap `-0.0002`
- Largest positive support-weighted gap vs oracle: `male` at `0.0465`
- Largest negative support-weighted gap vs oracle: `suv` at `-0.0077`

### 3 views
- Highest support-weighted fused quality: `tent` at `0.9829`
- Closest support-weighted result to oracle: `rock` at gap `0.0079`
- Largest positive support-weighted gap vs oracle: `male` at `0.0463`
- Largest negative support-weighted gap vs oracle: `suv` at `-0.0146`

