# Cross-View Matched-Box Fusion Report

## What This Adds

- This experiment keeps the full-viewpoint detector fixed and tests conservative late fusion at matched-target-box level.
- It is more box-aware than the earlier `1-of-2` / `1-of-3` operational success rules, but it is still not geometric reprojection fusion.
- Because the dataset lacks camera calibration and cross-view object IDs, the fusion is restricted to the intended target object and is ground-truth anchored.

## 2-View Overall Quality

- Best-box baseline: `0.9100`
- Noisy-OR + best IoU: `0.9518`
- Support-weighted noisy-OR: `0.9228`
- Noisy-OR gain vs best-box: `+0.0419`
- Support-weighted gain vs best-box: `+0.0128`

## 3-View Overall Quality

- Best-box baseline: `0.9196`
- Noisy-OR + best IoU: `0.9661`
- Support-weighted noisy-OR: `0.9293`
- Noisy-OR gain vs best-box: `+0.0466`
- Support-weighted gain vs best-box: `+0.0097`

## Best Combinations Under Support-Weighted Fusion

- Best pair: `elmid-radnear-az135 + elmid-radnear-az315` with mean fused quality `0.9831`
- Best triple: `elmid-radnear-az270 + elhigh-radmid-az000 + elhigh-radfar-az045` with mean fused quality `0.9355`

## Interpretation

- If noisy-OR beats best-box, then there is usable cross-view confidence accumulation even without geometry.
- If support-weighted fusion beats best-box, then some viewpoint combinations provide genuinely corroborating target evidence rather than just a single rescue view.
- If support-weighted fusion loses to best-box, then the multiview gain is mostly from having a rescue view available, not from true agreement between matched boxes.

