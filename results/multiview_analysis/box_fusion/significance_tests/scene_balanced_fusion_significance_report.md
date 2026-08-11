# Scene-Balanced Late-Fusion Significance Tests

- The statistical unit is the scene, not the raw coalition row.
- For each scene, coalition scores were averaged across all available combinations for that scene.
- Uncertainty is reported with paired scene bootstrap confidence intervals.
- Significance is tested with a two-sided paired sign-flip permutation test on the per-scene deltas.

## 2-view scene-balanced comparisons

| Comparison | Left mean | Right mean | Mean delta | 95% CI | Positive scenes | Permutation p | Holm p |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Noisy-OR + best IoU minus Best box | 0.951529 | 0.908813 | 0.042717 | [0.038928, 0.046745] | 205/205 | 0.000050 | 0.000300 |
| Support-weighted noisy-OR minus Best box | 0.921235 | 0.908813 | 0.012422 | [0.006195, 0.018174] | 171/205 | 0.000150 | 0.000300 |
| Noisy-OR + best IoU minus Support-weighted noisy-OR | 0.951529 | 0.921235 | 0.030294 | [0.024756, 0.036310] | 205/205 | 0.000050 | 0.000300 |

## 3-view scene-balanced comparisons

| Comparison | Left mean | Right mean | Mean delta | 95% CI | Positive scenes | Permutation p | Holm p |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Noisy-OR + best IoU minus Best box | 0.966058 | 0.918863 | 0.047195 | [0.042387, 0.052234] | 205/205 | 0.000050 | 0.000300 |
| Support-weighted noisy-OR minus Best box | 0.926922 | 0.918863 | 0.008059 | [0.001294, 0.014212] | 169/205 | 0.014049 | 0.014049 |
| Noisy-OR + best IoU minus Support-weighted noisy-OR | 0.966058 | 0.926922 | 0.039136 | [0.032255, 0.046572] | 205/205 | 0.000050 | 0.000300 |
