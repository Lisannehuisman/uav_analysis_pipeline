# Bootstrap Uncertainty For Exact Ring Shapley

- Scenes are resampled with replacement within each ring.
- For each bootstrap sample, the exact 8-player Shapley values are recomputed from the resampled scene coalition values.
- The key inferential quantity is the bootstrap interval for the gap between the observed top azimuth and the observed runner-up in each ring.

| Ring | Scenes | Top azimuth | Runner-up | Observed gap | 95% bootstrap CI for gap | P(gap > 0) | Top stays top |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| `ellow-radnear` | 154 | `az225` | `az180` | 0.001979 | [-0.065879, 0.073453] | 0.525000 | 0.433750 |
| `ellow-radmid` | 149 | `az315` | `az045` | 0.020994 | [-0.050102, 0.094501] | 0.726250 | 0.620250 |
| `ellow-radfar` | 154 | `az000` | `az135` | 0.012155 | [-0.055278, 0.075977] | 0.639500 | 0.427750 |
| `elmid-radnear` | 151 | `az270` | `az135` | 0.038354 | [-0.038448, 0.112999] | 0.821750 | 0.809250 |
| `elmid-radmid` | 148 | `az000` | `az270` | 0.020520 | [-0.052921, 0.095305] | 0.715500 | 0.645500 |
| `elmid-radfar` | 145 | `az225` | `az090` | 0.012418 | [-0.054382, 0.080719] | 0.638750 | 0.392500 |
| `elhigh-radnear` | 150 | `az225` | `az135` | 0.009706 | [-0.061412, 0.080950] | 0.613500 | 0.403750 |
| `elhigh-radmid` | 159 | `az000` | `az225` | 0.033839 | [-0.041651, 0.106453] | 0.817000 | 0.771500 |
| `elhigh-radfar` | 147 | `az045` | `az315` | 0.042608 | [-0.028605, 0.119110] | 0.864500 | 0.809750 |