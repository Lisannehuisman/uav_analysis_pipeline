# Harmonized Multiview Method Comparison

This report compares the current multiview coalition methods on one shared headline metric:

- `mean_scene_expected_strict_quality`

This is the only metric that is principled across all currently used coalition methods.

Secondary metrics are only shown where they remain methodologically meaningful:

- `mean_scene_expected_target_ap50_95` is shown for threshold / protocol methods;
- `mean_scene_expected_target_found_rate` is shown for threshold / protocol methods;
- late-fusion methods are therefore compared directly on strict quality, not on AP50-95.

Odds-product note:

- The independent-detection idea `1 - product(1 - p_i)` is mathematically the same aggregation used by the existing `noisy-OR` fusion methods when detector confidence is treated as a probability proxy.
- The additional corroboration variant added here is the odds-product fusion rule.

## Key Equivalences

- `1-of-2 OR` and `best_box` are identical on strict quality for 2-view coalitions.
- `1-of-3 OR` and `best_box` are identical on strict quality for 3-view coalitions.
- `2-of-2` is not the same as `unanimous_best_box`: `2-of-2` uses the weaker of the two matched qualities, while `unanimous_best_box` keeps the strongest quality once both views support the target.
- `3-of-3` is not the same as a hypothetical 3-view unanimous best-box rule for the same reason.
- `noisy-OR` methods operationalize independent-detection accumulation; `odds-product` methods provide an alternative corroboration rule.

## 1-View Methods

| Rank | Method | Family | Strict quality | Delta vs single | Relative gain % | Regret vs best | AP50-95 | Found rate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Single-view reference` | Reference | 0.8732 | 0.0000 | 0.0000 | 0.0000 | 0.8516 | 0.9811 |

## 2-View Methods

| Rank | Method | Family | Strict quality | Delta vs single | Relative gain % | Regret vs best | AP50-95 | Found rate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Noisy-OR + best IoU` | Probabilistic late fusion | 0.9515 | 0.0784 | 8.9765 | 0.0000 | n/a | n/a |
| 2 | `Odds-product + best IoU` | Odds-product fusion | 0.9487 | 0.0756 | 8.6545 | 0.0028 | n/a | n/a |
| 3 | `Noisy-OR + mean IoU` | Probabilistic late fusion | 0.9356 | 0.0625 | 7.1532 | 0.0159 | n/a | n/a |
| 4 | `Odds-product + mean IoU` | Odds-product fusion | 0.9330 | 0.0598 | 6.8510 | 0.0186 | n/a | n/a |
| 5 | `Support-weighted OR` | Probabilistic late fusion | 0.9212 | 0.0481 | 5.5070 | 0.0303 | n/a | n/a |
| 6 | `1-of-2 OR` | Operational threshold | 0.9088 | 0.0357 | 4.0842 | 0.0427 | 0.9277 | 0.9986 |
| 6 | `Best box (max)` | Selection / rescue | 0.9088 | 0.0357 | 4.0842 | 0.0427 | n/a | n/a |
| 7 | `Unanimous best box` | Strict confirmation | 0.8801 | 0.0069 | 0.7917 | 0.0715 | n/a | n/a |
| 8 | `Mean quality` | Naive aggregation | 0.8732 | 0.0000 | 0.0000 | 0.0784 | n/a | n/a |
| 9 | `2-of-2 confirmation` | Operational threshold | 0.8375 | -0.0357 | -4.0842 | 0.1140 | 0.7755 | 0.9635 |

## 3-View Methods

| Rank | Method | Family | Strict quality | Delta vs single | Relative gain % | Regret vs best | AP50-95 | Found rate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Noisy-OR + best IoU` | Probabilistic late fusion | 0.9661 | 0.0929 | 10.6405 | 0.0000 | n/a | n/a |
| 2 | `Odds-product + best IoU` | Odds-product fusion | 0.9643 | 0.0912 | 10.4424 | 0.0017 | n/a | n/a |
| 3 | `Noisy-OR + mean IoU` | Probabilistic late fusion | 0.9436 | 0.0705 | 8.0693 | 0.0225 | n/a | n/a |
| 4 | `Odds-product + mean IoU` | Odds-product fusion | 0.9420 | 0.0689 | 7.8869 | 0.0240 | n/a | n/a |
| 5 | `Support-weighted OR` | Probabilistic late fusion | 0.9269 | 0.0538 | 6.1583 | 0.0391 | n/a | n/a |
| 6 | `1-of-3 OR` | Operational threshold | 0.9189 | 0.0457 | 5.2353 | 0.0472 | 0.9487 | 0.9999 |
| 6 | `Best box (max)` | Selection / rescue | 0.9189 | 0.0457 | 5.2353 | 0.0472 | n/a | n/a |
| 7 | `2-of-3 confirmation` | Operational threshold | 0.8887 | 0.0156 | 1.7821 | 0.0773 | 0.8858 | 0.9961 |
| 8 | `Unanimous best box` | Strict confirmation | 0.8743 | 0.0012 | 0.1318 | 0.0918 | n/a | n/a |
| 9 | `Mean quality` | Naive aggregation | 0.8732 | 0.0000 | 0.0000 | 0.0929 | n/a | n/a |
| 10 | `3-of-3 unanimous` | Operational threshold | 0.8119 | -0.0613 | -7.0174 | 0.1542 | 0.7204 | 0.9473 |

## Best Method Per View Count

- `1` view(s): `Single-view reference` with strict quality `0.8732`.
- `2` view(s): `Noisy-OR + best IoU` with strict quality `0.9515`.
- `3` view(s): `Noisy-OR + best IoU` with strict quality `0.9661`.

## Interpretation

- Use this harmonized table when you want to compare methods directly on one shared coalition-quality metric.
- Use protocol-specific AP50-95 and found-rate columns only as secondary information inside the threshold family.
- Keep marginal analyses and ring Shapley separate from coalition-method ranking: they measure value attribution, not coalition quality.
