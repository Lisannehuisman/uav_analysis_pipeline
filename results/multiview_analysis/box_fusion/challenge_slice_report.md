# Challenge-Slice Fusion Summary

This report re-evaluates the ordered-pair matched-box fusion cache on harder slices of the fixed-detector benchmark.

## Why this exists

- The main late-fusion figure shows overall means on the full ordered-pair cache.
- This companion artifact checks whether accumulation-based fusion pulls further ahead when the primary observation is harder.
- The analysis stays on the ordered-pair cache so the results remain directly comparable to the existing matched-box fusion section.

## Slice definitions

- `Full ordered-pair cache`: All ordered primary-secondary view pairs. (rows `23692`, scenes `205`).
- `Weak primary quality`: Primary strict quality in the bottom quartile (<= 0.8752). (rows `5835`, scenes `99`).
- `Small target box`: Primary-view target-box scale proxy in the bottom quartile based on the largest target-class ground-truth box ratio (<= 0.0213). (rows `5858`, scenes `86`).
- `Low primary IoU`: Primary matched IoU in the bottom quartile (<= 0.9408). (rows `5800`, scenes `112`).
- `Far primary view`: Primary radius is `far`. (rows `7530`, scenes `198`).
- `Hard scene`: Scene-level mean single-view strict quality in the bottom quartile (<= 0.8249). (rows `5888`, scenes `52`).
- `Primary miss only`: Primary view does not detect the target. (rows `419`, scenes `29`).

## Headline deltas versus best-box selection

| Slice | Best box | Support-weighted OR | Odds-product + mean IoU | Noisy-OR + mean IoU | Odds-product + best IoU | Noisy-OR + best IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full ordered-pair cache | 0.9100 | +0.0128 | +0.0226 | +0.0263 | +0.0379 | +0.0419 |
| Weak primary quality | 0.7836 | +0.0046 | +0.0318 | +0.0456 | +0.0675 | +0.0824 |
| Small target box | 0.7878 | +0.0108 | +0.0385 | +0.0521 | +0.0675 | +0.0821 |
| Low primary IoU | 0.7868 | +0.0032 | +0.0304 | +0.0440 | +0.0653 | +0.0799 |
| Far primary view | 0.9004 | +0.0126 | +0.0234 | +0.0282 | +0.0395 | +0.0447 |
| Hard scene | 0.7859 | +0.0058 | +0.0342 | +0.0473 | +0.0680 | +0.0821 |
| Primary miss only | 0.7591 | -0.3795 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Reading note

- `Primary miss only` is mostly a rescue slice, so accumulation methods collapse toward the same score as best-box selection once only one supporting view remains.
- The strongest corroboration pattern is therefore expected on weak-primary, small-target, low-IoU, and hard-scene slices rather than on pure primary-miss rescue rows.