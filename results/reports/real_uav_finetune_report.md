# Real UAV Finetune Evaluation

Images evaluated: 39
Supported classes in this 39-image subset: tent, tank, tower, container, whitevan, suv, male, rock, barrel, tree
Stable classes with at least 10 objects: whitevan,suv,male

The class order used by these runs is the order in the synthetic/review-set class order, not the copied validation-set YAML order.
Classes with zero ground-truth support should be reported as not applicable, not as precision=1.

## Supported-Class Summary

| run | P@0.25/IoU50 | R@0.25/IoU50 | F1@0.25/IoU50 | mAP50 | TP | FP | FN | preds >=0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_only | 0.015 | 0.007 | 0.009 | 0.006 | 1 | 67 | 143 | 68 |
| finetuned | 0.638 | 0.514 | 0.569 | 0.637 | 74 | 42 | 70 | 116 |

## Stable-Class Summary (support >= 10)

| run | classes | P@0.25/IoU50 | R@0.25/IoU50 | F1@0.25/IoU50 | mAP50 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_only | whitevan,suv,male | 0.083 | 0.008 | 0.014 | 0.008 | 1 | 11 | 129 |
| finetuned | whitevan,suv,male | 0.644 | 0.500 | 0.563 | 0.432 | 65 | 36 | 65 |

## Class Support

- tent: 1
- tank: 1
- tower: 3
- container: 2
- whitevan: 33
- suv: 84
- male: 13
- rock: 3
- barrel: 2
- tree: 2

## Per-Class Metrics

| run | class | support | P | R | F1 | AP50 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_only | tent | 1 | 0.000 | 0.000 |  | 0.000 | 0 | 2 | 1 |
| synthetic_only | tank | 1 | 0.000 | 0.000 |  | 0.030 | 0 | 6 | 1 |
| synthetic_only | tower | 3 |  | 0.000 |  | 0.000 | 0 | 0 | 3 |
| synthetic_only | container | 2 | 0.000 | 0.000 |  | 0.004 | 0 | 15 | 2 |
| synthetic_only | whitevan | 33 | 0.333 | 0.030 | 0.056 | 0.024 | 1 | 2 | 32 |
| synthetic_only | suv | 84 |  | 0.000 |  | 0.000 | 0 | 0 | 84 |
| synthetic_only | male | 13 | 0.000 | 0.000 |  | 0.000 | 0 | 9 | 13 |
| synthetic_only | rock | 3 | 0.000 | 0.000 |  | 0.000 | 0 | 26 | 3 |
| synthetic_only | barrel | 2 |  | 0.000 |  | 0.000 | 0 | 0 | 2 |
| synthetic_only | tree | 2 | 0.000 | 0.000 |  | 0.000 | 0 | 7 | 2 |
| finetuned | tent | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| finetuned | tank | 1 | 0.333 | 1.000 | 0.500 | 1.000 | 1 | 2 | 0 |
| finetuned | tower | 3 | 0.667 | 0.667 | 0.667 | 0.917 | 2 | 1 | 1 |
| finetuned | container | 2 | 0.000 | 0.000 |  | 0.000 | 0 | 1 | 2 |
| finetuned | whitevan | 33 | 0.655 | 0.576 | 0.613 | 0.479 | 19 | 10 | 14 |
| finetuned | suv | 84 | 0.636 | 0.500 | 0.560 | 0.500 | 42 | 24 | 42 |
| finetuned | male | 13 | 0.667 | 0.308 | 0.421 | 0.316 | 4 | 2 | 9 |
| finetuned | rock | 3 | 0.500 | 0.667 | 0.571 | 0.656 | 2 | 2 | 1 |
| finetuned | barrel | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| finetuned | tree | 2 | 1.000 | 0.500 | 0.667 | 0.500 | 1 | 0 | 1 |
