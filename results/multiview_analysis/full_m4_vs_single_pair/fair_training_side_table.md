# Fair Training-Side Comparison

## What This Table Shows

- This table compares restricted-view mean and best results against full-M4 controls trained with the same train/val image counts.
- All models are evaluated on the same full fixed M4 test split.

## Reference

- Full M4 baseline: train=`all`, val=`all`, `mAP50-95 = 0.6396`, `mAP50 = 0.8367`, `F1 = 0.8457`

## Single-View Mean Budget

- Single-view mean: train=`144`, val=`31`, `mAP50-95 = 0.3384`, `mAP50 = 0.5336`, `F1 = 0.5743`, `delta vs equal-budget M4 = -0.1460`
- Equal-budget M4 (single mean): train=`144`, val=`31`, `mAP50-95 = 0.4843`, `mAP50 = 0.7074`, `F1 = 0.7308`, `delta vs equal-budget M4 = +0.0000`

## Single-View Best Budget

- Single-view best: train=`135`, val=`35`, `mAP50-95 = 0.4164`, `mAP50 = 0.6216`, `F1 = 0.6558`, `delta vs equal-budget M4 = -0.0638`
- Equal-budget M4 (best single): train=`135`, val=`35`, `mAP50-95 = 0.4803`, `mAP50 = 0.6965`, `F1 = 0.7228`, `delta vs equal-budget M4 = +0.0000`

## Pair-View Mean Budget

- Pair-view mean: train=`287`, val=`61`, `mAP50-95 = 0.4207`, `mAP50 = 0.6306`, `F1 = 0.6624`, `delta vs equal-budget M4 = -0.1028`
- Equal-budget M4 (pair mean): train=`287`, val=`61`, `mAP50-95 = 0.5234`, `mAP50 = 0.7403`, `F1 = 0.7605`, `delta vs equal-budget M4 = +0.0000`

## Pair-View Best Budget

- Pair-view best: train=`298`, val=`59`, `mAP50-95 = 0.4958`, `mAP50 = 0.7252`, `F1 = 0.7478`, `delta vs equal-budget M4 = -0.0238`
- Equal-budget M4 (best pair): train=`298`, val=`59`, `mAP50-95 = 0.5196`, `mAP50 = 0.7363`, `F1 = 0.7586`, `delta vs equal-budget M4 = +0.0000`
