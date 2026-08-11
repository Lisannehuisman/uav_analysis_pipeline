# Original S4 vs Single-Viewpoint-Trained Comparison

## What Is Being Compared

- `Original S4 viewpoint strength`: how strong a viewpoint is as a test-time observation under the original full-viewpoint model.
- `Single-trained viewpoint strength`: how well a model generalizes when trained on only that one viewpoint and evaluated on the full fixed M4 test set.
- These are related but not identical roles, so rank shifts are more interpretable than raw score gaps.

## Headline Results

- Viewpoints compared: `72`
- Spearman rank correlation between the two orderings: `0.514`
- Best original S4 observation viewpoint: `elhigh-radnear-az225` with mean per-image AP50-95 `0.9081`
- Best single-trained viewpoint: `elmid-radmid-az045` with full-test mAP50-95 `0.4164`
- Overlap between top-10 original viewpoints and top-10 trained viewpoints: `0 / 10`

## Largest Positive Training Shifts

- `elmid-radfar-az090`: original rank `44` -> trained rank `4` (`+40` places better as a training viewpoint)
- `elmid-radfar-az225`: original rank `46` -> trained rank `6` (`+40` places better as a training viewpoint)
- `elmid-radmid-az270`: original rank `40` -> trained rank `3` (`+37` places better as a training viewpoint)
- `elmid-radfar-az180`: original rank `42` -> trained rank `7` (`+35` places better as a training viewpoint)
- `elmid-radfar-az135`: original rank `45` -> trained rank `11` (`+34` places better as a training viewpoint)
- `elmid-radfar-az000`: original rank `47` -> trained rank `14` (`+33` places better as a training viewpoint)
- `elmid-radfar-az270`: original rank `41` -> trained rank `9` (`+32` places better as a training viewpoint)
- `elmid-radfar-az315`: original rank `48` -> trained rank `16` (`+32` places better as a training viewpoint)

## Largest Negative Training Shifts

- `elhigh-radnear-az045`: original rank `2` -> trained rank `38` (`-36` places worse as a training viewpoint)
- `elhigh-radmid-az315`: original rank `8` -> trained rank `42` (`-34` places worse as a training viewpoint)
- `elhigh-radnear-az135`: original rank `14` -> trained rank `46` (`-32` places worse as a training viewpoint)
- `elhigh-radfar-az000`: original rank `4` -> trained rank `33` (`-29` places worse as a training viewpoint)
- `elhigh-radnear-az225`: original rank `1` -> trained rank `28` (`-27` places worse as a training viewpoint)
- `elhigh-radfar-az270`: original rank `18` -> trained rank `44` (`-26` places worse as a training viewpoint)
- `elmid-radnear-az090`: original rank `11` -> trained rank `37` (`-26` places worse as a training viewpoint)
- `elhigh-radnear-az315`: original rank `16` -> trained rank `41` (`-25` places worse as a training viewpoint)

## Factor-Level Signals

### Elevation

- `elhigh`: original mean `0.8786`, trained mean `0.3408`, mean percentile shift `-0.217`
- `ellow`: original mean `0.6340`, trained mean `0.2943`, mean percentile shift `+0.001`
- `elmid`: original mean `0.8400`, trained mean `0.3800`, mean percentile shift `+0.216`

### Radius

- `radfar`: original mean `0.7540`, trained mean `0.3426`, mean percentile shift `+0.124`
- `radmid`: original mean `0.7899`, trained mean `0.3484`, mean percentile shift `+0.057`
- `radnear`: original mean `0.8088`, trained mean `0.3241`, mean percentile shift `-0.181`

## Top-10 Overlap

- No overlap between the two top-10 lists.
