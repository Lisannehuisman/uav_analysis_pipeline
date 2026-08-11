# Angle-Agnostic Image-Count Shapley Proxy

This report answers the supervisor's count-first question:

> what is the marginal contribution of adding more images, even before exact angle labels are used?

## Where these results come from

- Source file: `results/multiview_analysis/two_drone_operational/scene_view_records.csv`.
- Each row in that source file is one scene-view observation identified by `scene_key` and `viewpoint`.
- The source rows already contain target-centric detector outputs such as `target_match_confidence_iou50`, `target_match_iou_at_confidence_iou50`, and `target_strict_quality_iou50`.
- In other words, this report is not computed from raw images directly. It is computed from the cached per-view target metrics that were generated earlier in the pipeline.

## What the two coalition values mean

- `Best strict-quality value` means: if a coalition of images is available, keep the single best target strict-quality score among those images.
- `Target strict quality` for one image is `matched target confidence x matched target IoU`.
- `Noisy-OR x best IoU value` means: fuse the matched target confidences across the coalition with `Noisy-OR(confidences) = 1 - product(1 - confidence_i)`, then multiply by the best matched IoU seen anywhere in that coalition.
- The first coalition value is a selection game: keep the best single image.
- The second coalition value is a fusion game: combine confidence evidence across images, then attach the best localization quality available in the coalition.

## Why this report uses IoU50-based target quality

- The source cache stores one matched target confidence and one matched target IoU per image using an IoU>=0.50 target match.
- That gives one stable per-image target quality signal: `target_strict_quality_iou50 = confidence x matched IoU`.
- This is useful for coalition analysis because every coalition rule then operates on the same bounded 0--1 target-quality quantity.
- By contrast, AP50:95 is still reported elsewhere in the project, but it is less natural as the common coalition value for selection and fusion games because those games need one per-view matched target signal that can be combined image-by-image.
- Using IoU50 also keeps the match rule less brittle than a stricter IoU threshold would, which matters when the question is rescue availability and marginal multiview value rather than only very-tight localization.
- So the benefit of IoU50 here is not that it is universally better than AP50:95; it is that it is the most practical target-match definition for this particular coalition-growth analysis.

## How the calculation works

- Images are grouped by target instance (`scene_key`).
- Angle labels are ignored after grouping; only the set of available images matters.
- For each scene, `256` random image orderings are sampled up to `8` images using random seed `42`.
- For one sampled ordering, the first image defines a 1-image coalition, the first two images define a 2-image coalition, and so on.
- At each coalition size `k`, the script computes a best strict-quality prefix value: the maximum strict-quality score among the first `k` images in that sampled ordering.
- At each coalition size `k`, the script also computes a fusion prefix value: `Noisy-OR(first k matched confidences) x best IoU(first k images)`.
- The marginal contribution of the `k`th added image is then `prefix_value_at_k - prefix_value_at_(k-1)`.
- These marginal gains are averaged over all sampled orderings within a scene.
- The final table then averages those scene-level means over all scenes that have at least `k` available images.
- This is a Shapley-style permutation expectation over image order, but it does not yet assign value to named angles.

## How to read the table

- `Added image number`: the position `k` of the added image in the coalition-growth process.
- `Scene support`: how many scenes had at least `k` images available, so could contribute to that row.
- `Mean best strict-quality value`: the mean coalition value at size `k` under the best-single-image rule.
- `Mean Noisy-OR x best IoU value`: the mean coalition value at size `k` under the fusion rule.
- `Mean strict-quality marginal gain`: the expected increase caused by adding the `k`th image under the best-single-image rule.
- `Mean Noisy-OR x best IoU marginal gain`: the expected increase caused by adding the `k`th image under the fusion rule.
- The support drops from 205 to 177 by `k=8` because not every scene has 8 available images.

## What angles can appear in a coalition

- Across the full cache, all `72` M4 viewpoints are present.
- But one coalition is never formed from all 72 at once. A coalition is formed only within one `scene_key`, using whatever viewpoints are available for that specific scene in `scene_view_records.csv`.
- Because this is the count-first angle-agnostic report, there is no restriction that a coalition stay inside one ring, one elevation, or one radius. Low, mid, and high elevations can be mixed if they are available in the same scene. The same is true for radius and azimuth.
- The sampled orderings therefore can use angles from 'all over' within the scene-level viewpoint set.
- Example scene `S0-SM_barrel_1` has available viewpoints such as `elhigh-radfar-az270`, `elhigh-radmid-az045`, `elhigh-radnear-az090`, `ellow-radfar-az135`, `ellow-radmid-az135`, `ellow-radnear-az090`, `ellow-radnear-az135`, `ellow-radnear-az270`, `elmid-radfar-az225`, and `elmid-radmid-az090`.
- One sampled ordering for that scene can therefore produce coalitions like `k=2: [ellow-radnear-az090, ellow-radnear-az135]`, `k=3: [ellow-radnear-az090, ellow-radnear-az135, elhigh-radfar-az270]`, or `k=5: [ellow-radnear-az090, ellow-radnear-az135, elhigh-radfar-az270, ellow-radnear-az270, ellow-radfar-az135]`.
- If the research question becomes `which specific angle is the best teammate?`, then the ring- or subset-based Shapley analyses are the more angle-controlled methods.

## Overall progression

| Added image number | Scene support | Mean best strict-quality value | Mean Noisy-OR x best IoU value | Mean strict-quality marginal gain | Mean Noisy-OR x best IoU marginal gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 205 | 0.8728 | 0.8728 | 0.8728 | 0.8728 |
| 2 | 205 | 0.9087 | 0.9513 | 0.0359 | 0.0785 |
| 3 | 205 | 0.9190 | 0.9661 | 0.0103 | 0.0148 |
| 4 | 205 | 0.9245 | 0.9712 | 0.0055 | 0.0051 |
| 5 | 200 | 0.9281 | 0.9737 | 0.0037 | 0.0026 |
| 6 | 194 | 0.9315 | 0.9759 | 0.0028 | 0.0018 |
| 7 | 188 | 0.9330 | 0.9769 | 0.0022 | 0.0013 |
| 8 | 177 | 0.9355 | 0.9780 | 0.0018 | 0.0009 |

## Headline interpretation

- The row for `k=1` is the expected value of the first available image because the empty coalition starts at value `0`.
- The **2nd image** adds `0.0359` on the best-strict-quality game and `0.0785` on the fusion game.
- The **3rd image** adds `0.0103` on the best-strict-quality game and `0.0148` on the fusion game.
- For example, under the best strict-quality game, the mean coalition value rises from `0.8728` at 1 image to `0.9087` at 2 images, so the average 2nd-image gain is `0.0359`.
- Under the fusion game, the mean coalition value rises from `0.8728` at 1 image to `0.9513` at 2 images, so the average 2nd-image gain is `0.0785`.
- This is the count-first result you can report even before estimating exact angles on real imagery.
- The pattern shows strong diminishing returns: most of the gain comes from the 2nd image, the 3rd still helps, and later added images contribute progressively smaller average improvements.

## How to use this in the thesis

- Use this report for the question `how many different images are worth adding?`.
- Use ring or subset Shapley only after that, when the question becomes `which viewpoint is the most valuable teammate?`.
- The count-first report is the cleaner bridge from synthetic results to raw real-drone images, because it does not require trusted pose metadata.

## Classes with the largest 2nd-image gain

- `barrel`: 2nd-image strict-quality gain `0.0837`, fusion gain `0.1744`.
- `male`: 2nd-image strict-quality gain `0.0680`, fusion gain `0.1685`.
- `suv`: 2nd-image strict-quality gain `0.0581`, fusion gain `0.0974`.
- `tank`: 2nd-image strict-quality gain `0.0494`, fusion gain `0.0938`.
- `whitevan`: 2nd-image strict-quality gain `0.0352`, fusion gain `0.0626`.
- `rock`: 2nd-image strict-quality gain `0.0169`, fusion gain `0.0415`.
- `tree`: 2nd-image strict-quality gain `0.0143`, fusion gain `0.0423`.
- `container`: 2nd-image strict-quality gain `0.0130`, fusion gain `0.0411`.

