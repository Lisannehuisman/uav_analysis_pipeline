# Noisy-OR Shapley Explanation for elmid-radnear

- Input scenes: `151`
- Players: the 8 azimuth viewpoints in this fixed elevation-radius ring.
- Coalition value: per scene, fuse the selected viewpoints with `Noisy-OR + best IoU`, then average over scenes.
- Singleton coalition value means `v({u})`: the score of a one-view coalition containing exactly one azimuth in this fixed ring.
- Shapley value: for each azimuth, average how much it increases coalition value when added to all possible subsets of the other 7 azimuths.
- `Highest Shapley azimuth` therefore means: the azimuth with the largest exact average marginal contribution **within this ring only**, not one universal best angle across all rings.
- For one concrete current coalition `C`, the next-view rule is `Delta(u | C) = v(C union {u}) - v(C)` under the same Noisy-OR + best IoU coalition value.

## Reading the plot

- Left panel: the coalition value grows as more viewpoints are available, but with diminishing returns.
- Right panel: singleton value and Shapley value are not identical.
- In this ring, the best singleton azimuth is `az135` with `v({az135}) = 0.2629`, but the highest Shapley azimuth is `az270` with `phi = 0.1918`.
- That means the best individual viewpoint is not automatically the best collaborative viewpoint.

## Concrete next-view example in the same ring

- Starting from the best singleton coalition `C = {az135}`, the best second azimuth is `az270`.
- That raises coalition value from `0.2629` to `0.4789`, so `Delta(u | C) = 0.2160`.
- Starting from the best pair `C = {az135 + az270}`, the best third azimuth is `az180`.
- That raises coalition value from `0.4789` to `0.6166`, so `Delta(u | C) = 0.1376`.
- This is the right operational quantity if the question is not `who is best on average?` but `which extra drone should I add to the team I already have?`.

## Why this is still the best fusion method

- The coalition game uses the thesis-best coalition rule directly: `Noisy-OR + best IoU`.
- So Shapley is no longer answering `which viewpoint has the best max score?`.
- Instead, Shapley answers `which viewpoint adds the most value on average when coalition value is defined by the best fusion rule?`.
- The conditional gain table then complements that with the concrete decision rule for extending one specific current coalition.