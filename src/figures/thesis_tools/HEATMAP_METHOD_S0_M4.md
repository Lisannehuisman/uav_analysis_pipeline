# How The Current `S0_M4` Heatmaps Are Calculated

The current heatmaps are produced by:

- `plot_object_viewpoint_heatmaps.py`

and are based on:

- `comparison_output/per_image_metrics_model_b.csv`

## Input Data

The per-image CSV contains one row per test image with:

- image path
- precision
- recall
- F1
- AP50
- AP50-95

For the heatmaps we currently use:

- `ap50_95` by default

because the cached per-image outputs do **not** include a raw per-image IoU column.

## Where The Viewpoint Comes From

The script parses the image filename to recover:

- object class
- azimuth
- elevation
- radius

For example, a filename like:

`S0-SM_barrel_1-elhigh-radfar-az270.png`

is interpreted as:

- object = `barrel`
- elevation = `high`
- radius = `far`
- azimuth = `270`

## What A Heatmap Cell Means

For each object class, the script creates a matrix over:

- rows = elevation
- columns = azimuth

and it does this separately for each radius.

So each object gets:

- one heatmap for `near`
- one heatmap for `mid`
- one heatmap for `far`

Each cell is:

- the **mean** of the chosen metric
- over all test images that match that exact:
  - object
  - azimuth
  - elevation
  - radius

In compact form:

`cell_value = mean(metric_value for all images with same object + azimuth + elevation + radius)`

## How The Best Cell Is Chosen

Within each radius-specific heatmap:

- the best cell is the cell with the **highest mean value**

This is why the heatmaps are descriptive of average performance, not single-image extremes.

## Important Limitation

Some exact cells have very few supporting images.

That means:

- the heatmaps are still useful for pattern discovery
- but exact best `azimuth + elevation + radius` cells should be interpreted carefully

This is also why the separate factor-level analysis in:

- `factor_level_viewpoint_analysis/results_s0_m4`

is stronger for the thesis main text, because it pools over the other dimensions and gives much larger support per result.
