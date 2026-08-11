bet# Dataset Structure Audit

## Headline findings
- Total images: 14760
- Split sizes: train 10332, val 2214, test 2214
- Unique object instances inferred from filenames: 205
- Instances appearing in more than one split: 205/205
- Instances appearing in all three splits: 205/205
- Viewpoint grid balance: overall elevation, radius, and azimuth counts are exactly balanced.

## What the dataset is doing well
- The dataset is already very strong on viewpoint coverage density.
- Each object instance is usually covered by a near-complete 72-view grid.
- The global counts over elevation, radius, and azimuth are not the main source of bias.

## Structural risks
- The main bias is not missing viewpoints but repeated viewpoints of the same object instances.
- Because the same instances appear in train, val, and test, the benchmark mainly measures viewpoint interpolation on known objects.
- That setup can make results look much better than they would on unseen instances or scenes.

## Class imbalance by unique instances
- tower: 13 instances, 936 images, 72.00 views per instance on average
- tent: 18 instances, 1296 images, 72.00 views per instance on average
- whitevan: 19 instances, 1368 images, 72.00 views per instance on average
- tank: 20 instances, 1440 images, 72.00 views per instance on average
- male: 20 instances, 1440 images, 72.00 views per instance on average
- barrel: 20 instances, 1440 images, 72.00 views per instance on average
- rock: 21 instances, 1512 images, 72.00 views per instance on average
- container: 22 instances, 1584 images, 72.00 views per instance on average
- suv: 22 instances, 1584 images, 72.00 views per instance on average
- tree: 30 instances, 2160 images, 72.00 views per instance on average

## Label visibility issues in the test split
- barrel: absent target fraction 0.0660 (14/212)
- suv: absent target fraction 0.0415 (10/241)
- male: absent target fraction 0.0283 (6/212)
- whitevan: absent target fraction 0.0221 (4/181)
- tank: absent target fraction 0.0100 (2/200)

## Weak performance slices in the current test set
- tank | elevation=low: mean test AP50-95 0.5426 over 63 images
- whitevan | elevation=low: mean test AP50-95 0.5721 over 55 images
- tree | elevation=low: mean test AP50-95 0.5831 over 106 images
- suv | elevation=low: mean test AP50-95 0.5929 over 80 images
- tank | azimuth=270: mean test AP50-95 0.6047 over 20 images
- male | elevation=low: mean test AP50-95 0.6071 over 66 images
- tower | elevation=low: mean test AP50-95 0.6071 over 60 images
- barrel | elevation=low: mean test AP50-95 0.6305 over 82 images

## Best next data additions
- Add new object instances before adding more viewpoints of existing instances.
- Build an instance-disjoint test split so that no object instance appears in both train and test.
- Use extra data to strengthen the low-performing slices, but collect those slices on new instances and new scenes.
- If you want equal class balance by unique instances, the largest gains come from the classes below.
- tower: add 17 new instances (about 1224 images for a full 72-view capture)
- tent: add 12 new instances (about 864 images for a full 72-view capture)
- whitevan: add 11 new instances (about 792 images for a full 72-view capture)
- tank: add 10 new instances (about 720 images for a full 72-view capture)
- male: add 10 new instances (about 720 images for a full 72-view capture)
- barrel: add 10 new instances (about 720 images for a full 72-view capture)

## Output files
- split_summary.csv
- class_summary.csv
- factor_summary.csv
- class_factor_summary.csv
- visibility_summary.csv
- instance_split_overlap.csv
- weak_slices.csv
- expansion_plan_equalize_instances.csv