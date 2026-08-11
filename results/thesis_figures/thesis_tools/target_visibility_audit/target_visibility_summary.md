# Target visibility audit

- Total images checked: 2214
i - Images where filename target is absent from GT labels: 40
- Overall absent fraction: 0.0181

Interpretation:
- These images should not be treated as clean positive evidence for the filename target object.
- For viewpoint analysis, they are better interpreted as target-absent or fully occluded target-view samples.
- The detector-family benchmark itself is unaffected, because that evaluation uses the actual GT labels.