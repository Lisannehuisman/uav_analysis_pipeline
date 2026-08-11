# Data Collection Notes

This folder contains the data that directly supports the thesis experiments.

## Included Datasets

| Folder | Contents | Why it is included |
| --- | --- | --- |
| `raw_data/synthetic_subset` | 200 AirSim images with YOLO labels, split into train/val/test | Representative subset of the full synthetic capture used for code inspection and small reruns |
| `raw_data/same_taxonomy_54_images` | Full 54-image real same-taxonomy dataset | Used for the same-taxonomy transfer experiment |
| `raw_data/self_collected_uav_validation` | Full 156-image self-collected UAV validation dataset | Used for real-world validation and fine-tune evaluation |
| `raw_data/real_same_object_multiview` | 36 labeled real UAV source views of five object groups | Used for the real same-object multiview Shapley analysis; one Ooij tower view has no target box, leaving 35 usable views |

The synthetic subset keeps the AirSim filename convention:

```text
S0-SM_<object>-el<low|mid|high>-rad<near|mid|far>-az<000..315>.png
```

The full synthetic source manifest is kept as `raw_data/synthetic_subset/source_manifest_full.csv`. This preserves the complete capture inventory without copying the full image dataset.

## Class Orders

The synthetic AirSim data uses the 10-class order:

```text
tent, tank, tower, container, whitevan, suv, male, rock, barrel, tree
```

The real validation and Roboflow exports keep their original YAML files because the class order differs between some exports. Scripts that compare predictions against real labels explicitly translate class IDs where needed.
