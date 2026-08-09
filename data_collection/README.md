# Data Collection

This folder contains the datasets and preparation scripts used in the thesis experiments.

## Datasets

| Folder | Contents | Why it is included |
| --- | --- | --- |
| `raw_data/synthetic_subset` | 200 AirSim images with YOLO labels, split into train/validation/test | Representative subset of the full 14,760-image synthetic capture |
| `raw_data/same_taxonomy_54_images` | 54 real images using the thesis taxonomy | Used for the same-taxonomy zero-shot transfer experiment |
| `raw_data/self_collected_uav_validation` | Full 156-image self-collected UAV validation dataset | Used for real-world validation and fine-tune evaluation |
| `raw_data/real_same_object_multiview` | 36 labeled real UAV source views of five object groups | Used for the real same-object multiview Shapley analysis; one Ooij tower view has no target box, leaving 35 usable views |

The `same_taxonomy_54_images` folder keeps both `complete_set/` and the train/validation/test split folders, so a recursive file count includes split copies as well as the 54 unique source images.

The synthetic subset keeps the AirSim filename convention:

```text
S0-SM_<object>-el<low|mid|high>-rad<near|mid|far>-az<000..315>.png
```

The full synthetic source manifest is kept as `raw_data/synthetic_subset/source_manifest_full.csv`. This keeps the complete capture inventory without storing the whole generated image dataset in Git.

## Class Orders

The synthetic AirSim data uses the 10-class order:

```text
tent, tank, tower, container, whitevan, suv, male, rock, barrel, tree
```

The real validation and Roboflow exports keep their original YAML files because the class order differs between some exports. Scripts that compare predictions against real labels explicitly translate class IDs where needed.

## External Data

VisDrone and AU-AIR are not stored here. They can be downloaded from their official dataset pages:

- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- AU-AIR: https://bozcani.github.io/auairdataset
