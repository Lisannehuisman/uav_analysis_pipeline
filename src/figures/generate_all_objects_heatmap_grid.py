from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_DIR = WORKSPACE / "s0_m4" / "object_viewpoint_heatmaps_s0_m4"
OUTPUT_DIR = WORKSPACE / "s0_m4" / "thesis_viewpoint_analysis_s0_m4" / "heatmaps_avgoverradius"
OUTPUT_PATH = OUTPUT_DIR / "all_objects_ap50_95_heatmaps.png"


def main() -> None:
    image_paths = sorted(SOURCE_DIR.glob("*_ap50_95_heatmaps.png"))
    if not image_paths:
        raise FileNotFoundError(f"No heatmaps found in {SOURCE_DIR}")

    columns = 2
    rows = ceil(len(image_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 4.6))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, image_path in zip(axes_list, image_paths, strict=False):
        image = mpimg.imread(image_path)
        axis.imshow(image)
        axis.set_title(image_path.stem.replace("_ap50_95_heatmaps", ""), fontsize=11)
        axis.axis("off")

    for axis in axes_list[len(image_paths):]:
        axis.axis("off")

    fig.suptitle("Azimuth-elevation heatmaps averaged over radius by object class", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
