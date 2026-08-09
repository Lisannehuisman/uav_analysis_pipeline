import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "figure_3_2_viewpoint_sampling_recomputed.png"
RAW = ROOT / "data_collection" / "raw_data" / "synthetic_subset" / "images" / "test"


AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
RADII = [("near", 1, "10 m"), ("mid", 2, "16 m"), ("far", 3, "22 m")]
HEIGHT_BANDS = [
    ("low", "Low band", "2 m above target", "5-11 deg LOS elevation"),
    ("mid", "Mid band", "12 m above target", "29-50 deg LOS elevation"),
    ("high", "High band", "22 m above target", "45-66 deg LOS elevation"),
]

EXAMPLES = {
    "A": {
        "elevation": "low",
        "radius": "near",
        "azimuth": 45,
        "path": RAW / "S0-SM_tank1-ellow-radnear-az045.png",
        "title": "A. low / near / az045",
    },
    "B": {
        "elevation": "high",
        "radius": "far",
        "azimuth": 45,
        "path": RAW / "S0-SM_tank1-elhigh-radfar-az045.png",
        "title": "B. high / far / az045",
    },
}

def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def draw_slice(ax, elevation_key, title, height_note, angle_note):
    theta = np.deg2rad(AZIMUTHS)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 3.4)
    ax.set_yticks([1, 2, 3], labels=["near", "mid", "far"])
    ax.set_rlabel_position(220)
    ax.set_xticks(theta)
    ax.set_xticklabels([f"{az}" for az in AZIMUTHS], fontsize=9)
    ax.grid(color="#d8dce3", linewidth=0.8)
    ax.spines["polar"].set_color("#c7ccd4")
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_facecolor("#f7f9fc")

    ax.scatter([0], [0], s=220, color="#2f3b52", zorder=5)
    ax.text(0, 0, "Target", color="white", ha="center", va="center", fontsize=8, zorder=6)

    for radius_name, radius_plot, _ in RADII:
        ax.scatter(
            theta,
            np.full(len(theta), radius_plot),
            s=38,
            color="#667085",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

    ax.set_title(
        f"{title}\n{height_note} ({angle_note})",
        fontsize=11,
        pad=18,
        color="#243447",
    )
    ax.text(
        0.5,
        -0.17,
        "24 viewpoints in this slice",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#586174",
    )

    for key, example in EXAMPLES.items():
        if example["elevation"] != elevation_key:
            continue
        radius_map = {name: plot_r for name, plot_r, _ in RADII}
        theta_value = math.radians(example["azimuth"])
        radius_value = radius_map[example["radius"]]
        color = "#f97316" if key == "A" else "#0ea5e9"
        ax.scatter(
            [theta_value],
            [radius_value],
            s=180,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=7,
        )
        ax.text(
            theta_value,
            radius_value + 0.28,
            key,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=8,
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(
        nrows=3,
        ncols=2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.18,
        hspace=0.48,
        figure=fig,
    )

    polar_axes = {}
    for row, (key, title, height_note, angle_note) in enumerate(HEIGHT_BANDS):
        ax = fig.add_subplot(gs[row, 0], projection="polar")
        draw_slice(ax, key, title, height_note, angle_note)
        polar_axes[key] = ax

    ax_img_a = fig.add_subplot(gs[0:2, 1])
    ax_img_b = fig.add_subplot(gs[2, 1])

    img_a = load_image(EXAMPLES["A"]["path"])
    img_b = load_image(EXAMPLES["B"]["path"])
    for ax, image, title, color in [
        (ax_img_a, img_a, EXAMPLES["A"]["title"], "#f97316"),
        (ax_img_b, img_b, EXAMPLES["B"]["title"], "#0ea5e9"),
    ]:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(title, loc="left", fontsize=12, color="#243447", pad=8)

    fig.suptitle(
        "Deterministic 72-view sampling around each object instance",
        fontsize=18,
        y=0.98,
        color="#17212f",
    )
    fig.text(
        0.09,
        0.03,
        "Each object is rendered from 8 azimuths x 3 radius shells x 3 height bands. "
        "The highlighted examples show how the same object changes from a shallow close-range view "
        "to a steeper, farther view while remaining on the same azimuth ray.",
        fontsize=10,
        color="#445063",
    )

    connections = [
        (
            polar_axes["low"],
            (math.radians(EXAMPLES["A"]["azimuth"]), 1),
            ax_img_a,
            (0.0, 0.5),
            "#f97316",
        ),
        (
            polar_axes["high"],
            (math.radians(EXAMPLES["B"]["azimuth"]), 3),
            ax_img_b,
            (0.0, 0.55),
            "#0ea5e9",
        ),
    ]

    for ax_a, xy_a, ax_b, xy_b, color in connections:
        connector = ConnectionPatch(
            xyA=xy_a,
            coordsA="data",
            axesA=ax_a,
            xyB=xy_b,
            coordsB="axes fraction",
            axesB=ax_b,
            arrowstyle="->",
            linewidth=2.2,
            color=color,
            shrinkB=8,
            mutation_scale=14,
        )
        fig.add_artist(connector)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
