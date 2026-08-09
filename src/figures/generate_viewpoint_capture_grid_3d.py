from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "figure_3_1_viewpoint_grid_recomputed.png"

AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
RADIUS_LEVELS = [
    ("near", 10.0, "10 m", "solid"),
    ("mid", 16.0, "16 m", (0, (5, 3))),
    ("far", 22.0, "22 m", (0, (1.5, 2.5))),
]
ELEVATION_LEVELS = [
    ("low", 2.0, "2 m", "#ef4444"),
    ("mid", 12.0, "12 m", "#a855f7"),
    ("high", 22.0, "22 m", "#06b6d4"),
]

RADIUS_LOOKUP = {name: value for name, value, _label, _style in RADIUS_LEVELS}
ELEVATION_LOOKUP = {name: value for name, value, _label, _color in ELEVATION_LEVELS}


def ring_xy(radius: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, 360)
    return radius * np.cos(theta), radius * np.sin(theta)


def azimuth_point(radius: float, azimuth_deg: float) -> tuple[float, float]:
    theta = math.radians(azimuth_deg)
    return radius * math.cos(theta), radius * math.sin(theta)


def draw_grid(ax: plt.Axes) -> None:
    for elevation_name, elevation_value, _elevation_label, elevation_color in ELEVATION_LEVELS:
        for _radius_name, radius_value, _radius_label, line_style in RADIUS_LEVELS:
            x_ring, y_ring = ring_xy(radius_value)
            ax.plot(
                x_ring,
                y_ring,
                np.full_like(x_ring, elevation_value),
                color=elevation_color,
                linestyle=line_style,
                linewidth=1.6,
                alpha=0.65,
                zorder=1,
            )

            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for azimuth in AZIMUTHS:
                x_pos, y_pos = azimuth_point(radius_value, azimuth)
                xs.append(x_pos)
                ys.append(y_pos)
                zs.append(elevation_value)

            ax.scatter(
                xs,
                ys,
                zs,
                s=46,
                color=elevation_color,
                edgecolors="white",
                linewidths=0.8,
                depthshade=False,
                alpha=0.9,
                zorder=3,
            )

    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        s=210,
        marker="s",
        color="#1d4ed8",
        edgecolors="white",
        linewidths=1.0,
        depthshade=False,
        zorder=6,
    )


def add_annotation_labels(ax: plt.Axes) -> None:
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.35}

    ax.text(
        4.8,
        4.5,
        0.9,
        "object",
        fontsize=11,
        color="#111827",
        bbox=label_box,
        zorder=20,
    )

    radius_angle = 210
    radius_start = 11.5
    radius_end = 22.0
    start_x, start_y = azimuth_point(radius_start, radius_angle)
    end_x, end_y = azimuth_point(radius_end, radius_angle)
    ax.quiver(
        start_x,
        start_y,
        ELEVATION_LOOKUP["high"] + 0.1,
        end_x - start_x,
        end_y - start_y,
        0.0,
        color="#f97316",
        linewidth=2.0,
        arrow_length_ratio=0.12,
        zorder=8,
    )
    ax.text(
        (start_x + end_x) / 2.0 - 1.5,
        (start_y + end_y) / 2.0 + 0.8,
        ELEVATION_LOOKUP["high"] + 1.2,
        "radius",
        fontsize=11,
        color="#111827",
        bbox=label_box,
        zorder=20,
    )

    arc_radius = 14.5
    arc_angles_deg = np.linspace(285, 360, 80)
    arc_x = arc_radius * np.cos(np.deg2rad(arc_angles_deg))
    arc_y = arc_radius * np.sin(np.deg2rad(arc_angles_deg))
    arc_z = np.full_like(arc_x, ELEVATION_LOOKUP["mid"] + 0.2)
    ax.plot(arc_x, arc_y, arc_z, color="#16a34a", linewidth=2.2, zorder=8)
    tail_x, tail_y = azimuth_point(arc_radius, 352)
    head_x, head_y = azimuth_point(arc_radius, 360)
    ax.quiver(
        tail_x,
        tail_y,
        ELEVATION_LOOKUP["mid"] + 0.2,
        head_x - tail_x,
        head_y - tail_y,
        0.0,
        color="#16a34a",
        linewidth=2.2,
        arrow_length_ratio=0.7,
        zorder=8,
    )
    ax.text(
        2.0,
        9.0,
        ELEVATION_LOOKUP["mid"] + 1.0,
        "azimuth",
        fontsize=11,
        color="#111827",
        bbox=label_box,
        zorder=20,
    )

    ax.quiver(
        23.0,
        2.0,
        0.0,
        0.0,
        0.0,
        15.5,
        color="#2563eb",
        linewidth=2.0,
        arrow_length_ratio=0.08,
        zorder=8,
    )
    ax.text(
        18.5,
        2.4,
        6.2,
        "elevation",
        fontsize=11,
        color="#111827",
        bbox=label_box,
        zorder=20,
    )


def build_legend() -> list[Line2D]:
    handles: list[Line2D] = []
    for elevation_name, _value, elevation_label, elevation_color in ELEVATION_LEVELS:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=8,
                markerfacecolor=elevation_color,
                markeredgecolor="white",
                label=f"{elevation_name.capitalize()} elevation ({elevation_label})",
            )
        )

    for radius_name, _value, radius_label, line_style in RADIUS_LEVELS:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#475569",
                linestyle=line_style,
                linewidth=2.0,
                label=f"{radius_name.capitalize()} radius ({radius_label})",
            )
        )

    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=9,
            markerfacecolor="#1d4ed8",
            markeredgecolor="white",
            label="Target object",
        )
    )
    return handles


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_zlim(0, 26)
    ax.set_xticks(np.arange(-30, 31, 10))
    ax.set_yticks(np.arange(-30, 31, 10))
    ax.set_zticks([value for _name, value, _label, _color in ELEVATION_LEVELS])
    ax.set_zticklabels([name.capitalize() for name, _value, _label, _color in ELEVATION_LEVELS])
    ax.set_xlabel("Relative x position (m)", labelpad=12)
    ax.set_ylabel("Relative y position (m)", labelpad=12)
    ax.set_zlabel("Elevation", labelpad=12)
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.45)
    ax.view_init(elev=24, azim=48)
    ax.set_box_aspect((1.0, 1.0, 0.8))

    # Keep the 3D panes light so the geometry stays readable in print.
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.zaxis.pane.set_facecolor((0.985, 0.99, 1.0, 1.0))
    ax.xaxis.pane.set_edgecolor("#dbe3ee")
    ax.yaxis.pane.set_edgecolor("#dbe3ee")
    ax.zaxis.pane.set_edgecolor("#dbe3ee")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig = plt.figure(figsize=(11.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    draw_grid(ax)
    add_annotation_labels(ax)
    style_axes(ax)

    legend_handles = build_legend()
    ax.legend(
        handles=legend_handles,
        title="Legend",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        ncol=1,
    )

    fig.subplots_adjust(left=0.05, right=0.78, bottom=0.06, top=0.98)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
