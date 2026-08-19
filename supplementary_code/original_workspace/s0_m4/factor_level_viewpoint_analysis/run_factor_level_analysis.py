from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]


@dataclass(frozen=True)
class Record:
    class_name: str
    azimuth: str
    elevation: str
    radius: str
    metric_value: float


@dataclass(frozen=True)
class FactorStats:
    object_class: str
    factor: str
    level: str
    mean_value: float
    std_value: float
    variance_value: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run higher-support factor-level viewpoint analysis from cached per-image metrics."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Per-image metrics CSV for one model.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to infer canonical class names.",
    )
    parser.add_argument(
        "--metric",
        default="ap50_95",
        choices=sorted(VALID_METRICS),
        help="Per-image metric to analyze.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label shown in figure titles.",
    )
    parser.add_argument(
        "--output-dir",
        default="factor_level_viewpoint_analysis/results_s0_m4",
        help="Output folder for the factor-level analysis results.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    seen: list[str] = []
    for row in rows:
        class_name = row.get("class_name", "").strip()
        if class_name and class_name not in seen:
            seen.append(class_name)
    return seen


def parse_viewpoint_tokens(image_path: str, class_names: list[str]) -> dict[str, str] | None:
    stem = Path(image_path).stem
    object_token_match = re.search(r"^S0-SM_([^-]+)-", stem, re.IGNORECASE)
    if not object_token_match:
        return None
    object_token = object_token_match.group(1).lower()

    class_name = None
    for candidate in sorted(class_names, key=len, reverse=True):
        if object_token.startswith(candidate.lower()):
            class_name = candidate
            break
    if class_name is None:
        fallback = re.match(r"([a-zA-Z]+)", object_token)
        if not fallback:
            return None
        class_name = fallback.group(1).lower()

    azimuth_match = re.search(r"-az(\d+)", stem, re.IGNORECASE)
    elevation_match = re.search(r"-el([a-z]+)", stem, re.IGNORECASE)
    radius_match = re.search(r"-rad([a-z]+)", stem, re.IGNORECASE)
    if not azimuth_match or not elevation_match or not radius_match:
        return None

    elevation = elevation_match.group(1).lower()
    if elevation == "ellow":
        elevation = "low"
    radius = radius_match.group(1).lower()

    return {
        "class_name": class_name,
        "azimuth": azimuth_match.group(1),
        "elevation": elevation,
        "radius": radius,
    }


def build_records(rows: list[dict[str, str]], class_names: list[str], metric: str) -> list[Record]:
    records: list[Record] = []
    for row in rows:
        tokens = parse_viewpoint_tokens(row["image"], class_names)
        if tokens is None:
            continue
        metric_value = row.get(metric)
        if metric_value in (None, ""):
            continue
        records.append(
            Record(
                class_name=tokens["class_name"],
                azimuth=tokens["azimuth"],
                elevation=tokens["elevation"],
                radius=tokens["radius"],
                metric_value=float(metric_value),
            )
        )
    return records


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def variance(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.var(np.asarray(values, dtype=float), ddof=0))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def level_order(factor: str, levels: set[str]) -> list[str]:
    if factor == "azimuth":
        return sorted(levels, key=int)
    if factor == "elevation":
        return [level for level in ELEVATION_ORDER if level in levels]
    return [level for level in RADIUS_ORDER if level in levels]


def compute_factor_stats(records: list[Record], factor: str) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[FactorStats]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        level = getattr(record, factor)
        grouped[record.class_name][level].append(record.metric_value)

    stats_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    profiles: dict[str, list[FactorStats]] = {}

    for class_name in sorted(grouped):
        class_rows: list[FactorStats] = []
        for level in level_order(factor, set(grouped[class_name])):
            values = grouped[class_name][level]
            stat = FactorStats(
                object_class=class_name,
                factor=factor,
                level=level,
                mean_value=mean(values),
                std_value=std(values),
                variance_value=variance(values),
                count=len(values),
            )
            class_rows.append(stat)
            stats_rows.append(
                {
                    "object_class": class_name,
                    "factor": factor,
                    "level": level,
                    "mean_value": stat.mean_value,
                    "std_value": stat.std_value,
                    "variance_value": stat.variance_value,
                    "n_images": stat.count,
                }
            )

        class_rows.sort(key=lambda row: (row.mean_value, -row.std_value, row.count), reverse=True)
        profiles[class_name] = class_rows
        best = class_rows[0]
        worst = min(class_rows, key=lambda row: row.mean_value)
        best_rows.append(
            {
                "object_class": class_name,
                "factor": factor,
                "best_level": best.level,
                "best_mean_value": best.mean_value,
                "best_std_value": best.std_value,
                "best_variance_value": best.variance_value,
                "best_n_images": best.count,
                "worst_level": worst.level,
                "worst_mean_value": worst.mean_value,
                "effect_size": best.mean_value - worst.mean_value,
                "mean_group_support": mean(row.count for row in class_rows),
                "min_group_support": min(row.count for row in class_rows),
            }
        )

    return stats_rows, best_rows, profiles


def plot_factor_profiles(
    factor: str,
    profiles: dict[str, list[FactorStats]],
    output_path: Path,
    label: str,
    metric: str,
) -> None:
    classes = sorted(profiles)
    ncols = 2
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        ordered = sorted(
            profiles[class_name],
            key=lambda row: int(row.level) if factor == "azimuth" else (
                ELEVATION_ORDER.index(row.level) if factor == "elevation" else RADIUS_ORDER.index(row.level)
            ),
        )
        x_labels = [row.level for row in ordered]
        x = np.arange(len(x_labels))
        y = [row.mean_value for row in ordered]
        yerr = [row.std_value for row in ordered]
        supports = [row.count for row in ordered]

        ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, color="#1f77b4")
        best_idx = int(np.argmax(y))
        ax.scatter(x[best_idx], y[best_idx], s=120, color="#d62728", zorder=3)
        for xi, yi, n_images in zip(x, y, supports, strict=True):
            ax.text(xi, min(yi + 0.04, 0.98), f"n={n_images}", ha="center", fontsize=7)

        ax.set_title(class_name)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)

    for idx in range(len(classes), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{factor.capitalize()}-only viewpoint profiles for {label}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_factor_effects(best_rows_by_factor: dict[str, list[dict[str, object]]], output_path: Path, label: str, metric: str) -> None:
    classes = sorted(row["object_class"] for row in best_rows_by_factor["azimuth"])
    rows_by_factor_and_class = {
        factor: {row["object_class"]: row for row in rows}
        for factor, rows in best_rows_by_factor.items()
    }

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width,
        [rows_by_factor_and_class["azimuth"][cls]["effect_size"] for cls in classes],
        width=width,
        label="Azimuth",
    )
    ax.bar(
        x,
        [rows_by_factor_and_class["elevation"][cls]["effect_size"] for cls in classes],
        width=width,
        label="Elevation",
    )
    ax.bar(
        x + width,
        [rows_by_factor_and_class["radius"][cls]["effect_size"] for cls in classes],
        width=width,
        label="Radius",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel(f"{metric.upper()} effect size")
    ax.set_title(f"Effect size by factor and object for {label}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_summary_rows(best_rows_by_factor: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    classes = sorted(row["object_class"] for row in best_rows_by_factor["azimuth"])
    rows_by_factor_and_class = {
        factor: {row["object_class"]: row for row in rows}
        for factor, rows in best_rows_by_factor.items()
    }
    summary_rows: list[dict[str, object]] = []
    for class_name in classes:
        az = rows_by_factor_and_class["azimuth"][class_name]
        el = rows_by_factor_and_class["elevation"][class_name]
        rad = rows_by_factor_and_class["radius"][class_name]
        strongest_factor = max(
            ("azimuth", az["effect_size"]),
            ("elevation", el["effect_size"]),
            ("radius", rad["effect_size"]),
            key=lambda item: item[1],
        )[0]
        summary_rows.append(
            {
                "object_class": class_name,
                "best_azimuth": az["best_level"],
                "best_azimuth_mean": az["best_mean_value"],
                "best_azimuth_n": az["best_n_images"],
                "best_elevation": el["best_level"],
                "best_elevation_mean": el["best_mean_value"],
                "best_elevation_n": el["best_n_images"],
                "best_radius": rad["best_level"],
                "best_radius_mean": rad["best_mean_value"],
                "best_radius_n": rad["best_n_images"],
                "strongest_factor": strongest_factor,
                "azimuth_effect_size": az["effect_size"],
                "elevation_effect_size": el["effect_size"],
                "radius_effect_size": rad["effect_size"],
            }
        )
    return summary_rows


def plot_summary_table(summary_rows: list[dict[str, object]], output_path: Path, label: str) -> None:
    headers = [
        "Object",
        "Best az",
        "n",
        "Best el",
        "n",
        "Best rad",
        "n",
        "Strongest",
    ]
    table_rows = [
        [
            row["object_class"],
            row["best_azimuth"],
            int(row["best_azimuth_n"]),
            row["best_elevation"],
            int(row["best_elevation_n"]),
            row["best_radius"],
            int(row["best_radius_n"]),
            row["strongest_factor"],
        ]
        for row in summary_rows
    ]

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(table_rows) + 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title(f"Factor-level best viewpoints per object for {label}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(
    summary_rows: list[dict[str, object]],
    best_rows_by_factor: dict[str, list[dict[str, object]]],
    total_images: int,
    output_path: Path,
    metric: str,
    label: str,
) -> None:
    strongest_counter = Counter(str(row["strongest_factor"]) for row in summary_rows)
    best_elevation_counter = Counter(str(row["best_elevation"]) for row in summary_rows)
    best_radius_counter = Counter(str(row["best_radius"]) for row in summary_rows)
    best_azimuth_counter = Counter(str(row["best_azimuth"]) for row in summary_rows)

    az_support_mean = mean(row["mean_group_support"] for row in best_rows_by_factor["azimuth"])
    el_support_mean = mean(row["mean_group_support"] for row in best_rows_by_factor["elevation"])
    rad_support_mean = mean(row["mean_group_support"] for row in best_rows_by_factor["radius"])

    top_effect_objects = sorted(
        summary_rows,
        key=lambda row: max(
            float(row["azimuth_effect_size"]),
            float(row["elevation_effect_size"]),
            float(row["radius_effect_size"]),
        ),
        reverse=True,
    )[:3]
    top_effect_summary = ", ".join(
        f"{row['object_class']} ({row['strongest_factor']})" for row in top_effect_objects
    )

    lines = [
        f"# Factor-level viewpoint summary for {label}",
        "",
        f"Metric analyzed: `{metric}`. The cached per-image file does not contain a raw IoU column, so `{metric}` is used as the available localization-quality proxy.",
        f"Total test images analyzed: {total_images}.",
        "",
        "## Why this factor-level analysis is stronger",
        f"- Mean support per azimuth bin: {az_support_mean:.1f} images.",
        f"- Mean support per elevation bin: {el_support_mean:.1f} images.",
        f"- Mean support per radius bin: {rad_support_mean:.1f} images.",
        "- These support levels are much stronger than exact azimuth+elevation+radius cells, so the conclusions below are more defensible for the thesis.",
        "",
        "## Main results",
        f"- Most common best elevation: {best_elevation_counter.most_common(1)[0][0]} ({best_elevation_counter.most_common(1)[0][1]} of {len(summary_rows)} objects).",
        f"- Most common best radius: {best_radius_counter.most_common(1)[0][0]} ({best_radius_counter.most_common(1)[0][1]} of {len(summary_rows)} objects).",
        f"- Most common best azimuth: {best_azimuth_counter.most_common(1)[0][0]} ({best_azimuth_counter.most_common(1)[0][1]} of {len(summary_rows)} objects).",
        f"- Strongest factor most often: {', '.join(f'{factor} ({count})' for factor, count in strongest_counter.most_common())}.",
        f"- Objects with the clearest factor-level viewpoint dependence: {top_effect_summary}.",
        "",
        "## Thesis-ready interpretation",
        "- It is more reliable to state which elevation, radius, or azimuth band tends to work best than to claim a single exact viewpoint cell.",
        "- Elevation and radius can now be discussed with substantially more statistical support because each factor pools over the other two dimensions.",
        "- These outputs are intended to replace the sparse exact-combination claims in the thesis discussion.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_path = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(per_image_path)
    class_names = load_class_names(per_class_path)
    records = build_records(rows, class_names, args.metric)
    if not records:
        raise RuntimeError(f"No usable records were found in {per_image_path}.")

    best_rows_by_factor: dict[str, list[dict[str, object]]] = {}
    for factor in ("azimuth", "elevation", "radius"):
        stats_rows, best_rows, profiles = compute_factor_stats(records, factor)
        write_csv(
            output_dir / f"{factor}_stats.csv",
            ["object_class", "factor", "level", "mean_value", "std_value", "variance_value", "n_images"],
            stats_rows,
        )
        write_csv(
            output_dir / f"best_{factor}_per_object.csv",
            [
                "object_class",
                "factor",
                "best_level",
                "best_mean_value",
                "best_std_value",
                "best_variance_value",
                "best_n_images",
                "worst_level",
                "worst_mean_value",
                "effect_size",
                "mean_group_support",
                "min_group_support",
            ],
            best_rows,
        )
        plot_factor_profiles(
            factor,
            profiles,
            output_dir / f"{factor}_profiles.png",
            args.label,
            args.metric,
        )
        best_rows_by_factor[factor] = best_rows

    summary_rows = make_summary_rows(best_rows_by_factor)
    write_csv(
        output_dir / "factor_best_summary.csv",
        [
            "object_class",
            "best_azimuth",
            "best_azimuth_mean",
            "best_azimuth_n",
            "best_elevation",
            "best_elevation_mean",
            "best_elevation_n",
            "best_radius",
            "best_radius_mean",
            "best_radius_n",
            "strongest_factor",
            "azimuth_effect_size",
            "elevation_effect_size",
            "radius_effect_size",
        ],
        summary_rows,
    )

    plot_factor_effects(
        best_rows_by_factor,
        output_dir / "factor_effect_sizes.png",
        args.label,
        args.metric,
    )
    plot_summary_table(
        summary_rows,
        output_dir / "factor_best_summary.png",
        args.label,
    )
    build_summary_text(
        summary_rows,
        best_rows_by_factor,
        total_images=len(records),
        output_path=output_dir / "factor_level_summary.md",
        metric=args.metric,
        label=args.label,
    )

    print(f"Saved factor-level viewpoint analysis to: {output_dir}")


if __name__ == "__main__":
    main()
