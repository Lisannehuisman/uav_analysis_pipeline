from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["precision", "recall", "f1", "ap50", "ap50_95"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-object-type boxplots from cached per-image model comparison CSV files."
    )
    parser.add_argument(
        "--per-image-a",
        default="comparison_output/per_image_metrics_model_a.csv",
        help="CSV generated earlier for model A per-image metrics.",
    )
    parser.add_argument(
        "--per-image-b",
        default=None,
        help="Optional CSV generated earlier for model B per-image metrics.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Optional per-class CSV used to infer class names and plotting order.",
    )
    parser.add_argument("--label-a", default="Model A", help="Plot label for model A.")
    parser.add_argument("--label-b", default="Model B", help="Plot label for model B.")
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay jittered per-image points. Slower; boxplots only by default.",
    )
    parser.add_argument(
        "--max-points-per-class",
        type=int,
        default=120,
        help="Maximum number of jittered points per class and metric when --show-points is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/per_object_boxplots",
        help="Folder where the new class-wise plots will be written.",
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


def infer_class_name(image_path: str, known_classes: list[str]) -> str:
    stem = Path(image_path).stem
    object_token_match = re.search(r"^S0-SM_([^-]+)-", stem, re.IGNORECASE)
    if object_token_match:
        object_token = object_token_match.group(1).lower()
        for class_name in sorted(known_classes, key=len, reverse=True):
            if object_token.startswith(class_name.lower()):
                return class_name

    parts = stem.lower().split("_")
    if len(parts) >= 2:
        match = re.match(r"[a-zA-Z]+", parts[1])
        if match:
            return match.group(0)
    return "unknown"


def group_rows_by_class(
    rows: list[dict[str, str]],
    known_classes: list[str],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        class_name = infer_class_name(row["image"], known_classes)
        for metric in METRICS:
            value = row.get(metric)
            if value is None or value == "":
                continue
            grouped[class_name][metric].append(float(value))
    return grouped


def ordered_classes(
    grouped_a: dict[str, dict[str, list[float]]],
    grouped_b: dict[str, dict[str, list[float]]],
    preferred_order: list[str],
) -> list[str]:
    combined = list({*grouped_a.keys(), *grouped_b.keys()})
    if preferred_order:
        preferred = [name for name in preferred_order if name in combined]
        extras = sorted(name for name in combined if name not in preferred)
        return preferred + extras
    return sorted(combined)


def create_metric_plot(
    output_path: Path,
    metric: str,
    class_names: list[str],
    grouped_a: dict[str, dict[str, list[float]]],
    grouped_b: dict[str, dict[str, list[float]]],
    label_a: str,
    label_b: str,
    include_b: bool,
    show_points: bool,
    max_points_per_class: int,
) -> None:
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 1.2), 6))
    x = np.arange(len(class_names)) + 1
    offset = 0.18 if include_b else 0.0
    width = 0.28 if include_b else 0.45

    data_a = [grouped_a.get(class_name, {}).get(metric, []) for class_name in class_names]
    data_b = [grouped_b.get(class_name, {}).get(metric, []) for class_name in class_names]

    bp_a = ax.boxplot(
        data_a,
        positions=x - offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    bp_b = None
    if include_b:
        bp_b = ax.boxplot(
            data_b,
            positions=x + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )

    for patch in bp_a["boxes"]:
        patch.set_facecolor("#cfe2f3")
    if bp_b is not None:
        for patch in bp_b["boxes"]:
            patch.set_facecolor("#f9d5b3")
    medians = bp_a["medians"] + (bp_b["medians"] if bp_b is not None else [])
    for median in medians:
        median.set_color("#d62728")
        median.set_linewidth(1.8)

    if show_points:
        rng = np.random.default_rng(42)
        for idx, values in enumerate(data_a, start=1):
            if values:
                sampled = values
                if len(values) > max_points_per_class:
                    chosen = rng.choice(len(values), size=max_points_per_class, replace=False)
                    sampled = [values[i] for i in chosen]
                ax.scatter(
                    rng.normal(idx - offset, 0.025, size=len(sampled)),
                    sampled,
                    s=8,
                    alpha=0.18,
                    color="#1f77b4",
                )
        if include_b:
            for idx, values in enumerate(data_b, start=1):
                if values:
                    sampled = values
                    if len(values) > max_points_per_class:
                        chosen = rng.choice(len(values), size=max_points_per_class, replace=False)
                        sampled = [values[i] for i in chosen]
                    ax.scatter(
                        rng.normal(idx + offset, 0.025, size=len(sampled)),
                        sampled,
                        s=8,
                        alpha=0.18,
                        color="#ff7f0e",
                    )

    ax.plot([], [], color="#1f77b4", linewidth=8, label=label_a)
    if include_b:
        ax.plot([], [], color="#ff7f0e", linewidth=8, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by object type")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def draw_metric_boxplot(
    ax,
    metric: str,
    class_names: list[str],
    grouped_a: dict[str, dict[str, list[float]]],
    grouped_b: dict[str, dict[str, list[float]]],
    label_a: str,
    label_b: str,
    include_b: bool,
    show_points: bool,
    max_points_per_class: int,
) -> None:
    x = np.arange(len(class_names)) + 1
    offset = 0.18 if include_b else 0.0
    width = 0.28 if include_b else 0.45

    data_a = [grouped_a.get(class_name, {}).get(metric, []) for class_name in class_names]
    data_b = [grouped_b.get(class_name, {}).get(metric, []) for class_name in class_names]

    bp_a = ax.boxplot(
        data_a,
        positions=x - offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    bp_b = None
    if include_b:
        bp_b = ax.boxplot(
            data_b,
            positions=x + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )

    for patch in bp_a["boxes"]:
        patch.set_facecolor("#cfe2f3")
    if bp_b is not None:
        for patch in bp_b["boxes"]:
            patch.set_facecolor("#f9d5b3")
    medians = bp_a["medians"] + (bp_b["medians"] if bp_b is not None else [])
    for median in medians:
        median.set_color("#d62728")
        median.set_linewidth(1.8)

    if show_points:
        rng = np.random.default_rng(42)
        for idx, values in enumerate(data_a, start=1):
            if values:
                sampled = values
                if len(values) > max_points_per_class:
                    chosen = rng.choice(len(values), size=max_points_per_class, replace=False)
                    sampled = [values[i] for i in chosen]
                ax.scatter(
                    rng.normal(idx - offset, 0.025, size=len(sampled)),
                    sampled,
                    s=8,
                    alpha=0.18,
                    color="#1f77b4",
                )
        if include_b:
            for idx, values in enumerate(data_b, start=1):
                if values:
                    sampled = values
                    if len(values) > max_points_per_class:
                        chosen = rng.choice(len(values), size=max_points_per_class, replace=False)
                        sampled = [values[i] for i in chosen]
                    ax.scatter(
                        rng.normal(idx + offset, 0.025, size=len(sampled)),
                        sampled,
                        s=8,
                        alpha=0.18,
                        color="#ff7f0e",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by object type")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.plot([], [], color="#1f77b4", linewidth=8, label=label_a)
    if include_b:
        ax.plot([], [], color="#ff7f0e", linewidth=8, label=label_b)
    ax.legend()


def main() -> None:
    args = parse_args()
    per_image_a = Path(args.per_image_a).resolve()
    include_b = bool(args.per_image_b)
    per_image_b = Path(args.per_image_b).resolve() if include_b else None
    per_class_csv = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    known_classes = load_class_names(per_class_csv)
    rows_a = read_csv_rows(per_image_a)
    rows_b = read_csv_rows(per_image_b) if include_b and per_image_b is not None else []
    grouped_a = group_rows_by_class(rows_a, known_classes)
    grouped_b = group_rows_by_class(rows_b, known_classes)
    class_names = ordered_classes(grouped_a, grouped_b, known_classes)

    for metric in METRICS:
        create_metric_plot(
            output_path=output_dir / f"{metric}_by_object_type.png",
            metric=metric,
            class_names=class_names,
            grouped_a=grouped_a,
            grouped_b=grouped_b,
            label_a=args.label_a,
            label_b=args.label_b,
            include_b=include_b,
            show_points=args.show_points,
            max_points_per_class=args.max_points_per_class,
        )

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()
    for idx, metric in enumerate(METRICS):
        draw_metric_boxplot(
            ax=axes[idx],
            metric=metric,
            class_names=class_names,
            grouped_a=grouped_a,
            grouped_b=grouped_b,
            label_a=args.label_a,
            label_b=args.label_b,
            include_b=include_b,
            show_points=args.show_points,
            max_points_per_class=args.max_points_per_class,
        )
    axes[-1].axis("off")
    title = f"Per-object-type metrics for {args.label_a}"
    if include_b:
        title = f"Per-object-type metrics: {args.label_a} vs {args.label_b}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "all_metrics_by_object_type.png", dpi=140)
    plt.close(fig)

    print(f"Saved per-object-type boxplots to: {output_dir}")


if __name__ == "__main__":
    main()
