from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(r"C:\DATA\airsim\thesis\captures\COPY OF ACTUAL DATA!\manifest.csv")
DEFAULT_LABELS_DIR = Path(r"C:\DATA\airsim\thesis\captures\COPY OF ACTUAL DATA!\labels")
DEFAULT_OUTPUT_DIR = ROOT / "geometry_ground_truth_analysis" / "outputs"
DEFAULT_DATA_YAML = ROOT / "my_data.yaml"
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a target-centric geometry ground-truth table by joining AirSim "
            "camera/target metadata from manifest.csv with YOLO label files."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR), help="Directory with YOLO label txt files.")
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML), help="Dataset YAML with class id to name mapping.")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Image width in pixels.")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Image height in pixels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV summaries, plots, and report.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_class_map(yaml_path: Path) -> dict[int, str]:
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    class_map: dict[int, str] = {}
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            if not stripped:
                continue
            if not re.match(r"^\d+\s*:", stripped):
                break
            key_text, value_text = stripped.split(":", 1)
            class_map[int(key_text.strip())] = value_text.strip()
    if not class_map:
        raise ValueError(f"Could not parse class names from {yaml_path}")
    return class_map


def extract_focus_class_name(focus_object: str, class_map: dict[int, str]) -> str | None:
    token = focus_object
    if token.startswith("SM_"):
        token = token[3:]
    token = token.lower()
    matches = [name for name in class_map.values() if token.startswith(name.lower())]
    if matches:
        return max(matches, key=len)
    token_without_digits = re.sub(r"\d+$", "", token)
    matches = [name for name in class_map.values() if token_without_digits.startswith(name.lower())]
    if matches:
        return max(matches, key=len)
    return None


def load_yolo_labels(label_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not label_path.exists():
        return rows
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id, cx, cy, w, h = parts[:5]
        rows.append(
            {
                "class_id": int(float(class_id)),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
        )
    return rows


def center_distance(cx: float, cy: float) -> float:
    return math.hypot(cx - 0.5, cy - 0.5)


def select_focus_bbox(
    label_rows: list[dict[str, float]], target_class_id: int | None
) -> tuple[dict[str, float] | None, int, str, bool]:
    if not label_rows:
        return None, 0, "no_labels", False

    global_best = min(label_rows, key=lambda row: center_distance(row["cx"], row["cy"]))
    target_candidates = (
        [row for row in label_rows if row["class_id"] == target_class_id]
        if target_class_id is not None
        else []
    )
    if target_candidates:
        selected = min(target_candidates, key=lambda row: center_distance(row["cx"], row["cy"]))
        return selected, len(target_candidates), "target_class_center", selected == global_best
    return global_best, 0, "global_center_fallback", True


def wrap_angle_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def build_geometry_rows(
    manifest_path: Path,
    labels_dir: Path,
    class_map: dict[int, str],
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    reverse_class_map = {name: idx for idx, name in class_map.items()}
    rows: list[dict[str, object]] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for manifest_row in reader:
            file_name = manifest_row["file_name"]
            label_path = labels_dir / Path(file_name).with_suffix(".txt").name
            label_rows = load_yolo_labels(label_path)

            focus_class_name = extract_focus_class_name(manifest_row["focus_object"], class_map)
            focus_class_id = reverse_class_map.get(focus_class_name) if focus_class_name is not None else None
            selected_bbox, target_class_box_count, selection_method, selected_is_global_center = select_focus_bbox(
                label_rows, focus_class_id
            )

            cam_x = float(manifest_row["cam_x"])
            cam_y = float(manifest_row["cam_y"])
            cam_z = float(manifest_row["cam_z"])
            cam_yaw = float(manifest_row["cam_yaw"])
            cam_pitch = float(manifest_row["cam_pitch"])
            tgt_x = float(manifest_row["tgt_x"])
            tgt_y = float(manifest_row["tgt_y"])
            tgt_z = float(manifest_row["tgt_z"])

            dx = tgt_x - cam_x
            dy = tgt_y - cam_y
            dz = tgt_z - cam_z
            horizontal_distance = math.hypot(dx, dy)
            camera_to_target_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            look_yaw_to_target_deg = wrap_angle_deg(math.degrees(math.atan2(dy, dx)))
            look_pitch_to_target_deg = -math.degrees(math.atan2(dz, horizontal_distance)) if horizontal_distance > 0 else 0.0
            yaw_error_deg = wrap_angle_deg(look_yaw_to_target_deg - cam_yaw)
            pitch_error_deg = look_pitch_to_target_deg - cam_pitch

            row: dict[str, object] = {
                "scene_id": manifest_row["scene_id"],
                "scene_key": manifest_row["episode_id"],
                "focus_object": manifest_row["focus_object"],
                "file_name": file_name,
                "view_idx": int(manifest_row["view_idx"]),
                "focus_class_name": focus_class_name,
                "focus_class_id": focus_class_id,
                "cam_x": cam_x,
                "cam_y": cam_y,
                "cam_z": cam_z,
                "cam_yaw": cam_yaw,
                "cam_pitch": cam_pitch,
                "tgt_x": tgt_x,
                "tgt_y": tgt_y,
                "tgt_z": tgt_z,
                "azimuth_deg": float(manifest_row["azimuth_deg"]),
                "azimuth_bin": int(manifest_row["azimuth_bin"]),
                "radius_m": float(manifest_row["radius_m"]),
                "radius_name": manifest_row["radius_name"],
                "radius_bin": int(manifest_row["radius_bin"]),
                "elevation_z_off": float(manifest_row["elevation_z_off"]),
                "elevation_name": manifest_row["elevation_name"],
                "elevation_bin": int(manifest_row["elevation_bin"]),
                "delta_x_target_minus_camera": dx,
                "delta_y_target_minus_camera": dy,
                "delta_z_target_minus_camera": dz,
                "horizontal_distance_m": horizontal_distance,
                "camera_to_target_distance_m": camera_to_target_distance,
                "look_yaw_to_target_deg": look_yaw_to_target_deg,
                "look_pitch_to_target_deg": look_pitch_to_target_deg,
                "yaw_error_deg": yaw_error_deg,
                "pitch_error_deg": pitch_error_deg,
                "num_label_boxes": len(label_rows),
                "num_target_class_boxes": target_class_box_count,
                "bbox_selection_method": selection_method,
                "selected_bbox_is_global_center_closest": selected_is_global_center,
            }

            if selected_bbox is None:
                row.update(
                    {
                        "selected_bbox_class_id": None,
                        "selected_bbox_class_name": None,
                        "bbox_norm_cx": None,
                        "bbox_norm_cy": None,
                        "bbox_norm_w": None,
                        "bbox_norm_h": None,
                        "bbox_px_x1": None,
                        "bbox_px_y1": None,
                        "bbox_px_x2": None,
                        "bbox_px_y2": None,
                        "bbox_px_w": None,
                        "bbox_px_h": None,
                        "bbox_area_px": None,
                        "bbox_area_norm": None,
                        "bbox_center_distance_norm": None,
                    }
                )
            else:
                cx = float(selected_bbox["cx"])
                cy = float(selected_bbox["cy"])
                w = float(selected_bbox["w"])
                h = float(selected_bbox["h"])
                cx_px = cx * image_width
                cy_px = cy * image_height
                w_px = w * image_width
                h_px = h * image_height
                x1 = cx_px - w_px / 2.0
                y1 = cy_px - h_px / 2.0
                x2 = cx_px + w_px / 2.0
                y2 = cy_px + h_px / 2.0
                row.update(
                    {
                        "selected_bbox_class_id": int(selected_bbox["class_id"]),
                        "selected_bbox_class_name": class_map.get(int(selected_bbox["class_id"])),
                        "bbox_norm_cx": cx,
                        "bbox_norm_cy": cy,
                        "bbox_norm_w": w,
                        "bbox_norm_h": h,
                        "bbox_px_x1": x1,
                        "bbox_px_y1": y1,
                        "bbox_px_x2": x2,
                        "bbox_px_y2": y2,
                        "bbox_px_w": w_px,
                        "bbox_px_h": h_px,
                        "bbox_area_px": w_px * h_px,
                        "bbox_area_norm": w * h,
                        "bbox_center_distance_norm": center_distance(cx, cy),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_episode_summary(view_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        view_df.groupby(["scene_key", "focus_object", "focus_class_name"], as_index=False)
        .agg(
            view_count=("file_name", "count"),
            azimuth_count=("azimuth_bin", "nunique"),
            radius_count=("radius_bin", "nunique"),
            elevation_count=("elevation_bin", "nunique"),
            mean_distance_m=("camera_to_target_distance_m", "mean"),
            min_distance_m=("camera_to_target_distance_m", "min"),
            max_distance_m=("camera_to_target_distance_m", "max"),
            mean_bbox_area_norm=("bbox_area_norm", "mean"),
            min_bbox_area_norm=("bbox_area_norm", "min"),
            max_bbox_area_norm=("bbox_area_norm", "max"),
            mean_bbox_center_distance_norm=("bbox_center_distance_norm", "mean"),
            max_bbox_center_distance_norm=("bbox_center_distance_norm", "max"),
            fallback_count=("bbox_selection_method", lambda values: int((pd.Series(values) == "global_center_fallback").sum())),
            no_label_count=("bbox_selection_method", lambda values: int((pd.Series(values) == "no_labels").sum())),
            mean_abs_yaw_error_deg=("yaw_error_deg", lambda values: pd.Series(values).abs().mean()),
            mean_abs_pitch_error_deg=("pitch_error_deg", lambda values: pd.Series(values).abs().mean()),
        )
    )
    summary["fallback_rate"] = summary["fallback_count"] / summary["view_count"]
    return summary


def plot_distance_vs_bbox_area(view_df: pd.DataFrame, output_path: Path) -> None:
    subset = view_df.dropna(subset=["bbox_area_norm"]).copy()
    if subset.empty:
        return

    colors = {"low": "#4c78a8", "mid": "#f58518", "high": "#54a24b"}
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for elevation_name, elevation_group in subset.groupby("elevation_name", sort=False):
        ax.scatter(
            elevation_group["camera_to_target_distance_m"],
            elevation_group["bbox_area_norm"],
            s=16,
            alpha=0.45,
            label=elevation_name,
            color=colors.get(str(elevation_name), "#777777"),
        )
    ax.set_xlabel("Camera-to-target distance (m)")
    ax.set_ylabel("Selected target bbox area (normalized)")
    ax.set_title("Distance versus selected target bbox size")
    ax.legend(title="Elevation", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_orientation_errors(view_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    axes[0].hist(view_df["yaw_error_deg"], bins=40, color="#4c78a8", alpha=0.85)
    axes[0].set_title("Yaw Error")
    axes[0].set_xlabel("look_yaw_to_target - cam_yaw (deg)")
    axes[0].set_ylabel("Count")

    axes[1].hist(view_df["pitch_error_deg"], bins=40, color="#f58518", alpha=0.85)
    axes[1].set_title("Pitch Error")
    axes[1].set_xlabel("look_pitch_to_target - cam_pitch (deg)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Manifest pose sanity check against target coordinates", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    manifest_path: Path,
    labels_dir: Path,
    view_df: pd.DataFrame,
    episode_df: pd.DataFrame,
) -> None:
    total_views = len(view_df)
    total_episodes = view_df["scene_key"].nunique()
    class_center_count = int((view_df["bbox_selection_method"] == "target_class_center").sum())
    fallback_count = int((view_df["bbox_selection_method"] == "global_center_fallback").sum())
    no_label_count = int((view_df["bbox_selection_method"] == "no_labels").sum())
    mean_abs_yaw_error = float(view_df["yaw_error_deg"].abs().mean())
    mean_abs_pitch_error = float(view_df["pitch_error_deg"].abs().mean())
    mean_center_distance = float(view_df["bbox_center_distance_norm"].dropna().mean())
    max_center_distance = float(view_df["bbox_center_distance_norm"].dropna().max())

    lines = [
        "# Geometry Ground Truth Report",
        "",
        "## Purpose",
        "",
        "This report joins AirSim camera/target metadata from `manifest.csv` with YOLO label files to create a target-centric geometry table per image.",
        "The result is intended as the first geometry-ready layer for multiview target analysis.",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Labels directory: `{labels_dir}`",
        "",
        "## What is available now",
        "",
        "- Camera world position per image (`cam_x`, `cam_y`, `cam_z`)",
        "- Camera orientation per image (`cam_yaw`, `cam_pitch`)",
        "- Target world position per image (`tgt_x`, `tgt_y`, `tgt_z`)",
        "- Viewpoint lattice metadata (`azimuth`, `radius`, `elevation`)",
        "- A selected target bbox per image from the YOLO label file",
        "",
        "## Main counts",
        "",
        f"- Views processed: `{total_views}`",
        f"- Focus-object episodes processed: `{total_episodes}`",
        f"- Target-class-centered bbox selections: `{class_center_count}`",
        f"- Global-center fallbacks: `{fallback_count}`",
        f"- Missing-label rows: `{no_label_count}`",
        "",
        "## Pose sanity check",
        "",
        f"- Mean absolute yaw error: `{mean_abs_yaw_error:.4f}` deg",
        f"- Mean absolute pitch error: `{mean_abs_pitch_error:.4f}` deg",
        "",
        "If these errors stay near zero, the manifest camera pose is internally consistent with the target coordinates and can be used for geometry-aware target analysis.",
        "",
        "## Selected target bbox sanity check",
        "",
        f"- Mean selected bbox center distance from image center: `{mean_center_distance:.4f}`",
        f"- Max selected bbox center distance from image center: `{max_center_distance:.4f}`",
        "",
        "A small center-distance is expected because the focus object is designed to remain near the image center across viewpoints.",
        "",
        "## Important remaining gap",
        "",
        "This dataset now supports target-center geometry and camera-target line-of-sight analysis.",
        "It still does not guarantee exact reprojection outlines across views because that would also need camera intrinsics / FOV and object shape or 3D size metadata.",
        "",
        "## Files",
        "",
        "- `view_geometry_table.csv`",
        "- `episode_geometry_summary.csv`",
        "- `distance_vs_bbox_area.png`",
        "- `orientation_error_hist.png`",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    data_yaml_path = Path(args.data_yaml)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    class_map = parse_class_map(data_yaml_path)
    view_df = build_geometry_rows(
        manifest_path=manifest_path,
        labels_dir=labels_dir,
        class_map=class_map,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    episode_df = build_episode_summary(view_df)

    view_df.to_csv(output_dir / "view_geometry_table.csv", index=False)
    episode_df.to_csv(output_dir / "episode_geometry_summary.csv", index=False)
    plot_distance_vs_bbox_area(view_df, output_dir / "distance_vs_bbox_area.png")
    plot_orientation_errors(view_df, output_dir / "orientation_error_hist.png")
    write_report(output_dir / "geometry_ground_truth_report.md", manifest_path, labels_dir, view_df, episode_df)

    print(f"Wrote geometry ground-truth outputs to: {output_dir}")


if __name__ == "__main__":
    main()
