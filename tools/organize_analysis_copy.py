"""
Copy the remaining thesis analyses into the cleaned analysis project.

This script is intentionally conservative: it never moves or deletes anything from
the original "New project" folder. I still want to keep that folder as the untouched
working archive while the GitHub version is cleaned up.

The script copies code and generated outputs into a more logical structure. Existing
files in the new project are not deleted. If a file already exists at the destination,
the source copy replaces that destination file only.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ORIGINAL = Path(r"C:\Users\lisan\OneDrive\Documents\New project")
CLEAN = Path(r"C:\Users\lisan\thesis_pipeline_analysis\uav_analysis_pipeline")

REPORT_DIR = CLEAN / "migration_audit"

CODE_EXTENSIONS = {
    ".py", ".ps1", ".sh", ".md", ".json", ".yaml", ".yml", ".txt"
}
RESULT_EXTENSIONS = {
    ".csv", ".png", ".jpg", ".jpeg", ".md", ".json", ".yaml", ".yml", ".txt"
}

SKIP_CODE_PARTS = {
    "__pycache__", ".git", ".venv", "venv", ".vs", ".vscode",
    "outputs", "output", "results", "data", "images", "labels", "weights",
    "runs", "cache", "pred_cache",
}

SKIP_RESULT_PARTS = {
    "__pycache__", ".git", ".venv", "venv", ".vs", ".vscode",
    "weights", "runs",
}


# Main analysis folders that were still outside lisannesmasterthesis.
# I keep sub-analyses separate because it is easier to trace them back to the thesis.
CODE_MAPPINGS = {
    "clutter_grouping_analysis": "src/viewpoint_analysis/clutter_analysis",
    "detector_family_comparison_code": "src/detector_analysis/detector_family_comparison",
    "full_m4_vs_single_pair_operational_analysis": "src/multiview_analysis/full_m4_vs_single_pair",
    "geometry_aware_fusion_analysis": "src/geometry_analysis/geometry_aware_fusion",
    "geometry_ground_truth_analysis": "src/geometry_analysis/ground_truth",
    "m4_cross_view_box_fusion_analysis": "src/multiview_analysis/box_fusion",
    "m4_marginal_viewpoint_value_analysis": "src/multiview_analysis/marginal_value",
    "m4_pair_results": "src/viewpoint_training/pair_result_analysis",
    "m4_two_drone_operational_analysis": "src/multiview_analysis/two_drone_operational",
    "m4_viewpoint_selection_analysis": "src/multiview_analysis/viewpoint_selection",
    "multiview_method_comparison_analysis": "src/multiview_analysis/method_comparison",
    "multiview_transformer": "src/experimental/multiview_transformer",
    "probability_fusion": "src/multiview_analysis/probability_fusion",
    "two_drone_vs_single_view_analysis": "src/multiview_analysis/two_drone_vs_single_view",
    "thesis_tools": "src/figures/thesis_tools",
}

RESULT_MAPPINGS = {
    "clutter_grouping_analysis/outputs": "results/viewpoint_analysis/clutter_analysis",
    "full_m4_vs_single_pair_operational_analysis/outputs": "results/multiview_analysis/full_m4_vs_single_pair",
    "geometry_aware_fusion_analysis/outputs": "results/geometry_analysis/geometry_aware_fusion",
    "geometry_ground_truth_analysis/outputs": "results/geometry_analysis/ground_truth",
    "m4_cross_view_box_fusion_analysis/outputs": "results/multiview_analysis/box_fusion",
    "m4_marginal_viewpoint_value_analysis/outputs": "results/multiview_analysis/marginal_value",
    "m4_pair_results/outputs": "results/viewpoint_training/pair_results",
    "m4_two_drone_operational_analysis/outputs": "results/multiview_analysis/two_drone_operational",
    "m4_two_drone_operational_analysis/outputs_test": "results/multiview_analysis/two_drone_operational_test",
    "m4_viewpoint_selection_analysis/outputs": "results/multiview_analysis/viewpoint_selection",
    "multiview_method_comparison_analysis/outputs": "results/multiview_analysis/method_comparison",
    "probability_fusion/outputs": "results/multiview_analysis/probability_fusion",
    "two_drone_vs_single_view_analysis/outputs": "results/multiview_analysis/two_drone_vs_single_view",
    "outputs/detector_family_comparison": "results/detector_analysis/detector_family_comparison",
    "outputs/thesis_tools": "results/thesis_figures/thesis_tools",
    "comparison_output_s0_m4": "results/viewpoint_analysis/s0_m4_viewpoint_outputs",
    "thesis_fix_pack/figures": "results/thesis_figures/thesis_fix_pack",
}

# Some useful report files sit under data/current_snapshot rather than outputs.
EXTRA_RESULT_MAPPINGS = {
    "m4_pair_results/data/current_snapshot/reports":
        "results/viewpoint_training/pair_results/current_snapshot_reports",
}

VIEWPOINT_ROOT_SCRIPTS = [
    "analyze_best_viewpoints.py",
    "create_best_viewpoint_report.py",
    "plot_best_viewpoint_3d.py",
    "plot_object_viewpoint_boxplots.py",
    "plot_object_viewpoint_heatmaps.py",
    "plot_object_viewpoint_metric_grid.py",
    "plot_per_object_boxplots.py",
    "rank_object_viewpoints.py",
    "thesis_viewpoint_analysis.py",
    "visualize_best_viewpoints.py",
]

DETECTOR_ROOT_SCRIPTS = [
    "compare_yolo_models.py",
]


def ensure_safe_paths() -> None:
    if not ORIGINAL.exists():
        raise SystemExit(f"Original project not found: {ORIGINAL}")
    if not CLEAN.exists():
        raise SystemExit(f"Clean project not found: {CLEAN}")
    if ORIGINAL.resolve() == CLEAN.resolve():
        raise SystemExit("Original and clean project point to the same folder. Stopping.")


def should_skip(relative: Path, skip_parts: set[str]) -> bool:
    return any(part.lower() in skip_parts for part in relative.parts)


def copy_selected_tree(
    source: Path,
    destination: Path,
    extensions: set[str],
    skip_parts: set[str],
) -> tuple[int, int]:
    """Copy selected file types while preserving the folder layout."""
    copied = 0
    skipped = 0

    if not source.exists():
        return copied, skipped

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(source)

        if should_skip(rel, skip_parts):
            skipped += 1
            continue

        if path.suffix.lower() not in extensions:
            skipped += 1
            continue

        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1

    return copied, skipped


def copy_result_tree(source: Path, destination: Path) -> tuple[int, int]:
    return copy_selected_tree(
        source,
        destination,
        RESULT_EXTENSIONS,
        SKIP_RESULT_PARTS,
    )


def copy_root_scripts(names: list[str], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in names:
        source = ORIGINAL / name
        if source.exists():
            shutil.copy2(source, destination / source.name)
            count += 1
    return count


def copy_notes() -> int:
    notes_dir = CLEAN / "docs" / "analysis_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for name in [
        "analyse_twee_drones_vs_een_viewpoint.md",
    ]:
        source = ORIGINAL / name
        if source.exists():
            shutil.copy2(source, notes_dir / source.name)
            count += 1
    return count


def copy_viewpoint_data_separated() -> tuple[int, int]:
    """
    This folder contains a mixture of old analyses and generated plots. I do not copy
    model folders or arbitrary data here; only code and normal result files.
    """
    source = ORIGINAL / "viewpoint_data_separated"
    code_dest = CLEAN / "src" / "viewpoint_training" / "additional_analyses"
    result_dest = CLEAN / "results" / "viewpoint_training" / "additional_analyses"

    code_count, _ = copy_selected_tree(
        source, code_dest, CODE_EXTENSIONS, SKIP_CODE_PARTS
    )

    result_count = 0
    if source.exists():
        for path in source.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in RESULT_EXTENSIONS:
                continue

            rel = path.relative_to(source)
            lower_parts = {p.lower() for p in rel.parts}

            # Do not copy obvious training/model/data artefacts into Git results.
            if lower_parts & {
                "weights", "runs", "images", "labels", "data",
                "__pycache__", ".git", ".venv", "venv"
            }:
                continue

            # Python files belong in src, not results.
            if path.suffix.lower() == ".py":
                continue

            dest = result_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            result_count += 1

    return code_count, result_count


def copy_real_imagery_analysis() -> tuple[int, int]:
    """
    Keep real imagery itself external, but preserve scripts, manifests and analysis outputs.
    """
    source = ORIGINAL / "real_imagery"
    code_dest = CLEAN / "src" / "real_world_transfer" / "real_imagery_analysis"
    result_dest = CLEAN / "results" / "real_world_transfer" / "real_imagery_analysis"

    code_count, _ = copy_selected_tree(
        source, code_dest, CODE_EXTENSIONS,
        SKIP_CODE_PARTS | {"train", "valid", "test", "complete_set"}
    )

    result_count = 0
    if source.exists():
        for path in source.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in RESULT_EXTENSIONS:
                continue

            rel = path.relative_to(source)
            lower_parts = {p.lower() for p in rel.parts}

            # Data images/labels are handled outside Git.
            if lower_parts & {
                "images", "labels", "train", "valid", "test", "complete_set",
                "__pycache__", ".git", ".venv", "venv"
            }:
                continue

            # For this folder only keep generated outputs, manifests and small metadata.
            useful = (
                "outputs" in lower_parts
                or "manifest" in path.name.lower()
                or path.name.lower().endswith(".yaml")
                or path.name.lower().endswith(".yml")
                or "summary" in path.name.lower()
                or "log" in path.name.lower()
            )
            if not useful or path.suffix.lower() == ".py":
                continue

            dest = result_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            result_count += 1

    return code_count, result_count


def top_level_inventory(mapped_names: set[str]) -> list[dict[str, str]]:
    """
    Find top-level folders that still look like analysis folders but were not in the
    explicit mapping above. This gives one last check before I call the migration complete.
    """
    rows = []

    ignore = {
        ".git", ".vs", ".vscode", "lisannesmasterthesis", "data_collection",
        "thesis", "thesis_pipeline_analysis",
    }

    for folder in sorted(p for p in ORIGINAL.iterdir() if p.is_dir()):
        if folder.name.lower() in {x.lower() for x in ignore}:
            continue
        if folder.name in mapped_names:
            continue

        py_files = [
            p for p in folder.rglob("*.py")
            if "__pycache__" not in {part.lower() for part in p.parts}
        ]
        output_dirs = [
            p for p in folder.rglob("*")
            if p.is_dir() and p.name.lower() in {"output", "outputs", "results"}
        ]

        if not py_files and not output_dirs:
            continue

        rows.append({
            "folder": folder.name,
            "python_files": str(len(py_files)),
            "output_or_result_dirs": str(len(output_dirs)),
            "example_python": (
                str(py_files[0].relative_to(ORIGINAL)) if py_files else ""
            ),
        })

    return rows


def write_inventory(rows: list[dict[str, str]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "unmapped_top_level.csv"

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder", "python_files", "output_or_result_dirs", "example_python"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_safe_paths()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for source_rel, dest_rel in CODE_MAPPINGS.items():
        copied, skipped = copy_selected_tree(
            ORIGINAL / source_rel,
            CLEAN / dest_rel,
            CODE_EXTENSIONS,
            SKIP_CODE_PARTS,
        )
        summary.append(("code", source_rel, dest_rel, copied, skipped))

    for source_rel, dest_rel in {**RESULT_MAPPINGS, **EXTRA_RESULT_MAPPINGS}.items():
        copied, skipped = copy_result_tree(
            ORIGINAL / source_rel,
            CLEAN / dest_rel,
        )
        summary.append(("results", source_rel, dest_rel, copied, skipped))

    viewpoint_root = copy_root_scripts(
        VIEWPOINT_ROOT_SCRIPTS,
        CLEAN / "src" / "viewpoint_analysis" / "exploratory_plots",
    )
    detector_root = copy_root_scripts(
        DETECTOR_ROOT_SCRIPTS,
        CLEAN / "src" / "detector_analysis" / "exploratory",
    )
    note_count = copy_notes()

    vp_code, vp_results = copy_viewpoint_data_separated()
    real_code, real_results = copy_real_imagery_analysis()

    mapped_top = set(CODE_MAPPINGS)
    mapped_top.update(x.split("/")[0] for x in RESULT_MAPPINGS)
    mapped_top.update(x.split("/")[0] for x in EXTRA_RESULT_MAPPINGS)
    mapped_top.update({
        "comparison_output_s0_m4",
        "viewpoint_data_separated",
        "real_imagery",
        "thesis_fix_pack",
        "outputs",
    })

    unmapped = top_level_inventory(mapped_top)
    write_inventory(unmapped)

    print("Analysis copy/restructure finished.")
    print()
    print(f"Original (read only): {ORIGINAL}")
    print(f"Clean project:        {CLEAN}")
    print()
    print("Main copy summary:")
    for kind, source_rel, dest_rel, copied, skipped in summary:
        print(f"  {kind:7} {source_rel} -> {dest_rel}: {copied} copied")
    print()
    print(f"Root viewpoint scripts copied: {viewpoint_root}")
    print(f"Root detector scripts copied:  {detector_root}")
    print(f"Analysis notes copied:          {note_count}")
    print(f"viewpoint_data_separated:       {vp_code} code, {vp_results} result files")
    print(f"real_imagery:                   {real_code} code, {real_results} result files")
    print()
    print(f"Unmapped analysis-like top-level folders: {len(unmapped)}")
    print(f"See: {REPORT_DIR / 'unmapped_top_level.csv'}")
    print()
    print("Nothing was moved or deleted from New project.")


if __name__ == "__main__":
    main()
