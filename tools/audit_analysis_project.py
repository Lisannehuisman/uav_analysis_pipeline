"""
Small audit script for reorganising the thesis analysis project.

It only reads from the old "New project" folder. The only files it writes are
CSV/text reports inside the new project. I made this before moving more scripts
around because there are quite a few copies of the same analysis by now.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ORIGINAL = Path(r"C:\Users\lisan\OneDrive\Documents\New project")
CLEAN = Path(r"C:\Users\lisan\thesis_pipeline_analysis\uav_analysis_pipeline")
REPORT_DIR = CLEAN / "migration_audit"

ANALYSIS_FOLDERS = [
    "clutter_grouping_analysis",
    "detector_family_comparison_code",
    "full_m4_vs_single_pair_operational_analysis",
    "geometry_aware_fusion_analysis",
    "geometry_ground_truth_analysis",
    "m4_cross_view_box_fusion_analysis",
    "m4_marginal_viewpoint_value_analysis",
    "multiview_method_comparison_analysis",
    "multiview_transformer",
    "probability_fusion",
]

ROOT_PYTHON_FILES = [
    "analyze_best_viewpoints.py",
    "compare_yolo_models.py",
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

OUTPUT_EXTENSIONS = {".csv", ".md", ".png", ".json", ".yaml", ".yml"}
IGNORE_PARTS = {"__pycache__", ".git", ".venv", "venv", ".vs", ".vscode"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)


def clean_index(root: Path, extensions: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and not ignored(path) and path.suffix.lower() in extensions:
            index[path.name.lower()].append(path)
    return index


def compare_file(source: Path, candidates: list[Path]) -> tuple[str, str, str]:
    if not candidates:
        return "MISSING", "", ""
    source_hash = sha256(source)
    exact = [p for p in candidates if sha256(p) == source_hash]
    if len(exact) == 1:
        return "SAME", str(exact[0].relative_to(CLEAN)), source_hash
    if len(exact) > 1:
        joined = " | ".join(str(p.relative_to(CLEAN)) for p in exact)
        return "SAME_MULTIPLE", joined, source_hash
    joined = " | ".join(str(p.relative_to(CLEAN)) for p in candidates)
    return "DIFFERENT", joined, source_hash


def collect_old_python() -> list[Path]:
    files = []
    for name in ROOT_PYTHON_FILES:
        path = ORIGINAL / name
        if path.exists():
            files.append(path)
    for folder_name in ANALYSIS_FOLDERS:
        folder = ORIGINAL / folder_name
        if folder.exists():
            files.extend(
                p for p in folder.rglob("*.py")
                if p.is_file() and not ignored(p)
            )
    return sorted(set(files))


def collect_old_outputs() -> list[Path]:
    files = []
    for folder_name in ANALYSIS_FOLDERS:
        folder = ORIGINAL / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if (
                path.is_file()
                and not ignored(path)
                and path.suffix.lower() in OUTPUT_EXTENSIONS
                and "outputs" in {part.lower() for part in path.parts}
            ):
                files.append(path)
    return sorted(set(files))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ORIGINAL.exists():
        raise SystemExit(f"Original project was not found: {ORIGINAL}")
    if not CLEAN.exists():
        raise SystemExit(f"New project was not found: {CLEAN}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    new_python = clean_index(CLEAN / "src", {".py"})
    source_rows = []
    for source in collect_old_python():
        status, match, digest = compare_file(source, new_python.get(source.name.lower(), []))
        source_rows.append({
            "status": status,
            "old_file": str(source.relative_to(ORIGINAL)),
            "same_name_in_new_project": match,
            "sha256_old": digest,
        })
    write_csv(
        REPORT_DIR / "source_audit.csv",
        ["status", "old_file", "same_name_in_new_project", "sha256_old"],
        source_rows,
    )

    new_outputs = clean_index(CLEAN / "results", OUTPUT_EXTENSIONS)
    output_rows = []
    for source in collect_old_outputs():
        status, match, digest = compare_file(source, new_outputs.get(source.name.lower(), []))
        output_rows.append({
            "status": status,
            "old_file": str(source.relative_to(ORIGINAL)),
            "same_name_in_new_results": match,
            "sha256_old": digest,
        })
    write_csv(
        REPORT_DIR / "output_audit.csv",
        ["status", "old_file", "same_name_in_new_results", "sha256_old"],
        output_rows,
    )

    dependency_rows = []
    search_terms = [
        r"C:\Users\lisan",
        "OneDrive",
        "New project",
        "lisannesmasterthesis",
        "data_collection",
    ]
    for path in sorted((CLEAN / "src").rglob("*.py")):
        if ignored(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="latin-1").splitlines()
        for line_number, line in enumerate(lines, start=1):
            hits = [term for term in search_terms if term.lower() in line.lower()]
            if hits:
                dependency_rows.append({
                    "file": str(path.relative_to(CLEAN)),
                    "line": str(line_number),
                    "matched": " | ".join(hits),
                    "text": line.strip(),
                })
    write_csv(
        REPORT_DIR / "path_dependencies.csv",
        ["file", "line", "matched", "text"],
        dependency_rows,
    )

    def counts(rows):
        out = defaultdict(int)
        for row in rows:
            out[row["status"]] += 1
        return dict(sorted(out.items()))

    summary = [
        "# Migration audit",
        "",
        "This is only an audit. Nothing in the original project was changed.",
        "",
        "## Old analysis code compared with the new src folder",
        "",
    ]
    for key, value in counts(source_rows).items():
        summary.append(f"- {key}: {value}")
    summary += ["", "## Old analysis outputs compared with the new results folder", ""]
    for key, value in counts(output_rows).items():
        summary.append(f"- {key}: {value}")
    summary += [
        "",
        "## Path dependencies",
        "",
        f"- Hard-coded/path-related lines found: {len(dependency_rows)}",
        "",
        "The detailed reports are source_audit.csv, output_audit.csv and path_dependencies.csv.",
        "I would first look at MISSING and DIFFERENT before copying or moving anything else.",
    ]
    (REPORT_DIR / "audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Audit finished.")
    print(f"Original project: {ORIGINAL}")
    print(f"New project:      {CLEAN}")
    print(f"Reports:          {REPORT_DIR}")


if __name__ == "__main__":
    main()
