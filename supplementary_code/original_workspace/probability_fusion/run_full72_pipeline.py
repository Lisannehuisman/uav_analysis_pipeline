from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "detector_family_comparison_code"))

from standardized_test_eval import (
    load_or_build_coco_gt,
    load_paths_from_official_coco_gt,
    predict_yolo_to_coco_json,
    resolve_official_coco_gt,
)
DATA_YAML = Path(r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml")
TRAIN_GT_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "ground_truth" / "M4_train_gt.json"
TRAIN_PRED_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "predictions" / "YOLOv8l_M4_train_predictions.json"
TRAIN_WEIGHTS = ROOT / "outputs" / "imported_runs" / "yolov8l_m4" / "M4_clean_yolov8l_run1" / "weights" / "best.pt"
ALIGNED_CACHE_DIR = ROOT / "probability_fusion" / "outputs" / "full72_manifest_aligned_cache"
ALIGNED_CACHE_CSV = ALIGNED_CACHE_DIR / "full72_manifest_aligned_cache.csv"
FULL72_EXPERIMENT_DIR = ROOT / "probability_fusion" / "outputs" / "full72_probability_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing M4 train predictions, rebuild the full 72-view aligned cache, "
            "and rerun the probability-fusion experiment over the full 72-view per-instance data."
        )
    )
    parser.add_argument("--python", default=str((ROOT / ".venv" / "Scripts" / "python.exe").resolve()))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coalition-sizes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-combinations-per-k", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-train-predictions", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    python_exe = Path(args.python)
    if not python_exe.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    if not TRAIN_WEIGHTS.is_file():
        raise FileNotFoundError(f"Imported M4 weights not found: {TRAIN_WEIGHTS}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"M4_fixed.yaml not found: {DATA_YAML}")

    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str((ROOT / "Ultralytics").resolve())

    if not args.skip_train_predictions and not TRAIN_PRED_JSON.is_file():
        os.environ["YOLO_CONFIG_DIR"] = env["YOLO_CONFIG_DIR"]
        official_gt = resolve_official_coco_gt(DATA_YAML, split="train")
        if official_gt is not None:
            image_paths, image_id_map = load_paths_from_official_coco_gt(official_gt, DATA_YAML, split="train")
        else:
            image_paths, image_id_map = load_or_build_coco_gt(DATA_YAML, TRAIN_GT_JSON, split="train")
        TRAIN_PRED_JSON.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Generating train predictions for {len(image_paths)} train_M4 images into {TRAIN_PRED_JSON}",
            flush=True,
        )
        predict_yolo_to_coco_json(
            weights_path=TRAIN_WEIGHTS,
            image_paths=image_paths,
            image_id_map=image_id_map,
            out_json=TRAIN_PRED_JSON,
            imgsz=args.imgsz,
            conf=args.conf,
            batch=args.batch,
            device=args.device,
        )

    run_command(
        [
            str(python_exe),
            str((ROOT / "probability_fusion" / "build_full72_manifest_aligned_cache.py").resolve()),
            "--output-dir",
            str(ALIGNED_CACHE_DIR),
            "--overwrite",
        ],
        env=env,
    )

    command = [
        str(python_exe),
        str((ROOT / "probability_fusion" / "run_probability_fusion_experiment.py").resolve()),
        "--aligned-cache-csv",
        str(ALIGNED_CACHE_CSV),
        "--evaluation-split",
        "full72",
        "--calibration",
        "none",
        "--coalition-sizes",
        *[str(value) for value in args.coalition_sizes],
        "--max-combinations-per-k",
        str(args.max_combinations_per_k),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(FULL72_EXPERIMENT_DIR),
    ]
    if args.overwrite:
        command.append("--overwrite")

    run_command(command, env=env)
    print("Full72 probability-fusion pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
