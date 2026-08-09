import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
DATASET_ROOT = PROJECT_ROOT / "data_collection" / "raw_data" / "synthetic_subset"

REGIME_ORDER = ["M1", "M2a", "M2b", "M3", "M4"]
DETECTOR_ORDER = ["YOLOv8n", "YOLOv8l", "Faster R-CNN"]


MODEL_RUNS = {
    "YOLOv8l": {
        "M1": str(MODEL_ROOT / "yolov8l" / "S0_M1_yolov8l"),
        "M2a": str(MODEL_ROOT / "yolov8l" / "S0_M2a_yolov8l"),
        "M2b": str(MODEL_ROOT / "yolov8l" / "S0_M2b_yolov8l"),
        "M3": str(MODEL_ROOT / "yolov8l" / "S0_M3_yolov8l"),
        "M4": str(MODEL_ROOT / "yolov8l" / "S0_M4_yolov8l"),
    },
    "YOLOv8n": {
        "M1": str(MODEL_ROOT / "yolov8n" / "S0_M1_yolov8n"),
        "M2a": str(MODEL_ROOT / "yolov8n" / "S0_M2a_yolov8n"),
        "M2b": str(MODEL_ROOT / "yolov8n" / "S0_M2b_yolov8n"),
        "M3": str(MODEL_ROOT / "yolov8n" / "S0_M3_yolov8n"),
        "M4": str(MODEL_ROOT / "yolov8n" / "S0_M4_yolov8n"),
    },
    "Faster R-CNN": {
        "M1": str(MODEL_ROOT / "faster_rcnn" / "S0_M1_run1"),
        "M2a": str(MODEL_ROOT / "faster_rcnn" / "S0_M2a_run1"),
        "M2b": str(MODEL_ROOT / "faster_rcnn" / "S0_M2b_run1"),
        "M3": str(MODEL_ROOT / "faster_rcnn" / "S0_M3_run1"),
        "M4": str(MODEL_ROOT / "faster_rcnn" / "S0_M4_run1"),
    },
}


# The compact thesis archive includes a representative synthetic subset. Full
# M1-M4 reruns can be configured with DETECTOR_COMPARISON_CONFIG_JSON.
REGIME_DATA_YAMLS = {
    regime: str(DATASET_ROOT / "data.yaml") for regime in REGIME_ORDER
}


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "recomputed" / "detector_family_comparison"


def _load_override_config() -> None:
    override_path = os.environ.get("DETECTOR_COMPARISON_CONFIG_JSON")
    if not override_path:
        return

    config_path = Path(override_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    global REGIME_ORDER, DETECTOR_ORDER, MODEL_RUNS, REGIME_DATA_YAMLS, DEFAULT_OUTPUT_DIR

    if "regime_order" in config:
        REGIME_ORDER = list(config["regime_order"])
    if "detector_order" in config:
        DETECTOR_ORDER = list(config["detector_order"])
    if "model_runs" in config:
        MODEL_RUNS = {
            str(detector): {str(regime): str(path) for regime, path in runs.items()}
            for detector, runs in config["model_runs"].items()
        }
    if "regime_data_yamls" in config:
        REGIME_DATA_YAMLS = {str(regime): str(path) for regime, path in config["regime_data_yamls"].items()}
    if "default_output_dir" in config:
        DEFAULT_OUTPUT_DIR = Path(config["default_output_dir"])


_load_override_config()
