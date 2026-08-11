"""Paths used by the local thesis analysis scripts."""

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "paths.example.yaml"


def _load_paths():
    config_file = LOCAL_CONFIG if LOCAL_CONFIG.exists() else EXAMPLE_CONFIG

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PATHS = _load_paths()


def data_path(name):
    """Return one of the configured data directories."""
    if name not in PATHS:
        raise KeyError(f"No path called '{name}' is defined in {LOCAL_CONFIG.name}")

    path = Path(PATHS[name])

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()