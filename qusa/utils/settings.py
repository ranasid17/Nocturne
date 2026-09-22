"""Portable local settings with explicit environment overrides."""

import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"


class SettingsError(ValueError):
    """Raised when local application settings are invalid."""


def _resolve_path(value, root):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path)


def load_settings(config_path=None, environ=None, project_root=None):
    """Load YAML settings without changing the caller's working directory."""

    environment = environ if environ is not None else os.environ
    root = Path(project_root or PROJECT_ROOT)
    load_dotenv(root / ".env", override=False)
    selected_path = Path(
        config_path or environment.get("QUSA_CONFIG_PATH", root / "qusa" / "utils" / "config.yaml")
    ).expanduser()
    if not selected_path.exists():
        raise SettingsError(f"Configuration file does not exist: {selected_path}")

    with selected_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise SettingsError("Configuration root must be an object.")

    config = copy.deepcopy(config)
    data_root = environment.get("QUSA_DATA_ROOT")
    if data_root:
        paths = config.setdefault("data", {}).setdefault("paths", {})
        root = Path(data_root).expanduser()
        paths.update(
            {
                "raw_data_dir": str(root / "raw"),
                "processed_data_dir": str(root / "processed"),
                "figures_dir": str(root / "figures"),
                "predictions_dir": str(root / "predictions"),
                "reports_dir": str(root / "reports"),
            }
        )

    path_sections = [
        config.get("data", {}).get("paths", {}),
        config.get("model", {}).get("output", {}),
        config.get("prediction", {}),
        config.get("storage", {}),
    ]
    for section in path_sections:
        for key, value in list(section.items()):
            if isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")):
                section[key] = _resolve_path(value, selected_path.parent)
    if environment.get("QUSA_DATABASE_PATH"):
        config.setdefault("storage", {})["database_path"] = _resolve_path(
            environment["QUSA_DATABASE_PATH"], selected_path.parent
        )
    return config
