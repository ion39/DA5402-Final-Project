from __future__ import annotations

from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "base.yaml"


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(relative_path: str) -> Path:
    return ROOT_DIR / relative_path

