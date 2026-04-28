from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

