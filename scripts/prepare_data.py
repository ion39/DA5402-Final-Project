from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clv_app.config import load_config, resolve_path
from src.clv_app.data import ensure_parent, load_dataset
from src.clv_app.logging_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    paths = config["paths"]

    source_path = resolve_path(paths["raw_source"])
    raw_data_path = resolve_path(paths["raw_data"])
    sample_payload_path = resolve_path(paths["sample_payload"])

    dataset = load_dataset(source_path)

    ensure_parent(raw_data_path)
    dataset.to_csv(raw_data_path, index=False)

    sample_payload = dataset.iloc[0].drop(labels=["LTV"]).to_dict()
    ensure_parent(sample_payload_path)
    sample_payload_path.write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")

    logger.info("Prepared %s rows into %s", len(dataset), raw_data_path)


if __name__ == "__main__":
    main()
