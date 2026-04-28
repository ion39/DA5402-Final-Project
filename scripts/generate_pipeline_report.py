from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clv_app.config import load_config, resolve_path
from src.clv_app.data import load_dataset
from src.clv_app.logging_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    paths = config["paths"]
    raw_data_path = resolve_path(paths["raw_data"])
    metrics_path = resolve_path(paths["metrics"])
    baseline_path = resolve_path(paths["baseline_stats"])
    report_path = resolve_path(paths["pipeline_report"])

    dataset = load_dataset(raw_data_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stages": [
            {
                "name": "prepare_data",
                "status": "completed",
                "records": int(len(dataset)),
                "description": "Raw financial services customer data copied to versionable raw storage.",
            },
            {
                "name": "train_model",
                "status": "completed",
                "records": int(len(dataset)),
                "description": "Churn classifier and CLV regressor trained with MLflow tracking.",
            },
            {
                "name": "build_report",
                "status": "completed",
                "records": int(len(dataset)),
                "description": "Operational summary assembled for UI and demo visibility.",
            },
        ],
        "throughput": {
            "records_processed": int(len(dataset)),
            "features_baselined": int(len(baseline)),
        },
        "quality": metrics,
        "sample_columns": dataset.columns.tolist(),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Pipeline report written to %s", report_path)


if __name__ == "__main__":
    main()
