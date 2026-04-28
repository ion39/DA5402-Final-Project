from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clv_app.logging_utils import get_logger
from src.clv_app.pipeline import run_training


logger = get_logger(__name__)


def main() -> None:
    metrics = run_training()
    logger.info("Classifier ROC-AUC: %.4f", metrics["classifier"]["roc_auc"])
    logger.info("Regressor R2: %.4f", metrics["regressor"]["r2"])


if __name__ == "__main__":
    main()
