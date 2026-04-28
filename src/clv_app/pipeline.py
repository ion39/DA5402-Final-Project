from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import load_config, resolve_path
from .data import ensure_parent, load_dataset
from .features import (
    CHURN_LABEL_COLUMN,
    CHURN_PROBABILITY_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
    build_feature_frame,
    compute_baseline_stats,
    derive_churn_label,
    get_model_columns,
)
from .logging_utils import get_logger
from .modeling import CLVModelBundle, make_classifier, make_preprocessor, make_regressor


logger = get_logger(__name__)


def _model_candidates(modeling_config: dict, model_key: str, params_key: str) -> list[dict]:
    candidates = modeling_config.get(f"{model_key}_candidates")
    if candidates:
        return candidates
    return [
        {
            "name": modeling_config[model_key],
            "params": modeling_config.get(params_key, {}),
        }
    ]


def _classifier_metrics(y_true, probability, predictions) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def _regressor_metrics(y_true, predictions) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _feature_importance_frame(model_pipeline, feature_names) -> pd.DataFrame:
    estimator = model_pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = [0.0] * len(feature_names)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)


def run_training() -> dict:
    config = load_config()
    paths = config["paths"]
    random_state = config["project"]["random_state"]
    test_size = config["project"]["test_size"]

    raw_data_path = resolve_path(paths["raw_data"])
    model_path = resolve_path(paths["model_bundle"])
    metrics_path = resolve_path(paths["metrics"])
    baseline_path = resolve_path(paths["baseline_stats"])
    importance_path = resolve_path(paths["feature_importance"])
    manifest_path = resolve_path(paths["training_manifest"])
    local_temp_dir = resolve_path(".tmp")
    local_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(local_temp_dir)
    os.environ["TEMP"] = str(local_temp_dir)
    tempfile.tempdir = str(local_temp_dir)

    dataset = load_dataset(raw_data_path)
    dataset[CHURN_LABEL_COLUMN] = derive_churn_label(dataset)
    feature_df = build_feature_frame(dataset)
    baseline_stats = compute_baseline_stats(feature_df)

    classifier_features = get_model_columns(
        feature_df,
        excluded=[TARGET_COLUMN, CHURN_LABEL_COLUMN],
    )
    classifier_input = feature_df[classifier_features]
    classifier_target = feature_df[CHURN_LABEL_COLUMN]

    (
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
        train_index,
        test_index,
    ) = train_test_split(
        classifier_input,
        classifier_target,
        feature_df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=classifier_target,
    )

    classifier_candidates = []
    for candidate in _model_candidates(
        config["modeling"],
        "classifier",
        "classifier_params",
    ):
        logger.info("Training classifier candidate: %s", candidate["name"])
        classifier_preprocessor, _, _ = make_preprocessor(X_train_cls)
        candidate_model = make_classifier(
            classifier_preprocessor,
            candidate["name"],
            candidate.get("params", {}),
            random_state,
        )
        candidate_model.fit(X_train_cls, y_train_cls)
        candidate_probability = candidate_model.predict_proba(X_test_cls)[:, 1]
        candidate_predictions = candidate_model.predict(X_test_cls)
        candidate_metrics = _classifier_metrics(
            y_test_cls,
            candidate_probability,
            candidate_predictions,
        )
        classifier_candidates.append(
            {
                "name": candidate["name"],
                "params": candidate.get("params", {}),
                "model": candidate_model,
                "metrics": candidate_metrics,
            }
        )

    best_classifier = max(
        classifier_candidates,
        key=lambda candidate: (
            candidate["metrics"]["roc_auc"],
            candidate["metrics"]["recall"],
            candidate["metrics"]["accuracy"],
        ),
    )
    classifier = best_classifier["model"]

    train_churn_probability = classifier.predict_proba(X_train_cls)[:, 1]
    test_churn_probability = classifier.predict_proba(X_test_cls)[:, 1]
    churn_test_predictions = classifier.predict(X_test_cls)

    regression_feature_df = feature_df.copy()
    regression_feature_df[CHURN_PROBABILITY_COLUMN] = 0.0
    regression_feature_df.loc[train_index, CHURN_PROBABILITY_COLUMN] = train_churn_probability
    regression_feature_df.loc[test_index, CHURN_PROBABILITY_COLUMN] = test_churn_probability

    regression_features = get_model_columns(
        regression_feature_df,
        excluded=[TARGET_COLUMN, CHURN_LABEL_COLUMN],
    )

    X_train_reg = regression_feature_df.loc[train_index, regression_features]
    X_test_reg = regression_feature_df.loc[test_index, regression_features]
    y_train_reg = regression_feature_df.loc[train_index, TARGET_COLUMN]
    y_test_reg = regression_feature_df.loc[test_index, TARGET_COLUMN]

    regressor_candidates = []
    for candidate in _model_candidates(
        config["modeling"],
        "regressor",
        "regressor_params",
    ):
        logger.info("Training regressor candidate: %s", candidate["name"])
        (
            regressor_preprocessor,
            categorical_columns,
            numeric_columns,
        ) = make_preprocessor(X_train_reg)
        candidate_model = make_regressor(
            regressor_preprocessor,
            candidate["name"],
            candidate.get("params", {}),
            random_state,
        )
        candidate_model.fit(X_train_reg, y_train_reg)
        candidate_predictions = candidate_model.predict(X_test_reg)
        candidate_metrics = _regressor_metrics(y_test_reg, candidate_predictions)
        regressor_candidates.append(
            {
                "name": candidate["name"],
                "params": candidate.get("params", {}),
                "model": candidate_model,
                "metrics": candidate_metrics,
                "categorical_columns": categorical_columns,
                "numeric_columns": numeric_columns,
            }
        )

    best_regressor = max(
        regressor_candidates,
        key=lambda candidate: (
            candidate["metrics"]["r2"],
            -candidate["metrics"]["rmse"],
            -candidate["metrics"]["mae"],
        ),
    )
    regressor = best_regressor["model"]
    selected_categorical_columns = best_regressor["categorical_columns"]
    selected_numeric_columns = best_regressor["numeric_columns"]
    reg_predictions = regressor.predict(X_test_reg)

    metrics = {
        "classifier": _classifier_metrics(
            y_test_cls,
            test_churn_probability,
            churn_test_predictions,
        ),
        "regressor": _regressor_metrics(y_test_reg, reg_predictions),
        "data": {
            "row_count": int(len(dataset)),
            "feature_count_classifier": int(len(classifier_features)),
            "feature_count_regressor": int(len(regression_features)),
            "categorical_feature_count": int(len(selected_categorical_columns)),
            "numeric_feature_count": int(len(selected_numeric_columns)),
            "churn_rate": float(dataset[CHURN_LABEL_COLUMN].mean()),
        },
        "selection": {
            "classifier": {
                "selected_model": best_classifier["name"],
                "candidates": {
                    candidate["name"]: candidate["metrics"]
                    for candidate in classifier_candidates
                },
            },
            "regressor": {
                "selected_model": best_regressor["name"],
                "candidates": {
                    candidate["name"]: candidate["metrics"]
                    for candidate in regressor_candidates
                },
            },
        },
    }

    mlflow.set_tracking_uri("file:" + str(resolve_path("mlruns")))
    mlflow.set_experiment("clv-mlops-app")

    with mlflow.start_run(run_name="train-clv-model") as run:
        mlflow.log_params(
            {
                "classifier": best_classifier["name"],
                "regressor": best_regressor["name"],
                "test_size": test_size,
                "random_state": random_state,
            }
        )
        mlflow.log_metrics(
            {
                "classifier_roc_auc": metrics["classifier"]["roc_auc"],
                "classifier_accuracy": metrics["classifier"]["accuracy"],
                "classifier_precision": metrics["classifier"]["precision"],
                "classifier_recall": metrics["classifier"]["recall"],
                "regressor_rmse": metrics["regressor"]["rmse"],
                "regressor_mae": metrics["regressor"]["mae"],
                "regressor_r2": metrics["regressor"]["r2"],
            }
        )

        bundle = CLVModelBundle(
            classifier=classifier,
            regressor=regressor,
            feature_columns=classifier_features,
            baseline_stats=baseline_stats,
            metrics=metrics,
        )
        ensure_parent(model_path)
        joblib.dump(bundle, model_path)

        transformed_feature_names = regressor.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        feature_importance = _feature_importance_frame(
            regressor,
            transformed_feature_names,
        )

        ensure_parent(metrics_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        baseline_path.write_text(json.dumps(baseline_stats, indent=2), encoding="utf-8")
        feature_importance.to_csv(importance_path, index=False)

        manifest = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run.info.run_id,
            "artifacts": {
                "model_bundle": str(model_path),
                "metrics": str(metrics_path),
                "baseline_stats": str(baseline_path),
                "feature_importance": str(importance_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(baseline_path))
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(model_path))
        try:
            mlflow.sklearn.log_model(regressor, artifact_path="regressor_model")
            mlflow.sklearn.log_model(classifier, artifact_path="classifier_model")
        except PermissionError as exc:
            logger.warning("Skipping MLflow model artifact logging due to local permission issue: %s", exc)

        metrics["mlflow_run_id"] = run.info.run_id

    logger.info("Training complete with R2 %.4f", metrics["regressor"]["r2"])
    return metrics
