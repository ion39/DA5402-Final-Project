from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CHURN_PROBABILITY_COLUMN, build_feature_frame


def make_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )
    return preprocessor, categorical_columns, numeric_columns


CLASSIFIER_FACTORIES = {
    "random_forest": RandomForestClassifier,
}

REGRESSOR_FACTORIES = {
    "random_forest": RandomForestRegressor,
}


def _build_estimator(model_name: str, factories: dict, params: dict, random_state: int):
    if model_name not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    estimator_class = factories[model_name]
    estimator_params = dict(params or {})
    if "random_state" in estimator_class().get_params():
        estimator_params.setdefault("random_state", random_state)
    return estimator_class(**estimator_params)


def make_classifier(
    preprocessor: ColumnTransformer,
    model_name: str,
    params: dict,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                _build_estimator(
                    model_name,
                    CLASSIFIER_FACTORIES,
                    params,
                    random_state,
                ),
            ),
        ]
    )


def make_regressor(
    preprocessor: ColumnTransformer,
    model_name: str,
    params: dict,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                _build_estimator(
                    model_name,
                    REGRESSOR_FACTORIES,
                    params,
                    random_state,
                ),
            ),
        ]
    )


@dataclass
class CLVModelBundle:
    classifier: Pipeline
    regressor: Pipeline
    feature_columns: list[str]
    baseline_stats: dict
    metrics: dict

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        feature_df = build_feature_frame(records.copy())
        model_input = feature_df[self.feature_columns].copy()
        churn_probability = self.classifier.predict_proba(model_input)[:, 1]
        regression_input = model_input.copy()
        regression_input[CHURN_PROBABILITY_COLUMN] = churn_probability
        clv_prediction = self.regressor.predict(regression_input)
        prediction_frame = records.copy()
        prediction_frame[CHURN_PROBABILITY_COLUMN] = churn_probability
        prediction_frame["Predicted_CLV"] = clv_prediction
        prediction_frame["Predicted_Churn_Label"] = (churn_probability >= 0.5).astype(int)
        return prediction_frame
