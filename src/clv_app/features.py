from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "LTV"
ID_COLUMN = "Customer_ID"
CHURN_LABEL_COLUMN = "Churn_Label"
CHURN_PROBABILITY_COLUMN = "Churn_Probability"


def derive_churn_label(df: pd.DataFrame) -> pd.Series:
    inactivity_flag = df["Last_Transaction_Days_Ago"] >= 220
    satisfaction_flag = df["Customer_Satisfaction_Score"] <= 3
    support_flag = (df["App_Usage_Frequency"] == "Monthly") & (
        df["Support_Tickets_Raised"] >= 12
    )
    return (inactivity_flag | satisfaction_flag | support_flag).astype(int)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()

    active_days = feature_df["Active_Days"].clip(lower=1)
    transactions = feature_df["Total_Transactions"].clip(lower=1)
    total_spent = feature_df["Total_Spent"].clip(lower=1)

    feature_df["Spend_Per_Active_Day"] = feature_df["Total_Spent"] / active_days
    feature_df["Transactions_Per_Active_Day"] = (
        feature_df["Total_Transactions"] / active_days
    )
    feature_df["Recency_Activity_Ratio"] = (
        feature_df["Last_Transaction_Days_Ago"] / active_days
    )
    feature_df["Cashback_Ratio"] = feature_df["Cashback_Received"] / total_spent
    feature_df["Loyalty_Points_Per_Transaction"] = (
        feature_df["Loyalty_Points_Earned"] / transactions
    )
    feature_df["Support_Intensity"] = feature_df["Support_Tickets_Raised"] / active_days
    feature_df["Referral_Rate"] = feature_df["Referral_Count"] / active_days
    feature_df["Value_Spread"] = (
        feature_df["Max_Transaction_Value"] - feature_df["Min_Transaction_Value"]
    )
    feature_df["Satisfaction_Support_Interaction"] = (
        feature_df["Customer_Satisfaction_Score"]
        / (feature_df["Support_Tickets_Raised"] + 1)
    )

    return feature_df


def get_model_columns(df: pd.DataFrame, excluded: Iterable[str] | None = None) -> list[str]:
    excluded_columns = set(excluded or [])
    return [column for column in df.columns if column not in excluded_columns]


def compute_baseline_stats(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=["number"])
    baseline = {}
    for column in numeric_df.columns:
        series = numeric_df[column]
        baseline[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return baseline


def detect_drift(
    feature_df: pd.DataFrame,
    baseline_stats: dict,
    threshold: float,
    zscore_threshold: float = 3.0,
) -> dict:
    numeric_df = feature_df.select_dtypes(include=["number"])
    drift_summary = {}
    single_record_mode = len(numeric_df) == 1

    for column, stats in baseline_stats.items():
        if column not in numeric_df.columns:
            continue

        current = numeric_df[column]
        current_mean = float(current.mean())
        current_std = float(current.std(ddof=0))
        baseline_mean = stats["mean"]
        baseline_std = stats["std"] or 1.0

        if single_record_mode:
            zscore = abs(current_mean - baseline_mean) / max(baseline_std, 1.0)
            mean_shift = zscore
            std_shift = 0.0
            is_drifted = zscore > zscore_threshold
        else:
            mean_shift = abs(current_mean - baseline_mean) / max(abs(baseline_mean), 1.0)
            std_shift = abs(current_std - baseline_std) / max(abs(baseline_std), 1.0)
            is_drifted = mean_shift > threshold or std_shift > threshold

        drift_summary[column] = {
            "current_mean": current_mean,
            "current_std": current_std,
            "mean_shift_ratio": mean_shift,
            "std_shift_ratio": std_shift,
            "drift_detected": is_drifted,
        }

    return drift_summary
