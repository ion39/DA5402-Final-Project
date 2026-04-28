import pandas as pd

from src.clv_app.features import build_feature_frame, derive_churn_label


def sample_df():
    return pd.DataFrame(
        [
            {
                "Customer_ID": "cust_a",
                "Age": 45,
                "Location": "Urban",
                "Income_Level": "High",
                "Total_Transactions": 100,
                "Avg_Transaction_Value": 1000.0,
                "Max_Transaction_Value": 3000.0,
                "Min_Transaction_Value": 100.0,
                "Total_Spent": 100000.0,
                "Active_Days": 100,
                "Last_Transaction_Days_Ago": 260,
                "Loyalty_Points_Earned": 500,
                "Referral_Count": 5,
                "Cashback_Received": 200.0,
                "App_Usage_Frequency": "Monthly",
                "Preferred_Payment_Method": "UPI",
                "Support_Tickets_Raised": 15,
                "Issue_Resolution_Time": 12.0,
                "Customer_Satisfaction_Score": 2,
                "LTV": 25000.0,
            }
        ]
    )


def test_feature_engineering_adds_expected_columns():
    features = build_feature_frame(sample_df())
    assert "Spend_Per_Active_Day" in features.columns
    assert "Recency_Activity_Ratio" in features.columns
    assert "Cashback_Ratio" in features.columns


def test_churn_label_is_binary():
    label = derive_churn_label(sample_df())
    assert set(label.tolist()).issubset({0, 1})
    assert label.iloc[0] == 1

