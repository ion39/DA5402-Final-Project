from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Customer_ID: str = Field(default="manual_input")
    Age: int = Field(ge=16, le=100)
    Location: Literal["Urban", "Suburban", "Rural"]
    Income_Level: Literal["Low", "Middle", "High"]
    Total_Transactions: int = Field(ge=0)
    Avg_Transaction_Value: float = Field(ge=0)
    Max_Transaction_Value: float = Field(ge=0)
    Min_Transaction_Value: float = Field(ge=0)
    Total_Spent: float = Field(ge=0)
    Active_Days: int = Field(ge=1, le=3650)
    Last_Transaction_Days_Ago: int = Field(ge=0, le=3650)
    Loyalty_Points_Earned: int = Field(ge=0)
    Referral_Count: int = Field(ge=0)
    Cashback_Received: float = Field(ge=0)
    App_Usage_Frequency: Literal["Daily", "Weekly", "Monthly"]
    Preferred_Payment_Method: Literal["UPI", "Credit Card", "Debit Card", "Wallet Balance"]
    Support_Tickets_Raised: int = Field(ge=0)
    Issue_Resolution_Time: float = Field(ge=0)
    Customer_Satisfaction_Score: int = Field(ge=1, le=10)


class PredictionResponse(BaseModel):
    customer_id: str
    predicted_clv: float
    churn_probability: float
    predicted_churn_label: int
    drift_detected_features: list[str]

