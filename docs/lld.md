# Low-Level Design

## Backend endpoints

### `GET /health`

- Purpose: liveness probe
- Response:

```json
{"status": "ok"}
```

### `GET /ready`

- Purpose: readiness probe for deployment orchestration
- Response:

```json
{
  "ready": true,
  "model_artifact": "artifacts/model_bundle.joblib"
}
```

### `GET /sample-input`

- Purpose: fetch a valid example payload for the UI

### `GET /model/info`

- Purpose: return offline evaluation metrics and data summary

### `GET /pipeline/summary`

- Purpose: return pipeline stage visibility for UI/demo consumption

### `GET /metrics`

- Purpose: Prometheus scrape endpoint

### `POST /predict`

- Purpose: return CLV estimate and churn risk for a customer

#### Request body

```json
{
  "Customer_ID": "manual_input",
  "Age": 45,
  "Location": "Urban",
  "Income_Level": "High",
  "Total_Transactions": 400,
  "Avg_Transaction_Value": 10000.0,
  "Max_Transaction_Value": 30000.0,
  "Min_Transaction_Value": 2500.0,
  "Total_Spent": 4000000.0,
  "Active_Days": 180,
  "Last_Transaction_Days_Ago": 30,
  "Loyalty_Points_Earned": 2200,
  "Referral_Count": 10,
  "Cashback_Received": 1800.0,
  "App_Usage_Frequency": "Weekly",
  "Preferred_Payment_Method": "UPI",
  "Support_Tickets_Raised": 3,
  "Issue_Resolution_Time": 24.5,
  "Customer_Satisfaction_Score": 8
}
```

#### Response body

```json
{
  "customer_id": "manual_input",
  "predicted_clv": 525000.42,
  "churn_probability": 0.24,
  "predicted_churn_label": 0,
  "drift_detected_features": []
}
```

## Core modules

- `src/clv_app/features.py`
  - churn label heuristic
  - engineered features
  - baseline statistics
  - simple drift detection

- `src/clv_app/modeling.py`
  - preprocessing pipeline
  - classifier and regressor builders
  - serialized model bundle

- `src/clv_app/pipeline.py`
  - end-to-end training and MLflow logging

- `src/clv_app/api.py`
  - online inference and monitoring

