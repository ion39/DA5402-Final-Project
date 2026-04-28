# Architecture Diagram

```mermaid
flowchart LR
    A[Raw CSV Dataset] --> B[Prepare Data Script]
    B --> C[Versioned Raw Data]
    C --> D[Feature Engineering]
    D --> E[Churn Classifier]
    E --> F[Churn Probability]
    F --> G[CLV Regressor]
    G --> H[Model Bundle Artifact]
    H --> I[FastAPI Inference Service]
    I --> J[Frontend UI]
    I --> K[Prometheus Metrics]
    K --> L[Grafana Dashboards]
    D --> M[Baseline Statistics]
    M --> I
    D --> N[MLflow Tracking]
    E --> N
    G --> N
```

## Block summary

- `Prepare Data Script`: copies the source CSV into a reproducible raw-data location and creates a sample payload.
- `Feature Engineering`: derives churn-sensitive behavioral features.
- `Churn Classifier`: predicts probability of customer churn.
- `CLV Regressor`: predicts lifetime value using customer features and churn probability.
- `Model Bundle Artifact`: serialized inference package used by the API.
- `FastAPI Inference Service`: provides prediction, health, readiness, model info, and metrics endpoints.
- `Frontend UI`: standalone user interface that calls the backend through REST only.
- `Prometheus/Grafana`: production-style monitoring path for latency and request metrics.
- `MLflow`: tracks experiment metadata, parameters, metrics, and models.

