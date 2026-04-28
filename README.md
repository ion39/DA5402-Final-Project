# Customer Lifetime Value AI Application with MLOps

This project builds a production-oriented AI application for predicting Customer Lifetime Value (CLV) in a financial services setting. It uses customer behavior, engagement, and support signals to estimate CLV while explicitly modeling churn probability as an input feature to the final regressor.

The implementation is aligned to the provided MLOps and evaluation guidelines:

- Automated data preparation and model training scripts
- Reproducible experiment tracking with MLflow
- DVC pipeline definition for orchestration
- FastAPI backend with `/health`, `/ready`, and `/metrics`
- Separate frontend and backend services via Docker Compose
- Monitoring hooks for Prometheus and Grafana
- Design, API, test, and user documentation

## Project structure

```text
.
|-- configs/
|-- data/
|-- docs/
|-- frontend/
|-- prometheus/
|-- scripts/
|-- src/clv_app/
|-- tests/
|-- digital_wallet_ltv_dataset.csv
|-- dvc.yaml
|-- docker-compose.yml
|-- Dockerfile.api
|-- Dockerfile.frontend
|-- MLproject
```

## Business framing

The goal is to estimate customer CLV for retention and revenue planning. Because the available dataset is snapshot-based rather than a true event time series, the project approximates dynamic customer behavior using recency, activity, engagement, and support-derived features. Churn risk is estimated first, and the resulting churn probability is fed into the CLV model.

## Modeling approach

1. Prepare customer records from the raw CSV.
2. Engineer churn-sensitive behavior features.
3. Create a churn label heuristic using inactivity, poor satisfaction, and support burden.
4. Train a churn classifier.
5. Use predicted churn probability as an additional input to a CLV regressor.
6. Log parameters, metrics, and artifacts to MLflow.

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare data

```bash
python scripts/prepare_data.py
```

### 3. Train models

```bash
python scripts/train.py
python scripts/generate_pipeline_report.py
```

### 4. Run the API

```bash
uvicorn src.clv_app.api:app --host 0.0.0.0 --port 8000
```

### 5. Run the frontend locally

```bash
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080` and keep the backend running at `http://localhost:8000`.

## Docker Compose

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:8080`
- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow: `http://localhost:5000`

## DVC pipeline

```bash
dvc repro
```

Stages:

- `prepare_data`
- `train_model`
- `build_report`

## Testing

```bash
pytest
```

## Documentation

- [Architecture](docs/architecture.md)
- [High-Level Design](docs/hld.md)
- [Low-Level Design](docs/lld.md)
- [Test Plan](docs/test_plan.md)
- [User Manual](docs/user_manual.md)

