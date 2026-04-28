# High-Level Design

## Objective

Build an AI application that predicts Customer Lifetime Value for financial services customers while incorporating churn risk as a direct predictive signal. The design must be reproducible, observable, and easy to operate locally without cloud dependencies.

## Design choices

1. Snapshot-based surrogate for dynamic behavior:
   The provided dataset is not an event stream, so the system approximates time-varying behavior through recency, activity, spend intensity, and service-friction features.

2. Two-stage modeling:
   A churn classifier is trained first. Its churn probability output becomes an input feature to the final CLV regressor.

3. Loose coupling:
   The frontend is fully decoupled from the backend. It communicates only through REST APIs.

4. MLOps orientation:
   Data preparation, training, reporting, and monitoring are represented as explicit stages with versionable scripts and artifacts.

5. Local-first operations:
   The stack is designed to run with local files, MLflow local tracking, and Docker Compose.

## Main subsystems

- Data subsystem
  - Raw data preparation
  - Feature engineering
  - Baseline statistics for drift detection

- Modeling subsystem
  - Churn classifier
  - CLV regressor
  - Artifact packaging

- Serving subsystem
  - FastAPI inference layer
  - Prometheus instrumentation
  - Readiness and health probes

- Presentation subsystem
  - Static responsive frontend
  - Prediction form
  - Pipeline visualization summary

- Operations subsystem
  - DVC stage definitions
  - MLflow experiment tracking
  - Docker Compose orchestration

