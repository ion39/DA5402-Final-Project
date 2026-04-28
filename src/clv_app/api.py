from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .config import load_config, resolve_path
from .features import build_feature_frame, detect_drift
from .monitoring import record_prediction, render_metrics, track_inference
from .schemas import CustomerFeatures, PredictionResponse


config = load_config()
paths = config["paths"]
drift_threshold = config["monitoring"]["drift_threshold_mean_shift"]
zscore_threshold = config["monitoring"]["drift_threshold_zscore"]
model_bundle_path = resolve_path(paths["model_bundle"])
metrics_path = resolve_path(paths["metrics"])
pipeline_report_path = resolve_path(paths["pipeline_report"])
sample_payload_path = resolve_path(paths["sample_payload"])

app = FastAPI(title="CLV Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None and model_bundle_path.exists():
        _bundle = joblib.load(model_bundle_path)
    return _bundle


@app.get("/")
def root():
    return {
        "message": "CLV prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    ready_state = model_bundle_path.exists()
    return {"ready": ready_state, "model_artifact": str(model_bundle_path)}


@app.get("/sample-input")
def sample_input():
    if not sample_payload_path.exists():
        raise HTTPException(status_code=404, detail="Sample payload not generated yet.")
    return JSONResponse(json.loads(sample_payload_path.read_text(encoding="utf-8")))


@app.get("/model/info")
def model_info():
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not available.")
    return JSONResponse(json.loads(metrics_path.read_text(encoding="utf-8")))


@app.get("/pipeline/summary")
def pipeline_summary():
    if not pipeline_report_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline report not available.")
    return JSONResponse(json.loads(pipeline_report_path.read_text(encoding="utf-8")))


@app.get("/metrics")
def metrics():
    content, media_type = render_metrics()
    return Response(content=content, media_type=media_type)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model artifact is not ready.")

    frame = pd.DataFrame([payload.model_dump()])
    with track_inference():
        predictions = bundle.predict(frame)

    feature_frame = build_feature_frame(frame)
    drift_summary = detect_drift(
        feature_frame,
        bundle.baseline_stats,
        drift_threshold,
        zscore_threshold=zscore_threshold,
    )
    drifted_features = [
        feature_name
        for feature_name, stats in drift_summary.items()
        if stats["drift_detected"]
    ]

    predicted_clv = float(predictions.iloc[0]["Predicted_CLV"])
    churn_probability = float(predictions.iloc[0]["Churn_Probability"])
    churn_label = int(predictions.iloc[0]["Predicted_Churn_Label"])
    record_prediction(predicted_clv, churn_probability)

    return PredictionResponse(
        customer_id=payload.Customer_ID,
        predicted_clv=predicted_clv,
        churn_probability=churn_probability,
        predicted_churn_label=churn_label,
        drift_detected_features=drifted_features,
    )
