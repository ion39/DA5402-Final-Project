from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


REQUEST_COUNTER = Counter(
    "clv_prediction_requests_total",
    "Total number of CLV prediction requests",
)
ERROR_COUNTER = Counter(
    "clv_prediction_errors_total",
    "Total number of CLV prediction errors",
)
LATENCY_HISTOGRAM = Histogram(
    "clv_prediction_latency_seconds",
    "Latency for CLV inference requests",
)
PREDICTED_CLV_GAUGE = Gauge(
    "clv_latest_prediction_value",
    "Latest predicted CLV value",
)
CHURN_PROBABILITY_GAUGE = Gauge(
    "clv_latest_churn_probability",
    "Latest predicted churn probability",
)


@contextmanager
def track_inference():
    start = time.perf_counter()
    REQUEST_COUNTER.inc()
    try:
        yield
    except Exception:
        ERROR_COUNTER.inc()
        raise
    finally:
        LATENCY_HISTOGRAM.observe(time.perf_counter() - start)


def record_prediction(clv_value: float, churn_probability: float) -> None:
    PREDICTED_CLV_GAUGE.set(clv_value)
    CHURN_PROBABILITY_GAUGE.set(churn_probability)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

