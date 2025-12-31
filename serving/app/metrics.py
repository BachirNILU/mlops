from __future__ import annotations

from prometheus_client import Counter, Histogram

# These names match what your tests look for
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "http_status"],
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_name"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Latency for model prediction endpoint in seconds",
)
