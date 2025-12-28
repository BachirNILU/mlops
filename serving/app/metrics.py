from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "http_status"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions produced by the model",
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)


@contextmanager
def track_request(endpoint: str):
    start = time.perf_counter()
    status = "500"
    try:
        yield lambda s: _set_status(s)
        status = "200"
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", http_status=status).inc()


def _set_status(_status: int) -> None:
    # kept for future flexibility
    return


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
