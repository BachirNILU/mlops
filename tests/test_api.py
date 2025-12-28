from __future__ import annotations

import os

from fastapi.testclient import TestClient

# Force dummy model so tests do not require MLflow running.
os.environ["MODEL_LOAD_STRATEGY"] = "dummy"

from serving.app.main import app  # noqa: E402


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_uri"].startswith("dummy://")


def test_predict() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "predicted_class" in body
        assert "probabilities" in body
        assert len(body["probabilities"]) == 3


def test_metrics() -> None:
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "api_requests_total" in r.text or "model_predictions_total" in r.text
