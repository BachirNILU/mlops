from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from serving.app.metrics import (
    API_REQUESTS_TOTAL,
    MODEL_PREDICTIONS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
)
from serving.app.model_loader import IRIS_CLASSES, IRIS_FEATURES, LoadedModel, load_model_from_env
from serving.app.schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_from_env()
    yield


app = FastAPI(title="Iris FastAI API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model: LoadedModel | None = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_uri=(model.model_uri if model else None),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.time()
    try:
        model: LoadedModel = app.state.model
        df = pd.DataFrame(
            [[getattr(payload, f) for f in IRIS_FEATURES]],
            columns=IRIS_FEATURES,
        )

        predicted_class = model.predict(df)[0]
        probs = model.predict_proba(df)[0]
        probabilities = {c: float(p) for c, p in zip(IRIS_CLASSES, probs, strict=False)}

        MODEL_PREDICTIONS_TOTAL.labels(model_name="iris-fastai").inc()

        API_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", http_status="200").inc()
        PREDICTION_LATENCY_SECONDS.observe(time.time() - start)

        return PredictResponse(
            predicted_class=predicted_class,
            probabilities=probabilities,
            model_uri=model.model_uri,
        )

    except Exception as e:
        API_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", http_status="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
