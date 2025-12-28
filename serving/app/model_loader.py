from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def _set_local_minio_defaults() -> None:
    # Only set defaults if not already provided (so production can override)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@dataclass(frozen=True)
class LoadedModel:
    model_uri: str
    predict_proba: Callable[[pd.DataFrame], np.ndarray]


def _dummy_model() -> LoadedModel:
    # Deterministic dummy: always returns uniform probabilities.
    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.full((n, 3), 1.0 / 3.0, dtype=float)

    return LoadedModel(model_uri="dummy://uniform", predict_proba=predict_proba)


def load_model_from_env() -> LoadedModel:
    """
    Loads a model based on env vars.

    MODEL_LOAD_STRATEGY:
      - "mlflow" (default): load from MLflow Registry
      - "dummy": do not touch MLflow, return a deterministic dummy model
    """
    strategy = os.getenv("MODEL_LOAD_STRATEGY", "mlflow").strip().lower()
    if strategy == "dummy":
        return _dummy_model()

    _set_local_minio_defaults()

    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").strip()
    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv("MODEL_NAME", "iris-fastai").strip()
    model_stage = os.getenv("MODEL_STAGE", "Production").strip()
    model_version = os.getenv("MODEL_VERSION", "").strip()

    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        model_uri = f"models:/{model_name}/{model_stage}"

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        # Ensure column order
        df = df[IRIS_FEATURES].copy()

        y = pyfunc_model.predict(df)

        # We support:
        # - probabilities shape (n,3)
        # - labels shape (n,)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        y = np.asarray(y)

        if y.ndim == 2 and y.shape[1] == 3:
            return y.astype(float)

        # If it returns class indices or labels, convert to one-hot
        probs = np.zeros((len(df), 3), dtype=float)
        for i, v in enumerate(y):
            if isinstance(v, (int, np.integer)):
                idx = int(v)
            else:
                # try label string
                idx = IRIS_CLASSES.index(str(v))
            probs[i, idx] = 1.0
        return probs

    return LoadedModel(model_uri=model_uri, predict_proba=predict_proba)
