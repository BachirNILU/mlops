from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@dataclass
class LoadedModel:
    pyfunc: PyFuncModel
    model_uri: str

    def predict(self, df: pd.DataFrame) -> list[str]:
        preds, _ = _parse_pyfunc_output(self.pyfunc.predict(df))
        return preds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        preds, probs = _parse_pyfunc_output(self.pyfunc.predict(df))

        if probs is not None:
            return probs

        # Fallback: one-hot from predicted class
        out = np.zeros((len(preds), len(IRIS_CLASSES)), dtype=float)
        for i, p in enumerate(preds):
            p_str = str(p)
            if p_str in IRIS_CLASSES:
                out[i, IRIS_CLASSES.index(p_str)] = 1.0
        return out


def _dummy_model() -> LoadedModel:
    class _DummyPyfunc:
        def predict(self, df: pd.DataFrame) -> np.ndarray:
            n = len(df)
            row = ["setosa", 0.99, 0.005, 0.005]
            return np.array([row for _ in range(n)], dtype=object)

    return LoadedModel(pyfunc=_DummyPyfunc(), model_uri="dummy://iris-fastai")


def _resolve_model_uri() -> str:
    direct = os.getenv("MLFLOW_MODEL_URI")
    if direct:
        return direct

    name = os.getenv("MLFLOW_MODEL_NAME", "iris-fastai")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    return f"models:/{name}/{stage}"


def load_model_from_env() -> LoadedModel:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = _resolve_model_uri()

    # Allow fallback if explicitly enabled
    if os.getenv("ALLOW_MISSING_MODEL", "0") == "1":
        try:
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            return LoadedModel(pyfunc=pyfunc_model, model_uri=model_uri)
        except Exception:
            return _dummy_model()

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    return LoadedModel(pyfunc=pyfunc_model, model_uri=model_uri)


def _coerce_nested_rows(arr: np.ndarray) -> np.ndarray:
    """
    If arr is shape (n,) and each element is like [label, p1, p2, p3],
    turn it into shape (n, 4).
    """
    if arr.ndim != 1 or arr.size == 0:
        return arr

    first = arr[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        try:
            stacked = np.stack([np.array(x, dtype=object) for x in arr.tolist()], axis=0)
            return stacked
        except Exception:
            return arr

    return arr


def _parse_pyfunc_output(raw: Any) -> tuple[list[str], np.ndarray | None]:
    """
    Normalize pyfunc output into:
      preds: list[str]
      probs: np.ndarray shape (n, 3) or None
    Supports:
      - ndarray/list with shape (n, 4) => [label, p1, p2, p3]
      - ndarray/list with shape (n, 3) => probabilities only
      - 1d element that is itself [label, p1, p2, p3] repeated per row
      - DataFrame outputs
      - scalar string outputs
    """
    # DataFrame output
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return ["unknown"], None

        if "predicted_class" in raw.columns:
            preds = raw["predicted_class"].astype(str).tolist()
        elif "prediction" in raw.columns:
            preds = raw["prediction"].astype(str).tolist()
        else:
            preds = raw.iloc[:, 0].astype(str).tolist()

        # Probabilities in columns named as classes
        if all(c in raw.columns for c in IRIS_CLASSES):
            probs = raw[IRIS_CLASSES].to_numpy(dtype=float)
            return preds, probs

        # Probabilities in columns prob_setosa, prob_versicolor, ...
        prob_cols = [c for c in raw.columns if str(c).startswith("prob_")]
        if prob_cols:
            mapping = {str(c)[5:]: c for c in prob_cols}
            ordered = [mapping[c] for c in IRIS_CLASSES if c in mapping]
            if len(ordered) == 3:
                probs = raw[ordered].to_numpy(dtype=float)
                return preds, probs

        return preds, None

    # ndarray/list/tuple output
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.array(raw, dtype=object)
        arr = _coerce_nested_rows(arr)

        # Single row [label, p1, p2, p3]
        if arr.ndim == 1 and arr.shape[0] == 1 + len(IRIS_CLASSES):
            preds = [str(arr[0])]
            probs = np.array([arr[1:]], dtype=float)
            return preds, probs

        # Labels only
        if arr.ndim == 1:
            preds = [str(x) for x in arr.tolist()]
            return preds, None

        # Probabilities only: (n, 3)
        if arr.ndim == 2 and arr.shape[1] == len(IRIS_CLASSES):
            probs = arr.astype(float)
            idx = probs.argmax(axis=1)
            preds = [IRIS_CLASSES[int(i)] for i in idx]
            return preds, probs

        # [label, p1, p2, p3]: (n, 4)
        if arr.ndim == 2 and arr.shape[1] == 1 + len(IRIS_CLASSES):
            preds = [str(x) for x in arr[:, 0].tolist()]
            probs = arr[:, 1:].astype(float)
            return preds, probs

    # Scalar output
    return [str(raw)], None
