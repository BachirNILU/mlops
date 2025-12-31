from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_uri: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str | None = None
