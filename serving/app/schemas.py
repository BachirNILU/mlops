from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: list[float]
    model_uri: str
