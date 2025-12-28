from __future__ import annotations

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

from serving.app.metrics import PREDICTION_COUNT, render_metrics
from serving.app.model_loader import IRIS_CLASSES, LoadedModel, load_model_from_env
from serving.app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Iris FastAI Inference API", version="0.2.0")


@app.on_event("startup")
def _startup() -> None:
    app.state.model = load_model_from_env()


@app.get("/health")
def health() -> dict[str, str]:
    model: LoadedModel = app.state.model
    return {"status": "ok", "model_uri": model.model_uri}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <body>
        <h2>Iris FastAI Inference API</h2>
        <ul>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/health">Health</a></li>
          <li><a href="/metrics">Prometheus metrics</a></li>
        </ul>
      </body>
    </html>
    """


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model: LoadedModel = app.state.model

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "sepal_length": req.sepal_length,
                "sepal_width": req.sepal_width,
                "petal_length": req.petal_length,
                "petal_width": req.petal_width,
            }
        ]
    )

    probs = model.predict_proba(df)[0].tolist()
    class_index = int(max(range(len(probs)), key=lambda i: probs[i]))
    predicted_class = IRIS_CLASSES[class_index]

    PREDICTION_COUNT.inc()

    return PredictResponse(
        predicted_class=predicted_class,
        class_index=class_index,
        probabilities=[float(x) for x in probs],
        model_uri=model.model_uri,
    )


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
