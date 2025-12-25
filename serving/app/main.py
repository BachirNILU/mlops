from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

app = FastAPI(title="toy-mlops-serving", version="0.0.0")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received by the API",
    ["path", "method", "status"],
)


@app.middleware("http")
async def prom_middleware(request, call_next):
    response: Response = await call_next(request)
    HTTP_REQUESTS_TOTAL.labels(
        path=request.url.path, method=request.method, status=str(response.status_code)
    ).inc()
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
