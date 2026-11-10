"""15 - Delivery: the agent as a FastAPI service (then Docker, then CI).

WHAT IT SHOWS
  POST /ask     {"question": "..."}  ->  typed JSON: report, blocked reason, trace summary
  GET  /health  liveness check for Docker / Kubernetes
  Request and response bodies are Pydantic models, so the OpenAPI docs at /docs are generated.
  The Dockerfile and docker-compose.yml package this service; the repo CI workflow runs the
  tests and the evaluation gate (script 14) on every push.

RUN
  python scripts/15_api.py                 then open http://localhost:8000/docs
  curl -s localhost:8000/ask -H 'content-type: application/json' \\
       -d '{"question": "Is pump P-101 vibration normal?"}'
  docker compose up --build api            (same service in a container)

EXPECTED OUTPUT
  "Uvicorn running on http://0.0.0.0:8000". /ask returns JSON with a report and a trace summary.

WHAT IT DOES NOT SHOW
  Authentication, rate limiting, async/background jobs for long agent runs, streaming
  responses, or horizontal scaling. The human-approval step is not exposed over HTTP here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI  # noqa: E402
from lab.agent import PROMPT_VERSION, run_agent  # noqa: E402
from lab.config import mode  # noqa: E402
from lab.schemas import Report  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        examples=["The T2 guide bearing temperature is rising. What should we do?"],
    )


class AskResponse(BaseModel):
    report: Report | None
    blocked: str = ""
    grounding_problems: list[str] = []
    trace: dict
    prompt_version: str
    mode: str


app = FastAPI(
    title="Turbine Assistant",
    version="0.1.0",
    description="Answers maintenance questions with cited reports. Read-only. AI-generated output.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": mode(), "prompt_version": PROMPT_VERSION}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # plain `def` (not async): FastAPI runs it in a thread pool, so a slow agent does not block the server
    r = run_agent(req.question, verbose=False, use_memory=False)
    return AskResponse(
        report=r.report,
        blocked=r.blocked,
        grounding_problems=r.grounding_problems,
        trace=r.trace,
        prompt_version=PROMPT_VERSION,
        mode=mode(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
