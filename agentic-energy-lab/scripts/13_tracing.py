"""13 - Observability: trace every step, then look at cost and latency.

WHAT IT SHOWS
  1. One agent run traced step by step: guardrails, each LLM call, each tool call, the report.
     Per step: latency, tokens in/out, cost.
  2. Three more runs, then aggregate numbers across runs from runs/traces.jsonl:
     cost per question, p50/p95 latency per step type, error count.
  3. Langfuse: if LANGFUSE_* keys are set in .env, the same spans are sent to Langfuse
     (cloud or self-hosted) and you can browse them in its web UI.

RUN
  python scripts/13_tracing.py          (mock: costs shown are what real calls would cost)
  python scripts/13_tracing.py --real   (real API, about $0.40-1.00 for 4 runs)

EXPECTED OUTPUT (mock)
  A table of ~10 spans for the first run, then an aggregate table.
  "Langfuse: not configured" unless you set the keys.

WHAT IT DOES NOT SHOW
  Dashboards and alerts, OpenTelemetry collectors, user feedback scores, or sampling of traces
  in production. Mock latencies are tiny and not meaningful; mock token counts are estimates.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from lab.agent import run_agent  # noqa: E402
from lab.config import RUNS, banner  # noqa: E402
from lab.tracing import Tracer, langfuse_enabled  # noqa: E402

QUESTIONS = [
    "Is pump P-101 vibration normal? Has it happened before?",
    "What is the normal cooling water flow per unit?",
    "Is the T1 bearing temperature OK right now?",
]


def main() -> None:
    banner("13 - Tracing, cost and latency")
    (RUNS / "traces.jsonl").unlink(missing_ok=True)
    print(
        f"Langfuse: {'ENABLED, spans are sent to ' + __import__('os').environ['LANGFUSE_HOST'] if langfuse_enabled() else 'not configured (local trace only; see README to enable)'}"
    )

    q = "The T2 guide bearing temperature is rising. Known issue? What does the manual say?"
    tracer = Tracer("turbine-assistant", q)
    r = run_agent(q, verbose=False, tracer=tracer, faults={"search_docs": 2}, use_memory=False)
    tracer.print_table()
    print(f"\n  run summary: {r.trace}")

    for q in QUESTIONS:
        run_agent(q, verbose=False, use_memory=False)

    spans = [json.loads(line) for line in (RUNS / "traces.jsonl").read_text().splitlines()]
    runs = {s["trace_id"] for s in spans}
    print(f"\nAcross {len(runs)} runs ({len(spans)} spans):")
    cost_per_run = [sum(s["cost_usd"] for s in spans if s["trace_id"] == t) for t in runs]
    print(f"  cost per question: mean ${np.mean(cost_per_run):.4f}, max ${max(cost_per_run):.4f}")
    print(f"  {'step type':<12}{'count':>6}{'p50 s':>9}{'p95 s':>9}{'errors':>8}")
    for kind in ["generation", "tool", "guardrail"]:
        lat = [s["latency_s"] for s in spans if s["kind"] == kind]
        err = sum(1 for s in spans if s["kind"] == kind and s["error"])
        if lat:
            print(
                f"  {kind:<12}{len(lat):>6}{np.percentile(lat, 50):>9.3f}{np.percentile(lat, 95):>9.3f}{err:>8}"
            )
    print(
        "\nTake-away: without traces you cannot debug an agent, explain a bad answer, or control cost."
    )


if __name__ == "__main__":
    main()
