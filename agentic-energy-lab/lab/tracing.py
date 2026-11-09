"""Tracing: record every step of a run (input, output, time, tokens, cost).

Always writes a local trace to runs/traces.jsonl.
If LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are set, it also sends the same
spans to Langfuse (checked against langfuse==5.2.0: Langfuse().start_as_current_observation).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any

from lab.config import RUNS


def langfuse_enabled() -> bool:
    return all(
        os.getenv(k) for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    )


def _short(x: Any, n: int = 300) -> Any:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + "..."


class Tracer:
    def __init__(self, name: str, user_input: Any = None):
        self.trace_id = uuid.uuid4().hex[:12]
        self.name = name
        self.spans: list[dict] = []
        self._lf = None
        self._root = None
        if langfuse_enabled():
            from langfuse import Langfuse

            self._lf = Langfuse()
            self._root_cm = self._lf.start_as_current_observation(
                name=name, as_type="agent", input=user_input
            )
            self._root = self._root_cm.__enter__()

    @contextmanager
    def span(self, name: str, kind: str = "span", input: Any = None, model: str | None = None):
        """kind: span | generation | tool | retriever | guardrail | evaluator (Langfuse types)."""
        rec = {
            "trace_id": self.trace_id,
            "name": name,
            "kind": kind,
            "input": _short(input),
            "start": time.time(),
            "output": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "error": None,
        }
        lf_cm = (
            self._lf.start_as_current_observation(name=name, as_type=kind, input=input, model=model)
            if self._lf
            else None
        )
        lf_obs = lf_cm.__enter__() if lf_cm else None
        try:
            yield rec
        except Exception as e:
            rec["error"] = repr(e)
            raise
        finally:
            rec["latency_s"] = round(time.time() - rec["start"], 4)
            self.spans.append(rec)
            if lf_obs is not None:
                kw: dict[str, Any] = {"output": rec["output"]}
                if kind == "generation":
                    kw["usage_details"] = {"input": rec["tokens_in"], "output": rec["tokens_out"]}
                    kw["cost_details"] = {"total": rec["cost_usd"]}
                if rec["error"]:
                    kw["level"] = "ERROR"
                    kw["status_message"] = rec["error"]
                lf_obs.update(**kw)
                lf_cm.__exit__(None, None, None)

    def record_llm(self, rec: dict, resp) -> None:
        rec.update(
            output=_short(resp.text or [t.name for t in resp.tool_calls]),
            tokens_in=resp.input_tokens,
            tokens_out=resp.output_tokens,
            cost_usd=round(resp.cost_usd, 6),
            model=resp.model,
        )

    def close(self, output: Any = None) -> None:
        RUNS.mkdir(exist_ok=True)
        with open(RUNS / "traces.jsonl", "a") as f:
            for s in self.spans:
                f.write(json.dumps(s, default=str) + "\n")
        if self._root is not None:
            self._root.update(output=output)
            self._root_cm.__exit__(None, None, None)
            self._lf.flush()

    def summary(self) -> dict:
        llm = [s for s in self.spans if s["kind"] == "generation"]
        return {
            "trace_id": self.trace_id,
            "steps": len(self.spans),
            "llm_calls": len(llm),
            "tokens_in": sum(s["tokens_in"] for s in llm),
            "tokens_out": sum(s["tokens_out"] for s in llm),
            "cost_usd": round(sum(s["cost_usd"] for s in llm), 5),
            "latency_s": round(sum(s["latency_s"] for s in self.spans if s["kind"] != "agent"), 3),
        }

    def print_table(self) -> None:
        print(f"\n  trace {self.trace_id}")
        print(
            f"  {'step':<34}{'kind':<11}{'latency':>9}{'tok in':>8}{'tok out':>8}{'cost $':>10}  error"
        )
        for s in self.spans:
            print(
                f"  {s['name'][:33]:<34}{s['kind']:<11}{s['latency_s']:>8.3f}s{s['tokens_in']:>8}"
                f"{s['tokens_out']:>8}{s['cost_usd']:>10.5f}  {s['error'] or ''}"
            )
