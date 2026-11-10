"""One function, `chat()`, used by every script. Mock or real behind the same interface.

Message format used inside the lab (simple and provider-neutral):
  {"role": "user", "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "name": "...", "args": {...}}]}
  {"role": "tool", "tool_call_id": "...", "name": "...", "content": "<JSON string>"}

Real mode uses the Anthropic Python SDK (Messages API). Checked against anthropic==2.7.0:
  - client.beta.messages.create / .parse with `output_format=<Pydantic model>` for structured output
  - tools: {"name", "description", "input_schema"}; results go back as "tool_result" blocks
  - server-side refusal fallback: betas=["server-side-fallback-2026-07-01"], fallbacks="default"
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from lab import mock_llm
from lab.config import cost_warning, is_real, model_name

# USD per million tokens (input, output). Checked 2026-11; prices change, see the Anthropic pricing page.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed: BaseModel | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    stop_reason: str = ""
    raw_content: Any = None  # real mode: the exact content blocks, sent back unchanged next turn

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICES.get(self.model.replace("mock:", ""), PRICES["claude-opus-5-5"])
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000

    def as_message(self) -> dict:
        """Turn the response into an assistant message to append to the history."""
        return {
            "role": "assistant",
            "content": self.text,
            "raw": self.raw_content,
            "tool_calls": [{"id": t.id, "name": t.name, "args": t.args} for t in self.tool_calls],
        }


def chat(
    messages: list[dict],
    system: str = "",
    tools: list | None = None,
    response_model: type[BaseModel] | None = None,
    max_tokens: int = 4000,
) -> LLMResponse:
    """Send a conversation to the LLM. `tools` is a list of lab.tools.Tool."""
    t0 = time.perf_counter()
    if is_real():
        cost_warning()
        resp = _real_chat(messages, system, tools, response_model, max_tokens)
    else:
        resp = _mock_chat(messages, system, tools, response_model)
    resp.latency_s = time.perf_counter() - t0
    return resp


# ---------------------------------------------------------------- mock


def _mock_chat(messages, system, tools, response_model) -> LLMResponse:
    out = mock_llm.respond(system, messages, tools or [], response_model)
    prompt_chars = len(system) + sum(len(str(m.get("content", ""))) for m in messages)
    if tools:
        prompt_chars += sum(len(json.dumps(t.json_schema())) for t in tools)
    parsed = None
    if response_model is not None:
        parsed = response_model.model_validate_json(
            out["text"]
        )  # raises if the mock returned bad JSON
    return LLMResponse(
        text=out["text"],
        tool_calls=[ToolCall(**tc) for tc in out.get("tool_calls", [])],
        parsed=parsed,
        input_tokens=prompt_chars // 4,  # rough rule: 1 token is about 4 characters
        output_tokens=max(1, len(out["text"]) // 4) + 30 * len(out.get("tool_calls", [])),
        model=f"mock:{model_name()}",
        stop_reason="tool_use" if out.get("tool_calls") else "end_turn",
    )


# ---------------------------------------------------------------- real (Anthropic)


def _to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            if m.get("raw") is not None:
                content = m["raw"]  # pass thinking + tool_use blocks back unchanged
            else:
                content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                content += [
                    {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]}
                    for t in m.get("tool_calls", [])
                ]
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if m.get("is_error"):
                block["is_error"] = True
            # all results of one turn must go back in ONE user message
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
                and out[-1]["content"]
                and out[-1]["content"][0].get("type") == "tool_result"
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _real_chat(messages, system, tools, response_model, max_tokens) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment / .env
    kwargs: dict[str, Any] = dict(
        model=model_name(), max_tokens=max_tokens, messages=_to_anthropic(messages)
    )
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.json_schema()}
            for t in tools
        ]
    if os.getenv("LLM_EFFORT"):
        kwargs["output_config"] = {"effort": os.environ["LLM_EFFORT"]}
    if os.getenv("LLM_FALLBACKS", "1") == "1":
        # if a safety classifier declines, the API retries on a suitable fallback model
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    else:
        kwargs["betas"] = []

    try:
        if response_model is not None:
            r = client.beta.messages.parse(output_format=response_model, **kwargs)
        else:
            r = client.beta.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise SystemExit(
            "ANTHROPIC_API_KEY missing or invalid. Copy .env.example to .env and set it."
        ) from e
    except anthropic.RateLimitError as e:
        raise SystemExit("Rate limited by the API. Wait a minute and try again.") from e

    if r.stop_reason == "refusal":
        raise RuntimeError(f"The model declined: {getattr(r, 'stop_details', None)}")

    text = "".join(b.text for b in r.content if b.type == "text")
    calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
    return LLMResponse(
        text=text,
        tool_calls=calls,
        parsed=getattr(r, "parsed_output", None),
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
        model=r.model,
        stop_reason=r.stop_reason,
        raw_content=[
            {k: v for k, v in b.model_dump(exclude_none=True).items() if k != "parsed_output"}
            for b in r.content
        ],
    )
