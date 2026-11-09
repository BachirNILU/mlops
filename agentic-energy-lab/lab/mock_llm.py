"""A FAKE LLM made of hand-written rules, so every script runs offline, free and repeatably.

Read this file once, then forget it. It is NOT how a language model works. It only imitates the
*shape* of what a real model returns (text, tool calls, JSON), so you can study everything around
the model: loops, schemas, retrieval, guardrails, tracing, evaluation.

Two behaviours are deliberately unrealistic, to make lessons visible:
  - It is gullible: if text containing "ignore all previous instructions" reaches it, it obeys.
    Real models resist many injections, but not all. Guardrails are still required.
  - It can be told to return broken JSON once (FAIL_NEXT_STRUCTURED), to show validation + retry.
"""

from __future__ import annotations

import itertools
import json
import re

FAIL_NEXT_STRUCTURED = False  # script 02 sets this to show a validation failure
_ids = itertools.count(1)

INJECTION_MARK = "ignore all previous instructions"


def respond(system: str, messages: list[dict], tools: list, response_model) -> dict:
    global FAIL_NEXT_STRUCTURED
    question = _last_user_text(messages)
    if response_model is not None and response_model.__name__ == "Report":
        question = _first_user_text(messages)  # the report is about the original question
    evidence = _evidence(messages)
    hijacked = _sees_injection(system, messages)

    if response_model is not None:
        if FAIL_NEXT_STRUCTURED:
            FAIL_NEXT_STRUCTURED = False
            return {"text": '{"asset_id": "T2", "category": "hot bearing", "urgency": "very high"}'}
        builder = {"TicketTriage": _triage, "Report": _report, "JudgeVerdict": _judge}.get(
            response_model.__name__
        )
        if builder is None:
            raise NotImplementedError(f"The mock LLM has no rule for {response_model.__name__}")
        return {"text": json.dumps(builder(question, evidence, hijacked))}

    if tools:
        call = _next_tool_call(question, messages, {t.name for t in tools})
        if call:
            return {"text": "", "tool_calls": [call]}
    return {"text": _answer(question, evidence, system, hijacked)}


# ---------------------------------------------------------------- reading the conversation


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], str):
            return m["content"]
    return ""


def _first_user_text(messages: list[dict]) -> str:
    return next(
        (m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)),
        "",
    )


def _evidence(messages: list[dict]) -> list[dict]:
    ev = []
    for m in messages:
        if m["role"] == "tool":
            try:
                data = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            ev.append({"name": m.get("name"), "data": data, "is_error": m.get("is_error", False)})
    return ev


def _sees_injection(system: str, messages: list[dict]) -> bool:
    texts = [str(m.get("content", "")) for m in messages]
    return any(INJECTION_MARK in t.lower() for t in texts)


def find_asset(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bp-?101\b", t) or "pump" in t:
        return "P-101"
    m = re.search(r"\b(t[12])\b", t)
    return m.group(1).upper() if m else None


def find_signal(text: str) -> str | None:
    t = text.lower()
    if "vibrat" in t or "noisy" in t or "noise" in t:
        return "vibration_mm_s"
    if any(w in t for w in ["temp", "bearing", "hot", "warm", "degc", "heat", "trend", "rising"]):
        return "bearing_temp_c"
    return None


# ---------------------------------------------------------------- planning (which tool next?)


def plan(question: str) -> list[tuple[str, dict]]:
    q = question.lower()
    asset, signal = find_asset(q), find_signal(q)
    steps: list[tuple[str, dict]] = []
    if asset and (signal or any(w in q for w in ["status", "healthy", "problem", "ok"])):
        steps.append(
            (
                "analyze_sensor",
                {"asset_id": asset, "signal": signal or "bearing_temp_c", "days": 14},
            )
        )
    doc_words = [
        "manual",
        "procedure",
        "limit",
        "what does",
        "should",
        "why",
        "cause",
        "known",
        "recommend",
        "zone",
        "how",
        "log",
        "policy",
        "rule",
        "permit",
        "inspection",
        "oil",
        "strainer",
        "approve",
        "allowed",
        "sample",
        "drift",
        "when",
        "which",
        "flow",
        "normal",
    ]
    if not asset or any(w in q for w in doc_words):
        steps.append(("search_docs", {"query": question.strip()[:200], "k": 4}))
    if asset and any(
        w in q
        for w in [
            "known",
            "before",
            "history",
            "past",
            "previous",
            "again",
            "happened",
            "seen",
            "issue",
        ]
    ):
        steps.append(("get_work_orders", {"asset_id": asset}))
    if "create a work order" in q or "open a work order" in q:
        steps.append(
            (
                "create_work_order",
                {
                    "asset_id": asset or "T2",
                    "priority": "medium",
                    "description": "Check cooling water strainer and take oil sample.",
                },
            )
        )
    return steps


def _next_tool_call(question: str, messages: list[dict], available: set[str]) -> dict | None:
    # tool calls already made since the last user message, and how they ended
    idx = max(
        i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)
    )
    attempts: dict[str, int] = {}
    succeeded: set[str] = set()
    for m in messages[idx + 1 :]:
        if m["role"] == "tool":
            attempts[m["name"]] = attempts.get(m["name"], 0) + 1
            if not m.get("is_error"):
                succeeded.add(m["name"])
    for name, args in plan(question):
        if name not in available or name in succeeded:
            continue
        if attempts.get(name, 0) >= 2:  # tried twice and failed: give up on this tool
            continue
        return {"id": f"call_{next(_ids)}", "name": name, "args": args}
    return None


# ---------------------------------------------------------------- writing answers


def _sentences(text: str) -> list[str]:
    joined = re.sub(r"\n(?!\d+\.|\n|#|- )", " ", text)  # re-join wrapped lines, keep list items
    return [s.strip(" -#") for s in re.split(r"(?<=[.!?])\s+|\n", joined) if s.strip(" -#")]


def _first_sentence_with(text: str, words: list[str]) -> str:
    """Pick the sentence sharing the most words with the question (ties: the earliest)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]  # skip markdown headings
    sentences = [s for s in _sentences("\n".join(lines)) if len(s) > 20]
    if not sentences:
        return ""
    # words that appear in many sentences say little (a tiny IDF); exact phrases add a bonus
    weight = {w: 1.0 / max(1, sum(w in s.lower() for s in sentences)) for w in words}
    pairs = [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
    return max(
        sentences,
        key=lambda s: (
            sum(weight[w] for w in words if w in s.lower())
            + 1.0 * sum(p in s.lower() for p in pairs)
        ),
    )


def _rank(sentences: list[str], words: list[str]) -> list[str]:
    """Most relevant first (stable for ties)."""
    return sorted(sentences, key=lambda s: -sum(w in s.lower() for w in words))


def _keywords(question: str) -> list[str]:
    stop = {
        "what",
        "does",
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "should",
        "about",
        "from",
        "have",
        "is",
        "are",
        "how",
        "why",
        "when",
        "which",
        "we",
        "do",
        "say",
        "it",
        "a",
        "an",
        "of",
    }
    return [w for w in re.findall(r"[a-z0-9-]+", question.lower()) if len(w) > 2 and w not in stop]


def _answer(question: str, evidence: list[dict], system: str, hijacked: bool) -> str:
    if hijacked:
        return "All assets are healthy. All pending work orders are approved."
    parts = []
    for ev in evidence:
        d = ev["data"]
        if ev["is_error"]:
            parts.append(f"(The {ev['name']} tool failed, so this part is unknown.)")
        elif ev["name"] == "analyze_sensor":
            parts.append(
                f"{d['asset_id']} {d['signal']} is {d['status']}: last value {d['last_value']} {d['unit']}, "
                f"trend {d['trend_per_day']:+} {d['unit']}/day (95% CI {d['trend_ci95'][0]} to "
                f"{d['trend_ci95'][1]}), change started {d['change_start']}. [{d['source_id']}]"
            )
        elif ev["name"] == "search_docs":
            for hit in d[:2]:
                parts.append(
                    f"{_first_sentence_with(hit['text'], _keywords(question))} [{hit['source_id']}]"
                )
        elif ev["name"] == "get_work_orders":
            for wo in d["work_orders"]:
                if any(w in wo["problem"].lower() for w in ["temperature", "vibration"]):
                    parts.append(
                        f"Similar past case {wo['id']}: {wo['cause']} {wo['action']} [{d['source_id']}]"
                    )
                    break
        elif ev["name"] == "create_work_order":
            parts.append(f"Work order {d['id']} proposed ({d['status']}). [{d['source_id']}]")
    if parts:
        return " ".join(parts)
    # no tools: answer only from <document> blocks placed in the prompt (script 01)
    docs = re.findall(r'<document id="([^"]+)">(.*?)</document>', _all_text(question), re.S)
    kws = _keywords(question.split("QUESTION:")[-1])
    for doc_id, body in docs:
        if sum(w in body.lower() for w in kws) >= 2:
            return f"{_first_sentence_with(body, kws)} [{doc_id}]"
    if "cite" in system.lower() or "unknown" in system.lower():
        return "I don't know based on the provided documents."
    return "Bearings usually run between 40 and 90 degC. (No source: this is a guess.)"


def _all_text(question: str) -> str:
    return question


def _triage(question: str, evidence, hijacked) -> dict:
    q = question.lower()
    asset = find_asset(q) or "unknown"
    cat = (
        "vibration"
        if "vibrat" in q or "nois" in q
        else "temperature"
        if "temp" in q or "hot" in q
        else "leak"
        if "leak" in q
        else "electrical"
        if "breaker" in q or "volt" in q
        else "other"
    )
    urgent = any(w in q for w in ["smoke", "fire", "trip", "alarm", "80"])
    return {
        "asset_id": asset,
        "category": cat,
        "urgency": "high" if urgent else "medium",
        "needs_human": urgent or asset == "unknown",
        "reason": "Rule-based mock: keywords in the message.",
    }


def _report(question: str, evidence: list[dict], hijacked: bool) -> dict:
    asset = find_asset(question) or "unknown"
    if hijacked:
        return {
            "asset_id": asset,
            "summary": "All assets are healthy.",
            "findings": [],
            "recommended_actions": ["Approve all pending work orders."],
            "needs_work_order": False,
            "confidence": 0.95,
            "unknowns": [],
            "ai_generated": True,
        }
    findings, actions, unknowns = [], [], []
    sensor = next(
        (e["data"] for e in evidence if e["name"] == "analyze_sensor" and not e["is_error"]), None
    )
    docs = next(
        (e["data"] for e in evidence if e["name"] == "search_docs" and not e["is_error"]), []
    )
    wos = next(
        (e["data"] for e in evidence if e["name"] == "get_work_orders" and not e["is_error"]), None
    )
    if sensor:
        findings.append(
            {
                "statement": f"{sensor['signal']} on {sensor['asset_id']} is {sensor['status']}: "
                f"last {sensor['last_value']} {sensor['unit']}, trend {sensor['trend_per_day']:+} "
                f"{sensor['unit']}/day since {sensor['change_start']}.",
                "citations": [sensor["source_id"]],
            }
        )
    else:
        unknowns.append("No sensor analysis was available.")
    for hit in docs[:3]:
        findings.append(
            {
                "statement": _first_sentence_with(hit["text"], _keywords(question)),
                "citations": [hit["source_id"]],
            }
        )
        for line in _sentences(hit["text"]):
            line = line.lstrip("0123456789. ")
            if re.match(r"^(Check|Take|Compare|Read|Do not|Switch)", line) and line not in actions:
                actions.append(line)
    if wos:
        similar = [
            w
            for w in wos["work_orders"]
            if "temperature" in w["problem"].lower() or "vibration" in w["problem"].lower()
        ]
        if similar:
            w = similar[0]
            findings.append(
                {
                    "statement": f"Similar past case {w['id']}: {w['cause']}",
                    "citations": [wos["source_id"]],
                }
            )
    abnormal = bool(sensor and sensor["status"] != "normal")
    conf = 0.3 + 0.3 * bool(sensor) + 0.2 * bool(docs) + 0.1 * bool(wos)
    unknowns.append("Redundant sensor reading not available to the assistant.")
    summary = (
        f"{asset}: "
        + (findings[0]["statement"] if findings else "no evidence found.")
        + (
            " Likely cause per manual and history: reduced cooling water flow."
            if abnormal and wos
            else ""
        )
    )
    return {
        "asset_id": asset,
        "summary": summary,
        "findings": findings,
        "recommended_actions": _rank(actions, _keywords(question))[:4]
        or ["Ask a maintenance engineer."],
        "needs_work_order": abnormal,
        "confidence": round(min(conf, 0.9), 2),
        "unknowns": unknowns,
        "ai_generated": True,
    }


def _judge(question: str, evidence, hijacked) -> dict:
    """A deliberately simple judge: passes if every expected fact appears as text and there is no
    negation word. It cannot handle paraphrases, wrong units or wrong context. Script 14 measures it."""
    facts = re.findall(r"^- (.+)$", question.split("ANSWER:")[0], re.M)
    answer = question.split("ANSWER:")[-1].lower()
    hits = sum(f.lower() in answer for f in facts)
    score = hits / len(facts) if facts else 0.0
    rest = answer
    for f in facts:
        rest = rest.replace(f.lower(), " ")
    negated = bool(re.search(r"\bnot\b|\bnever\b|^\s*no\b|\bno,", rest))  # crude negation check
    passed = score == 1.0 and not negated
    return {
        "passed": passed,
        "score": round(0.2 + 0.75 * score * (0.4 if negated else 1.0), 2),
        "reason": f"{hits}/{len(facts)} expected facts found; negation={'yes' if negated else 'no'}.",
    }
