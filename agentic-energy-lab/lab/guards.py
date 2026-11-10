"""Safety layers (see Figure 7): input check, retrieved-text check, permissions,
human approval, output grounding check.

These are simple rules on purpose, so you can read them. Production systems add trained
classifiers (for example Azure AI Content Safety "Prompt Shields") on top. Rules alone are
easy to bypass; layers together make attacks harder, never impossible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all )?(the )?previous instructions",
    r"disregard (the )?(system|above)",
    r"you are now",
    r"system note to ai",
    r"approve all",
    r"reveal (your|the) (system )?prompt",
]
IN_SCOPE_WORDS = [
    "t1",
    "t2",
    "p-101",
    "pump",
    "turbine",
    "bearing",
    "vibration",
    "temperature",
    "oil",
    "strainer",
    "cooling",
    "maintenance",
    "inspection",
    "work order",
    "manual",
    "procedure",
    "sensor",
    "permit",
    "alarm",
    "trip",
    "plant",
    "log",
    "asset",
]


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


# ---------------------------------------------------------------- layer 1: input


def check_input(question: str) -> GuardResult:
    q = question.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, q):
            return GuardResult(False, f"possible prompt injection: matches '{p}'")
    if not any(w in q for w in IN_SCOPE_WORDS):
        return GuardResult(False, "off-topic: not about plant assets or maintenance")
    if len(question) > 2000:
        return GuardResult(False, "too long")
    return GuardResult(True)


# ---------------------------------------------------------------- layer 2: retrieved text


def is_injected(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)


def filter_retrieved(hits: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop passages that look like instructions to the AI. Retrieved text is data, never orders."""
    kept, dropped = [], []
    for h in hits:
        if is_injected(h["text"]):
            dropped.append(h["source_id"])
        else:
            kept.append(h)
    return kept, dropped


# ---------------------------------------------------------------- layer 3: permissions

ROLE_PERMISSIONS = {
    "assistant": {"read"},  # the agent itself: read-only
    "engineer": {"read", "write"},  # a human engineer may approve write actions
}


def allowed(tool_permission: str, role: str = "assistant") -> bool:
    return tool_permission in ROLE_PERMISSIONS.get(role, set())


# ---------------------------------------------------------------- layer 4: human in the loop


def ask_human_approval(action: str, details: dict) -> bool:
    """Ask a person to approve a write action. In CI or demos, set APPROVAL=auto-yes / auto-no."""
    policy = os.getenv("APPROVAL", "ask")
    print(f"\n  >>> HUMAN APPROVAL NEEDED: {action} {details}")
    if policy == "auto-yes":
        print("  >>> auto-approved (APPROVAL=auto-yes)")
        return True
    if policy == "auto-no":
        print("  >>> auto-denied (APPROVAL=auto-no)")
        return False
    try:
        return input("  >>> Approve? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


# ---------------------------------------------------------------- layer 5: output grounding

NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def check_grounding(report, evidence: dict[str, str]) -> list[str]:
    """Return a list of problems. `evidence` maps source_id -> the text/JSON the agent actually saw.
    Checks: (1) every citation exists, (2) every finding has a citation,
    (3) every number in a finding appears in one of its cited sources."""
    problems = []
    for f in report.findings:
        if not f.citations:
            problems.append(f"uncited finding: {f.statement!r}")
        for c in f.citations:
            if c not in evidence:
                problems.append(f"citation not in evidence: {c}")
        cited_text = " ".join(evidence.get(c, "") for c in f.citations)
        for n in NUMBER.findall(f.statement):
            if n.lstrip("+") not in cited_text and str(float(n)) not in cited_text:
                problems.append(f"number {n} not found in cited sources: {f.statement!r}")
    return problems
