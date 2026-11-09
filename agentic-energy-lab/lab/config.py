"""Shared settings. Every script reads its mode from here.

LAB_MODE=mock (default)  -> no network, no cost, a fake LLM made of rules.
LAB_MODE=real            -> calls the real Claude API. Costs money.
A script can also be switched with the command-line flag --real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"
EVAL = ROOT / "eval"
RUNS = ROOT / "runs"  # local outputs (traces, memory, chroma); git-ignored

load_dotenv(ROOT / ".env")

_warned = False


def mode() -> str:
    """Return 'mock' or 'real'. The --real flag wins over the env variable."""
    if "--real" in sys.argv:
        return "real"
    return os.getenv("LAB_MODE", "mock").strip().lower()


def is_real() -> bool:
    return mode() == "real"


def model_name() -> str:
    return os.getenv("LLM_MODEL", "claude-opus-5-5")


def cost_warning() -> None:
    """Print a clear cost warning once per run in real mode."""
    global _warned
    if is_real() and not _warned:
        _warned = True
        print(
            "\n" + "!" * 72 + "\n"
            f"  REAL API MODE: calls {model_name()} and is billed to your API key.\n"
            "  Each script usually costs well under $0.10, the evaluation a few dollars at most.\n"
            "  Set LLM_MODEL=claude-haiku-5-5 in .env for a much cheaper model.\n" + "!" * 72 + "\n"
        )


def banner(title: str) -> None:
    print("=" * 72)
    print(f"{title}   [mode: {mode()}]")
    print("=" * 72)
