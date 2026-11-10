"""01 - Prompting: the same question, three prompts.

WHAT IT SHOWS
  v1  bare question                         -> the model guesses (no source)
  v2  system prompt with rules              -> the model says "I don't know" instead of guessing
  v3  rules + documents in <document> tags  -> a short answer with a citation
  Then an unanswerable question with documents -> "I don't know" (the rule works both ways).
  For each call: tokens and the cost the call would have in real mode.

RUN
  python scripts/01_prompting.py          (mock, free)
  python scripts/01_prompting.py --real   (real API, about $0.01)

EXPECTED OUTPUT (mock)
  v1 prints a guess, v2 "I don't know...", v3 "The alarm limit is 70 degC ... [MAN-T-01]".

WHAT IT DOES NOT SHOW
  Prompt optimisation, few-shot example selection, or prompt caching. Real models answer with
  more nuance than the mock; the lesson is the structure of the prompt, not the wording.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.config import DOCS, banner  # noqa: E402
from lab.llm import chat  # noqa: E402

QUESTION = "What is the alarm limit for the guide bearing temperature?"
RULES = (
    "You are a maintenance assistant for a hydropower plant.\n"
    "Answer only from the documents given. Cite the document id in square brackets.\n"
    "If the documents do not contain the answer, say exactly: I don't know based on the provided documents."
)


def with_documents(question: str, doc_ids: list[str]) -> str:
    # Put data in clearly delimited blocks, and the question last. Tags help the model
    # separate data from instructions (and make prompt injection easier to spot).
    docs = "\n".join(
        f'<document id="{d}">{(DOCS / f"{d}.md").read_text()}</document>' for d in doc_ids
    )
    return f"{docs}\n\nQUESTION: {question}"


def show(label: str, system: str, user: str) -> None:
    print(f"\n--- {label} " + "-" * (66 - len(label)))
    print(f"SYSTEM: {system[:120] or '(none)'}{'...' if len(system) > 120 else ''}")
    print(f"USER:   {user[:160]}{'...' if len(user) > 160 else ''}")
    r = chat([{"role": "user", "content": user}], system=system, max_tokens=300)
    print(f"ANSWER: {r.text}")
    print(f"        tokens in/out = {r.input_tokens}/{r.output_tokens}, cost = ${r.cost_usd:.5f}")


def main() -> None:
    banner("01 - Prompting")
    show("v1: bare question", "", QUESTION)
    show("v2: rules, no documents", RULES, QUESTION)
    show("v3: rules + documents", RULES, with_documents(QUESTION, ["MAN-T-01", "MAN-T-02"]))
    show(
        "v3 on an unanswerable question",
        RULES,
        with_documents("Who manufactured the T2 generator?", ["MAN-T-01", "MAN-T-02"]),
    )
    print(
        "\nTake-away: rules + delimited context + 'say I don't know' turn guessing into grounded answers."
    )
    print(
        "Version your prompts (see PROMPT_VERSION in lab/agent.py): a prompt change is a code change."
    )


if __name__ == "__main__":
    main()
