"""02 - Structured output with Pydantic.

WHAT IT SHOWS
  1. A Pydantic model (TicketTriage) and the JSON schema generated from it.
  2. The LLM fills the schema for three operator messages; you get typed Python objects.
  3. What happens when the output is invalid: Pydantic raises, you retry with the error message.

RUN
  python scripts/02_structured_output.py          (mock, free)
  python scripts/02_structured_output.py --real   (real API, about $0.02)

EXPECTED OUTPUT (mock)
  The JSON schema, three triage objects, then "attempt 1 failed: ... validation error",
  and "attempt 2 OK".

WHAT IT DOES NOT SHOW
  In real mode the API uses constrained decoding (structured outputs), so invalid JSON is rare;
  the forced failure only happens in mock mode. Validation still matters: the schema guarantees
  the shape, not that the values are true.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import mock_llm  # noqa: E402
from lab.config import banner, is_real  # noqa: E402
from lab.llm import chat  # noqa: E402
from lab.schemas import TicketTriage  # noqa: E402
from pydantic import ValidationError  # noqa: E402

SYSTEM = "Classify operator messages from a hydropower plant. Fill every field."
MESSAGES = [
    "T2 bearing feels hot, display shows 66 degC and still climbing.",
    "Pump P-101 is noisier than usual since Tuesday.",
    "Smoke near the T1 excitation cabinet, breaker tripped!",
]


def triage(text: str, max_attempts: int = 3) -> TicketTriage:
    messages = [{"role": "user", "content": text}]
    for attempt in range(1, max_attempts + 1):
        try:
            r = chat(messages, system=SYSTEM, response_model=TicketTriage, max_tokens=500)
            print(f"  attempt {attempt} OK")
            return r.parsed
        except ValidationError as e:
            first = e.errors()[0]
            print(
                f"  attempt {attempt} failed: {e.error_count()} validation error(s), "
                f"e.g. {first['loc']}: {first['msg']}"
            )
            # feed the error back so the model can correct itself
            messages.append(
                {"role": "user", "content": f"Your output was invalid: {e}. Return valid JSON."}
            )
    raise RuntimeError("no valid output after retries")


def main() -> None:
    banner("02 - Structured output (Pydantic)")
    print("JSON schema sent to the model (generated from the Pydantic class):")
    print(json.dumps(TicketTriage.model_json_schema(), indent=1)[:700], "...\n")

    for text in MESSAGES:
        print(f"> {text}")
        t = triage(text)
        print(f"  -> {t!r}\n")

    if not is_real():
        print("Now forcing one invalid output (mock only) to see validation + retry:")
        mock_llm.FAIL_NEXT_STRUCTURED = True
        print(f"  -> {triage(MESSAGES[0])!r}")
    print(
        "\nTake-away: never pass raw LLM text to code. Validate against a schema; retry or stop on failure."
    )


if __name__ == "__main__":
    main()
