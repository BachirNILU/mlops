"""03 - Tool calling, by hand.

WHAT IT SHOWS
  The full message exchange when an LLM uses a tool:
    user question -> model returns a tool call (name + JSON args) -> YOUR code runs the function
    -> you send the result back -> model writes the answer from the result.
  The model never runs anything itself. Arguments are validated with Pydantic before running.

RUN
  python scripts/03_tool_calling.py          (mock, free)
  python scripts/03_tool_calling.py --real   (real API, about $0.03)

EXPECTED OUTPUT (mock)
  The tool schema, then 4 messages: user, assistant(tool_call analyze_sensor T2),
  tool(result JSON with trend about +0.85 degC/day), assistant(answer citing the tool).

WHAT IT DOES NOT SHOW
  Several tools, loops of several steps, or errors. That is script 07.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.agent import SYSTEM_PROMPT  # noqa: E402
from lab.config import banner  # noqa: E402
from lab.llm import chat  # noqa: E402
from lab.tools import TOOLS  # noqa: E402


def main() -> None:
    banner("03 - Tool calling")
    tool = TOOLS["analyze_sensor"]
    print("Tool definition sent to the model:")
    print(
        json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.json_schema(),
            },
            indent=1,
        )[:800],
        "\n",
    )

    messages = [{"role": "user", "content": "Is the T2 bearing temperature trend abnormal?"}]
    print(f"[1 user]      {messages[0]['content']}")

    r = chat(messages, system=SYSTEM_PROMPT, tools=[tool])
    messages.append(r.as_message())
    for call in r.tool_calls:
        print(f"[2 assistant] tool call: {call.name}({call.args})   <- the model only ASKS")
        result = tool.run(call.args)  # validates args with Pydantic, then runs plain Python
        print(f"[3 tool]      result: {json.dumps(result)[:300]}...")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result),
            }
        )

    r2 = chat(messages, system=SYSTEM_PROMPT, tools=[tool])
    print(f"[4 assistant] {r2.text}")
    print(f"\nTotal cost: ${r.cost_usd + r2.cost_usd:.5f} (two LLM calls)")
    print(
        "Take-away: numbers come from your code, not from the model. Validate the arguments it sends."
    )


if __name__ == "__main__":
    main()
