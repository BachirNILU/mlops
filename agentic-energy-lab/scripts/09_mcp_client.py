"""09b - MCP client: discover tools from the MCP server and let the agent use them.

WHAT IT SHOWS
  1. The client starts 09_mcp_server.py as a subprocess and connects over stdio.
  2. list_tools(): the server describes its tools (name, description, JSON schema).
  3. call_tool(): a direct call, like the LLM would request.
  4. A small agent loop where the tools come from MCP, not from local Python imports.
     The agent code does not know or care where the tools run.

RUN
  python scripts/09_mcp_client.py          (mock, free)
  python scripts/09_mcp_client.py --real   (real API for the loop, about $0.05)

EXPECTED OUTPUT
  3 tools listed with their parameters, a direct analyze_sensor result for P-101
  (status "warning", vibration about 5.2 mm/s), then an agent answer that cites MCP tool results.

WHAT IT DOES NOT SHOW
  Connecting several servers at once, authentication, or remote (HTTP) transport.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.agent import SYSTEM_PROMPT  # noqa: E402
from lab.config import banner  # noqa: E402
from lab.llm import chat  # noqa: E402
from mcp import StdioServerParameters  # noqa: E402
from mcp.client.client import Client  # noqa: E402

SERVER = StdioServerParameters(
    command=sys.executable, args=[str(Path(__file__).with_name("09_mcp_server.py"))]
)


class McpTool:
    """Adapter: makes an MCP tool look like a lab Tool (name, description, json_schema)."""

    def __init__(self, t):
        self.name, self.description, self.schema = t.name, t.description or "", t.input_schema

    def json_schema(self) -> dict:
        return self.schema


def result_text(res) -> str:
    """MCP returns structured content (JSON) when the tool has a typed return, plus text blocks."""
    sc = getattr(res, "structured_content", None)
    if sc:
        # a list return value is wrapped as {"result": [...]} by the server
        return json.dumps(sc["result"] if list(sc) == ["result"] else sc)
    return "\n".join(getattr(c, "text", "") for c in res.content)


async def main() -> None:
    banner("09 - MCP client")
    async with Client(SERVER) as client:
        listed = await client.list_tools()
        print("Tools offered by the server:")
        for t in listed.tools:
            print(
                f"  - {t.name}({', '.join(t.input_schema.get('properties', {}))}): "
                f"{(t.description or '').splitlines()[0]}"
            )

        res = await client.call_tool(
            "analyze_sensor", {"asset_id": "P-101", "signal": "vibration_mm_s"}
        )
        print(
            f"\nDirect call analyze_sensor(P-101, vibration): {' '.join(result_text(res).split())[:250]}..."
        )

        tools = [McpTool(t) for t in listed.tools]
        messages = [
            {"role": "user", "content": "Is pump P-101 vibration normal? Has it happened before?"}
        ]
        for step in range(1, 6):
            r = chat(messages, system=SYSTEM_PROMPT, tools=tools)
            messages.append(r.as_message())
            if not r.tool_calls:
                break
            for call in r.tool_calls:
                res = await client.call_tool(
                    call.name, call.args
                )  # the tool runs in the server process
                print(
                    f"  step {step}: MCP call {call.name}({call.args}) -> is_error={res.is_error}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result_text(res),
                        "is_error": bool(res.is_error),
                    }
                )
        print(f"\nAgent answer: {r.text}")
    print(
        "\nTake-away: write a tool once as an MCP server; every MCP-capable agent or app can use it."
    )


if __name__ == "__main__":
    asyncio.run(main())
