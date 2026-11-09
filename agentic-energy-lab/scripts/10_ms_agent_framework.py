"""10 - The same assistant in Microsoft Agent Framework; real mode runs on Azure AI Foundry.

WHAT IT SHOWS
  1. Tools declared with the @tool decorator. The write tool is declared with
     approval_mode="always_require" (the framework would pause for a human), but it is NOT given
     to this agent: least privilege. Script 12 shows the approval flow.
  2. An Agent = chat client + instructions + tools. agent.run() runs the tool loop for you
     (compare with the ~100 lines of lab/agent.py).
  3. Mock mode: a custom chat client (subclass of BaseChatClient) that wraps the lab's fake LLM,
     so it runs offline. Real mode: FoundryChatClient, a model deployed in your Azure AI Foundry project.

RUN
  python scripts/10_ms_agent_framework.py           (mock, free)
  python scripts/10_ms_agent_framework.py --real    (Azure AI Foundry; needs FOUNDRY_PROJECT_ENDPOINT,
                                                     FOUNDRY_MODEL and `az login`; billed by Azure)

EXPECTED OUTPUT (mock)
  Three tool calls made by the framework, then the agent's cited answer.

WHAT IT DOES NOT SHOW
  Agent Framework workflows (multi-agent graphs), hosted agents, or Foundry evaluations.
  Real mode is written against agent-framework-core 1.25.0 / agent-framework-foundry 1.16.1
  but was NOT run here (needs an Azure subscription).
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_framework import (  # noqa: E402
    Agent,
    BaseChatClient,
    ChatResponse,
    Content,
    Message,
    tool,
)
from agent_framework._tools import FunctionInvocationLayer  # noqa: E402
from lab import mock_llm, tools  # noqa: E402
from lab.agent import SYSTEM_PROMPT  # noqa: E402
from lab.config import banner, cost_warning, is_real  # noqa: E402

# ---------------------------------------------------------------- tools, Agent Framework style


@tool
def search_docs(query: Annotated[str, "What to look for in the maintenance documents."]) -> str:
    """Search maintenance manuals, procedures, inspection reports and logs."""
    return json.dumps(tools.search_docs(query, 4))


@tool
def analyze_sensor(
    asset_id: Annotated[str, "T1, T2 or P-101"],
    signal: Annotated[str, "bearing_temp_c or vibration_mm_s"],
) -> str:
    """Trend, z-score and status for one sensor signal of one asset."""
    return json.dumps(tools.TOOLS["analyze_sensor"].run({"asset_id": asset_id, "signal": signal}))


@tool
def get_work_orders(asset_id: Annotated[str, "T1, T2 or P-101"]) -> str:
    """Past repairs and inspections for one asset."""
    return json.dumps(tools.get_work_orders(asset_id))


@tool(approval_mode="always_require")  # the framework pauses and asks a human before running it
def create_work_order(asset_id: str, description: str, priority: str) -> str:
    """Propose a new work order."""
    return json.dumps(tools.create_work_order(asset_id, description, priority))


# ---------------------------------------------------------------- mock chat client (offline)


class MockChatClient(FunctionInvocationLayer, BaseChatClient):
    """Translates Agent Framework messages to the lab format and asks the fake LLM.
    FunctionInvocationLayer gives it the automatic tool-calling loop."""

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        lab_msgs, names = [], {}
        for m in messages:
            for c in m.contents:
                if c.type == "text":
                    lab_msgs.append({"role": m.role, "content": c.text})
                elif c.type == "function_call":
                    names[c.call_id] = c.name
                    lab_msgs.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": c.call_id, "name": c.name, "args": c.parse_arguments() or {}}
                            ],
                        }
                    )
                elif c.type == "function_result":
                    lab_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": c.call_id,
                            "name": names.get(c.call_id),
                            "content": c.result
                            if isinstance(c.result, str)
                            else json.dumps(c.result),
                        }
                    )
        available = [type("T", (), {"name": t.name}) for t in (options.get("tools") or [])]
        out = mock_llm.respond(options.get("instructions") or "", lab_msgs, available, None)
        if out.get("tool_calls"):
            call = out["tool_calls"][0]
            print(f"  [framework] model asked for {call['name']}({call['args']})")
            contents = [
                Content.from_function_call(call["id"], call["name"], arguments=call["args"])
            ]
        else:
            contents = [out["text"]]
        return ChatResponse(messages=[Message("assistant", contents)])


def make_client():
    if not is_real():
        return MockChatClient()
    cost_warning()
    from agent_framework.foundry import FoundryChatClient
    from azure.identity import AzureCliCredential

    # reads FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL (a deployment name) from the environment
    return FoundryChatClient(credential=AzureCliCredential())


async def main() -> None:
    banner("10 - Microsoft Agent Framework")
    agent = Agent(
        client=make_client(),
        instructions=SYSTEM_PROMPT,
        tools=[search_docs, analyze_sensor, get_work_orders],
    )
    question = "T2 guide bearing temperature is rising. Known issue? What does the manual say?"
    print(f"Q: {question}")
    result = await agent.run(question)
    print(f"\nA: {result.text}")
    print("\nTake-away: the framework runs the tool loop, tool schemas and approvals for you;")
    print("you still own the prompt, the tools, the guardrails and the evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
