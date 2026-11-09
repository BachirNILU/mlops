"""11 - A two-agent crew with CrewAI (role-based multi-agent).

WHAT IT SHOWS
  Two agents with roles, each with a goal and a backstory, working in sequence:
    1. Sensor analyst  (has the analyze_sensor and get_work_orders tools)
    2. Report writer   (no tools; turns the analyst's notes into a short report)
  Tasks pass their output to the next task as context. CrewAI runs a ReAct-style loop per agent:
  Thought -> Action -> Observation -> ... -> Final Answer.

RUN  (CrewAI needs older pydantic/mcp versions than the rest of the lab: use its own venv)
  python -m venv .venv-crew && .venv-crew/bin/pip install -r requirements-crewai.txt
  .venv-crew/bin/python scripts/11_crewai.py           (mock, free)
  .venv-crew/bin/python scripts/11_crewai.py --real    (real API through CrewAI's Anthropic provider)

EXPECTED OUTPUT (mock)
  The analyst calls analyze_sensor then get_work_orders, the writer returns a 3-line report.

WHAT IT DOES NOT SHOW
  Hierarchical crews (a manager agent), CrewAI Flows, memory or knowledge sources.
  When to choose multi-agent: only when roles really need different tools, prompts or permissions.
  One agent with good tools is often simpler and cheaper. Checked against crewai==1.15.39;
  real mode was not run here.
"""

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OTEL_SDK_DISABLED", "true")  # CrewAI sends anonymous telemetry by default
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewai import LLM, Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from lab import tools as lab_tools  # noqa: E402
from lab.config import banner, cost_warning, is_real, model_name  # noqa: E402


@tool("analyze_sensor")
def analyze_sensor(asset_id: str, signal: str) -> str:
    """Trend, z-score and status for one sensor signal. asset_id: T1, T2 or P-101.
    signal: bearing_temp_c or vibration_mm_s."""
    return json.dumps(lab_tools.analyze_sensor(asset_id, signal))


@tool("get_work_orders")
def get_work_orders(asset_id: str) -> str:
    """Past repairs for one asset (T1, T2 or P-101)."""
    return json.dumps(lab_tools.get_work_orders(asset_id))


class MockCrewLLM(BaseLLM):
    """Fake LLM speaking CrewAI's ReAct text format. NOT a real model: see lab/mock_llm.py."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        text = (
            messages
            if isinstance(messages, str)
            else "\n".join(str(m.get("content", "")) for m in messages)
        )
        observations = re.findall(r"Observation:\s*(\{.*?\})\s*(?:\n|$)", text, re.S)
        if "analyze_sensor" in text and "Sensor analyst" in text:
            if '"source_id": "tool:analyze_sensor' not in text:
                return (
                    "Thought: I need the sensor trend first.\nAction: analyze_sensor\n"
                    'Action Input: {"asset_id": "T2", "signal": "bearing_temp_c"}'
                )
            if '"work_orders"' not in text:
                return (
                    "Thought: Now check if this happened before.\nAction: get_work_orders\n"
                    'Action Input: {"asset_id": "T2"}'
                )
            s = json.loads(next(o for o in observations if "analyze_sensor" in o))
            return (
                f"Thought: I now know the final answer\nFinal Answer: T2 bearing temperature is "
                f"{s['status']}: {s['last_value']} degC, trend {s['trend_per_day']:+} degC/day since "
                f"{s['change_start']}. Past case WO-2025-031 had the same pattern: blocked cooling "
                f"water strainer."
            )
        notes = re.search(r"T2 bearing temperature is .*?strainer\.", text, re.S)
        return (
            "Thought: I now know the final answer\nFinal Answer: REPORT T2 (AI-generated)\n"
            f"- Finding: {notes.group(0) if notes else 'no analyst notes found'}\n"
            "- Action: check strainer differential pressure and cooling water flow; take an oil sample.\n"
            "- Owner: shift engineer must approve any work order."
        )

    def supports_function_calling(self) -> bool:
        return False  # use the text ReAct format, easier to read in a demo

    def get_context_window_size(self) -> int:
        return 200_000


def main() -> None:
    banner("11 - CrewAI two-agent crew")
    if is_real():
        cost_warning()
        llm = LLM(model=f"anthropic/{model_name()}")  # needs ANTHROPIC_API_KEY
    else:
        llm = MockCrewLLM(model="mock")

    analyst = Agent(
        role="Sensor analyst",
        goal="Find abnormal trends in turbine sensor data and past cases.",
        backstory="Reliability engineer at a hydropower plant. Uses only tool results.",
        tools=[analyze_sensor, get_work_orders],
        llm=llm,
        verbose=True,
        max_iter=5,
    )
    writer = Agent(
        role="Report writer",
        goal="Write a short, cited report for the shift engineer.",
        backstory="Writes plain, short maintenance notes. Marks them as AI-generated.",
        llm=llm,
        verbose=True,
        max_iter=2,
    )
    t1 = Task(
        description="Analyse the T2 guide bearing temperature and check past work orders.",
        expected_output="Two sentences of findings with numbers.",
        agent=analyst,
    )
    t2 = Task(
        description="Write a 3-line report from the analyst's findings.",
        expected_output="Finding, action, owner.",
        agent=writer,
        context=[t1],
    )
    crew = Crew(agents=[analyst, writer], tasks=[t1, t2], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    print("\nFINAL OUTPUT:\n" + str(result))
    print(
        "\nTake-away: roles split prompts and tools. Each extra agent adds LLM calls, cost and failure points."
    )


if __name__ == "__main__":
    main()
