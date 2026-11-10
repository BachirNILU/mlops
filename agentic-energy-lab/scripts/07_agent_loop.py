"""07 - The agent loop from scratch: state, memory, planning, stopping, error handling.

WHAT IT SHOWS
  Run A: the full Turbine Assistant question. The sensor tool is made to fail twice
         (simulated historian timeout): one automatic retry with backoff, then the error goes
         back to the LLM, which tries again. Final report with citations.
  Run B: the same question with a step limit of 2: the loop stops early, the report admits gaps.
         Memory from run A is recalled ("Notes from earlier sessions").
  Code: lab/agent.py (read run_agent from top to bottom).

RUN
  python scripts/07_agent_loop.py          (mock, free)
  python scripts/07_agent_loop.py --real   (real API, about $0.10-0.30)

EXPECTED OUTPUT (mock)
  Run A: ERROR on analyze_sensor, then success, 3 tools used, "stop: model finished", a report.
  Run B: "[memory] recalled 1 earlier note", "stop: step limit (2)", lower confidence.

WHAT IT DOES NOT SHOW
  Parallel tool calls, context window management for long runs, or multi-agent setups.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.agent import MEMORY_FILE, print_report, run_agent  # noqa: E402
from lab.config import banner  # noqa: E402

QUESTION = (
    "The T2 guide bearing temperature has been rising since Monday. Is this a known issue, "
    "what does the manual say, and what should we do?"
)


def main() -> None:
    banner("07 - Agent loop from scratch")
    MEMORY_FILE.unlink(missing_ok=True)  # start with empty long-term memory

    print("\nRUN A: analyze_sensor fails twice (simulated timeout)")
    a = run_agent(QUESTION, faults={"analyze_sensor": 2})
    print_report(a.report)
    print(
        f"\n  state: {a.state.step} steps, stop reason = {a.state.stop_reason!r}, "
        f"{len(a.state.evidence)} evidence items, tools = {[t['tool'] for t in a.state.tool_log]}"
    )
    print(f"  trace: {a.trace}")

    print("\nRUN B: step limit = 2, memory from run A available")
    b = run_agent(QUESTION, max_steps=2)
    print_report(b.report)
    print(
        "\nTake-away: an agent is a loop you control. Bound it (steps, cost), handle tool errors,"
    )
    print("and make it say what it does not know when it stops early.")


if __name__ == "__main__":
    main()
