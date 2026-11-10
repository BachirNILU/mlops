"""08 - The same agent as a LangGraph graph, with checkpoints and a human approval pause.

WHAT IT SHOWS
  1. The agent loop as an explicit graph: guard -> plan <-> act -> report -> approve -> create_wo.
     (The graph is printed as Mermaid text: paste it into any Mermaid viewer.)
  2. A checkpointer saves the state after every node (thread_id = one conversation).
  3. interrupt(): the graph PAUSES before the write action, waits for a human, then resumes
     from the saved checkpoint with Command(resume=...).

RUN
  python scripts/08_langgraph_agent.py               (asks you y/N at the approval step)
  APPROVAL=auto-yes python scripts/08_langgraph_agent.py
  python scripts/08_langgraph_agent.py --real        (real API, about $0.10-0.30)

EXPECTED OUTPUT (mock)
  Mermaid graph text, the node-by-node progress, "PAUSED: waiting for approval",
  then a proposed work order (if approved) and the number of saved checkpoints.

WHAT IT DOES NOT SHOW
  A persistent checkpointer (SQLite/Postgres) that survives a restart, streaming tokens to a UI,
  or sub-graphs. Checked against langgraph==1.2.20.
"""

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import guards  # noqa: E402
from lab.agent import (  # noqa: E402
    REPORT_REQUEST,
    SYSTEM_PROMPT,
    FaultInjector,
    print_report,
    run_tool,
)
from lab.config import banner  # noqa: E402
from lab.llm import chat  # noqa: E402
from lab.schemas import Report  # noqa: E402
from lab.tools import READ_ONLY, TOOLS  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

MAX_STEPS = 8


class State(TypedDict, total=False):
    question: str
    messages: Annotated[list, operator.add]  # reducer: new messages are appended, not replaced
    evidence: dict
    step: int
    blocked: str
    report: dict
    approved: bool
    work_order: dict


# ---------------------------------------------------------------- nodes (each returns a state update)


def guard(state: State) -> dict:
    g = guards.check_input(state["question"])
    print(f"[guard]   {'ok' if g.ok else 'BLOCKED: ' + g.reason}")
    return {
        "blocked": "" if g.ok else g.reason,
        "messages": [{"role": "user", "content": state["question"]}],
        "step": 0,
        "evidence": {},
    }


def plan(state: State) -> dict:
    r = chat(state["messages"], system=SYSTEM_PROMPT, tools=READ_ONLY)
    print(f"[plan]    step {state['step'] + 1}: {[c.name for c in r.tool_calls] or 'final answer'}")
    return {"messages": [r.as_message()], "step": state["step"] + 1}


def act(state: State) -> dict:
    last = state["messages"][-1]
    new_msgs, evidence = [], dict(state["evidence"])
    for call in last["tool_calls"]:
        result, is_err = run_tool(TOOLS[call["name"]], call["args"], FaultInjector())
        if call["name"] == "search_docs" and not is_err:
            hits, dropped = guards.filter_retrieved(json.loads(result))
            result = json.dumps(hits)
        if not is_err:
            data = json.loads(result)
            evidence.update(
                {h["source_id"]: h["text"] for h in data}
                if isinstance(data, list)
                else {data["source_id"]: result}
            )
        print(f"[act]     {call['name']} -> {'ERROR' if is_err else 'ok'}")
        new_msgs.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": result,
                "is_error": is_err,
            }
        )
    return {"messages": new_msgs, "evidence": evidence}


def report(state: State) -> dict:
    r = chat(
        state["messages"] + [{"role": "user", "content": REPORT_REQUEST}],
        system=SYSTEM_PROMPT,
        response_model=Report,
    )
    problems = guards.check_grounding(r.parsed, state["evidence"])
    print(f"[report]  confidence={r.parsed.confidence}, grounding problems={len(problems)}")
    return {"report": r.parsed.model_dump()}


def approve(state: State) -> dict:
    rep = state["report"]
    proposal = {
        "asset_id": rep["asset_id"],
        "priority": "medium",
        "description": "; ".join(rep["recommended_actions"][:2]),
    }
    # The graph stops here. The state is saved by the checkpointer. Whatever is passed to
    # Command(resume=...) later becomes the return value of interrupt().
    decision = interrupt({"action": "create_work_order", "proposal": proposal})
    return {"approved": bool(decision)}


def create_wo(state: State) -> dict:
    rep = state["report"]
    wo = TOOLS["create_work_order"].run(
        {
            "asset_id": rep["asset_id"],
            "priority": "medium",
            "description": "; ".join(rep["recommended_actions"][:2]),
        }
    )
    print(f"[write]   {wo['id']} {wo['status']}")
    return {"work_order": wo}


# ---------------------------------------------------------------- routing (conditional edges)


def after_guard(state: State) -> str:
    return END if state["blocked"] else "plan"


def after_plan(state: State) -> str:
    has_calls = bool(state["messages"][-1].get("tool_calls"))
    return "act" if has_calls and state["step"] < MAX_STEPS else "report"  # stop rules


def after_report(state: State) -> str:
    return "approve" if state["report"]["needs_work_order"] else END


def after_approve(state: State) -> str:
    return "create_wo" if state["approved"] else END


def build():
    g = StateGraph(State)
    for name, fn in [
        ("guard", guard),
        ("plan", plan),
        ("act", act),
        ("report", report),
        ("approve", approve),
        ("create_wo", create_wo),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, ["plan", END])
    g.add_conditional_edges("plan", after_plan, ["act", "report"])
    g.add_edge("act", "plan")
    g.add_conditional_edges("report", after_report, ["approve", END])
    g.add_conditional_edges("approve", after_approve, ["create_wo", END])
    g.add_edge("create_wo", END)
    return g.compile(checkpointer=InMemorySaver())


def main() -> None:
    banner("08 - LangGraph agent")
    graph = build()
    print("Graph as Mermaid:\n" + graph.get_graph().draw_mermaid())

    config = {"configurable": {"thread_id": "shift-2026-11-01-T2"}}
    question = "T2 guide bearing temperature is rising. Known issue? What does the manual say?"
    result = graph.invoke({"question": question}, config)

    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        print(f"\nPAUSED: waiting for approval of {pending['action']}: {pending['proposal']}")
        ok = guards.ask_human_approval(pending["action"], pending["proposal"])
        result = graph.invoke(Command(resume=ok), config)  # resumes from the checkpoint

    print_report(Report(**result["report"]) if result.get("report") else None)
    print(f"\nwork order: {result.get('work_order')}")
    print(f"checkpoints saved for this thread: {len(list(graph.get_state_history(config)))}")
    print(
        "\nTake-away: a graph makes the loop explicit and testable; checkpoints make pause/resume"
    )
    print("and human approval possible without keeping a process waiting.")


if __name__ == "__main__":
    main()
