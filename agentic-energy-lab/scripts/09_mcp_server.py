"""09a - MCP server: expose the plant tools through the Model Context Protocol.

WHAT IT SHOWS
  The same three read-only tools (search_docs, analyze_sensor, get_work_orders), published once
  as an MCP server. Any MCP client (Claude Code, an IDE, LangGraph, Agent Framework...) can now
  discover and call them. The write tool is NOT exposed: least privilege at the protocol level.

RUN
  You normally do not run this file yourself: 09_mcp_client.py starts it as a subprocess
  and talks to it over stdin/stdout ("stdio" transport).
  To plug it into Claude Code:  claude mcp add turbine-tools -- python scripts/09_mcp_server.py

EXPECTED OUTPUT
  Nothing on screen (stdout carries the protocol messages).

WHAT IT DOES NOT SHOW
  Remote servers over HTTP, authentication (OAuth), resources and prompts.
  Checked against mcp==2.3.0, where FastMCP was renamed MCPServer (mcp.server.mcpserver).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import tools  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

server = MCPServer("turbine-tools")


@server.tool()
def search_docs(query: str, k: int = 4) -> list[dict]:
    """Search maintenance manuals, procedures, inspection reports and logs.
    Returns passages with a source_id to cite."""
    return tools.search_docs(query, k)


@server.tool()
def analyze_sensor(asset_id: str, signal: str, days: int = 14) -> dict:
    """Trend, z-score and status for one sensor signal.
    asset_id: T1, T2 or P-101. signal: bearing_temp_c or vibration_mm_s."""
    return tools.TOOLS["analyze_sensor"].run({"asset_id": asset_id, "signal": signal, "days": days})


@server.tool()
def get_work_orders(asset_id: str) -> dict:
    """Past repairs and inspections for one asset (T1, T2 or P-101)."""
    return tools.get_work_orders(asset_id)


if __name__ == "__main__":
    server.run()  # stdio transport by default
