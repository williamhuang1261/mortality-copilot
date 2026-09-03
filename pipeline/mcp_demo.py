"""Scripted transcript of the MCP server, for `make mcp-demo` and the README.

Runs a real in-process MCP client session against `pipeline/mcp_server.py`
(the same connection helper `tests/test_mcp_server.py` uses) and prints each
tool call and its result, so the README's transcript is captured output,
not hand-typed.
"""

from __future__ import annotations

import asyncio
import json

from mcp.shared.memory import create_connected_server_and_client_session

from pipeline.mcp_server import mcp

CALLS = [
    ("mcp_lookup_case", {"case_id": "case_001"}),
    ("mcp_what_if", {"case_id": "case_001", "feature": "age", "new_value": "80"}),
    ("mcp_query_model_card", {"question": "what is the AUC?"}),
]


async def run() -> None:
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        print(f"$ python -m pipeline.mcp_server  (stdio, {len(tools.tools)} tools)")
        for name, arguments in CALLS:
            print(f"\n> call {name}({json.dumps(arguments)})")
            result = await client.call_tool(name, arguments)
            payload = json.loads(result.content[0].text)
            print(json.dumps(payload, indent=2)[:400])


if __name__ == "__main__":
    asyncio.run(run())
