"""Scripted transcript of the MCP server, for `make mcp-demo` and the README.

Boots a real FastAPI server (`pipeline/api.py`) on a local port, points the
MCP server at it via `MCP_API_BASE_URL`, then runs a real in-process MCP
client session against `pipeline/mcp_server.py`'s tools -- the same round
trip a deployed MCP service makes to the deployed API service over the
cluster network, just with both processes local to this one demo run.
Prints each tool call and its result, so the README's transcript is captured
output, not hand-typed.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import httpx

_DEMO_API_PORT = 8010
# Must be set before pipeline.mcp_server is imported: it reads
# MCP_API_BASE_URL once, at module load, into a module-level constant.
os.environ.setdefault("MCP_API_BASE_URL", f"http://127.0.0.1:{_DEMO_API_PORT}")

import uvicorn

from mcp.shared.memory import create_connected_server_and_client_session

from pipeline.api import app as api_app
from pipeline.mcp_server import mcp

CALLS = [
    ("mcp_lookup_case", {"case_id": "case_001"}),
    ("mcp_what_if", {"case_id": "case_001", "feature": "age", "new_value": "80"}),
    ("mcp_query_model_card", {"question": "what is the AUC?"}),
]


def _start_api_server() -> uvicorn.Server:
    """Run pipeline/api.py's FastAPI app on a real local port in a
    background thread, so the MCP tools below reach it over real HTTP,
    not an import."""
    config = uvicorn.Config(
        api_app, host="127.0.0.1", port=_DEMO_API_PORT, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{_DEMO_API_PORT}/health", timeout=0.5)
            return server
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("API server did not start in time")


async def run() -> None:
    server = _start_api_server()
    print(f"$ uvicorn pipeline.api:app --port {_DEMO_API_PORT}  "
          f"(background, for this demo)")
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        print(f"$ python -m pipeline.mcp_server  (stdio, {len(tools.tools)} tools, "
              f"calling http://127.0.0.1:{_DEMO_API_PORT})")
        for name, arguments in CALLS:
            print(f"\n> call {name}({json.dumps(arguments)})")
            result = await client.call_tool(name, arguments)
            payload = json.loads(result.content[0].text)
            print(json.dumps(payload, indent=2)[:400])
    server.should_exit = True


if __name__ == "__main__":
    asyncio.run(run())
