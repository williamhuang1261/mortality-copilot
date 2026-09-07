"""Round-trip tests over a real MCP client session against the server in
pipeline/mcp_server.py, which itself makes real HTTP requests to a live
pipeline/api.py FastAPI server on a real local TCP port.

No mocked transport at either layer: `create_connected_server_and_client_session`
(part of the `mcp` SDK's own testing surface) wires an in-memory MCP client
and server together and speaks the real MCP protocol between them; the MCP
server in turn reaches a real `uvicorn` process over a real socket for every
tool call, via `MCP_API_BASE_URL`. Every assertion compares the round-tripped
MCP result against calling the FastAPI service's own endpoints directly with
`httpx`, proving the MCP layer does not drift from the API it wraps -- and
proving the MCP server's tool calls actually reach the API service over the
network rather than importing `pipeline/tools.py` in-process.
"""

from __future__ import annotations

import json
import os
import threading
import time

import httpx
import pytest
import uvicorn

_TEST_API_PORT = 8011
os.environ["MCP_API_BASE_URL"] = f"http://127.0.0.1:{_TEST_API_PORT}"

from mcp.shared.memory import create_connected_server_and_client_session

from pipeline.api import app as api_app
from pipeline.mcp_server import mcp


def _tool_json(result) -> dict:
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


@pytest.fixture(scope="module")
def live_api_server():
    """A real uvicorn server for pipeline/api.py, bound to a real local
    port, running for the duration of this test module. The MCP server
    under test reaches it purely over HTTP via MCP_API_BASE_URL -- no
    import of pipeline.api or pipeline.tools on the MCP-server side."""
    config = uvicorn.Config(
        api_app, host="127.0.0.1", port=_TEST_API_PORT, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    base_url = f"http://127.0.0.1:{_TEST_API_PORT}"
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base_url}/health", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        raise RuntimeError("test API server did not start in time")
    yield base_url
    server.should_exit = True


@pytest.mark.anyio
async def test_list_tools_exposes_all_three(live_api_server):
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.list_tools()
        names = {tool.name for tool in result.tools}
    assert names == {"mcp_lookup_case", "mcp_query_model_card", "mcp_what_if"}


@pytest.mark.anyio
async def test_lookup_case_matches_live_api_call(live_api_server):
    direct = httpx.get(f"{live_api_server}/cases/case_001").json()

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mcp_lookup_case", {"case_id": "case_001"})
    via_mcp = _tool_json(result)

    assert via_mcp == direct


@pytest.mark.anyio
async def test_lookup_case_unknown_id_is_a_tool_error(live_api_server):
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mcp_lookup_case", {"case_id": "case_999"})
    assert result.isError
    assert "No case with id" in result.content[0].text


@pytest.mark.anyio
async def test_query_model_card_matches_live_api_call(live_api_server):
    direct = httpx.get(
        f"{live_api_server}/model-card", params={"question": "what is the AUC?"}
    ).json()["sections"]

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_query_model_card", {"question": "what is the AUC?"}
        )
    via_mcp = _tool_json(result)

    assert via_mcp == direct


@pytest.mark.anyio
async def test_what_if_matches_live_api_call(live_api_server):
    direct = httpx.post(
        f"{live_api_server}/what-if",
        json={"case_id": "case_001", "feature": "age", "new_value": "80"},
    ).json()

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_what_if",
            {"case_id": "case_001", "feature": "age", "new_value": "80"},
        )
    via_mcp = _tool_json(result)

    assert via_mcp["base_risk"] == pytest.approx(direct["base_risk"])
    assert via_mcp["new_risk"] == pytest.approx(direct["new_risk"])
    assert via_mcp["risk_delta_pct_points"] == pytest.approx(
        direct["risk_delta_pct_points"]
    )


@pytest.mark.anyio
async def test_what_if_categorical_feature_is_not_parsed_as_a_number(live_api_server):
    direct = httpx.post(
        f"{live_api_server}/what-if",
        json={"case_id": "case_001", "feature": "smoker", "new_value": "current"},
    ).json()

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_what_if",
            {"case_id": "case_001", "feature": "smoker", "new_value": "current"},
        )
    via_mcp = _tool_json(result)

    assert via_mcp["new_risk"] == pytest.approx(direct["new_risk"])


@pytest.fixture
def anyio_backend():
    return "asyncio"
