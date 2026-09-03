"""Round-trip tests over a real MCP client session against the server in
pipeline/mcp_server.py.

No mocked transport: `create_connected_server_and_client_session` (part of
the `mcp` SDK's own testing surface) wires an in-memory client and server
together and speaks the real protocol between them. Every assertion
compares the round-tripped MCP result against calling `pipeline/tools.py`
directly on the same artifacts, proving the MCP layer does not drift from
the tools it wraps.
"""

from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from pipeline.agent import load_data
from pipeline.mcp_server import mcp
from pipeline.tools import lookup_case, query_model_card, what_if


def _tool_json(result) -> dict:
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_list_tools_exposes_all_three():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.list_tools()
        names = {tool.name for tool in result.tools}
    assert names == {"mcp_lookup_case", "mcp_query_model_card", "mcp_what_if"}


@pytest.mark.anyio
async def test_lookup_case_matches_direct_call():
    cases, _ = load_data()
    direct = lookup_case(cases, "case_001")

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mcp_lookup_case", {"case_id": "case_001"})
    via_mcp = _tool_json(result)

    assert via_mcp == direct


@pytest.mark.anyio
async def test_lookup_case_unknown_id_is_a_tool_error():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mcp_lookup_case", {"case_id": "case_999"})
    assert result.isError
    assert "No case with id" in result.content[0].text


@pytest.mark.anyio
async def test_query_model_card_matches_direct_call():
    _, model_card = load_data()
    direct = query_model_card(model_card, "what is the AUC?")

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_query_model_card", {"question": "what is the AUC?"}
        )
    via_mcp = _tool_json(result)

    assert via_mcp == direct


@pytest.mark.anyio
async def test_what_if_matches_direct_call():
    cases, model_card = load_data()
    direct = what_if(cases, model_card, "case_001", "age", 80.0)

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_what_if",
            {"case_id": "case_001", "feature": "age", "new_value": "80"},
        )
    via_mcp = _tool_json(result)

    assert via_mcp["base_risk"] == direct.base_risk
    assert via_mcp["new_risk"] == pytest.approx(direct.new_risk)
    assert via_mcp["risk_delta_pct_points"] == pytest.approx(
        direct.risk_delta_pct_points
    )


@pytest.mark.anyio
async def test_what_if_categorical_feature_is_not_parsed_as_a_number():
    cases, model_card = load_data()
    direct = what_if(cases, model_card, "case_001", "smoker", "current")

    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "mcp_what_if",
            {"case_id": "case_001", "feature": "smoker", "new_value": "current"},
        )
    via_mcp = _tool_json(result)

    assert via_mcp["new_risk"] == pytest.approx(direct.new_risk)


@pytest.fixture
def anyio_backend():
    return "asyncio"
