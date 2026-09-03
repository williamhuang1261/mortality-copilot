"""An MCP server exposing the agent's tools over the Model Context Protocol.

    python -m pipeline.mcp_server

This is a second, additive interface over the same pure functions
`pipeline/agent.py`'s deterministic dispatcher and `--llm` Ollama path
already call (`pipeline/tools.py`). No tool logic is reimplemented here --
each MCP tool loads the same artifacts `agent.py.load_data()` reads and
delegates straight into `pipeline/tools.py`. `agent.py`'s own Ollama
`tools=[...]` path is untouched and stays the default entry point; this
server exists so a generic MCP client (Claude Desktop, an MCP inspector,
another agent framework) can reach the same three tools over a standard
protocol instead of Ollama's function-calling API specifically.

Transport is stdio, the standard local-MCP-server transport: an MCP client
launches this process and talks to it over its stdin/stdout, so there is no
network port to bind and no auth to configure.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pipeline.agent import load_data
from pipeline.tools import ToolError
from pipeline.tools import lookup_case as _lookup_case
from pipeline.tools import query_model_card as _query_model_card
from pipeline.tools import what_if as _what_if

mcp = FastMCP("mortality-copilot")


@mcp.tool()
def mcp_lookup_case(case_id: str) -> dict:
    """Return the full case record for `case_id` (e.g. "case_017")."""
    cases, _ = load_data()
    try:
        return _lookup_case(cases, case_id)
    except ToolError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def mcp_query_model_card(question: str) -> dict:
    """Return the model-card section(s) matching a question about the
    fitted mortality model (cohort, validation metrics, predictors,
    limitations, coefficients, provenance)."""
    _, model_card = load_data()
    try:
        return _query_model_card(model_card, question)
    except ToolError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def mcp_what_if(case_id: str, feature: str, new_value: str) -> dict:
    """Recompute `case_id`'s predicted 36-month mortality risk with one
    feature changed to `new_value`, using the fitted GLM's own log-odds
    coefficients. `new_value` is parsed as a number for a continuous
    feature (age, bmi, sbp, dbp, hdl, hba1c, income_ratio) or a category
    label for a categorical one (sex, smoker, diabetes, prior_chd,
    prior_cancer)."""
    cases, model_card = load_data()
    parsed_value: object = new_value
    if feature not in {"sex", "smoker", "diabetes", "prior_chd", "prior_cancer"}:
        try:
            parsed_value = float(new_value)
        except ValueError:
            pass
    try:
        result = _what_if(cases, model_card, case_id, feature, parsed_value)
    except ToolError as exc:
        raise ValueError(str(exc)) from exc
    return {
        "case_id": result.case_id,
        "feature": result.feature,
        "old_value": result.old_value,
        "new_value": result.new_value,
        "base_risk": result.base_risk,
        "new_risk": result.new_risk,
        "risk_delta_pct_points": result.risk_delta_pct_points,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
