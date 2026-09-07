"""An MCP server exposing the agent's tools over the Model Context Protocol.

    python -m pipeline.mcp_server

This is a second, additive service in front of `pipeline/api.py`'s FastAPI
service: each MCP tool makes a real HTTP request to that service's REST
endpoints (`GET /cases/{id}`, `GET /model-card`, `POST /what-if`) instead of
importing `pipeline/tools.py` or reading `artifacts/` directly. No tool logic
is duplicated here or in `api.py` -- both eventually call the same
`pipeline/tools.py` functions, but this process only ever reaches them over
the network, through the API service. `pipeline/agent.py`'s Ollama
`tools=[...]` path is untouched and stays the default local entry point;
this server exists so a generic MCP client (Claude Desktop, an MCP
inspector, another agent framework) can reach the same three tools over a
standard protocol, backed by the same deployable API service used elsewhere.

The API service's base URL is configurable via `MCP_API_BASE_URL` (default
`http://localhost:8000`, matching `make api-serve`). In Kubernetes
(`k8s/mcp-*.yaml`) it is set to the API Service's in-cluster DNS name, so the
two Deployments talk to each other over the cluster network.

Transport defaults to stdio, the standard local-MCP-server transport used
when a client (Claude Desktop, an MCP inspector) launches this process and
talks to it over its own stdin/stdout -- no port to bind, no auth to
configure. Set `MCP_TRANSPORT=streamable-http` (plus `MCP_HOST`/`MCP_PORT`)
to run it as a standalone network service instead, which is how the
Kubernetes Deployment in `k8s/mcp-deployment.yaml` runs it.
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE_URL = os.environ.get("MCP_API_BASE_URL", "http://localhost:8000")

mcp = FastMCP(
    "mortality-copilot",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8001")),
)


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=10.0)


def _error_detail(response: httpx.Response, fallback: str) -> str:
    try:
        return str(response.json().get("detail", fallback))
    except ValueError:
        return fallback


@mcp.tool()
def mcp_lookup_case(case_id: str) -> dict:
    """Return the full case record for `case_id` (e.g. "case_017"), fetched
    from the FastAPI service's `GET /cases/{case_id}` endpoint."""
    with _client() as client:
        response = client.get(f"/cases/{case_id}")
    if response.status_code == 404:
        raise ValueError(_error_detail(response, "case not found"))
    response.raise_for_status()
    return response.json()


@mcp.tool()
def mcp_query_model_card(question: str) -> dict:
    """Return the model-card section(s) matching a question about the
    fitted mortality model (cohort, validation metrics, predictors,
    limitations, coefficients, provenance), fetched from the FastAPI
    service's `GET /model-card` endpoint."""
    with _client() as client:
        response = client.get("/model-card", params={"question": question})
    if response.status_code == 400:
        raise ValueError(_error_detail(response, "no matching section"))
    response.raise_for_status()
    return response.json()["sections"]


@mcp.tool()
def mcp_what_if(case_id: str, feature: str, new_value: str) -> dict:
    """Recompute `case_id`'s predicted 36-month mortality risk with one
    feature changed to `new_value`, using the fitted GLM's own log-odds
    coefficients. `new_value` is a number for a continuous feature (age,
    bmi, sbp, dbp, hdl, hba1c, income_ratio) or a category label for a
    categorical one (sex, smoker, diabetes, prior_chd, prior_cancer);
    parsing happens in the FastAPI service's `POST /what-if` endpoint."""
    with _client() as client:
        response = client.post(
            "/what-if",
            json={"case_id": case_id, "feature": feature, "new_value": new_value},
        )
    if response.status_code == 400:
        raise ValueError(_error_detail(response, "invalid what-if request"))
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
