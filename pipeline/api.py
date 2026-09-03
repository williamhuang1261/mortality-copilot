"""A FastAPI service over the agent's tools.

    uvicorn pipeline.api:app --reload

A third additive interface over `pipeline/tools.py`'s pure functions,
alongside the CLI agent (`pipeline/agent.py`) and the MCP server
(`pipeline/mcp_server.py`). No tool logic is reimplemented here -- every
endpoint delegates straight into the same functions those two already
call. Every request and response is a typed Pydantic model, not a bare
dict, so FastAPI's generated OpenAPI schema is real.

`GET /cases/{case_id}` reads from PostgreSQL when `DATABASE_URL` is set in
the environment (see `pipeline/db.py`) and falls back to the JSON artifact
otherwise -- the same fail-closed-to-a-simpler-path pattern `copilot.py`'s
Ollama fallback and `agent.py`'s dispatcher fallback already use.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.agent import load_data
from pipeline.tools import ToolError
from pipeline.tools import lookup_case as _lookup_case
from pipeline.tools import query_model_card as _query_model_card
from pipeline.tools import what_if as _what_if

app = FastAPI(
    title="mortality-copilot API",
    description=(
        "REST interface over the mortality-risk copilot's tools: case "
        "lookup, model-card queries and what-if recomputation."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------- data models


class CaseResponse(BaseModel):
    case_id: str
    predicted_risk_36mo: float
    risk_decile: int
    features: dict[str, Any]
    top_drivers: list[dict[str, Any]]


class ModelCardResponse(BaseModel):
    sections: dict[str, Any]


class WhatIfRequest(BaseModel):
    case_id: str
    feature: str
    new_value: str


class WhatIfResponse(BaseModel):
    case_id: str
    feature: str
    old_value: Any
    new_value: Any
    base_risk: float
    new_risk: float
    risk_delta_pct_points: float


class HealthResponse(BaseModel):
    status: str
    database: str


# ------------------------------------------------------------------ routes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check -- the Kubernetes manifest's probes hit
    this endpoint. Reports whether DATABASE_URL is configured, but never
    fails the health check on the fallback path being active; the
    JSON-artifact fallback is a deliberate, working mode, not a degraded
    one."""
    return HealthResponse(
        status="ok",
        database="configured" if os.environ.get("DATABASE_URL") else "fallback",
    )


@app.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str) -> CaseResponse:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from sqlalchemy.exc import SQLAlchemyError

        from pipeline.db import read_case  # local import: no psycopg/SQLAlchemy
        # dependency for callers that never set DATABASE_URL.

        try:
            record = read_case(database_url, case_id)
        except SQLAlchemyError:
            # DATABASE_URL is set but the database itself is unreachable
            # (wrong credentials, network partition, Postgres down). Same
            # fail-closed-to-a-simpler-path pattern as the Ollama/RAG
            # fallbacks elsewhere in this project: fall back to the JSON
            # artifact rather than surface a raw 500 for a transient
            # infrastructure problem the caller cannot fix.
            record = None
        if record is not None:
            return CaseResponse(**record)

    cases, _ = load_data()
    try:
        case = _lookup_case(cases, case_id)
    except ToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CaseResponse(
        case_id=case["case_id"],
        predicted_risk_36mo=case["predicted_risk_36mo"],
        risk_decile=case["risk_decile"],
        features=case["features"],
        top_drivers=case["top_drivers"],
    )


@app.get("/model-card", response_model=ModelCardResponse)
def get_model_card(question: str) -> ModelCardResponse:
    _, model_card = load_data()
    try:
        sections = _query_model_card(model_card, question)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelCardResponse(sections=sections)


@app.post("/what-if", response_model=WhatIfResponse)
def post_what_if(request: WhatIfRequest) -> WhatIfResponse:
    cases, model_card = load_data()
    parsed_value: object = request.new_value
    if request.feature not in {"sex", "smoker", "diabetes", "prior_chd",
                                "prior_cancer"}:
        try:
            parsed_value = float(request.new_value)
        except ValueError:
            pass
    try:
        result = _what_if(cases, model_card, request.case_id, request.feature,
                           parsed_value)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WhatIfResponse(
        case_id=result.case_id,
        feature=result.feature,
        old_value=result.old_value,
        new_value=result.new_value,
        base_risk=result.base_risk,
        new_risk=result.new_risk,
        risk_delta_pct_points=result.risk_delta_pct_points,
    )
