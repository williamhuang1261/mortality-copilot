"""Tests for pipeline/api.py's FastAPI endpoints.

Uses FastAPI's real TestClient (a real ASGI request through the app, not a
call to the route function directly) and checks every response against
calling pipeline/tools.py on the same artifacts.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from pipeline.agent import load_data
from pipeline.api import app
from pipeline.tools import lookup_case, query_model_card, what_if

client = TestClient(app)


def test_health_reports_fallback_when_no_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "fallback"}


def test_get_case_matches_direct_call(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cases, _ = load_data()
    direct = lookup_case(cases, "case_001")

    response = client.get("/cases/case_001")
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == direct["case_id"]
    assert body["predicted_risk_36mo"] == direct["predicted_risk_36mo"]
    assert body["risk_decile"] == direct["risk_decile"]
    assert body["features"] == direct["features"]
    assert body["top_drivers"] == direct["top_drivers"]


def test_get_case_falls_back_to_json_when_database_is_unreachable(monkeypatch):
    """DATABASE_URL is set but points nowhere real -- a real, unmocked
    connection failure, the same discipline as the Ollama fail-closed
    test (`tests/test_agent.py`)."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://nobody:nobody@localhost:1/does-not-exist",
    )
    cases, _ = load_data()
    direct = lookup_case(cases, "case_001")

    response = client.get("/cases/case_001")
    assert response.status_code == 200
    assert response.json()["predicted_risk_36mo"] == direct["predicted_risk_36mo"]


def test_get_case_unknown_id_is_404():
    response = client.get("/cases/case_999")
    assert response.status_code == 404
    assert "No case with id" in response.json()["detail"]


def test_get_model_card_matches_direct_call():
    _, model_card = load_data()
    direct = query_model_card(model_card, "what is the AUC?")

    response = client.get("/model-card", params={"question": "what is the AUC?"})
    assert response.status_code == 200
    assert response.json()["sections"] == direct


def test_what_if_matches_direct_call():
    cases, model_card = load_data()
    direct = what_if(cases, model_card, "case_001", "age", 80.0)

    response = client.post(
        "/what-if",
        json={"case_id": "case_001", "feature": "age", "new_value": "80"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["base_risk"] == direct.base_risk
    assert body["new_risk"] == pytest.approx(direct.new_risk)
    assert body["risk_delta_pct_points"] == pytest.approx(
        direct.risk_delta_pct_points
    )


def test_what_if_categorical_feature():
    cases, model_card = load_data()
    direct = what_if(cases, model_card, "case_001", "smoker", "current")

    response = client.post(
        "/what-if",
        json={"case_id": "case_001", "feature": "smoker", "new_value": "current"},
    )
    assert response.status_code == 200
    assert response.json()["new_risk"] == pytest.approx(direct.new_risk)


def test_what_if_out_of_scope_feature_is_400():
    response = client.post(
        "/what-if",
        json={"case_id": "case_001", "feature": "race_eth", "new_value": "x"},
    )
    assert response.status_code == 400
