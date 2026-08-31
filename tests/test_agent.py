"""Tests for the agent's multi-turn state and fail-closed LLM path.

No Ollama needed: the deterministic-dispatch tests exercise the same parser
the CLI uses, and the fail-closed test relies on the real, always-true fact
in CI that nothing is listening on localhost:11434 -- so `--llm` genuinely
fails to connect and the fallback path is what actually runs, not a mock of
one.
"""

from __future__ import annotations

import pytest

from pipeline.agent import Session, dispatch, parse_utterance, run_turn
from pipeline.tools import ToolError

CASES = [
    {
        "case_id": "case_001",
        "predicted_risk_36mo": 0.04628,
        "risk_decile": 9,
        "features": {
            "age": 67, "sex": "male", "bmi": 28.8, "sbp": 133.3, "dbp": 81.3,
            "smoker": "never", "diabetes": "yes", "prior_chd": "no",
            "prior_cancer": "no", "hdl": 57, "hba1c": 8.6,
            "income_ratio": 2.04, "income_ratio_imputed": False,
        },
        "top_drivers": [
            {"variable": "age", "label": "Age", "value": "67",
             "statement": "Age = 67", "contribution_log_odds": 1.34,
             "direction": "increases risk", "relative_to": "cohort mean"},
        ],
    },
    {
        "case_id": "case_002",
        "predicted_risk_36mo": 0.10,
        "risk_decile": 10,
        "features": {
            "age": 71, "sex": "female", "bmi": 31.2, "sbp": 158.0, "dbp": 82.0,
            "smoker": "current", "diabetes": "no", "prior_chd": "no",
            "prior_cancer": "no", "hdl": 41.0, "hba1c": 7.4,
            "income_ratio": 1.15, "income_ratio_imputed": False,
        },
        "top_drivers": [],
    },
]

MODEL_CARD = {
    "validation": {"metrics": [{"Model": "Logistic GLM", "AUC": "0.853"}]},
    "cohort": {"n": 4906},
    "coefficients": {
        "logistic_glm": [
            {"term": "(Intercept)", "estimate_log_odds": -8.35214},
            {"term": "age", "estimate_log_odds": 0.07634},
            {"term": "hba1c", "estimate_log_odds": 0.00119},
            {"term": "smokercurrent", "estimate_log_odds": 0.57854},
            {"term": "smokerformer", "estimate_log_odds": 0.31443},
        ]
    },
}


def new_session() -> Session:
    return Session(cases=CASES, model_card=MODEL_CARD)


# --------------------------------------------------------------- dispatcher

def test_lookup_by_explicit_case_number_sets_session_state():
    session = new_session()
    reply = dispatch(session, "case 1")
    assert "case_001" in reply
    assert session.current_case_id == "case_001"


def test_what_if_uses_session_state_when_case_not_repeated():
    session = new_session()
    dispatch(session, "case 1")
    reply = dispatch(session, "what if age were 80")
    assert "case_001" in reply
    assert "67" in reply and "80" in reply


def test_what_if_with_no_case_in_context_raises():
    session = new_session()
    with pytest.raises(ToolError, match="No case in context"):
        dispatch(session, "what if age were 80")


def test_what_if_percentage_and_categorical_dispatch():
    session = new_session()
    dispatch(session, "case 1")
    pct_reply = dispatch(session, "raise hba1c by 20%")
    assert "hba1c" in pct_reply
    cat_reply = dispatch(session, "what if smoker is current")
    assert "'never' to 'current'" in cat_reply


def test_switching_case_number_mid_conversation():
    session = new_session()
    dispatch(session, "case 1")
    dispatch(session, "case 2")
    assert session.current_case_id == "case_002"
    reply = dispatch(session, "what if age were 90")
    assert "case_002" in reply


def test_model_card_question_dispatches():
    session = new_session()
    reply = dispatch(session, "what's the AUC?")
    assert "0.853" in reply


def test_unparseable_what_if_raises_helpful_error():
    session = new_session()
    dispatch(session, "case 1")
    with pytest.raises(ToolError, match="Say how"):
        dispatch(session, "what if age changes")


def test_parse_utterance_is_pure_and_does_not_mutate_session():
    session = new_session()
    dispatch(session, "case 1")
    before = session.current_case_id
    parse_utterance(session, "what if age were 99")
    assert session.current_case_id == before


# --------------------------------------------------------- fail-closed LLM

def test_llm_path_falls_back_when_ollama_unreachable():
    """No Ollama server runs in CI, so this hits a real connection failure
    -- not a mocked one -- and must still produce the correct answer via
    the deterministic dispatcher, never raise."""
    session = new_session()
    reply = run_turn(session, "case 1", use_llm=True, model="llama3.2:3b")
    assert "fallback" in reply
    assert "case_001" in reply
    assert session.current_case_id == "case_001"


def test_llm_path_without_model_uses_dispatcher_directly():
    session = new_session()
    reply = run_turn(session, "case 1", use_llm=False, model=None)
    assert "deterministic dispatcher" in reply
    assert "fallback" not in reply
