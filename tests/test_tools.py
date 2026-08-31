"""Tests for the agent's tool functions.

No Ollama, no faiss, no torch: these are pure functions over small in-memory
fixtures, so they run with only the core requirements installed, same as
every other test in this repo.
"""

from __future__ import annotations

import math

import pytest

from pipeline.tools import (
    ToolError,
    lookup_case,
    lookup_case_by_number,
    query_model_card,
    what_if,
)

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
        "top_drivers": [],
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
    "purpose": "Educational demonstration. Not an underwriting system.",
    "outcome": {"definition": "All-cause mortality within 36 months"},
    "cohort": {"n": 4906, "deaths_36_months": 151},
    "predictors": {"used": ["age", "sex", "bmi"]},
    "validation": {"metrics": [{"Model": "Logistic GLM", "AUC": "0.853"}]},
    "limitations": ["Survey weights are not applied."],
    "provenance": {"source": "NHANES 2015-2016"},
    "coefficients": {
        "logistic_glm": [
            {"term": "(Intercept)", "estimate_log_odds": -8.35214},
            {"term": "age", "estimate_log_odds": 0.07634},
            {"term": "income_ratio", "estimate_log_odds": -0.28537},
            {"term": "bmi", "estimate_log_odds": -0.00247},
            {"term": "sbp", "estimate_log_odds": 0.00379},
            {"term": "dbp", "estimate_log_odds": -0.01007},
            {"term": "hdl", "estimate_log_odds": 0.00885},
            {"term": "hba1c", "estimate_log_odds": 0.00119},
            {"term": "sexmale", "estimate_log_odds": 0.70168},
            {"term": "smokercurrent", "estimate_log_odds": 0.57854},
            {"term": "smokerformer", "estimate_log_odds": 0.31443},
            {"term": "diabetesborderline", "estimate_log_odds": 0.08124},
            {"term": "diabetesyes", "estimate_log_odds": 0.48193},
            {"term": "prior_chdyes", "estimate_log_odds": 0.22506},
            {"term": "prior_canceryes", "estimate_log_odds": 0.61376},
        ]
    },
}


# ------------------------------------------------------------- case lookup

def test_lookup_case_found():
    assert lookup_case(CASES, "case_002")["risk_decile"] == 10


def test_lookup_case_not_found_raises():
    with pytest.raises(ToolError, match="No case with id"):
        lookup_case(CASES, "case_999")


def test_lookup_case_by_number():
    assert lookup_case_by_number(CASES, 1)["case_id"] == "case_001"


def test_lookup_case_by_number_out_of_range():
    with pytest.raises(ToolError, match="between 1 and 2"):
        lookup_case_by_number(CASES, 5)


# --------------------------------------------------------- model-card query

def test_query_model_card_matches_validation():
    result = query_model_card(MODEL_CARD, "how good is the model, what's the AUC?")
    assert "validation" in result
    assert result["validation"]["metrics"][0]["AUC"] == "0.853"


def test_query_model_card_matches_multiple_sections():
    result = query_model_card(MODEL_CARD,
                               "what predictors were used and were any dropped?")
    assert "predictors" in result


def test_query_model_card_no_match_raises():
    with pytest.raises(ToolError, match="No model-card section"):
        query_model_card(MODEL_CARD, "what's the weather today?")


# ----------------------------------------------------------------- what-if

def test_what_if_continuous_matches_manual_sigmoid():
    result = what_if(CASES, MODEL_CARD, "case_001", "age", 77)
    base_logit = math.log(0.04628 / (1 - 0.04628))
    expected_logit = base_logit + 0.07634 * (77 - 67)
    expected_risk = 1 / (1 + math.exp(-expected_logit))
    assert result.new_risk == pytest.approx(expected_risk, abs=1e-9)
    assert result.old_value == 67
    assert result.new_value == 77
    assert result.risk_delta_pct_points > 0


def test_what_if_continuous_negative_coefficient_lowers_risk():
    result = what_if(CASES, MODEL_CARD, "case_001", "income_ratio", 4.0)
    assert result.new_risk < result.base_risk


def test_what_if_categorical_swaps_dummy():
    result = what_if(CASES, MODEL_CARD, "case_001", "smoker", "current")
    base_logit = math.log(0.04628 / (1 - 0.04628))
    expected_logit = base_logit + (0.57854 - 0.0)  # never is the reference
    expected_risk = 1 / (1 + math.exp(-expected_logit))
    assert result.new_risk == pytest.approx(expected_risk, abs=1e-9)


def test_what_if_categorical_to_reference_level():
    result = what_if(CASES, MODEL_CARD, "case_002", "smoker", "never")
    base_logit = math.log(0.10 / 0.90)
    expected_logit = base_logit + (0.0 - 0.57854)  # current -> never
    expected_risk = 1 / (1 + math.exp(-expected_logit))
    assert result.new_risk == pytest.approx(expected_risk, abs=1e-9)


def test_what_if_unknown_categorical_level_raises():
    with pytest.raises(ToolError, match="must be one of"):
        what_if(CASES, MODEL_CARD, "case_001", "smoker", "vaper")


def test_what_if_out_of_scope_feature_raises():
    with pytest.raises(ToolError, match="not exported per-case"):
        what_if(CASES, MODEL_CARD, "case_001", "education", "college_grad")


def test_what_if_unknown_feature_raises():
    with pytest.raises(ToolError, match="Unknown feature"):
        what_if(CASES, MODEL_CARD, "case_001", "shoe_size", 10)
