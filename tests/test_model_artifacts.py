"""Checks on the committed model artifacts.

These are the files the copilot reads and cites, so a defect here becomes a
confident false statement in a generated note. Two of these tests exist because
the first version of R/05_export.R got them wrong: it labelled drivers by
design-matrix column, so a female participant showed a "Sex: male" driver, and
it presented an imputed income as though it had been measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "artifacts" / "model_card.json"
CASES = ROOT / "artifacts" / "cases.json"


@pytest.fixture(scope="module")
def card():
    if not CARD.exists():
        pytest.skip("artifacts/model_card.json not generated yet (run `make models`)")
    return json.loads(CARD.read_text())


@pytest.fixture(scope="module")
def cases():
    if not CASES.exists():
        pytest.skip("artifacts/cases.json not generated yet (run `make models`)")
    return json.loads(CASES.read_text())


# ------------------------------------------------------------- model card

@pytest.mark.parametrize("key", [
    "purpose", "outcome", "cohort", "predictors",
    "validation", "coefficients", "limitations", "provenance",
])
def test_model_card_has_section(card, key):
    assert key in card and card[key]


def test_model_card_carries_the_disclaimer(card):
    purpose = card["purpose"].lower()
    assert "not an underwriting system" in purpose
    assert "must not be used" in purpose


def test_model_card_states_the_survey_weight_limitation(card):
    joined = " ".join(card["limitations"]).lower()
    assert "survey weights are not applied" in joined
    assert "not the u.s. population" in joined


def test_model_card_explains_the_36_month_endpoint(card):
    why = card["outcome"]["why_36_months"].lower()
    assert "censoring" in why and "2019" in why


def test_every_model_reports_a_plausible_auc(card):
    metrics = card["validation"]["metrics"]
    assert len(metrics) == 3
    for row in metrics:
        auc = float(row["AUC"])
        assert 0.5 < auc < 1.0, f"{row['Model']} AUC {auc} is not plausible"


def test_proportional_hazards_result_is_reported_either_way(card):
    ph = card["validation"]["proportional_hazards"]
    assert ph["test"] == "cox.zph"
    assert "violations" in ph
    assert ph["interpretation"]


def test_coefficients_are_present_for_both_models(card):
    assert len(card["coefficients"]["logistic_glm"]) > 10
    assert len(card["coefficients"]["cox_proportional_hazards"]) > 10


# ------------------------------------------------------------------ cases

def test_case_file_shape(cases):
    assert cases["n_cases"] == len(cases["cases"]) == 50
    assert cases["horizon_months"] == 36


def test_every_case_has_an_out_of_fold_risk_and_five_drivers(cases):
    for case in cases["cases"]:
        assert 0.0 <= case["predicted_risk_36mo"] <= 1.0
        assert case["prediction_is_out_of_fold"] is True
        assert len(case["top_drivers"]) == 5
        assert 1 <= case["risk_decile"] <= 10


def test_cases_span_the_risk_distribution(cases):
    """A demo where every case is low-risk teaches a reviewer nothing."""
    deciles = {c["risk_decile"] for c in cases["cases"]}
    assert len(deciles) >= 8, f"only deciles {sorted(deciles)} represented"


def test_drivers_are_never_labelled_by_design_matrix_column(cases):
    """`sexmale`, `smokercurrent` etc. must not leak into a human-facing label."""
    leaked = {"sexmale", "smokercurrent", "smokerformer", "diabetesyes",
              "prior_chdyes", "prior_canceryes", "race_ethnh_asian"}
    for case in cases["cases"]:
        for driver in case["top_drivers"]:
            assert driver["variable"] not in leaked, (
                f"{case['case_id']} attributes to dummy column {driver['variable']}"
            )
            assert "=" in driver["statement"]


@pytest.mark.parametrize("variable,feature", [
    ("sex", "sex"), ("smoker", "smoker"), ("diabetes", "diabetes"),
    ("prior_chd", "prior_chd"), ("prior_cancer", "prior_cancer"),
])
def test_categorical_driver_states_this_participants_own_value(cases, variable, feature):
    """The bug this catches: reporting "Sex = male" for a female participant."""
    checked = 0
    for case in cases["cases"]:
        for driver in case["top_drivers"]:
            if driver["variable"] == variable:
                expected = str(case["features"][feature]).replace("_", " ")
                assert driver["value"] == expected, (
                    f"{case['case_id']}: driver says {variable}={driver['value']!r} "
                    f"but the participant is {expected!r}"
                )
                checked += 1
    if checked == 0:
        pytest.skip(f"{variable} is not a top driver for any sampled case")


def test_imputed_income_is_flagged_and_never_shown_as_measured(cases):
    for case in cases["cases"]:
        assert "income_ratio_imputed" in case["features"]
        if case["features"]["income_ratio_imputed"]:
            for driver in case["top_drivers"]:
                if driver["variable"] == "income_ratio":
                    assert "imputed" in driver["value"], (
                        f"{case['case_id']} presents an imputed income as measured"
                    )


def test_attribution_method_is_described(cases):
    text = cases["attribution"].lower()
    assert "log-odds" in text and "cohort mean" in text
