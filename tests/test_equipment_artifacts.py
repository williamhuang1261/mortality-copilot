"""Checks on the committed equipment artifacts, mirroring
tests/test_model_artifacts.py's pattern for the mortality domain.

These skip cleanly when the artifacts have not been generated (CI never runs
R), and otherwise check the two things a reviewer would actually rely on: the
model card states its limitations honestly, and every case's driver
decomposition reconstructs the health score exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "artifacts" / "equipment_model_card.json"
CASES = ROOT / "artifacts" / "equipment_cases.json"


@pytest.fixture(scope="module")
def card():
    if not CARD.exists():
        pytest.skip("artifacts/equipment_model_card.json not generated yet (run `make equipment-models`)")
    return json.loads(CARD.read_text())


@pytest.fixture(scope="module")
def cases():
    if not CASES.exists():
        pytest.skip("artifacts/equipment_cases.json not generated yet (run `make equipment-models`)")
    return json.loads(CASES.read_text())


# ------------------------------------------------------------- model card

@pytest.mark.parametrize("key", [
    "purpose", "outcome", "training_cohort", "external_holdout",
    "predictors", "validation", "coefficients", "limitations", "provenance",
])
def test_model_card_has_section(card, key):
    assert key in card and card[key]


def test_model_card_carries_the_simulated_data_disclaimer(card):
    purpose = card["purpose"].lower()
    assert "simulated" in purpose
    assert "must not be used" in purpose


def test_model_card_states_the_quasi_separation_limitation(card):
    """A GLM this discriminative on near-deterministic simulated data shows
    signs of quasi-complete separation -- the model card must say so rather
    than presenting the coefficient table as reliable effect sizes."""
    joined = " ".join(card["limitations"]).lower()
    assert "separation" in joined
    assert "not reliable effect-size" in joined


def test_model_card_states_the_proportional_hazards_violation(card):
    ph = card["validation"]["proportional_hazards"]
    assert len(ph["violations"]) > 0
    assert "structurally poor fit" in ph["interpretation"].lower()


def test_external_holdout_is_disjoint_from_training_cohort_by_construction(card):
    assert card["training_cohort"]["engines"] == 100
    assert card["external_holdout"]["engines"] == 100
    assert "none of these 100 engines" in card["external_holdout"]["note"].lower()


def test_dropped_predictors_match_the_measured_constant_sensors(card):
    dropped = set(card["predictors"]["dropped"])
    assert dropped == {"op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"}


# ------------------------------------------------------------------ cases

def test_every_case_is_labelled_external_holdout(cases):
    assert cases["n_cases"] == len(cases["cases"])
    assert all(c["prediction_is_external_holdout"] for c in cases["cases"])


def test_every_case_has_five_or_fewer_ranked_drivers(cases):
    for case in cases["cases"]:
        assert 1 <= len(case["top_drivers"]) <= 5


def test_case_health_scores_are_valid_probabilities(cases):
    for case in cases["cases"]:
        assert 0.0 <= case["asset_health_score"] <= 1.0


def test_driver_directions_agree_with_the_sign_of_their_contribution(cases):
    for case in cases["cases"]:
        for driver in case["top_drivers"]:
            if driver["contribution_log_odds"] > 0:
                assert driver["direction"] == "increases risk"
            else:
                assert driver["direction"] == "decreases risk"
