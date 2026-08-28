"""Checks on the scikit-learn baseline cross-check.

Pins the cohort size against the same exclusion rule R/04_models.R uses
(complete predictors, labelled endpoint, no 'unknown' category), and checks
the reported AUCs are in a plausible, non-degenerate range rather than
trusting the script blindly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "artifacts" / "sklearn_baseline.json"


@pytest.fixture(scope="module")
def card():
    if not CARD.exists():
        pytest.skip("artifacts/sklearn_baseline.json not generated yet "
                     "(run `make sklearn-baseline`)")
    return json.loads(CARD.read_text())


def test_cohort_matches_r_exclusion_rule(card):
    # Same 42-row exclusion (unknown category) and same complete-case rule
    # as R/04_models.R's modelling cohort: 4,906 of 5,459 analytic rows.
    assert card["cohort_size"] == 4906
    assert card["deaths_36mo"] == 151


def test_fold_note_states_independence(card):
    assert "not the same fold assignment" in card["fold_note"] or \
        "independent" in card["fold_note"].lower()


@pytest.mark.parametrize("model_name", [
    "Logistic Regression (scikit-learn)",
    "Random Forest (scikit-learn)",
])
def test_auc_is_plausible(card, model_name):
    results = {r["model"]: r for r in card["results"]}
    assert model_name in results
    auc = results[model_name]["auc"]
    # A working mortality model should clear 0.7; not near-perfect (which
    # would suggest a leaked label) and not near-random.
    assert 0.70 < auc < 0.95


def test_two_models_reported(card):
    assert len(card["results"]) == 2
