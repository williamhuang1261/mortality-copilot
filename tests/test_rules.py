"""Tests for the versioned rule evaluator.

No Ollama, no faiss, no torch, no R: pure functions over small in-memory
fixtures and a tiny fixture YAML file, so this runs with only the core
requirements installed, same as every other test in this repo.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from pipeline.rules import (
    RuleError,
    active_version,
    evaluate_case,
    evaluate_rule,
    load_rules,
)

CASE_HIGH_RISK = {
    "case_id": "case_001",
    "predicted_risk_36mo": 0.15,
    "features": {"diabetes": "yes", "smoker": "current"},
}

CASE_LOW_RISK = {
    "case_id": "case_002",
    "predicted_risk_36mo": 0.03,
    "features": {"diabetes": "no", "smoker": "never"},
}

TWO_VERSIONS_YAML = """
- rule_id: high_risk_flag
  version: 1
  effective_from: "2026-01-01"
  effective_to: "2026-06-01"
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.10
  reason: "risk {value:.4f} >= v1 threshold 0.10"

- rule_id: high_risk_flag
  version: 2
  effective_from: "2026-06-01"
  effective_to: null
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.08
  reason: "risk {value:.4f} >= v2 threshold 0.08"
"""

OVERLAPPING_YAML = """
- rule_id: high_risk_flag
  version: 1
  effective_from: "2026-01-01"
  effective_to: "2026-07-01"
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.10
  reason: "v1 fired"

- rule_id: high_risk_flag
  version: 2
  effective_from: "2026-06-01"
  effective_to: null
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.08
  reason: "v2 fired"
"""


@pytest.fixture
def two_versions_path(tmp_path: Path) -> Path:
    p = tmp_path / "two_versions.yaml"
    p.write_text(TWO_VERSIONS_YAML)
    return p


@pytest.fixture
def overlapping_path(tmp_path: Path) -> Path:
    p = tmp_path / "overlapping.yaml"
    p.write_text(OVERLAPPING_YAML)
    return p


# ------------------------------------------------------------- load_rules


def test_load_rules_parses_the_real_shipped_file():
    from pipeline.rules import RULES_PATH

    rules = load_rules(RULES_PATH)
    assert len(rules) >= 3
    assert {r.rule_id for r in rules} >= {"high_risk_flag", "comorbidity_review_flag"}


def test_load_rules_rejects_unknown_operator(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
- rule_id: bad_rule
  version: 1
  effective_from: "2026-01-01"
  effective_to: null
  condition:
    field: predicted_risk_36mo
    operator: between
    value: 0.10
  reason: "nope"
"""
    )
    with pytest.raises(RuleError):
        load_rules(bad)


# ---------------------------------------------------------- active_version


def test_active_version_picks_the_version_covering_the_date(two_versions_path):
    rules = load_rules(two_versions_path)

    v1 = active_version("high_risk_flag", date(2026, 3, 1), rules)
    assert v1.version == 1

    v2 = active_version("high_risk_flag", date(2026, 6, 1), rules)
    assert v2.version == 2

    v2_later = active_version("high_risk_flag", date(2026, 12, 31), rules)
    assert v2_later.version == 2


def test_active_version_raises_when_no_version_covers_the_date(two_versions_path):
    rules = load_rules(two_versions_path)
    with pytest.raises(RuleError):
        active_version("high_risk_flag", date(2025, 1, 1), rules)


def test_active_version_raises_on_overlapping_windows(overlapping_path):
    rules = load_rules(overlapping_path)
    with pytest.raises(RuleError):
        active_version("high_risk_flag", date(2026, 6, 15), rules)


# ----------------------------------------------------------- evaluate_rule


def test_evaluate_rule_fires_and_formats_the_reason(two_versions_path):
    rules = load_rules(two_versions_path)
    rule = active_version("high_risk_flag", date(2026, 3, 1), rules)

    result = evaluate_rule(rule, CASE_HIGH_RISK)
    assert result.fired is True
    assert result.version == 1
    assert "0.1500" in result.reason


def test_evaluate_rule_does_not_fire_below_threshold(two_versions_path):
    rules = load_rules(two_versions_path)
    rule = active_version("high_risk_flag", date(2026, 3, 1), rules)

    result = evaluate_rule(rule, CASE_LOW_RISK)
    assert result.fired is False


def test_evaluate_rule_reads_dotted_feature_fields():
    from pipeline.rules import Rule

    diabetes_rule = Rule(
        rule_id="comorbidity_review_flag",
        version=1,
        effective_from=date(2026, 1, 1),
        effective_to=None,
        field="features.diabetes",
        op="eq",
        value="yes",
        reason_template="diabetes = {value}",
    )
    result = evaluate_rule(diabetes_rule, CASE_HIGH_RISK)
    assert result.fired is True
    assert result.reason == "diabetes = yes"


# ------------------------------------------------------------ evaluate_case


def test_evaluate_case_runs_every_rule_id_once(two_versions_path):
    rules = load_rules(two_versions_path)
    results = evaluate_case(CASE_HIGH_RISK, date(2026, 3, 1), rules)
    assert len(results) == 1
    assert results[0].rule_id == "high_risk_flag"
    assert results[0].version == 1


def test_evaluate_case_against_the_real_shipped_rules():
    from pipeline.rules import RULES_PATH

    rules = load_rules(RULES_PATH)
    results = evaluate_case(CASE_HIGH_RISK, date(2026, 8, 1), rules)
    by_id = {r.rule_id: r for r in results}

    assert by_id["high_risk_flag"].version == 2
    assert by_id["high_risk_flag"].fired is True
    assert by_id["comorbidity_review_flag"].fired is True

    low_risk_results = evaluate_case(CASE_LOW_RISK, date(2026, 8, 1), rules)
    by_id_low = {r.rule_id: r for r in low_risk_results}
    assert by_id_low["high_risk_flag"].fired is False
    assert by_id_low["comorbidity_review_flag"].fired is False
