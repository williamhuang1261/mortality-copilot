"""Tests for the append-only audit log.

Every test points `log_path` and `rules_path` at `tmp_path` fixtures, so
none of them touch the real, committed `artifacts/audit_log.jsonl`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pipeline.audit import evaluate_and_log, read_audit_log

RULES_YAML = """
- rule_id: high_risk_flag
  version: 1
  effective_from: "2026-01-01"
  effective_to: null
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.10
  reason: "risk {value:.4f} >= threshold 0.10"
"""

CASES = [
    {
        "case_id": "case_001",
        "predicted_risk_36mo": 0.15,
        "features": {"diabetes": "yes"},
    },
    {
        "case_id": "case_002",
        "predicted_risk_36mo": 0.03,
        "features": {"diabetes": "no"},
    },
]


@pytest.fixture
def rules_path(tmp_path: Path) -> Path:
    p = tmp_path / "rules.yaml"
    p.write_text(RULES_YAML)
    return p


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit_log.jsonl"


def test_evaluate_and_log_appends_one_entry_per_rule(rules_path, log_path):
    entries = evaluate_and_log("case_001", date(2026, 3, 1), CASES, rules_path, log_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["case_id"] == "case_001"
    assert entry["rule_id"] == "high_risk_flag"
    assert entry["version"] == 1
    assert entry["fired"] is True
    assert entry["as_of"] == "2026-03-01"
    assert "0.1500" in entry["reason"]


def test_evaluate_and_log_writes_to_the_file(rules_path, log_path):
    evaluate_and_log("case_001", date(2026, 3, 1), CASES, rules_path, log_path)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk["case_id"] == "case_001"
    assert on_disk["fired"] is True


def test_evaluate_and_log_appends_rather_than_overwrites(rules_path, log_path):
    evaluate_and_log("case_001", date(2026, 3, 1), CASES, rules_path, log_path)
    evaluate_and_log("case_002", date(2026, 3, 1), CASES, rules_path, log_path)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["case_id"] == "case_001"
    assert second["case_id"] == "case_002"
    assert second["fired"] is False


def test_read_audit_log_returns_entries_in_order(rules_path, log_path):
    evaluate_and_log("case_001", date(2026, 3, 1), CASES, rules_path, log_path)
    evaluate_and_log("case_002", date(2026, 3, 1), CASES, rules_path, log_path)

    entries = read_audit_log(log_path)
    assert len(entries) == 2
    assert [e["case_id"] for e in entries] == ["case_001", "case_002"]


def test_read_audit_log_returns_empty_list_when_no_file_exists(log_path):
    assert read_audit_log(log_path) == []
