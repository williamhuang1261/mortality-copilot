"""Proves the core claim of this extension: changing a rule's effective-date
range (or adding a new version) never rewrites an audit entry that was
already written for a past date.

Every test points `log_path` and `rules_path` at `tmp_path` fixtures, so
none of them touch the real, committed `artifacts/audit_log.jsonl`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pipeline.audit import evaluate_and_log, read_audit_log

CASE = {
    "case_id": "case_001",
    "predicted_risk_36mo": 0.09,
    "features": {"diabetes": "no"},
}

RULES_V1_ONLY = """
- rule_id: high_risk_flag
  version: 1
  effective_from: "2026-01-01"
  effective_to: null
  condition:
    field: predicted_risk_36mo
    operator: gte
    value: 0.10
  reason: "risk {value:.4f} >= v1 threshold 0.10"
"""

# Same rule_id, but a v2 is now in force for dates on/after 2026-06-01,
# lowering the threshold to 0.08 -- which WOULD flip case_001's evaluation
# (0.09 >= 0.08) if it were applied to the earlier date.
RULES_V1_AND_V2 = """
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


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit_log.jsonl"


def test_a_later_rule_version_does_not_alter_a_past_audit_entry(tmp_path, log_path):
    past_date = date(2026, 3, 1)

    # Step A: evaluate and log the case as of a past date, under the rule
    # set that existed at the time (v1 only).
    rules_v1_path = tmp_path / "rules_v1.yaml"
    rules_v1_path.write_text(RULES_V1_ONLY)
    original_entries = evaluate_and_log(
        "case_001", past_date, [CASE], rules_v1_path, log_path
    )

    assert original_entries == [
        {
            "case_id": "case_001",
            "rule_id": "high_risk_flag",
            "version": 1,
            "effective_from": "2026-01-01",
            "effective_to": None,
            "as_of": "2026-03-01",
            "fired": False,  # 0.09 < 0.10
            "reason": "risk 0.0900 >= v1 threshold 0.10",
            "logged_at": original_entries[0]["logged_at"],
        }
    ]

    original_log_bytes = log_path.read_bytes()
    original_log_lines = original_log_bytes.decode().strip().splitlines()
    assert len(original_log_lines) == 1

    # Step B: a new rule version is added, effective from a LATER date
    # (2026-06-01), lowering the threshold to 0.08. If this were applied
    # retroactively to the March evaluation, case_001 (0.09) would flip
    # from not-fired to fired.
    rules_v1_and_v2_path = tmp_path / "rules_v1_and_v2.yaml"
    rules_v1_and_v2_path.write_text(RULES_V1_AND_V2)

    # Step C: re-evaluate and log the SAME case at the SAME past date,
    # under the now-amended rule set.
    replay_entries = evaluate_and_log(
        "case_001", past_date, [CASE], rules_v1_and_v2_path, log_path
    )

    # The replayed evaluation still resolves to v1 (the version whose
    # window covers 2026-03-01) and still does not fire -- adding a v2
    # effective later did not retroactively reinterpret this date.
    assert replay_entries[0]["version"] == 1
    assert replay_entries[0]["fired"] is False
    assert replay_entries[0]["reason"] == original_entries[0]["reason"]

    # The original bytes already on disk are untouched: the file only grew
    # by the newly-appended line, the first line is byte-identical.
    log_bytes_after = log_path.read_bytes()
    log_lines_after = log_bytes_after.decode().strip().splitlines()
    assert len(log_lines_after) == 2
    assert log_lines_after[0] == original_log_lines[0]
    assert log_bytes_after.startswith(original_log_bytes)

    # And read_audit_log's first entry -- the historical record -- still
    # matches exactly what was originally written.
    all_entries = read_audit_log(log_path)
    assert all_entries[0] == original_entries[0]


def test_a_case_evaluated_after_the_new_version_takes_effect_does_use_it(
    tmp_path, log_path
):
    """Sanity check the other direction: the new version DOES apply once
    its own effective window has begun -- immutability protects the past,
    it does not freeze the rule set forever."""
    rules_v1_and_v2_path = tmp_path / "rules_v1_and_v2.yaml"
    rules_v1_and_v2_path.write_text(RULES_V1_AND_V2)

    future_date = date(2026, 7, 1)
    entries = evaluate_and_log(
        "case_001", future_date, [CASE], rules_v1_and_v2_path, log_path
    )

    assert entries[0]["version"] == 2
    assert entries[0]["fired"] is True  # 0.09 >= 0.08
