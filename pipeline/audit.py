"""Append-only audit trail for versioned rule evaluations.

`evaluate_and_log()` evaluates every rule against a case as of a given date
and appends one JSON line per rule result to an audit log file. The only
file operation this module performs is opening the log in append ("a")
mode -- no function here reads a log file back in order to rewrite it, so a
later rule-data change (a new version, an edited effective-date range)
cannot retroactively alter an entry that has already been written.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.rules import RULES_PATH, Rule, evaluate_case, load_rules
from pipeline.tools import lookup_case

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "audit_log.jsonl"


def _audit_entry(
    case_id: str,
    as_of: date,
    rule: Rule,
    fired: bool,
    reason: str,
    logged_at: datetime,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "rule_id": rule.rule_id,
        "version": rule.version,
        "effective_from": rule.effective_from.isoformat(),
        "effective_to": rule.effective_to.isoformat() if rule.effective_to else None,
        "as_of": as_of.isoformat(),
        "fired": fired,
        "reason": reason,
        "logged_at": logged_at.isoformat(),
    }


def evaluate_and_log(
    case_id: str,
    as_of: date,
    cases: list[dict],
    rules_path: Path = RULES_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> list[dict[str, Any]]:
    """Evaluate every rule against `case_id` as of `as_of` and append the
    results to `log_path`. Returns the entries that were appended.

    `cases` is the already-loaded list from `artifacts/cases.json` (same
    shape `pipeline.tools.lookup_case` expects) -- this function does not
    read that file itself, so a caller controls exactly which case data
    a given evaluation used.
    """
    case = lookup_case(cases, case_id)
    rules = load_rules(rules_path)
    results = evaluate_case(case, as_of, rules)

    rules_by_id_version = {(r.rule_id, r.version): r for r in rules}
    logged_at = datetime.now(timezone.utc)

    entries = []
    for result in results:
        rule = rules_by_id_version[(result.rule_id, result.version)]
        entries.append(
            _audit_entry(case_id, as_of, rule, result.fired, result.reason, logged_at)
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return entries


def read_audit_log(log_path: Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """Read every entry ever appended to the audit log, in order."""
    if not log_path.exists():
        return []
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
