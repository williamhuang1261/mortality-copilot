"""Versioned eligibility / risk-flagging rules over case records.

Rule conditions live in `rules/eligibility_rules.yaml` as data, not code, so
a non-engineer could review them the way a compliance analyst would review a
policy document. A rule can carry several versions over time; each version
declares the date range during which it is the one in force
(`effective_from` inclusive, `effective_to` exclusive, or open-ended when
`effective_to` is null). `active_version()` looks up the single version whose
window covers a given date -- it never silently picks one when two versions'
windows overlap, since that would mean the rule data itself is wrong.

Nothing here fits a new model or reads `R/04_models.R`'s pinned numbers --
these functions only read case records already exported to
`artifacts/cases.json` by the existing pipeline.
"""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "eligibility_rules.yaml"

_OPERATORS = {
    "gte": _operator.ge,
    "lte": _operator.le,
    "gt": _operator.gt,
    "lt": _operator.lt,
    "eq": _operator.eq,
}


class RuleError(Exception):
    """Raised on a rule-data problem (bad operator, overlapping versions,
    unparseable date) -- never on a case simply not matching a condition."""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    version: int
    effective_from: date
    effective_to: date | None
    field: str
    op: str
    value: Any
    reason_template: str

    def covers(self, as_of: date) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to is not None and as_of >= self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    version: int
    fired: bool
    reason: str


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_rules(path: Path = RULES_PATH) -> list[Rule]:
    """Parse every rule version out of the YAML file."""
    raw = yaml.safe_load(path.read_text())
    if not raw:
        raise RuleError(f"{path} contains no rules.")

    rules: list[Rule] = []
    for entry in raw:
        condition = entry["condition"]
        if condition["operator"] not in _OPERATORS:
            raise RuleError(
                f"{entry['rule_id']} v{entry['version']}: unknown operator "
                f"{condition['operator']!r}."
            )
        effective_to = entry.get("effective_to")
        rules.append(
            Rule(
                rule_id=entry["rule_id"],
                version=entry["version"],
                effective_from=_parse_date(entry["effective_from"]),
                effective_to=_parse_date(effective_to) if effective_to else None,
                field=condition["field"],
                op=condition["operator"],
                value=condition["value"],
                reason_template=entry["reason"],
            )
        )
    return rules


def active_version(rule_id: str, as_of: date, rules: list[Rule]) -> Rule:
    """Return the single rule version whose effective window covers `as_of`.

    Raises RuleError if zero versions cover the date (the rule did not
    exist yet, or was retired) or if more than one does (an overlapping
    effective-date range is a data error in the YAML, not something to
    guess through).
    """
    candidates = [
        rule for rule in rules if rule.rule_id == rule_id and rule.covers(as_of)
    ]
    if not candidates:
        raise RuleError(f"No version of {rule_id!r} is active on {as_of}.")
    if len(candidates) > 1:
        versions = [c.version for c in candidates]
        raise RuleError(
            f"{len(candidates)} versions of {rule_id!r} are active on {as_of} "
            f"({versions}) -- overlapping effective-date ranges in the rule data."
        )
    return candidates[0]


def _lookup_field(case: dict, dotted_field: str) -> Any:
    value: Any = case
    for part in dotted_field.split("."):
        value = value[part]
    return value


def evaluate_rule(rule: Rule, case: dict) -> RuleResult:
    """Evaluate one rule version's condition against one case record."""
    field_value = _lookup_field(case, rule.field)
    fired = _OPERATORS[rule.op](field_value, rule.value)
    reason = rule.reason_template.format(value=field_value)
    return RuleResult(rule_id=rule.rule_id, version=rule.version, fired=fired, reason=reason)


def evaluate_case(case: dict, as_of: date, rules: list[Rule]) -> list[RuleResult]:
    """Evaluate every distinct rule_id's active version against one case."""
    rule_ids = sorted({rule.rule_id for rule in rules})
    results = []
    for rule_id in rule_ids:
        rule = active_version(rule_id, as_of, rules)
        results.append(evaluate_rule(rule, case))
    return results
